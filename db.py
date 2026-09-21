"""
Persistence layer -- hosted Postgres, not local SQLite, because this app
runs on free-tier hosting with ephemeral disk (a local file would be wiped
on every restart/sleep/redeploy). Set DATABASE_URL to a standard
connection string (postgresql://user:password@host:5432/dbname); works
with any Postgres host (Supabase, Neon, etc.).

Uses a connection pool, not one connection per call or a single shared
one: a Streamlit app serves concurrent users, so sharing one connection
isn't safe, and reconnecting to a remote DB on every call would be slow.
"""

import os
import urllib.parse

import pandas as pd
import psycopg2
import psycopg2.pool

_pool = None


def _get_pool():
    """DATABASE_URL is read here, not as a module-level constant, because
    this module is imported before app.py's load_dotenv() call runs --
    a module-level read would freeze in as None too early.

    The URL is parsed with urllib and passed to psycopg2 as separate
    keyword args, not as a raw connection string: passwords containing
    "%", "@", etc. (common in provider-generated passwords) aren't valid
    inside a postgresql:// URI unless percent-encoded, and psycopg2's own
    parser enforces that strictly. Keyword args skip URI parsing entirely,
    so any character in the password just works."""
    global _pool
    if _pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is not set -- point it at a Postgres connection "
                "string (e.g. from Supabase or Neon) before running the app."
            )
        parsed = urllib.parse.urlsplit(database_url)
        _pool = psycopg2.pool.ThreadedConnectionPool(
            1, 10,
            host=parsed.hostname,
            port=parsed.port or 5432,
            dbname=parsed.path.lstrip("/"),
            user=urllib.parse.unquote(parsed.username) if parsed.username else None,
            password=urllib.parse.unquote(parsed.password) if parsed.password else None,
        )
    return _pool


def get_connection():
    """Acquires a connection from the pool. Always pair with
    release_connection() in a try/finally -- this does not close the
    connection, it returns it to the pool for reuse."""
    return _get_pool().getconn()


def release_connection(conn):
    _get_pool().putconn(conn)


def init_db():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id SERIAL PRIMARY KEY,
                expert_id TEXT NOT NULL,
                forecast_date TEXT NOT NULL,
                timestamp_slot TEXT NOT NULL,
                forecast REAL NOT NULL,
                adjusted REAL NOT NULL,
                flagged INTEGER NOT NULL,
                confidence INTEGER,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (expert_id) REFERENCES users(username) ON DELETE CASCADE
            )
        """)
        # Backstop against a double-submit race (e.g. two tabs open on the
        # same date) -- has_submitted() alone can't fully prevent this.
        # Note: "ADD CONSTRAINT IF NOT EXISTS" isn't valid Postgres syntax;
        # this DO-block is the actual portable way to make it idempotent.
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'submissions_unique_slot'
                ) THEN
                    ALTER TABLE submissions ADD CONSTRAINT submissions_unique_slot
                        UNIQUE (expert_id, forecast_date, timestamp_slot);
                END IF;
            END $$;
        """)
        # One-time gate per user: research-purposes disclaimer (consented) and
        # the step-by-step tutorial (completed_tutorial).
        cur.execute("""
            CREATE TABLE IF NOT EXISTS onboarding_status (
                username TEXT PRIMARY KEY,
                consented INTEGER NOT NULL,
                completed_tutorial INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            )
        """)
        # One-time profile per user -- asked only once (see get_user_profile).
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                username TEXT PRIMARY KEY,
                epf_experience TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            )
        """)
        # Reflection survey shown right after a successful submission --
        # linked to (username, forecast_date) so it can be joined against
        # the "submissions" table's MAE evaluation later.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS submission_survey (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                forecast_date TEXT NOT NULL,
                usability_rating INTEGER NOT NULL,
                comprehension_rating INTEGER NOT NULL,
                context_relevance_rating INTEGER NOT NULL,
                comment TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            )
        """)
        conn.commit()
    finally:
        release_connection(conn)


def load_users():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT username, email, password_hash, role FROM users")
        rows = cur.fetchall()
    finally:
        release_connection(conn)
    return {u: {"email": e, "password": p, "role": r} for u, e, p, r in rows}


def save_new_user(username, password_hash, email, role):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s)",
            (username, email.strip().lower(), password_hash, role),
        )
        conn.commit()
    finally:
        release_connection(conn)


class DuplicateSubmissionError(Exception):
    """Raised when a (expert_id, forecast_date, timestamp_slot) submission
    already exists -- enforced by the submissions_unique_slot constraint."""


def save_submission(rows_df):
    conn = get_connection()
    try:
        cur = conn.cursor()
        for _, row in rows_df.iterrows():
            cur.execute(
                """
                INSERT INTO submissions
                    (expert_id, forecast_date, timestamp_slot, forecast, adjusted, flagged, confidence, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row["expert_id"], str(row["forecast_date"]), str(row["timestamp_slot"]),
                    float(row["forecast"]), float(row["adjusted"]), int(bool(row["flagged"])),
                    int(row["confidence"]),
                    row["timestamp"],
                ),
            )
        conn.commit()
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        raise DuplicateSubmissionError(
            "A submission for this expert and date already exists."
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def load_submissions(expert_id=None, forecast_date=None):
    """Full table by default (scoreboard/reveal pages need every row), but
    accepts optional filters so the review page -- which only ever needs
    one (expert, date) pair to restore an in-progress session -- doesn't
    pull every submission ever made just to check one."""
    conn = get_connection()
    try:
        query = "SELECT * FROM submissions"
        conditions, params = [], []
        if expert_id is not None:
            conditions.append("expert_id = %s")
            params.append(expert_id)
        if forecast_date is not None:
            conditions.append("forecast_date = %s")
            params.append(str(forecast_date))
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        df_submissions = pd.read_sql_query(query, conn, params=params)
    finally:
        release_connection(conn)
    if not df_submissions.empty:
        df_submissions["forecast_date"] = pd.to_datetime(df_submissions["forecast_date"]).dt.date
        df_submissions["timestamp_slot"] = pd.to_datetime(df_submissions["timestamp_slot"])
        df_submissions["flagged"] = df_submissions["flagged"].astype(bool)
    return df_submissions


def has_submitted(expert_id, forecast_date):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM submissions WHERE expert_id = %s AND forecast_date = %s",
            (expert_id, str(forecast_date)),
        )
        count = cur.fetchone()[0]
    finally:
        release_connection(conn)
    return count > 0


def get_user_profile(username):
    """Returns the stored epf_experience string for this user, or None if
    they've never answered it -- used to decide whether to ask the
    experience question again (skip if already answered once)."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT epf_experience FROM user_profile WHERE username = %s", (username,))
        row = cur.fetchone()
    finally:
        release_connection(conn)
    return row[0] if row else None


def save_user_profile(username, epf_experience, timestamp):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_profile (username, epf_experience, timestamp)
            VALUES (%s, %s, %s)
            ON CONFLICT (username) DO UPDATE SET
                epf_experience = EXCLUDED.epf_experience,
                timestamp = EXCLUDED.timestamp
            """,
            (username, epf_experience, timestamp),
        )
        conn.commit()
    finally:
        release_connection(conn)


def load_all_user_profiles():
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM user_profile", conn)
    finally:
        release_connection(conn)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def save_submission_survey(username, forecast_date, usability_rating, comprehension_rating,
                            context_relevance_rating, comment, timestamp):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO submission_survey
                (username, forecast_date, usability_rating, comprehension_rating, context_relevance_rating, comment, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (username, str(forecast_date), int(usability_rating), int(comprehension_rating),
             int(context_relevance_rating), comment, timestamp),
        )
        conn.commit()
    finally:
        release_connection(conn)


def load_submission_survey():
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM submission_survey ORDER BY timestamp DESC", conn)
    finally:
        release_connection(conn)
    if not df.empty:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def has_completed_survey(username):
    """True if this user has ever submitted the reflection survey, any
    date -- not just today's. Skipping never inserts a row, so a skip
    means it's offered again next time rather than marked done."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM submission_survey WHERE username = %s LIMIT 1", (username,))
        row = cur.fetchone()
    finally:
        release_connection(conn)
    return row is not None


def get_onboarding_status(username):
    """Returns (consented, completed_tutorial) as booleans, or (False, False)
    if this user has never started onboarding."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT consented, completed_tutorial FROM onboarding_status WHERE username = %s", (username,)
        )
        row = cur.fetchone()
    finally:
        release_connection(conn)
    if row is None:
        return False, False
    return bool(row[0]), bool(row[1])


def save_onboarding_status(username, consented, completed_tutorial, timestamp):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO onboarding_status (username, consented, completed_tutorial, timestamp)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (username) DO UPDATE SET
                consented = EXCLUDED.consented,
                completed_tutorial = EXCLUDED.completed_tutorial,
                timestamp = EXCLUDED.timestamp
            """,
            (username, int(consented), int(completed_tutorial), timestamp),
        )
        conn.commit()
    finally:
        release_connection(conn)