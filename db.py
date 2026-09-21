"""
Persistence layer -- hosted Postgres, not local SQLite, because this app
runs on free-tier hosting with ephemeral disk (a local file would be wiped
on every restart/sleep/redeploy). Set DATABASE_URL to a standard
connection string (postgresql://user:password@host:5432/dbname); works
with any Postgres host (Supabase, Neon, etc.).

Uses SQLAlchemy's engine-level connection pool, not psycopg2's own
ThreadedConnectionPool (an earlier version of this file did): psycopg2's
pool tracks checked-out connections by hand (an id(conn) -> key map) with
no validation step, which surfaced as intermittent "PoolError: trying to
put unkeyed connection" crashes under Streamlit's concurrent per-session
threads. SQLAlchemy's pool is the standard, thread-safe alternative, and
pool_pre_ping=True pings a connection before handing it out, so a
connection Supabase's pooler silently closed while idle gets transparently
replaced instead of surfacing as an error deep in a query.
"""

import os
import threading

import pandas as pd
import psycopg2
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

_engine = None
_engine_lock = threading.Lock()  # guards first-time engine creation, see _get_engine()


def _get_engine():
    """DATABASE_URL is read here, not as a module-level constant, because
    this module is imported before app.py's load_dotenv() call runs -- a
    module-level read would freeze in as None too early.

    The URL is parsed with urllib.parse, and username/password are handed
    to sqlalchemy.engine.URL.create() as plain (already-decoded) strings
    rather than re-assembled into a URI: URL.create() encodes them itself
    when it builds the DBAPI connection string, so a password containing
    "%", "!", "&", "$", etc. (common in provider-generated passwords)
    just works without needing to be percent-encoded first.

    Double-checked locking around creation: Streamlit runs each active
    session in its own thread of the same process, so two sessions could
    both see `_engine is None` at once with no lock and each build a
    separate Engine. The lock makes creation atomic; the `is None`
    re-check inside it means the lock is only ever taken once, on the
    very first call."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:  # re-check: another thread may have won the race while we waited
                database_url = os.getenv("DATABASE_URL")
                if not database_url:
                    raise RuntimeError(
                        "DATABASE_URL is not set -- point it at a Postgres connection "
                        "string (e.g. from Supabase or Neon) before running the app."
                    )
                import urllib.parse
                parsed = urllib.parse.urlsplit(database_url)
                url = sqlalchemy.engine.URL.create(
                    drivername="postgresql+psycopg2",
                    username=urllib.parse.unquote(parsed.username) if parsed.username else None,
                    password=urllib.parse.unquote(parsed.password) if parsed.password else None,
                    host=parsed.hostname,
                    port=parsed.port or 5432,
                    database=parsed.path.lstrip("/"),
                )
                _engine = sqlalchemy.create_engine(
                    url,
                    pool_size=5,
                    max_overflow=5,
                    pool_pre_ping=True,   # validates a connection before use -- see module docstring
                    pool_recycle=280,     # recycle before Supabase's own idle timeout can close it under us
                )
    return _engine


def init_db():
    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL
            )
        """))
        conn.execute(text("""
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
        """))
        # Backstop against a double-submit race (e.g. two tabs open on the
        # same date) -- has_submitted() alone can't fully prevent this.
        # Note: "ADD CONSTRAINT IF NOT EXISTS" isn't valid Postgres syntax;
        # this DO-block is the actual portable way to make it idempotent.
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'submissions_unique_slot'
                ) THEN
                    ALTER TABLE submissions ADD CONSTRAINT submissions_unique_slot
                        UNIQUE (expert_id, forecast_date, timestamp_slot);
                END IF;
            END $$;
        """))
        # One-time gate per user: research-purposes disclaimer (consented) and
        # the step-by-step tutorial (completed_tutorial).
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS onboarding_status (
                username TEXT PRIMARY KEY,
                consented INTEGER NOT NULL,
                completed_tutorial INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            )
        """))
        # One-time profile per user -- asked only once (see get_user_profile).
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_profile (
                username TEXT PRIMARY KEY,
                epf_experience TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            )
        """))
        # Reflection survey shown right after a successful submission --
        # linked to (username, forecast_date) so it can be joined against
        # the "submissions" table's MAE evaluation later.
        conn.execute(text("""
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
        """))
    # engine.begin() commits automatically here, on a clean exit of the block.


def load_users():
    with _get_engine().connect() as conn:
        rows = conn.execute(text("SELECT username, email, password_hash, role FROM users")).all()
    return {r.username: {"email": r.email, "password": r.password_hash, "role": r.role} for r in rows}


def save_new_user(username, password_hash, email, role):
    with _get_engine().begin() as conn:
        conn.execute(
            text("INSERT INTO users (username, email, password_hash, role) VALUES (:username, :email, :password_hash, :role)"),
            {"username": username, "email": email.strip().lower(), "password_hash": password_hash, "role": role},
        )


class DuplicateSubmissionError(Exception):
    """Raised when a (expert_id, forecast_date, timestamp_slot) submission
    already exists -- enforced by the submissions_unique_slot constraint."""


def save_submission(rows_df):
    engine = _get_engine()
    try:
        with engine.begin() as conn:
            for _, row in rows_df.iterrows():
                conn.execute(
                    text("""
                        INSERT INTO submissions
                            (expert_id, forecast_date, timestamp_slot, forecast, adjusted, flagged, confidence, timestamp)
                        VALUES (:expert_id, :forecast_date, :timestamp_slot, :forecast, :adjusted, :flagged, :confidence, :timestamp)
                    """),
                    {
                        "expert_id": row["expert_id"],
                        "forecast_date": str(row["forecast_date"]),
                        "timestamp_slot": str(row["timestamp_slot"]),
                        "forecast": float(row["forecast"]),
                        "adjusted": float(row["adjusted"]),
                        "flagged": int(bool(row["flagged"])),
                        "confidence": int(row["confidence"]),
                        "timestamp": row["timestamp"],
                    },
                )
        # engine.begin() rolls back automatically if an exception propagates
        # out of the block above, so no manual rollback is needed here.
    except IntegrityError as e:
        if isinstance(e.orig, psycopg2.errors.UniqueViolation):
            raise DuplicateSubmissionError(
                "A submission for this expert and date already exists."
            )
        raise


def load_submissions(expert_id=None, forecast_date=None):
    """Full table by default (scoreboard/reveal pages need every row), but
    accepts optional filters so the review page -- which only ever needs
    one (expert, date) pair to restore an in-progress session -- doesn't
    pull every submission ever made just to check one."""
    engine = _get_engine()
    query = "SELECT * FROM submissions"
    conditions, params = [], {}
    if expert_id is not None:
        conditions.append("expert_id = :expert_id")
        params["expert_id"] = expert_id
    if forecast_date is not None:
        conditions.append("forecast_date = :forecast_date")
        params["forecast_date"] = str(forecast_date)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    df_submissions = pd.read_sql_query(text(query), engine, params=params)
    if not df_submissions.empty:
        df_submissions["forecast_date"] = pd.to_datetime(df_submissions["forecast_date"]).dt.date
        df_submissions["timestamp_slot"] = pd.to_datetime(df_submissions["timestamp_slot"])
        df_submissions["flagged"] = df_submissions["flagged"].astype(bool)
    return df_submissions


def has_submitted(expert_id, forecast_date):
    with _get_engine().connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM submissions WHERE expert_id = :expert_id AND forecast_date = :forecast_date"),
            {"expert_id": expert_id, "forecast_date": str(forecast_date)},
        ).scalar()
    return count > 0


def get_user_profile(username):
    """Returns the stored epf_experience string for this user, or None if
    they've never answered it -- used to decide whether to ask the
    experience question again (skip if already answered once)."""
    with _get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT epf_experience FROM user_profile WHERE username = :username"), {"username": username}
        ).fetchone()
    return row[0] if row else None


def save_user_profile(username, epf_experience, timestamp):
    with _get_engine().begin() as conn:
        conn.execute(
            text("""
                INSERT INTO user_profile (username, epf_experience, timestamp)
                VALUES (:username, :epf_experience, :timestamp)
                ON CONFLICT (username) DO UPDATE SET
                    epf_experience = EXCLUDED.epf_experience,
                    timestamp = EXCLUDED.timestamp
            """),
            {"username": username, "epf_experience": epf_experience, "timestamp": timestamp},
        )


def load_all_user_profiles():
    df = pd.read_sql_query(text("SELECT * FROM user_profile"), _get_engine())
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def save_submission_survey(username, forecast_date, usability_rating, comprehension_rating,
                            context_relevance_rating, comment, timestamp):
    with _get_engine().begin() as conn:
        conn.execute(
            text("""
                INSERT INTO submission_survey
                    (username, forecast_date, usability_rating, comprehension_rating, context_relevance_rating, comment, timestamp)
                VALUES (:username, :forecast_date, :usability_rating, :comprehension_rating, :context_relevance_rating, :comment, :timestamp)
            """),
            {
                "username": username, "forecast_date": str(forecast_date),
                "usability_rating": int(usability_rating), "comprehension_rating": int(comprehension_rating),
                "context_relevance_rating": int(context_relevance_rating), "comment": comment, "timestamp": timestamp,
            },
        )


def load_submission_survey():
    df = pd.read_sql_query(text("SELECT * FROM submission_survey ORDER BY timestamp DESC"), _get_engine())
    if not df.empty:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def has_completed_survey(username):
    """True if this user has ever submitted the reflection survey, any
    date -- not just today's. Skipping never inserts a row, so a skip
    means it's offered again next time rather than marked done."""
    with _get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM submission_survey WHERE username = :username LIMIT 1"), {"username": username}
        ).fetchone()
    return row is not None


def get_onboarding_status(username):
    """Returns (consented, completed_tutorial) as booleans, or (False, False)
    if this user has never started onboarding."""
    with _get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT consented, completed_tutorial FROM onboarding_status WHERE username = :username"),
            {"username": username},
        ).fetchone()
    if row is None:
        return False, False
    return bool(row[0]), bool(row[1])


def save_onboarding_status(username, consented, completed_tutorial, timestamp):
    with _get_engine().begin() as conn:
        conn.execute(
            text("""
                INSERT INTO onboarding_status (username, consented, completed_tutorial, timestamp)
                VALUES (:username, :consented, :completed_tutorial, :timestamp)
                ON CONFLICT (username) DO UPDATE SET
                    consented = EXCLUDED.consented,
                    completed_tutorial = EXCLUDED.completed_tutorial,
                    timestamp = EXCLUDED.timestamp
            """),
            {"username": username, "consented": int(consented), "completed_tutorial": int(completed_tutorial), "timestamp": timestamp},
        )