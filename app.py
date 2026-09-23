"""
EPF Expert Review (RLHF) -- Streamlit dashboard for human-in-the-loop
review of day-ahead electricity price forecasts for the Belgian market.
"RLHF" in the UI title reflects the research framing (does human feedback
improve a model's output); code/variable names still say "EPF Expert
Review" as the literal description.

WHAT IT DOES
------------
A DNN produces a day-ahead forecast at 15-min resolution (96 slots/day).
Experts adjust it by dragging the curve (see draggable_curve/) and rate
their confidence. Once a delivery day settles, admins compare each
expert's adjustment against the realized price (MAE) and aggregate results
on a scoreboard. Anomaly flagging is automatic (5th/95th percentile of
that day's own forecast) -- no manual override.

ROLES
-----
- expert: Review & Adjust only, own submissions, final once submitted.
- admin: everything, plus Reveal & Evaluate and Expert Scoreboard; can
  view but never submit on an expert's behalf.

DATA SOURCES
------------
Two CSVs pulled live from GitHub each page load (cached 30 min): the DNN
forecast and realized Belgian market data (price, load, solar, wind,
weather).

UNCERTAINTY BAND
-----------------
The chart's band is a single 80%-coverage margin, conformal-calibrated
from the DNN's own settled residuals via Adaptive Conformal Inference
(get_aci_margin()) -- self-adjusting, no manual recalibration needed.

PAGES
-----
1. Review & Adjust -- the core workflow above.
2. Deterministic Forecast Analysis -- DNN vs. actual, last 14 days.
3. Reveal & Evaluate (admin) -- MAE for one submission once settled.
4. Expert Scoreboard (admin) -- aggregated MAE improvement per expert.
5. Survey Results (admin) -- the reflection survey + experience profiles;
   joinable against "submissions" on (username, forecast_date).

THEMING
-------
Dark mode only. apply_theme() injects CSS; themed() applies a matching
Plotly template -- both read the same get_palette(), so nothing falls
out of sync.
"""

import streamlit as st
import datetime as dt
import os
import io
import re
from collections import deque
from datetime import timedelta

import smtplib
from email.mime.text import MIMEText

import bcrypt
import holidays
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv

from db import (
    init_db, load_users, save_new_user, save_submission, load_submissions, has_submitted,
    DuplicateSubmissionError,
    get_user_profile, save_user_profile, load_all_user_profiles,
    save_submission_survey, load_submission_survey, has_completed_survey,
    get_onboarding_status, save_onboarding_status,
    get_last_emailed_days_reviewed, mark_results_emailed,
)
from draggable_curve import draggable_curve

load_dotenv()

# Run the Streamlit Dashboard using 'streamlit run app.py'


# --------------------------------------------------------------------------
# CACHED DB READS
#
# Several DB-touching functions were being called uncached on every rerun --
# not just the review page's drag reruns, but literally every rerun of the
# whole app (get_onboarding_status() runs unconditionally in main(), before
# routing to any page). Short TTLs remove that from the hot path;
# correctness doesn't suffer because each cache is cleared the moment the
# underlying write actually happens, below, and the submissions_unique_slot
# DB constraint is the real backstop against a double submit regardless of
# cache staleness.
# --------------------------------------------------------------------------

@st.cache_data(ttl=15)
def _cached_has_submitted(expert_id, forecast_date):
    return has_submitted(expert_id, forecast_date)


@st.cache_data(ttl=15)
def _cached_load_submissions(expert_id=None, forecast_date=None):
    """Same short-TTL treatment, plus scoping: passing expert_id/forecast_date
    avoids pulling the whole submissions table just to restore one
    in-progress session (see load_submissions() in db.py)."""
    return load_submissions(expert_id, forecast_date)


@st.cache_data(ttl=30)
def _cached_get_onboarding_status(username):
    """main() calls get_onboarding_status() unconditionally, before routing
    to ANY page -- so uncached, it was a live Postgres round-trip on every
    single rerun of the entire app (every page, every click, every drag),
    not just the review page. Cleared the moment onboarding is actually
    completed, in render_onboarding() below, so the gate lifts immediately
    rather than waiting out the TTL."""
    return get_onboarding_status(username)


@st.cache_data(ttl=30)
def _cached_load_users():
    """load_users() was uncached and called on every admin rerun of the
    review page (every drag) plus every login/register click. Cleared in
    auth_screen() right after a new account is created."""
    return load_users()

st.set_page_config(page_title='EPF Expert Review', layout='wide')
st.title('Electricity Price Forecasting - RLHF')


# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

# GitHub repo that publishes the forecast/actuals CSVs this app consumes.
# Read access requires GITHUB_TOKEN (set in .env / environment); the repo is
# private, hence the auth header in fetch_csv_from_github() below.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = "margaridamascarenhas"
GITHUB_REPO = "DAM_Forecast_V4"
GITHUB_BRANCH = "main"

DNN_FILE = "DNN_forecasts_10AM.csv"     # DNN point forecast + Imputed flag
BE_DATA_FILE = "Data_BE_UTC.csv"        # Realized price, load, weather, renewables

STEPS_PER_DAY = 96  # 15-minute resolution: 24h * 4

EXPERT_ROLES = ["expert"]  # roles selectable at self-registration (no public admin signup)

# Results-email notifications (see maybe_send_results_email()): sent via
# Gmail SMTP using an App Password, not the account password -- see
# .env.example. Both must be set for sending to actually happen; if either
# is missing, maybe_send_results_email() no-ops rather than raising, so a
# missing config never breaks the app for anyone.
GMAIL_SENDER_ADDRESS = os.getenv("GMAIL_SENDER_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
RESULTS_EMAIL_THRESHOLD_DAYS = 10  # send once an expert has *more than* this many evaluated days
APP_URL = "https://epf-dashboard-expert-review.onrender.com/"  # linked from the results email



# --------------------------------------------------------------------------
# THEME (dark mode only)
#
# get_palette() is the single source of truth for both layers below:
#   - apply_theme() uses it to inject CSS that themes Streamlit's own chrome
#     (sidebar, buttons, inputs, popovers, calendar, icons, etc.)
#   - themed() uses it to give every Plotly chart matching colors, so charts
#     never look out of sync with the rest of the page.
# A light-mode toggle/palette used to exist here; removed by request -- the
# app is dark-only now, so there's nothing left to keep in sync.
# --------------------------------------------------------------------------


def get_palette() -> dict:
    """Single source of truth for theme colors, shared by the injected CSS
    and the Plotly charts, so both layers always agree on what's readable."""
    return {
        "bg": "#0e1117",
        "bg_secondary": "#161b22",
        "sidebar_bg": "#161b22",
        "card_bg": "#1c222b",
        "text": "#e6edf3",
        "text_muted": "#9aa5b1",
        "border": "#2d3540",
        "grid": "#2d3540",
        "input_bg": "#1c222b",
        "accent": "#6366f1",
        "accent_text": "#ffffff",
    }


def apply_theme():
    """Injects the (dark-only) theme CSS. Call this once, first thing, in main()."""
    palette = get_palette()

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background-color: {palette['bg']};
            color: {palette['text']};
        }}
        [data-testid="stSidebar"] {{
            background-color: {palette['sidebar_bg']};
            border-right: 1px solid {palette['border']};
        }}
        [data-testid="stSidebar"] * {{
            color: {palette['text']} !important;
        }}
        h1, h2, h3, h4, h5, h6, p, label, span, li, .stMarkdown {{
            color: {palette['text']};
        }}
        [data-testid="stMetric"] {{
            background-color: {palette['card_bg']};
            border: 1px solid {palette['border']};
            border-radius: 10px;
            padding: 0.75rem 1rem;
        }}
        [data-testid="stMetricLabel"] {{
            color: {palette['text_muted']} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {palette['text']} !important;
        }}
        [data-testid="stExpander"], [data-testid="stForm"] {{
            background-color: {palette['card_bg']};
            border: 1px solid {palette['border']};
            border-radius: 10px;
        }}
        div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="textarea"] {{
            background-color: {palette['input_bg']} !important;
            border-color: {palette['border']} !important;
        }}
        input, textarea {{
            background-color: {palette['input_bg']} !important;
            color: {palette['text']} !important;
        }}
        .stButton > button, .stFormSubmitButton > button {{
            background-color: {palette['accent']};
            color: {palette['accent_text']};
            border: none;
            border-radius: 8px;
        }}
        .stButton > button:hover, .stFormSubmitButton > button:hover {{
            filter: brightness(1.1);
        }}
        hr {{
            border-color: {palette['border']};
        }}
        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
            border: 1px solid {palette['border']};
            border-radius: 8px;
        }}
        /* Fallback only: every success/warning/error/info message in the
        app is actually rendered through render_banner() (colored per
        kind, see BANNER_COLORS), not st.success()/st.warning()/etc., so
        this rule normally never applies. Left in place in case any native
        Streamlit alert ever shows up (e.g. a library/future code path
        that isn't routed through render_banner()), so it isn't left
        unthemed rather than colored wrong. */
        [data-testid="stAlert"] {{
            background-color: {palette['card_bg']};
            color: {palette['text']} !important;
            border: 1px solid {palette['border']};
        }}
        [data-testid="stAlert"] * {{
            color: {palette['text']} !important;
        }}
        /* Selectbox/radio dropdown menus render in a portal attached to
           <body>, outside the sidebar/app containers above, so they need
           their own rule or their text is invisible against the popup's
           own background. */
        div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {{
            background-color: {palette['card_bg']} !important;
        }}
        div[data-baseweb="popover"] *, div[data-baseweb="menu"] *, ul[role="listbox"] * {{
            color: {palette['text']} !important;
        }}
        li[role="option"]:hover, li[aria-selected="true"] {{
            background-color: {palette['bg_secondary']} !important;
        }}
        /* The date picker's month/year header lives in a separate baseweb
        wrapper from the day grid itself — cover both, or the header text
        stays stuck on its default color regardless of mode. Background is
        applied to every descendant, not just the outer container: unlike
        color, background-color doesn't cascade down through nested elements
        -- a sub-element with its own background (e.g. the header bar) keeps
        it regardless of what the parent container is set to. */
        div[data-baseweb="datepicker"], div[data-baseweb="calendar"],
        div[data-baseweb="datepicker"] *, div[data-baseweb="calendar"] * {{
            background-color: {palette['card_bg']} !important;
        }}
        div[data-baseweb="datepicker"] *, div[data-baseweb="calendar"] * {{
            color: {palette['text']} !important;
        }}
        div[data-baseweb="calendar"] [role="gridcell"] > div {{
            background-color: transparent !important;
        }}
        /* Icon glyphs (password show/hide, calendar nav arrows, expander
        chevrons, sidebar icons) are SVGs with their own fixed color that
        doesn't follow the page text color automatically — scoped to
        Streamlit's own UI chrome only, never the Plotly charts, which
        manage their own colors via the template. */
        button svg, [role="button"] svg,
        div[data-baseweb="input"] svg, div[data-baseweb="select"] svg,
        div[data-baseweb="popover"] svg, div[data-baseweb="calendar"] svg,
        div[data-baseweb="datepicker"] svg,
        [data-testid="stExpander"] svg, [data-testid="stSidebar"] svg {{
            fill: {palette['text']} !important;
            stroke: {palette['text']} !important;
        }}
        /* Checkbox, toggle, and radio "selected" color. Streamlit's own
        React Aria-based widgets leave a checked/selected control's fill
        color to Streamlit's stock theme (#FF4B4B red) unless overridden --
        so without this rule, a checked checkbox, an "on" toggle, and the
        selected radio dot (including the sidebar page picker) all show
        that default red instead of this app's accent color. This used to
        be targeted via Streamlit's auto-generated Emotion class names, but
        those are tied to the exact Streamlit build and silently stop
        matching on any version bump -- which is exactly what happened once
        already. `data-selected="true"` is a stable attribute Streamlit
        sets on these widgets itself, so these rules key off that plus each
        widget's structural position instead. Verified directly against the
        live rendered DOM (right-click, then Inspect, on a checked/selected
        element, if this ever needs re-checking). */
        [data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type {{
            background-color: {palette['accent']} !important;
            border-color: {palette['accent']} !important;
        }}
        label[data-testid="stRadioOption"][data-selected="true"] > div:first-of-type > div:first-child {{
            background-color: {palette['accent']} !important;
            border-color: {palette['accent']} !important;
        }}
        /* The (?) help-tooltip icon next to metrics/sliders/etc. Verified
        against the actual rendered DOM: Streamlit's real testid here is
        "stTooltipHoverTarget", not "stTooltipIcon" (a stale/incorrect
        selector that never matched anything, in any Streamlit version
        tested) -- so this icon was silently falling back to Streamlit's
        own default stroke color, a dark charcoal meant for a light
        background. On this app's dark background that reads as a faint,
        low-contrast smudge -- easy to mistake for a plain dot rather than
        a "?". Setting a real, theme-matched stroke color that actually
        targets the element fixes that. */
        [data-testid="stTooltipHoverTarget"] svg {{
            fill: none !important;
            stroke: {palette['text_muted']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def current_plotly_template():
    """Plotly template name. Dark-only app, so this is always "plotly_dark"."""
    return "plotly_dark"


def themed(figure):
    """Applies the dark Plotly template and makes the chart background
    transparent so it blends with the page. Text, legend, and gridline colors
    are set explicitly (not just left to the template default) so they can
    never end up washed out or invisible."""
    palette = get_palette()

    figure.update_layout(
        template=current_plotly_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=palette["text"]),
        legend=dict(
            font=dict(color=palette["text"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor=palette["grid"],
            zerolinecolor=palette["grid"],
            linecolor=palette["border"],
            tickfont=dict(color=palette["text_muted"]),
            title_font=dict(color=palette["text"]),
        ),
        yaxis=dict(
            gridcolor=palette["grid"],
            zerolinecolor=palette["grid"],
            linecolor=palette["border"],
            tickfont=dict(color=palette["text_muted"]),
            title_font=dict(color=palette["text"]),
        ),
        yaxis2=dict(
            tickfont=dict(color=palette["text_muted"]),
            title_font=dict(color=palette["text"]),
        ),
        hoverlabel=dict(
            font=dict(color=palette["text"]),
            bgcolor=palette["card_bg"],
            bordercolor=palette["border"],
        ),
    )
    return figure


# --------------------------------------------------------------------------
# GITHUB DATA LOADING (live, cached with a TTL so daily 10AM/14h updates get picked up)
# --------------------------------------------------------------------------

def fetch_csv_from_github(owner, repo, branch, path, fname, token, usecols=None):
    """Downloads a single CSV file's raw content from a GitHub repo and
    parses it into a DataFrame. `token` is required for private repos (like
    this one) -- passed as a Bearer token; requests with no token still work
    against public repos but will 404/403 here since GITHUB_REPO is private."""
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/{fname}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(raw_url, headers=headers, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), usecols=usecols)


@st.cache_data(ttl=1800)  # 30 min -- catches both the 10AM and 14h daily refreshes without hammering GitHub
def get_dnn_df():
    """DNN point forecast: one row per 15-min slot, with DateTime,
    DNN_expanding (the forecast value), and Imputed (True = that day's model
    run failed and the previous day's forecast was carried forward)."""
    # Only 3 columns exist in this file already -- nothing to trim.
    df = fetch_csv_from_github(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, "Forecast", DNN_FILE, GITHUB_TOKEN)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["date_only"] = df["DateTime"].dt.date
    return df.sort_values("DateTime").reset_index(drop=True)



@st.cache_data(ttl=1800)
def get_be_df():
    """Realized Belgian market data: settled day-ahead price plus load,
    solar, wind, and weather -- everything needed for the context metrics,
    Reveal & Evaluate's "actual" price, and the Solar/Wind and Weather
    sub-charts on Review & Adjust."""
    # Already a lean 9-column file -- nothing to trim.
    df = fetch_csv_from_github(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, "datasets", BE_DATA_FILE, GITHUB_TOKEN)
    df["Date"] = pd.to_datetime(df["Date"])
    df["date_only"] = df["Date"].dt.date
    return df.sort_values("Date").reset_index(drop=True)


init_db()  # creates users/feedback tables in SQLite if they don't exist yet


# --------------------------------------------------------------------------
# SHARED HELPER FUNCTIONS
# --------------------------------------------------------------------------

def get_available_dates(dnn_df):
    """Dates with a complete 96-slot DNN forecast. A date with fewer rows
    means the model run for that day is only partially present (e.g. still
    in progress or truncated) -- excluded so the UI never shows a half-day."""
    counts = dnn_df.groupby("date_only").size()
    return sorted(counts[counts == STEPS_PER_DAY].index)


def dnn_forecast(forecast_date, dnn_df):
    """Returns (values_array, timestamps) for a 96-slot day, or (None, None)
    if that date doesn't have a complete forecast."""
    day_rows = dnn_df[dnn_df["date_only"] == forecast_date].sort_values("DateTime")
    if len(day_rows) != STEPS_PER_DAY:
        return None, None
    return day_rows["DNN_expanding"].values, day_rows["DateTime"].values


def dnn_imputed_flags(forecast_date, dnn_df):
    """Returns the per-slot Imputed flag array for the day, or None. All 96
    values are identical in practice (the flag is set at the day level, just
    stored per-slot) -- .any() is enough to know if the whole day was imputed."""
    day_rows = dnn_df[dnn_df["date_only"] == forecast_date].sort_values("DateTime")
    if len(day_rows) != STEPS_PER_DAY:
        return None
    return day_rows["Imputed"].values


def _forecast_vs_actual(dnn_df, be_df):
    """Settled (forecast, actual) pairs, shared prep for both conformal
    methods below. Excludes the unsettled current day.

    Drops rows with a missing value rather than leaving NaN: ACI's residual
    pool is a fixed-size rolling window, so one NaN poisons every
    np.quantile() call until it slides back out -- possibly the last one,
    silently returning a NaN margin overall."""
    last_evaluable = get_last_evaluable_ts()
    merged = (
        dnn_df.set_index("DateTime")["DNN_expanding"].rename("forecast")
        .to_frame()
        .join(be_df.set_index("Date")["Price"].rename("actual"), how="inner")
        .sort_index()
    )
    merged = merged.loc[merged.index <= last_evaluable]
    return merged.dropna(subset=["forecast", "actual"])


@st.cache_data(ttl=1800)
def get_aci_margin(dnn_df, be_df, alpha=0.2, gamma=0.01, calibration_days=30, min_calibration_days=5):
    """Adaptive Conformal Inference margin (ported from method_ACI() in the
    conformal-prediction notebook): after an initial calibration window,
    replays every settled slot, self-correcting a target quantile
    (alpha_t) based on whether each prediction covered the real price.
    Returns the margin as of the end of that replay -- applied to the
    currently-reviewed (unsettled) date.

    Calibration window is adaptive, not a hard 30 days: shrinks down to
    `min_calibration_days` rather than refusing a band just because the
    model is new and lacks history. At least one day is always reserved
    for the replay itself.

    Sequential by nature (each step depends on the last), so it can't be
    vectorized -- the ttl cache avoids recomputing it on every interaction.
    Returns None only if there's under min_calibration_days + 1 days total."""
    merged = _forecast_vs_actual(dnn_df, be_df)

    total_days = len(merged) // STEPS_PER_DAY
    if total_days < min_calibration_days + 1:
        return None

    cal_days = min(calibration_days, total_days - 1)  # leave >=1 day for the replay
    n_cal = cal_days * STEPS_PER_DAY

    cal, val = merged.iloc[:n_cal], merged.iloc[n_cal:]
    eps_pool = deque(np.abs(cal["actual"].values - cal["forecast"].values))
    alpha_t = alpha
    q = None

    for mu, y in zip(val["forecast"].values, val["actual"].values):
        q = np.quantile(np.array(eps_pool), 1 - alpha_t, method="linear")
        covered = (y >= mu - q) and (y <= mu + q)
        alpha_t = np.clip(alpha_t + gamma * (alpha - (0 if covered else 1)), 1e-6, 1 - 1e-6)
        eps_pool.append(abs(y - mu))
        eps_pool.popleft()

    return float(q) if q is not None else None


@st.cache_data(ttl=3600)  # holidays never change intra-session; long TTL just bounds process memory
def get_calendar_context(forecast_date):
    """Belgian holiday/bridge-day context -- both are known drivers of
    unusual demand/price shapes. A "bridge day" is a working day between a
    holiday and a weekend, often behaving like a de facto holiday.

    Was rebuilding a holidays.Belgium() calendar object from scratch on
    every rerun of the review page (every drag release) -- pure CPU work,
    no DB involved, but still real per-rerun cost for something that never
    changes."""
    be_holidays = holidays.Belgium(years=[forecast_date.year - 1, forecast_date.year, forecast_date.year + 1])
    is_holiday = forecast_date in be_holidays

    is_bridge_day = False
    if forecast_date.weekday() == 0:  # Monday
        tuesday = forecast_date + timedelta(days=1)
        is_bridge_day = tuesday in be_holidays
    elif forecast_date.weekday() == 4:  # Friday
        thursday = forecast_date - timedelta(days=1)
        is_bridge_day = thursday in be_holidays

    return {
        "is_holiday": is_holiday,
        "holiday_name": be_holidays.get(forecast_date, ""),
        "day_of_week": forecast_date.strftime("%a"),
        "is_bridge_day": is_bridge_day,
    }


def get_dnn_history_window(dnn_df, be_df, days=14):
    """DNN forecast vs. actual, last `days` days -- windowed the same way
    as Margarida's dashboard_app.py Section 1, DNN-only (not her scatter
    diagnostics or MAE tables).

    Two deliberate, non-obvious choices:
      - Window ends at the latest forecast timestamp, not the latest
        settled day -- so it can include tomorrow's forecast, same as hers.
      - Actual is reindexed onto forecast's dates, not inner-joined, so an
        unsettled day still shows its forecast line (with a gap where the
        actual isn't in yet) instead of disappearing entirely.
    """
    forecast = dnn_df.set_index("DateTime")["DNN_expanding"].rename("forecast").sort_index()
    if forecast.empty:
        return pd.DataFrame(columns=["DateTime", "forecast", "actual"])

    plot_end = forecast.index.max()
    plot_start = plot_end - pd.Timedelta(days=days)
    forecast_window = forecast.loc[(forecast.index >= plot_start) & (forecast.index <= plot_end)]

    actual = be_df.set_index("Date")["Price"].rename("actual")
    actual = actual[~actual.index.duplicated(keep="last")].sort_index()
    actual_window = actual.reindex(forecast_window.index)

    history_df = pd.concat([forecast_window, actual_window], axis=1).reset_index()
    return history_df.rename(columns={"index": "DateTime"})


def make_simple_chart(timestamps, values, y_title):
    """A single-series hourly chart used for the Weather sub-chart
    (Temperature or Humidity, one at a time, selected via a dropdown)."""

    ts_min = pd.Timestamp(np.asarray(timestamps).min())
    ts_max = pd.Timestamp(np.asarray(timestamps).max())

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=timestamps, y=values, mode="lines+markers", name=y_title, marker=dict(size=4)))
    figure.update_layout(
        xaxis_title="Time of day",
        yaxis_title=y_title,
        xaxis=dict(tickformat="%H:%M", dtick=3600000, range=[ts_min, ts_max]),
    )
    return themed(figure)


def make_renewables_chart(timestamps, solar=None, wind=None):
    """Standalone solar + wind chart -- both in MW, so they share one axis
    (unlike the main chart's secondary axis, this one doesn't need to share
    space with a EUR/MWh price series)."""
    figure = go.Figure()
    if solar is not None:
        figure.add_trace(go.Scatter(
            x=timestamps, y=solar, mode="lines", name="Solar (MW)",
            line=dict(color="#eab308", width=2.5),
            fill="tozeroy", fillcolor="rgba(234,179,8,0.15)",
        ))
    if wind is not None:
        figure.add_trace(go.Scatter(
            x=timestamps, y=wind, mode="lines", name="Wind total (MW)",
            line=dict(color="#14b8a6", width=2.5),
            fill="tozeroy", fillcolor="rgba(20,184,166,0.15)",
        ))
    ts_min = pd.Timestamp(np.asarray(timestamps).min())
    ts_max = pd.Timestamp(np.asarray(timestamps).max())
    figure.update_layout(
        xaxis_title="Time of day",
        yaxis_title="MW",
        xaxis=dict(tickformat="%H:%M", dtick=3600000, range=[ts_min, ts_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=40, l=50, r=20),
        height=280, 
        hovermode="x unified",
    )
    return themed(figure)


def make_comparison_chart(timestamps, forecast, adjusted, band_lower=None, band_upper=None, flagged=None):
    """Forecast vs. adjusted with the uncertainty band and flagged anomaly
    markers -- the read-only counterpart to draggable_curve, for
    admin-viewed or already-submitted work where dragging doesn't apply."""
    figure = go.Figure()

    if band_lower is not None and band_upper is not None:
        figure.add_trace(go.Scatter(x=timestamps, y=band_upper, mode="lines", line=dict(width=0), showlegend=False))
        figure.add_trace(go.Scatter(
            x=timestamps, y=band_lower, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(100,100,255,0.35)", name="80% interval (ACI)",
        ))

    figure.add_trace(go.Scatter(x=timestamps, y=forecast, mode="lines", name="DNN Forecast",
                                 line=dict(width=2, dash="dash")))
    figure.add_trace(go.Scatter(x=timestamps, y=adjusted, mode="lines+markers", name="Expert Adjusted",
                                 marker=dict(size=4)))

    # Anomaly markers: slots outside this day's own 5th/95th percentile
    # (computed by the caller, see the flagging comment in page_review_and_adjust).
    if flagged is not None and np.any(flagged):
        flagged = np.asarray(flagged)
        ts_arr = np.asarray(timestamps)
        adj_arr = np.asarray(adjusted)
        figure.add_trace(go.Scatter(
            x=ts_arr[flagged], y=adj_arr[flagged], mode="markers", name="Flagged (5th/95th pct)",
            marker=dict(size=6, symbol="diamond", color="orange", line=dict(color="white", width=1)),
        ))

    ts_min = pd.Timestamp(np.asarray(timestamps).min())
    ts_max = pd.Timestamp(np.asarray(timestamps).max())
    figure.update_layout(
        xaxis_title="Time of day",
        yaxis_title="EUR / MWh",
        xaxis=dict(tickformat="%H:%M", dtick=3600000, range=[ts_min, ts_max]),
        showlegend=False,  # replaced by render_chart_legend() below -- see its docstring
    )
    return themed(figure)


def render_chart_legend(show_forecast=True, show_band=True, show_flagged=True):
    """Custom HTML legend matching draggable_curve's own hand-built one --
    same colors/shapes/layout, so editing (drag widget) and read-only
    (static Plotly chart) look consistent. Plotly's native legend is off in
    make_comparison_chart() in favor of this; kept in sync by hand with
    DraggableCurve.tsx's version, not shared code."""
    palette = get_palette()

    items = ['<span style="display:flex;align-items:center;gap:5px;">'
             '<span style="width:14px;height:3px;background:#6366f1;display:inline-block;border-radius:1px;"></span>'
             'Adjusted</span>']
    if show_forecast:
        items.append('<span style="display:flex;align-items:center;gap:5px;">'
                      f'<span style="width:14px;height:0;border-top:2px dashed {palette["text_muted"]};display:inline-block;opacity:0.6;"></span>'
                      'DNN Forecast</span>')
    if show_band:
        items.append('<span style="display:flex;align-items:center;gap:5px;">'
                      '<span style="width:14px;height:10px;background:rgba(100,100,255,0.35);display:inline-block;border-radius:2px;"></span>'
                      '80% interval (ACI)</span>')
    if show_flagged:
        items.append('<span style="display:flex;align-items:center;gap:5px;">'
                      '<span style="width:9px;height:9px;background:#f59e0b;border-radius:50%;display:inline-block;"></span>'
                      'Flagged (5th/95th pct)</span>')

    html = (f'<div style="display:flex;flex-wrap:wrap;gap:14px;font-size:12px;'
            f'color:{palette["text_muted"]};margin-top:6px;">' + "".join(items) + '</div>')
    st.markdown(html, unsafe_allow_html=True)


def make_history_chart(history_df):
    """DNN vs. actual over a multi-day window, matching Margarida's
    dashboard_app.py look (terracotta line, unified hover, range slider) --
    but 'Actual' uses the theme's own text color instead of her hardcoded
    near-black, so it stays visible against the dark background."""
    palette = get_palette()

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=history_df["DateTime"], y=history_df["actual"],
                                 mode="lines", name="Actual",
                                 line=dict(width=2.4, color=palette["text"])))
    figure.add_trace(go.Scatter(x=history_df["DateTime"], y=history_df["forecast"],
                                 mode="lines", name="DNN forecast",
                                 line=dict(width=2.2, color="#C97B63")))  # Margarida's DNN color
    figure.update_layout(
        title=dict(text="DNN vs Actual", x=0.01, xanchor="left", font=dict(size=20)),
        xaxis_title="DateTime", yaxis_title="EUR / MWh",
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    return themed(figure)


# --------------------------------------------------------------------------
# AUTH HELPERS
# --------------------------------------------------------------------------

def hash_password(password):
    """One-way bcrypt hash for storing a new user's password. bcrypt embeds
    its own random salt in the output, so no separate salt column is needed
    in the users table -- check_password() below re-derives it from the hash."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def check_password(password, hashed_password):
    """Verifies a login attempt against the stored bcrypt hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))


def validate_registration(username, email, password):
    """Self-registration field checks. Returns (is_valid, error_message) --
    error_message is empty when is_valid is True. Duplicate username/email
    checks happen separately in auth_screen() since they need a DB lookup."""
    if not username or not email or not password:
        return False, "All fields (Username, Email, and Password) are required."
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "Please enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
        return False, "Password must contain both letters and numbers."
    return True, ""


# --------------------------------------------------------------------------
# PAGE 1: REVIEW & ADJUST
# --------------------------------------------------------------------------

def render_submission_survey(expert_id, forecast_date, survey_key):
    """One-time-ever reflection survey, shown after a user's first
    submission and never again (gated by has_completed_survey() in the
    caller, checked across all submissions, not just today's).

    Research purpose: usability/comprehension scores matter because a low
    one casts doubt on whether an adjustment reflects genuine judgment;
    context-relevance tests whether the weather/renewables data actually
    helped or was decoration. Experience level is a stable trait, asked
    once, not a per-session measure.

    Confidence is unaffected -- it's on a separate table ("submissions"),
    tied to each forecast_date, asked every time.

    Skippable, not mandatory (a required form would hurt response quality
    more than a miss costs). Skipping writes no row, so it's offered again
    next time rather than marked done."""
    st.divider()
    st.subheader("Quick reflection")
    st.caption("A few questions for the research behind this tool -- not about the forecast itself.")

    existing_experience = get_user_profile(expert_id)
    ask_experience = existing_experience is None

    with st.form(key=f"survey_form_{survey_key}"):
        if ask_experience:
            experience = st.select_slider(
                "How would you describe your experience with electricity price forecasting?",
                options=["None", "Some familiarity", "Experienced", "Expert"],
                value="Some familiarity",
            )

        usability = st.slider(
            "How easy was it to use this application for this task?", 
            1, 5, 3,
            help="1 = Very difficult | 5 = Very easy"
        )

        comprehension = st.slider(
            "How well did you understand the forecast chart and uncertainty band (shaded region)?", 
            1, 5, 3,
            help="1 = Did not understand at all | 5 = Understood perfectly"
        )

        context_relevance = st.slider(
            "How useful was the additional context (temperature, humidity, solar/wind) for making your adjustment?",
            1, 5, 3,
            help="1 = Not at all useful | 5 = Extremely useful"
        )
        comment = st.text_area("Anything else you'd like to share about this session? (optional)", height=80)

        submit_col, skip_col = st.columns(2)
        submit_survey = submit_col.form_submit_button("Submit reflection")
        skip_survey = skip_col.form_submit_button("Skip")

    if submit_survey:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        if ask_experience:
            save_user_profile(expert_id, experience, now)
        save_submission_survey(expert_id, forecast_date, usability, comprehension, context_relevance,
                                comment.strip(), now)
        st.session_state[survey_key] = False
        render_banner("Thanks for the reflection!", "success")
        st.rerun()
    elif skip_survey:
        st.session_state[survey_key] = False
        st.rerun()


def page_review_and_adjust():
    """Core page: load a forecast, show context/charts, let an expert
    adjust it. Behavior by state:
      - expert, not submitted -> editable (drag + confidence + submit)
      - expert, already submitted -> read-only, "already submitted"
      - admin -> always read-only, can view but never submit for someone
    """
    current_user = st.session_state["logged_in_user"]
    current_role = st.session_state["role"]

    dnn_df = get_dnn_df()
    be_df = get_be_df()

    with st.sidebar:
        if current_role == "admin":
            # Admins pick which expert's work to view; experts only ever see their own.
            # Cached: this reruns on every drag release, and a 30s-stale
            # expert list is harmless here (unlike at login/registration,
            # where correctness matters more than speed -- see auth_screen()).
            users = _cached_load_users()
            expert_list = [u for u, d in users.items() if d["role"] == "expert"]
            expert_id = st.selectbox("Expert ID (Admin View)", expert_list) if expert_list else None
        else:
            expert_id = current_user
            st.write(f"**Expert ID:** {expert_id}")

        available_dates = get_available_dates(dnn_df)
        if not available_dates:
            render_banner("No complete DNN forecast days available yet.", "error")
            return
        forecast_date = st.date_input(
            "Forecast date (day d+1)",
            value=available_dates[-1],
            min_value=available_dates[0],
            max_value=available_dates[-1],
            key="review_forecast_date",  # explicit key so this survives a page switch
        )

    forecast, timestamps = dnn_forecast(forecast_date, dnn_df)
    if forecast is None:
        render_banner(f"No complete DNN forecast for {forecast_date}.", "error")
        return

    # Warn if this day's forecast is a stale carry-forward from a failed model run.
    imputed_flags = dnn_imputed_flags(forecast_date, dnn_df)
    if imputed_flags is not None and imputed_flags.any():
        render_banner(
            f"This forecast run failed for {forecast_date}. The previous day's forecast was "
            "carried forward. Treat this forecast with extra caution.",
            "warning",
        )

    # ACI margin, not the retired QR model -- see get_aci_margin() for why.
    margin = get_aci_margin(dnn_df, be_df, alpha=0.2, gamma=0.01, calibration_days=30)

    calendar_ctx = get_calendar_context(forecast_date)
    day_rows = be_df.loc[be_df["date_only"] == forecast_date].sort_values("Date")

    # Weather/load context needs the actuals feed caught up to this date.
    # French demand isn't shown: different scale from Belgium's, not useful
    # for judging this forecast.
    #
    # Net (residual) demand, not gross: price is set by the generator
    # covering demand AFTER renewables (near-zero marginal cost) are
    # subtracted, so net demand tracks price in a way gross doesn't.
    if len(day_rows) == STEPS_PER_DAY:
        wind_total_day = day_rows["Wind_Offshore_BE"] + day_rows["Wind_Onshore_BE"]
        net_demand = (day_rows["Load_BE"] - day_rows["Solar_BE"] - wind_total_day).mean()
        avg_temp = day_rows["temperature_2m"].mean()
        avg_hum = day_rows["relative_humidity_2m"].mean()
        context_available = True
    else:
        context_available = False

    c1, c2, c3 = st.columns(3)
    c1.metric("Day of the week", calendar_ctx["day_of_week"])
    c2.metric("Holiday?", calendar_ctx["holiday_name"] if calendar_ctx["is_holiday"] else "No")
    c3.metric("Bridge day?", "Yes" if calendar_ctx["is_bridge_day"] else "No")

    # Toggles live in the sidebar; the charts they control render on the
    # main page (see after Solar & Wind), not here.
    with st.sidebar:
        st.divider()
        st.subheader("Day info")
        if context_available:
            st.metric(
                "Avg. Net BE demand (MW)", f"{net_demand:,.0f}",
                help="Gross demand minus solar and wind generation -- the portion of demand "
                     "that has to be covered by other (price-setting) generation.",
            )
            st.metric("Avg. temp (°C)", f"{avg_temp:.1f}")
            st.metric("Avg. humidity (%)", f"{avg_hum:.0f}")

            show_temp_sidebar = st.toggle("Show Temperature plot")
            show_hum_sidebar = st.toggle("Show Humidity plot")
        else:
            render_banner("Weather/load context not available for this date.", "info")
            show_temp_sidebar = False
            show_hum_sidebar = False

    with st.expander("What am I looking at?"):
        st.write(
            "This is the forecast chart for the chosen day. The :blue[shaded band] is an 80% "
            "confidence interval. Points shown in :orange[orange] are ones the model itself flagged as "
            "unusual for this particular day. The dashed line is the original, unadjusted "
            "forecast, useful for seeing how far an adjustment has drifted from it. Points "
            "can be dragged up or down accordingly."
        )
        st.write(
            "In the left side panel, the second page: :red[***Deterministic Forecast Analysis***] shows the DNN forecast against the "
            "actual prices across the last 14 days. Further down, there is relevant day "
            "information, as well as two slider buttons for displaying the temperature and "
            "humidity plots."
        )
        st.write(
            ":green[Renewables (Solar, Wind)] during the given day are plotted underneath the "
            "draggable forecast using their respective slider button. "
            "Indicate the confidence level in your results before submission."
        )

    # Auto-flag: 5th/95th percentile of THIS day's own forecast, not a
    # fixed EUR/MWh threshold -- so calm and volatile days each get flagged
    # relative to their own baseline.
    low_threshold = np.percentile(forecast, 5)
    high_threshold = np.percentile(forecast, 95)
    flagged = (forecast <= low_threshold) | (forecast >= high_threshold)

    hour_of_slot = np.array([pd.Timestamp(ts).hour for ts in timestamps])
    time_label = [pd.Timestamp(ts).strftime("%H:%M") for ts in timestamps]

    # "adjusted" starts as an exact copy of "forecast" (pre-rounded so they
    # stay bit-identical) until the expert changes a value. Built before the
    # chart so it can show forecast, band, and flags together.
    working_df = pd.DataFrame({
        "timestamp_slot": timestamps,
        "hour": hour_of_slot,
        "time_label": time_label,
        "forecast": np.round(forecast, 2),
        "adjusted": np.round(forecast, 2),
        "flagged": flagged,
    })

    key = f"{expert_id}_{forecast_date}"  # one working copy per (expert, date) pair in session state
    survey_key = f"survey_pending_{key}"  # set True right after a fresh submission, see below
    already_submitted = _cached_has_submitted(expert_id, forecast_date) if expert_id else False
    is_read_only = (current_role == "admin")

    if key not in st.session_state:
        # Restore a prior unsubmitted session (navigated away before
        # submitting) instead of resetting to the raw forecast. Scoped to
        # this one (expert, date) pair -- see _cached_load_submissions().
        log = _cached_load_submissions(expert_id, forecast_date)
        if not log.empty:
            past_sub = log[(log["expert_id"] == expert_id) & (log["forecast_date"] == forecast_date)]
            if not past_sub.empty:
                past_sub = past_sub.tail(STEPS_PER_DAY).sort_values("timestamp_slot")
                if len(past_sub) == STEPS_PER_DAY:
                    working_df["adjusted"] = past_sub["adjusted"].values
                    working_df["flagged"] = past_sub["flagged"].values
        st.session_state[key] = working_df

    working = st.session_state[key]

    # Band belongs to the original forecast, not the adjusted line -- ACI
    # is calibrated on the model's own error, unaffected by drags.
    band_lower = (working["forecast"] - margin).tolist() if margin is not None else None
    band_upper = (working["forecast"] + margin).tolist() if margin is not None else None

    # Draggable while editing, static once there's nothing left to drag --
    # both show forecast/adjustment/band/flags together.
    if is_read_only or already_submitted:
        st.plotly_chart(
            make_comparison_chart(
                working["timestamp_slot"].values, working["forecast"].values, working["adjusted"].values,
                band_lower=band_lower, band_upper=band_upper, flagged=working["flagged"].values,
            ),
            width="stretch",
        )
        render_chart_legend(
            show_forecast=True,  # make_comparison_chart always plots forecast, unlike draggable_curve where it's optional
            show_band=band_lower is not None,
            show_flagged=bool(np.any(working["flagged"].values)),
        )
    else:
        dragged = draggable_curve(
            values=working["adjusted"].tolist(),
            labels=working["time_label"].tolist(),
            radius=8,
            height=420,
            key=f"drag_{key}",
            forecast=working["forecast"].tolist(),
            band_lower=band_lower,
            band_upper=band_upper,
            flagged=working["flagged"].tolist(),
        )
        if dragged != working["adjusted"].tolist():
            working["adjusted"] = dragged
            st.session_state[key] = working
            st.rerun()

    if margin is None:
        st.caption("Not enough settled history yet to calibrate an ACI band for this date.")
    else:
        st.caption(f"Current ACI margin: ± {margin:.2f} EUR/MWh")


    # Solar+wind share one chart: their combined dip drives net demand and
    # price spikes.
    if context_available:
        with st.container(border=True):
            st.subheader("Solar & Wind - Renewables")
            sc1, sc2 = st.columns(2)
            show_solar = sc1.toggle("Solar", value=True)
            show_wind = sc2.toggle("Wind", value=True)

            wind_total = day_rows["Wind_Offshore_BE"] + day_rows["Wind_Onshore_BE"]
            solar_vals = day_rows["Solar_BE"].values if show_solar else None
            wind_vals = wind_total.values if show_wind else None

            if solar_vals is None and wind_vals is None:
                render_banner("Select at least one series to display.", "info")
            else:
                st.plotly_chart(make_renewables_chart(day_rows["Date"].values, solar=solar_vals, wind=wind_vals),
                                width="stretch")

        # Toggles live in the sidebar; charts render here, full width.
        if show_temp_sidebar and show_hum_sidebar:
            wc1, wc2 = st.columns(2)
            with wc1:
                st.plotly_chart(
                    make_simple_chart(day_rows["Date"].values, day_rows["temperature_2m"].values, "Temperature (°C)"),
                    width="stretch",
                )
            with wc2:
                st.plotly_chart(
                    make_simple_chart(day_rows["Date"].values, day_rows["relative_humidity_2m"].values, "Humidity (%)"),
                    width="stretch",
                )
        elif show_temp_sidebar:
            st.plotly_chart(
                make_simple_chart(day_rows["Date"].values, day_rows["temperature_2m"].values, "Temperature (°C)"),
                width="stretch",
            )
        elif show_hum_sidebar:
            st.plotly_chart(
                make_simple_chart(day_rows["Date"].values, day_rows["relative_humidity_2m"].values, "Humidity (%)"),
                width="stretch",
            )

    # Footer: the reflection survey is one-time-ever (has_completed_survey
    # checks all submissions, not just today's); confidence is unaffected,
    # asked every time via the separate "submissions" table.
    if is_read_only or already_submitted:
        if is_read_only:
            render_banner(f"Viewing {expert_id}'s submission (read-only, admins cannot submit on behalf of experts).", "info")
        elif st.session_state.get(survey_key, False) and not has_completed_survey(expert_id):
            render_submission_survey(expert_id, forecast_date, survey_key)
        else:
            render_banner(f"You've already submitted feedback for {forecast_date}. Submissions are final.", "info")
    else:
        # No form needed: dragging already updates session state and
        # reruns on release, so `working` is current by the time Submit
        # is clicked.
        #
        # Trade-off: the old per-slot "Flag" checkbox is gone -- `flagged`
        # is now purely automatic, no manual override.
        with st.container(border=True):
            st.subheader("Your confidence")
            confidence = st.slider("How confident are you in these adjustments? (1-no confidence, 5-highest confidence)", 1, 5, 3)
            submitted = st.button("Submit feedback")

        if submitted:
            if not expert_id:
                render_banner("Error: No Expert ID found.", "error")
            elif has_submitted(expert_id, forecast_date):
                # Guards against a double-submit race (e.g. two tabs open on the same date) --
                # the fast-path check for the normal case. The DuplicateSubmissionError catch
                # below is the actual, database-enforced backstop for the rare case where two
                # near-simultaneous submissions both pass this check before either commits.
                render_banner("A submission already exists for this date. Refresh the page.", "error")
            else:
                rows = working.copy()
                rows["expert_id"] = expert_id
                rows["forecast_date"] = forecast_date
                rows["timestamp"] = dt.datetime.now(dt.timezone.utc).isoformat()
                rows["confidence"] = confidence
                try:
                    save_submission(rows)
                except DuplicateSubmissionError:
                    render_banner("A submission already exists for this date. Refresh the page.", "error")
                else:
                    # Invalidate the short-TTL caches so this submission is
                    # immediately visible everywhere (this page's read-only
                    # branch, admin's Reveal & Evaluate / Scoreboard, the
                    # sidebar's results-email countdown) instead of waiting
                    # out the TTL. (This particular submission won't itself
                    # change the results-email count today, since it can't
                    # be "reviewed" until its price settles -- but clearing
                    # it here keeps that number always correct rather than
                    # relying on that timing coincidence.)
                    _cached_has_submitted.clear()
                    _cached_load_submissions.clear()
                    _cached_compute_expert_results_summary.clear()
                    _refresh_email_progress(expert_id)  # sidebar countdown reflects this submission right away
                    st.session_state[survey_key] = True  # triggers render_submission_survey() on the next render
                    render_banner(f"Forecast submitted for {forecast_date}! Saved {len(rows)} adjusted values.", "success")
                    st.rerun()  # forces the page back into the read-only branch above


# --------------------------------------------------------------------------
# PAGE 2: REVEAL & EVALUATE
# --------------------------------------------------------------------------

def get_last_evaluable_ts(now=None):
    """Cutoff for evaluable data: Belgian day-ahead prices publish the day
    before delivery, so "tomorrow" isn't settled yet even if the actuals
    feed has a stale placeholder for it. `now` is injectable for testing."""
    now = now if now is not None else pd.Timestamp.now(tz="Europe/Brussels").tz_localize(None)
    tomorrow_start = now.normalize() + pd.Timedelta(days=1)
    return tomorrow_start - pd.Timedelta(minutes=15)


# --------------------------------------------------------------------------
# RESULTS-EMAIL NOTIFICATIONS
#
# Once an expert has more than RESULTS_EMAIL_THRESHOLD_DAYS settled/graded
# days, they get an email with their current aggregate stats -- checked
# once per login (see main()), and re-sent only when days_reviewed has
# actually increased since the last email (tracked in email_notifications).
# Streamlit has no background scheduler, so "checked on login" is the
# closest practical approximation of "after every submission, once past
# the threshold": the count updates the moment a new day settles, but the
# email itself only goes out the next time that expert opens the app.
# --------------------------------------------------------------------------

def compute_expert_results_summary(expert_id):
    """Aggregates this one expert's settled (evaluated) days into the same
    stats shown on the Expert Scoreboard -- avg. MAE improvement, days
    reviewed, win rate, avg. confidence. Deliberately mirrors
    page_expert_scoreboard()'s per-expert aggregation rather than sharing
    code with it, so a future change to that page can't silently change
    what gets emailed (or vice versa).

    Returns None if this expert has no settled/evaluable days yet."""
    log = _cached_load_submissions(expert_id=expert_id)
    if log.empty:
        return None

    be_df = get_be_df()
    last_evaluable = get_last_evaluable_ts()

    results = []
    for forecast_date, group in log.groupby("forecast_date"):
        if pd.Timestamp(forecast_date) > last_evaluable:
            continue  # not settled yet, same rule as Reveal & Evaluate/Scoreboard

        actuals = be_df.loc[be_df["date_only"] == forecast_date, ["Date", "Price"]].rename(
            columns={"Date": "timestamp_slot", "Price": "actual"}
        )
        evaluation = group.merge(actuals, on="timestamp_slot", how="inner").dropna(subset=["actual"])
        if evaluation.empty:
            continue

        forecast_mae = (evaluation["forecast"] - evaluation["actual"]).abs().mean()
        adjusted_mae = (evaluation["adjusted"] - evaluation["actual"]).abs().mean()
        results.append({
            "forecast_date": forecast_date,
            "improvement": forecast_mae - adjusted_mae,
            "confidence": group["confidence"].iloc[0],
        })

    if not results:
        return None

    results_df = pd.DataFrame(results)
    return {
        "days_reviewed": int(results_df["forecast_date"].nunique()),
        "avg_improvement": round(float(results_df["improvement"].mean()), 2),
        "win_rate": round(float((results_df["improvement"] > 0).mean() * 100), 1),
        "avg_confidence": round(float(results_df["confidence"].mean()), 1),
    }


def send_results_email(to_email, username, stats):
    """Sends a plain-text summary of `stats` to `to_email` via Gmail SMTP.
    Raises on any failure (missing config, auth failure, network error) --
    the caller, maybe_send_results_email(), is responsible for catching
    that so a flaky send can never break a page render for the person
    using the app right now."""
    if not GMAIL_SENDER_ADDRESS or not GMAIL_APP_PASSWORD:
        raise RuntimeError("GMAIL_SENDER_ADDRESS / GMAIL_APP_PASSWORD not configured")

    body = (
        f"Hi {username},\n\n"
        f"You've now reviewed {stats['days_reviewed']} settled forecast days on the "
        f"EPF Expert Review app. Here are your current numbers:\n\n"
        f"  Avg. MAE improvement: {stats['avg_improvement']} EUR/MWh (Forecast MAE minus Adjusted MAE, averaged across your settled days)\n"
        f"  Win rate: {stats['win_rate']}% (share of your settled days where Adjusted MAE was lower than Forecast MAE)\n"
        f"  Avg. confidence: {stats['avg_confidence']} / 5 (your own self-reported rating, averaged across those same days)\n\n"
        f"{APP_URL}\n\n"
        f"Thank you for taking part in this study. Your reviews are what make this research possible.\n\n"
        f"EPF Expert Review (automated message, part of the research study)"
    )
    msg = MIMEText(body)
    msg["Subject"] = f"Your EPF Expert Review results ({stats['days_reviewed']} days reviewed)"
    msg["From"] = GMAIL_SENDER_ADDRESS
    msg["To"] = to_email

    # STARTTLS on 587 rather than implicit SSL on 465 -- some networks
    # (VPNs, campus/corporate wifi, certain routers) block 465 outright
    # while leaving 587 open, and 587+STARTTLS is Gmail's own documented
    # recommendation for SMTP clients.
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
        server.starttls()
        server.login(GMAIL_SENDER_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_SENDER_ADDRESS, [to_email], msg.as_string())


@st.cache_data(ttl=15)
def _cached_compute_expert_results_summary(expert_id):
    """Same short-TTL treatment as _cached_load_submissions: this wraps a
    per-day groupby/merge over an expert's full submission history.
    render_email_progress_sidebar() calls it on every single rerun of the
    app for a logged-in expert (every drag release, every button click,
    every page switch) -- not just once per login like
    maybe_send_results_email() -- so leaving it uncached meant redoing
    that whole aggregation on every rerun. Cleared the moment a new
    submission is actually saved (page_review_and_adjust()), so the
    displayed count updates immediately rather than waiting out the TTL."""
    return compute_expert_results_summary(expert_id)


@st.cache_data(ttl=30)
def _cached_get_last_emailed_days_reviewed(username):
    """Same treatment as _cached_get_onboarding_status: render_email_progress_sidebar()
    calls this on every rerun, so left uncached it was a live Postgres
    round-trip on every single click/drag for every logged-in expert, not
    just once per login. Cleared the moment a results email is actually
    sent, in maybe_send_results_email(), so the countdown updates right
    away rather than waiting out the TTL."""
    return get_last_emailed_days_reviewed(username)


def _refresh_email_progress(username):
    """Computes this expert's results-email progress and stores it in
    st.session_state. This is the only place that actually touches the
    database or recomputes the per-day aggregation for this feature --
    render_email_progress_sidebar() below only ever reads the stored
    result, since it's called on every single rerun of the app (including
    every chart drag, the single most rerun-heavy interaction in the app),
    and recomputing on a timer there was adding real per-drag latency
    even with short-TTL caching. Called once at login (see main()), and
    again only at the two moments these numbers can actually change: a
    new submission (page_review_and_adjust()) and a results email
    actually being sent (maybe_send_results_email())."""
    stats = _cached_compute_expert_results_summary(username)
    days_reviewed = stats["days_reviewed"] if stats else 0
    last_emailed = _cached_get_last_emailed_days_reviewed(username)
    # Mirrors maybe_send_results_email()'s own threshold logic exactly:
    # the first email needs days_reviewed > RESULTS_EMAIL_THRESHOLD_DAYS
    # (so day 11, not day 10), every one after that needs a further full
    # RESULTS_EMAIL_THRESHOLD_DAYS past the last email actually sent.
    next_target = (
        RESULTS_EMAIL_THRESHOLD_DAYS + 1 if last_emailed == 0
        else last_emailed + RESULTS_EMAIL_THRESHOLD_DAYS
    )
    st.session_state["email_progress"] = {"days_reviewed": days_reviewed, "next_target": next_target}


def render_email_progress_sidebar(username):
    """Shown in the sidebar for every expert, every page (see main()): how
    many more reviewed (settled) days remain until their next automatic
    results email. Pure read from st.session_state -- see
    _refresh_email_progress() for where and how often that's populated --
    so this costs nothing extra on an ordinary rerun."""
    if "email_progress" not in st.session_state:
        _refresh_email_progress(username)
    progress = st.session_state["email_progress"]
    days_reviewed = progress["days_reviewed"]
    next_target = progress["next_target"]
    remaining = max(next_target - days_reviewed, 0)

    st.caption(f"Results email: {days_reviewed}/{next_target} reviewed days")
    st.progress(min(days_reviewed / next_target, 1.0))
    if remaining > 0:
        st.caption(f"{remaining} more reviewed day{'s' if remaining != 1 else ''} until your next results email.")
    else:
        st.caption("Your next results email will be sent at your next login.")


def maybe_send_results_email(username):
    """Checked once per login (see main()): sends the first results email
    once this expert passes RESULTS_EMAIL_THRESHOLD_DAYS evaluated days,
    then a fresh one only every further RESULTS_EMAIL_THRESHOLD_DAYS days
    after that (milestone-based, e.g. 11, 21, 31...) rather than on every
    single new evaluated day -- daily emails would themselves be a signal
    that could influence how an expert reviews, which runs against the
    same neutral-framing goal as the email's plain, non-judgmental wording.
    No-ops quietly -- never raises into the page -- if email isn't
    configured, this expert has no email on file, or the send itself
    fails; a failed send is retried at their next login since
    mark_results_emailed() is only called after a successful send."""
    stats = compute_expert_results_summary(username)
    if stats is None or stats["days_reviewed"] <= RESULTS_EMAIL_THRESHOLD_DAYS:
        return

    last_emailed = get_last_emailed_days_reviewed(username)
    if stats["days_reviewed"] < last_emailed + RESULTS_EMAIL_THRESHOLD_DAYS:
        return  # not a full milestone past the last email yet

    user_email = _cached_load_users().get(username, {}).get("email")
    if not user_email:
        return

    try:
        send_results_email(user_email, username, stats)
        mark_results_emailed(username, stats["days_reviewed"], dt.datetime.now(dt.timezone.utc).isoformat())
        _cached_get_last_emailed_days_reviewed.clear()
        _refresh_email_progress(username)  # sidebar countdown reflects the new milestone right away
    except Exception as e:
        print(f"[results email] failed to send to {username!r}: {e}")


def page_reveal_and_evaluate():
    """Admin-only: pick one expert's one-day submission and, once settled,
    compare forecast vs. adjusted against the realized price (MAE each)."""
    st.title("Reveal & Evaluate")

    with st.expander("What is this page showing?"):
        st.write(
            "This page grades one expert's adjustment for one day, but only "
            "once that day has settled. A delivery day settles the moment "
            "its real, realized day-ahead price becomes known, which is "
            "always the day before delivery, so a day can't be graded until "
            "then."
        )
        st.write(
            "Once it's settled, both the original DNN forecast and the "
            "expert's dragged, adjusted version are compared against that "
            "real price using MAE (mean absolute error, in EUR/MWh): the "
            "average, across the day's 96 fifteen-minute slots, of how far "
            "off each line was. Lower MAE means a more accurate line."
        )
        st.write(
            "Forecast MAE is the model's own error, unaffected by the "
            "expert. Adjusted MAE is the expert's edited line's error. If "
            "Adjusted MAE is lower, the expert's changes made the forecast "
            "more accurate for that day; if it's higher, the changes made "
            "it worse."
        )

    log = _cached_load_submissions()
    if log.empty:
        render_banner("No submissions yet.", "warning")
        return

    be_df = get_be_df()

    expert_id = st.selectbox("Expert ID", sorted(log["expert_id"].dropna().unique()))
    available_dates = sorted(log.loc[log["expert_id"] == expert_id, "forecast_date"].dropna().unique())
    forecast_date = st.selectbox("Forecast date", available_dates)

    last_evaluable = get_last_evaluable_ts()
    if pd.Timestamp(forecast_date) > last_evaluable:
        render_banner("This delivery day's day-ahead prices haven't settled yet, so there is nothing to reveal.", "info")
        return

    submission = (
        log.loc[
            (log["expert_id"] == expert_id) & (log["forecast_date"] == forecast_date),
            ["timestamp_slot", "forecast", "adjusted", "confidence"],
        ]
        .sort_values("timestamp_slot")
    )

    # Join the saved submission to the realized price by timestamp -- this is
    # the only place "forecast"/"adjusted" (saved at submission time) meet
    # "actual" (fetched live), so MAE here always reflects the true settled price.
    actuals = be_df.loc[be_df["date_only"] == forecast_date, ["Date", "Price"]].rename(
        columns={"Date": "timestamp_slot", "Price": "actual"}
    )
    evaluation = submission.merge(actuals, on="timestamp_slot", how="inner")

    if evaluation.empty:
        render_banner("No realized prices available yet for this date.", "info")
        return

    # Some slots can have a matched timestamp but a missing/NaN price -- a
    # real upstream data gap (e.g. around when the model's pipeline stopped
    # updating), not a bug in the submission or the merge itself. Drop those
    # rather than letting them silently turn both MAE numbers into NaN.
    n_missing_actual = int(evaluation["actual"].isna().sum())
    evaluation = evaluation.dropna(subset=["actual"])

    if evaluation.empty:
        render_banner(f"Realized prices for {forecast_date} are missing from the data feed -- nothing to evaluate yet.", "info")
        return

    forecast_mae = (evaluation["forecast"] - evaluation["actual"]).abs().mean()
    adjusted_mae = (evaluation["adjusted"] - evaluation["actual"]).abs().mean()
    confidence_rating = submission["confidence"].iloc[0]  # constant across the day's 96 rows

    if n_missing_actual > 0:
        st.caption(f"{n_missing_actual} of {STEPS_PER_DAY} slots had a missing realized price and were excluded from these MAE numbers.")

    forecast_metric, adjusted_metric, confidence_metric = st.columns(3)
    forecast_metric.metric("Forecast MAE", f"{forecast_mae:.2f} EUR/MWh")
    adjusted_metric.metric("Adjusted MAE", f"{adjusted_mae:.2f} EUR/MWh")
    confidence_metric.metric("Expert confidence", f"{confidence_rating}/5")

    if adjusted_mae < forecast_mae:
        st.write("Verdict: the expert adjustment improved the forecast.")
    elif adjusted_mae > forecast_mae:
        st.write("Verdict: the expert adjustment worsened the forecast.")
    else:
        st.write("Verdict: the expert adjustment made no difference.")


# --------------------------------------------------------------------------
# PAGE 3: EXPERT SCOREBOARD
# --------------------------------------------------------------------------

def page_expert_scoreboard():
    """Admin-only: aggregates every settled (expert, date) submission into
    a per-expert leaderboard -- avg. MAE improvement, days reviewed, win
    rate, avg. confidence."""
    st.title("Expert Scoreboard")

    with st.expander("What is this page showing?"):
        st.write(
            "This aggregates the same per-day MAE comparison used on Reveal "
            "& Evaluate, across every settled day each expert has reviewed, "
            "into one leaderboard row per expert."
        )
        st.write(
            "Avg. improvement is the mean, across that expert's settled "
            "days, of (Forecast MAE minus Adjusted MAE) in EUR/MWh. A "
            "positive number means their adjustments made the forecast "
            "more accurate on average; a negative number means the "
            "opposite."
        )
        st.write(
            "Days reviewed counts distinct settled days this expert's "
            "numbers are based on. Win rate is the percentage of those days "
            "where the expert's adjustment beat the model (Adjusted MAE "
            "lower than Forecast MAE). Avg. confidence is the mean of the "
            "expert's own self-reported confidence rating (1 to 5) on "
            "those same days."
        )
        st.write(
            "Unsettled days are left out entirely until their real price "
            "is known, so a day never counts toward these numbers before "
            "it can actually be graded."
        )

    log = _cached_load_submissions()
    if log.empty:
        render_banner("No submissions yet.", "warning")
        return

    be_df = get_be_df()
    last_evaluable = get_last_evaluable_ts()

    results = []
    for (expert_id, forecast_date), group in log.groupby(["expert_id", "forecast_date"]):
        if pd.Timestamp(forecast_date) > last_evaluable:
            continue  # skip days that haven't settled yet, same rule as Reveal & Evaluate

        actuals = be_df.loc[be_df["date_only"] == forecast_date, ["Date", "Price"]].rename(
            columns={"Date": "timestamp_slot", "Price": "actual"}
        )
        evaluation = group.merge(actuals, on="timestamp_slot", how="inner")
        if evaluation.empty:
            continue

        # Same upstream data-gap issue as Reveal & Evaluate: drop slots with
        # a missing actual price rather than letting a NaN day silently
        # poison this expert's entire avg_improvement/win_rate aggregation
        # below, not just this one row.
        evaluation = evaluation.dropna(subset=["actual"])
        if evaluation.empty:
            continue

        forecast_mae = (evaluation["forecast"] - evaluation["actual"]).abs().mean()
        adjusted_mae = (evaluation["adjusted"] - evaluation["actual"]).abs().mean()

        results.append({
            "expert_id": expert_id,
            "forecast_date": forecast_date,
            "forecast_mae": forecast_mae,
            "adjusted_mae": adjusted_mae,
            "improvement": forecast_mae - adjusted_mae,  # positive = the expert helped
            "confidence": group["confidence"].iloc[0],
        })

    if not results:
        render_banner("No submissions overlap with settled actual prices yet.", "warning")
        return

    results_df = pd.DataFrame(results)

    scoreboard = (
        results_df.groupby("expert_id")
        .agg(
            avg_improvement=("improvement", "mean"),
            days_reviewed=("forecast_date", "nunique"),
            win_rate=("improvement", lambda s: (s > 0).mean()),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values("avg_improvement", ascending=False)
    )
    scoreboard["win_rate"] = (scoreboard["win_rate"] * 100).round(1)
    scoreboard["avg_improvement"] = scoreboard["avg_improvement"].round(2)
    scoreboard["avg_confidence"] = scoreboard["avg_confidence"].round(1)

    st.dataframe(scoreboard, hide_index=True)


# --------------------------------------------------------------------------
# ONBOARDING (research disclaimer + step-by-step tutorial, shown once)
# --------------------------------------------------------------------------

ONBOARDING_STEPS = [
    {
        "title": "Before you continue",
        "kind": "consent",
    },
    {
        "title": "What this app is for",
        "body": (
            "You'll review a day-ahead electricity price forecast produced by a "
            "neural network, and adjust it if you disagree with part of it. This is "
            "part of a research study measuring whether human review actually "
            "improves electricity price forecasts."
        ),
    },
    {
        "title": "Reading the chart",
        "body": (
            "The main chart shows the model's forecast as a line, with a shaded band "
            "around it; an 80% confidence interval, meaning the real price is "
            "expected to land inside it about 80% of the time. Points shown in orange "
            "are ones the model itself flagged as unusual for that particular day (5th/95th percentile)."
        ),
    },
    {
        "title": "Adjusting the forecast",
        "body": (
            "To adjust the forecast, drag any point on the curve up or down. Nearby "
            "points shift too, so you're reshaping a stretch of the curve rather than "
            "creating a single spike."
        ),
    },
    {
        "title": "Using the extra context",
        "body": (
            "Below the chart you'll find extra information: demand, temperature, "
            "humidity, solar and wind output. That might help you judge whether the "
            "model's forecast makes sense for that particular day."
        ),
    },
    {
        "title": "Submitting",
        "body": (
            "Once you're happy with your adjustment, rate how confident you are and "
            "click Submit. Submissions are final. There's no editing afterward. "
            "Right after submitting, you'll be asked a few short reflection "
            "questions. Please answer honestly."
        ),
    },
    {
        "title": "Your results by email",
        "body": (
            "Once a submitted day settles and can be graded, it counts toward your "
            "reviewed-days total. After you pass 10 reviewed days, you'll automatically "
            "get an email with your stats so far, and another every 10 reviewed days "
            "after that."
        ),
    },
]


def render_onboarding(username):
    """Blocks the rest of the app until a user has consented to the
    research disclaimer and completed the tutorial -- persisted in the DB
    (onboarding_status), so it's a true one-time gate, not per-session.
    Applies to every role, including admins."""
    step_idx_key = f"onboarding_step_idx_{username}"
    if step_idx_key not in st.session_state:
        st.session_state[step_idx_key] = 0

    step_idx = st.session_state[step_idx_key]
    step = ONBOARDING_STEPS[step_idx]
    total = len(ONBOARDING_STEPS)

    st.title(step["title"])
    st.caption(f"Step {step_idx + 1} of {total}")

    consent_given = True
    if step.get("kind") == "consent":
        render_banner(
            "**This application is intended for research purposes.** Your forecast "
            "adjustments, confidence ratings, and survey responses will be used in an "
            "academic study on human-in-the-loop electricity price forecasting. No "
            "real trading or operational decisions should be based on this tool.",
            "warning",
        )
        consent_given = st.checkbox(
            "I understand this application is intended for research purposes.",
            key=f"onboarding_consent_{username}",
        )
    else:
        st.write(step["body"])

    back_col, next_col = st.columns(2)
    if step_idx > 0:
        if back_col.button("Back", key=f"onboarding_back_{username}_{step_idx}"):
            st.session_state[step_idx_key] -= 1
            st.rerun()

    is_last = step_idx == total - 1
    next_label = "Get started" if is_last else "Next"
    if next_col.button(next_label, disabled=not consent_given, key=f"onboarding_next_{username}_{step_idx}"):
        if is_last:
            save_onboarding_status(username, True, True, dt.datetime.now(dt.timezone.utc).isoformat())
            _cached_get_onboarding_status.clear()  # lift the gate immediately, not after the TTL
            del st.session_state[step_idx_key]
        else:
            st.session_state[step_idx_key] += 1
        st.rerun()


BANNER_COLORS = {
    # One shared visual style (padding, border-radius, font-weight -- see
    # render_banner()) for every success/warning/error/info message in the
    # app; only these three values change per kind. Kept as a plain dict
    # rather than folded into get_palette(), since these are alert-specific
    # semantic colors, not part of the app's general UI/Plotly palette.
    "success": {"text": "#4ade80", "border": "#22c55e", "bg": "rgba(34,197,94,0.15)"},
    "error":   {"text": "#f87171", "border": "#ef4444", "bg": "rgba(239,68,68,0.15)"},
    "warning": {"text": "#fbbf24", "border": "#f59e0b", "bg": "rgba(245,158,11,0.15)"},
    "info":    {"text": "#93c5fd", "border": "#3b82f6", "bg": "rgba(59,130,246,0.15)"},
}


def render_banner(message, kind="info"):
    """Single reusable banner for every success/warning/error/info message
    in the app, so they all share one visual style and differ only by
    color, per `kind` (one of BANNER_COLORS' keys). Built as plain HTML
    rather than st.success()/st.warning()/st.error()/st.info(): those all
    render through the same [data-testid="stAlert"] testid, so a CSS
    override can't tell them apart by type -- this replaces reliance on
    that testid entirely rather than fighting it."""
    colors = BANNER_COLORS[kind]
    st.markdown(
        f'<div style="background-color:{colors["bg"]};border:1px solid {colors["border"]};'
        f'border-radius:8px;padding:0.75rem 1rem;color:{colors["text"]};font-weight:500;">'
        f'{message}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# AUTH SCREEN
# --------------------------------------------------------------------------

def auth_screen():
    """Login/registration, shown when nobody's logged in (see main()).
    Self-registration only offers "expert" -- admins are created directly
    in the database."""
    st.subheader("Welcome to EPF Expert Review")

    auth_mode = st.radio("Choose an option:", ["Log In", "Create Account"], horizontal=True)
    st.divider()

    if auth_mode == "Log In":
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")

        if st.button("Log In"):
            users = load_users()
            if login_user in users and check_password(login_pass, users[login_user]["password"]):
                st.session_state["logged_in_user"] = login_user
                st.session_state["role"] = users[login_user]["role"]
                st.rerun()
            else:
                render_banner("Invalid username or password.", "error")

    elif auth_mode == "Create Account":
        new_email = st.text_input("Email Address", key="new_email")
        new_user = st.text_input("New Username", key="new_user")
        new_pass = st.text_input("New Password", type="password", key="new_pass")
        new_role = st.selectbox("Role", EXPERT_ROLES, key="new_role")

        if st.button("Create Account"):
            users = load_users()
            username_input = new_user.strip()
            email_input = new_email.strip().lower()
            email_exists = any(account.get("email") == email_input for account in users.values())
            is_valid, validation_message = validate_registration(username_input, email_input, new_pass)

            if not is_valid:
                render_banner(validation_message, "error")
            elif username_input in users:
                render_banner("This username is already taken. Please choose another.", "error")
            elif email_exists:
                render_banner("This email address is already registered. Please use another or log in.", "error")
            else:
                save_new_user(username_input, hash_password(new_pass), email_input, new_role)
                _cached_load_users.clear()  # so the admin's expert-picker sees this account right away
                render_banner("Account created successfully! You can now switch to the Log In option.", "success")


def page_dnn_history():
    """DNN-only slice of Margarida's Section 1 chart, natively rendered in
    this app's own theme -- not her scatter diagnostics or MAE tables. The
    plot design itself is adapted from her public Hugging Face Space; see
    the explanatory text below for the link and what the chart means."""
    st.title("Deterministic Forecast Analysis, DNN")
    st.caption("DNN forecast vs. actual settled price, last 14 days.")

    st.markdown(
        "This chart's design is taken directly from "
        "[Margarida's DAM price forecast Space on Hugging Face]"
        "(https://huggingface.co/spaces/EDS-lab/DAM-price-forecast), "
        "the source of the underlying DNN model shown here. "
        "The model is a deep neural network trained to forecast the "
        "Belgian day-ahead electricity price. It is retrained and run "
        "on a daily schedule because day-ahead prices have to be "
        "forecast before each day's auction closes, so a new forecast "
        "is published every day for the next delivery day. "
        "The chart below compares that forecast against the price that "
        "actually settled, once it's known, over the last 14 days. It "
        "is a rolling accuracy check on the model itself, independent "
        "of any expert's manual adjustments, and it's what the review "
        "page's forecast (and its uncertainty band) are ultimately "
        "built on."
    )

    dnn_df = get_dnn_df()
    be_df = get_be_df()

    history_df = get_dnn_history_window(dnn_df, be_df, days=14)

    if history_df.empty:
        render_banner("No DNN forecast data available yet.", "info")
        return

    st.plotly_chart(make_history_chart(history_df), width="stretch")


def page_survey_results():
    """Admin-only: reflection survey data plus one-time experience
    profiles. Summaries + raw-data export only -- no analysis here."""
    st.title("Survey Results")

    with st.expander("What is this page showing?"):
        st.write(
            "This page isn't about forecast accuracy. It's about the app "
            "itself: after an expert's first-ever submission, they're "
            "asked a short reflection survey on how usable and "
            "understandable the tool was, and whether the extra weather "
            "and renewables context actually helped them adjust the "
            "forecast."
        )
        st.write(
            "The four numbers at the top are plain averages of the 1 to 5 "
            "ratings across every response collected so far. Usability and "
            "comprehension speak to whether the tool itself got in the "
            "way; context relevance speaks to whether the extra data "
            "shown alongside the chart was actually useful, or just "
            "decoration."
        )
        st.write(
            "Each row in Raw responses is one reflection answer, tied to "
            "the submission it followed. It can be joined, on username "
            "and forecast date, to that same submission's MAE result from "
            "Reveal & Evaluate or the Scoreboard, so a low usability "
            "rating can be checked against whether that expert's "
            "adjustment actually helped or hurt the forecast."
        )
        st.write(
            "Experience profile is a separate, one-time question (prior "
            "familiarity with electricity price forecasting), asked only "
            "on an expert's first submission, since it describes the "
            "person rather than any single day's forecast."
        )

    survey_df = load_submission_survey()
    if survey_df.empty:
        render_banner("No survey responses submitted yet.", "info")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Responses", len(survey_df))
    c2.metric("Avg. usability", f"{survey_df['usability_rating'].mean():.1f} / 5")
    c3.metric("Avg. comprehension", f"{survey_df['comprehension_rating'].mean():.1f} / 5")
    c4.metric("Avg. context relevance", f"{survey_df['context_relevance_rating'].mean():.1f} / 5")

    st.subheader("Raw responses")
    st.caption("One row per submission this was answered for, joinable against the feedback "
               "table's MAE evaluation on (username, forecast_date).")
    st.dataframe(survey_df, hide_index=True, width="stretch")
    st.download_button(
        "Download as CSV",
        survey_df.to_csv(index=False).encode("utf-8"),
        file_name="submission_survey.csv",
        mime="text/csv",
    )

    st.subheader("Experience profile per expert")
    st.caption("Answered once per user, on their first submission, since it's a moderator "
               "variable, not something that changes day to day.")
    profiles_df = load_all_user_profiles()
    if profiles_df.empty:
        render_banner("No experience-level responses yet.", "info")
    else:
        st.dataframe(profiles_df, hide_index=True, width="stretch")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    """Entry point. Applies the theme first (so even the login screen is
    themed), gates everything behind login, then renders the sidebar nav and
    routes to the selected page. Available pages depend on role -- see the
    module docstring at the top of this file for what each role can see."""
    apply_theme()

    if "logged_in_user" not in st.session_state:
        auth_screen()
        return

    current_user = st.session_state["logged_in_user"]
    current_role = st.session_state["role"]

    consented, completed_tutorial = _cached_get_onboarding_status(current_user)
    if not (consented and completed_tutorial):
        render_onboarding(current_user)
        return

    # Results-email check: once per login session (st.session_state persists
    # across reruns within one browser session, so this runs once after
    # login, not on every page switch/drag/rerun). Experts only -- admins
    # never submit, so they'd never have results to report anyway.
    if current_role == "expert" and not st.session_state.get("results_email_checked"):
        st.session_state["results_email_checked"] = True
        maybe_send_results_email(current_user)

    with st.sidebar:
        st.write(f"Logged in as: **{current_user}** ({current_role})")
        if st.button("Log Out"):
            st.session_state.clear()
            st.rerun()

        if current_role == "expert":
            render_email_progress_sidebar(current_user)

        st.divider()

        if current_role == "admin":
            pages = ["Review & Adjust", "Deterministic Forecast Analysis", "Reveal & Evaluate",
                     "Expert Scoreboard", "Survey Results"]
        else:
            pages = ["Review & Adjust", "Deterministic Forecast Analysis"]

        page = st.radio("Page", pages)

    if page == "Review & Adjust":
        page_review_and_adjust()
    elif page == "Reveal & Evaluate":
        page_reveal_and_evaluate()
    elif page == "Expert Scoreboard":
        page_expert_scoreboard()
    elif page == "Deterministic Forecast Analysis":
        page_dnn_history()
    elif page == "Survey Results":
        page_survey_results()


if __name__ == "__main__":
    main()