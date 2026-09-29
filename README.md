# EPF Expert Review

**Live**: [epfdashboard-expert-review-rlhf.streamlit.app](https://epfdashboard-expert-review-rlhf.streamlit.app/)

A Streamlit dashboard for human-in-the-loop review of day-ahead electricity
price forecasts for the Belgian market. Domain experts adjust a neural
network's forecast by dragging the curve on a chart, rate their confidence,
and submit. Once the delivery day settles, admins compare each expert's
adjustment against the realized price (MAE) and aggregate results across
experts. The research question is whether, and how, human adjustments can
improve a machine-learning forecasting model's output — applied here to a
deep neural network forecasting the Belgian day-ahead electricity price.
The model itself is never retrained or updated from expert input; reviews
are recorded and evaluated, not fed back into training. A natural
extension for future research is whether adjustments like these could
ever feed back into the model the way RLHF does for language models, but
that is, for the moment, outside the scope of this study.

## The underlying DNN model

The day-ahead forecast this app asks experts to review is not trained
here. It comes from Margarida Mascarenhas's deep neural network, published
in her [DAM price forecast Hugging Face Space](https://huggingface.co/spaces/EDS-lab/DAM-price-forecast).
The model is retrained and run daily, since a day-ahead price forecast has
to be published before each day's auction closes. This app's
Deterministic Forecast Analysis page reuses the plot design from that
Space, redrawn in this app's own theme, and both the forecast and the
uncertainty band that experts adjust on the Review & Adjust page trace
back to that same model.

## Origin

Forked from Margarida Mascarenhas's
[**EPF_Cross-border-Markets**](https://github.com/margaridamascarenhas/EPF_Cross-border-Markets)
to bootstrap the initial app, using the Belgian data bundled in that repo
to mimic her DNN forecast results as a starting point. That local-data
dependency was later removed in favor of pulling data directly and live
from a private data repository (see Architecture below). None of the
forked repo's code is used today, only its data initially, to mock results
during early development. The fork also vendored
[epftoolbox](https://github.com/jeslago/epftoolbox) (Lago et al., *Applied
Energy* 2021), the library that originally produced the DNN/LEAR
forecasts; that code has since been deleted from this project too, since
the review app never imported it directly.

## Architecture

- **`app.py`**: the Streamlit app, covering auth, theming, and all five pages.
- **`db.py`**: Postgres persistence, hosted on Supabase and connected via
  `DATABASE_URL`, with the schema managed through pgAdmin.
- **`draggable_curve/`**: custom React/TypeScript component (built with
  Vite) rendering the forecast chart as one drag-to-edit widget, with the
  forecast line, uncertainty band, and flagged points all in one view.
- **Data**: fetched at runtime from a private GitHub repo (two CSVs: the
  DNN forecast and realized market data), authenticated via `GITHUB_TOKEN`
  and parsed straight into memory (`requests` to `io.StringIO` to
  `pandas`, cached 30 min via `st.cache_data`). Nothing is ever saved to a
  local folder in this repo. Exact repo/path names are intentionally
  omitted here since this is a public README. See the
  `GITHUB_OWNER`/`GITHUB_REPO` constants in `app.py` if you have
  legitimate access.

## Project structure

```
.
├── app.py
├── db.py
├── requirements.txt
├── .env / .env.example
└── draggable_curve/
    ├── __init__.py
    └── frontend/
        ├── src/DraggableCurve.tsx
        ├── dist/                # built output (what actually runs)
        └── package.json
```

## Setup

```bash
# 1. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Environment variables
cp .env.example .env   # then fill in GITHUB_TOKEN and DATABASE_URL

# 3. Build the drag-to-edit component (one-time; only needs Node for this step)
cd draggable_curve/frontend && npm install && npm run build && cd ../..

# 4. Run
streamlit run app.py
```

`DATABASE_URL` points to a Postgres instance (this project runs on
[Supabase](https://supabase.com); any Postgres host works). Tables are
created automatically on first run against whatever database it points
to, and nothing is bundled locally. Self-registration only creates
`expert` accounts, so insert an `admin` row directly into `users`
(bcrypt-hash the password first) to create one.

## Deployment

Hosted on Streamlit Community Cloud. Secrets (`GITHUB_TOKEN`,
`DATABASE_URL`) are set there, not committed. Community Cloud only runs
`pip install`, not `npm build`, so `draggable_curve/frontend/dist/` must
already be built and committed before pushing.

## Roles & pages

- **expert**: Review & Adjust and Deterministic Forecast Analysis only,
  and can only view/submit their own work. Submissions are final.
- **admin**: everything above, plus Reveal & Evaluate, Expert Scoreboard,
  and Survey Results; can view but never submit on an expert's behalf.

To help experts judge the forecast rather than adjust it blind, Review &
Adjust also surfaces: day-of-week/holiday/bridge-day info, avg. net
demand/temperature/humidity, a solar & wind chart, the 80% uncertainty
band, auto-flagged anomalies (5th/95th percentile of that day's own
forecast), and a warning if the forecast is a stale imputed carry-forward.
Deterministic Forecast Analysis adds a longer-horizon check: DNN vs.
actual over the last 14 days.

| Page | Purpose |
|---|---|
| Review & Adjust | View forecast, drag to adjust, rate confidence, submit, answer reflection survey. |
| Deterministic Forecast Analysis | DNN forecast vs. actual, last 14 days. |
| Reveal & Evaluate (admin) | MAE for one expert's submission once the price settles. |
| Expert Scoreboard (admin) | Aggregated MAE improvement, win rate, per expert. |
| Survey Results (admin) | Reflection-survey data + experience profiles; CSV export. |

## Data model (Postgres)

| Table | Purpose |
|---|---|
| `users` | Username, email, bcrypt hash, role. |
| `submissions` | One row per 15-min slot per submission: forecast, adjusted, flagged, confidence. |
| `onboarding_status` | Research-disclaimer consent + tutorial completion, gates all access. |
| `user_profile` | Self-reported EPF experience, asked once per user. |
| `submission_survey` | Reflection-survey ratings per submission, joinable against `submissions` on `(username, forecast_date)`. |

## The uncertainty band

The chart's 80% band isn't a separate model. It's a single margin,
conformal-calibrated from the DNN forecast's own settled residuals via
Adaptive Conformal Inference (`get_aci_margin()`). It self-adjusts: recent
misses widen it, recent hits relax it, no manual recalibration needed.
