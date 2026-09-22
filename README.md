# EPF Expert Review (RLHF)

Live: **[epf-dashboard-expert-review.onrender.com](https://epf-dashboard-expert-review.onrender.com/)**

A Streamlit dashboard for human-in-the-loop review of day-ahead electricity price forecasts for the Belgian market. Domain experts adjust a neural network's forecast by dragging the curve on a chart, rate their confidence, and submit. Once the delivery day settles, admins compare each expert's adjustment against the realized price and aggregate results across experts. The research question is whether human review measurably improves a forecasting model's output; the same question RLHF asks for language models, applied here to electricity prices.

## The underlying DNN model

The day-ahead forecast this app asks experts to review is not trained here. It comes from Margarida Mascarenhas's deep neural network, published in her [DAM price forecast Hugging Face Space](https://huggingface.co/spaces/EDS-lab/DAM-price-forecast). The model is retrained and run daily, since a day-ahead price forecast has to be published before each day's auction closes. This app's Deterministic Forecast Analysis page reuses the plot design from that Space, redrawn in this app's own theme.

## Origin

Forked from Margarida Mascarenhas's [EPF_Cross-border-Markets](https://github.com/margaridamascarenhas/EPF_Cross-border-Markets) to bootstrap the initial app. None of the forked repo's code is used today.

## Tech stack

- **App**: Streamlit, with a custom React/TypeScript drag-to-edit chart component.
- **Database**: Postgres, hosted on [Supabase](https://supabase.com).
- **Hosting**: [Render](https://render.com), deployed from this repo as a Docker web service.

## Setup

```bash
# 1. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Environment variables
cp .env.example .env   # fill in your own values

# 3. Build the drag-to-edit component (one-time; only needs Node for this step)
cd draggable_curve/frontend && npm install && npm run build && cd ../..

# 4. Run
streamlit run app.py
```

`DATABASE_URL` points to a Postgres instance (this project runs on Supabase; any Postgres host works). Tables are created automatically on first run.

## Deployment

Hosted on Render as a Docker web service, connected to this repo for auto-deploy on push to `main`. Secrets are set in Render's dashboard, never committed. The database is a separate, always-on Supabase Postgres instance.

## Roles & pages

- **expert**: Review & Adjust and Deterministic Forecast Analysis only, and can only view/submit their own work. Submissions are final.
- **admin**: everything above, plus Reveal & Evaluate, Expert Scoreboard, and Survey Results; can view but never submit on an expert's behalf.

| Page | Purpose |
|---|---|
| Review & Adjust | View forecast, drag to adjust, rate confidence, submit, answer reflection survey. |
| Deterministic Forecast Analysis | DNN forecast vs. actual, last 14 days. |
| Reveal & Evaluate (admin) | Accuracy for one expert's submission once the price settles. |
| Expert Scoreboard (admin) | Aggregated accuracy improvement per expert. |
| Survey Results (admin) | Reflection-survey data + experience profiles; CSV export. |
