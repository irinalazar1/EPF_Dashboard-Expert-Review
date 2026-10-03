# EPF Expert Review

Live: **[epf-dashboard-expert-review.onrender.com](https://epf-dashboard-expert-review.onrender.com/)**

A Streamlit dashboard for human-in-the-loop review of day-ahead electricity price forecasts for the Belgian market. Domain experts adjust a neural network's forecast by dragging the curve on a chart, rate their confidence, and submit. Once the delivery day settles, admins compare each expert's adjustment against the realized price and aggregate results across experts.

## The study

The study asks one question: **can expert feedback improve a model's forecast?** Each expert reviews the DNN's day-ahead forecast for a given day and adjusts it where their judgment disagrees with the model. After the day settles, both the original forecast and the expert's adjusted forecast are scored against the realized Belgian day-ahead price.

## Helping the expert decide

The Review & Adjust page puts the context an electricity market expert would normally look at next to the forecast, so an adjustment can be grounded in something:

- **Forecast chart**: the DNN forecast at 15-minute resolution (96 slots), with an 80% uncertainty band calibrated from the model's own recent errors (Adaptive Conformal Inference). Slots in the top or bottom 5% of that day's forecast are highlighted as unusual, and the original curve stays visible as a dashed line while the expert drags.
- **Calendar**: day of the week, whether it's a Belgian public holiday, and whether it's a bridge day (a Monday or Friday between a holiday and the weekend).
- **Day info** (sidebar): average net Belgian demand (demand minus solar and wind, the part that price-setting generators have to cover), average temperature and average humidity, with optional temperature and humidity plots.
- **Renewables chart**: solar and wind generation across the day, each togglable.
- **Recent track record**: the Deterministic Forecast Analysis page shows the DNN forecast against the actual price over the last 14 days (as a time series and as an actual-vs-forecast error scatter), so the expert can see where the model has recently been over- or under-shooting.

After submitting, experts answer a short reflection survey, including how useful the extra context was for their adjustment.

## How results are measured

Accuracy is measured with **MAE (mean absolute error, in EUR/MWh)** only, computed over the day's 15-minute slots against the realized price. Slots with a missing realized price are excluded.

- **Forecast MAE**: the model's error on its own.
- **Adjusted MAE**: the error of the expert's adjusted curve.
- **Improvement** = Forecast MAE − Adjusted MAE. Positive means the expert made the forecast more accurate.

The Expert Scoreboard aggregates this per expert across all settled days: average improvement, number of days reviewed, win rate (share of days where the adjustment beat the model) and average self-reported confidence (1–5). Days are only scored once their price has settled.

## Future work

The expert adjustments collected here could later serve as feedback for retraining the forecasting model, in the spirit of RLHF (reinforcement learning from human feedback). This is **out of scope for now**: the current study only measures whether expert adjustments improve the forecast, and the model is not trained on them.

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
| Deterministic Forecast Analysis | DNN forecast vs. actual, last 14 days: time series + error scatter. |
| Reveal & Evaluate (admin) | Accuracy for one expert's submission once the price settles. |
| Expert Scoreboard (admin) | Aggregated accuracy improvement per expert. |
| Survey Results (admin) | Reflection-survey data + experience profiles; CSV export. |
