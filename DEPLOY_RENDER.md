# Deploying to Render

Render is the most commonly used host for exactly this situation: a Python web app
(Streamlit, in this case) that's outgrown Streamlit Community Cloud's free, shared,
sleep-after-inactivity containers. It's entirely web-dashboard driven, no command-line
tool required. Your database doesn't move, Supabase Postgres stays exactly as it is;
Render just runs the Streamlit process itself, on a small always-on instance you pay for.

This uses the same `Dockerfile` from this delivery, so if you ever want to switch to a
different host later, nothing about the app itself needs to change.

## 0. Before you start: push these files to GitHub

Render deploys from a GitHub (or GitLab) repo, so first commit and push everything from
this delivery, plus your existing `app.py`, `db.py`, `requirements.txt`, and
`draggable_curve/`, into your repo. Make sure a `.env` file is **not** committed
(it shouldn't be tracked already, but double-check `.gitignore` includes it), secrets go
into Render's dashboard instead, see step 3.

## 1. Create a Render account and connect your repo

1. Go to https://render.com and sign up (or log in).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub account if you haven't already, then select this repo.

## 2. Configure the service

Render should auto-detect the `Dockerfile` and offer Docker as the runtime. If it asks
you to pick a blueprint, you can instead click **New +** -> **Blueprint** and point it at
the repo, it will read `render.yaml` (included in this delivery) and pre-fill everything
below automatically. Either way, confirm these settings:

- **Runtime**: Docker
- **Region**: Frankfurt (closest Render region to Belgium and most EU Supabase projects;
  check your Supabase project's region under **Project Settings -> General** and pick
  whichever Render region is actually closest to it)
- **Instance type**: **Starter** (~$7/mo). This is the smallest tier that stays always-on;
  the free tier sleeps after inactivity, which is the exact problem we're fixing.
- **Health check path**: `/_stcore/health` (Streamlit's own health endpoint, already set
  in `render.yaml` if you used the Blueprint option)

## 3. Set your environment variables

In the service's **Environment** tab, add:

- `DATABASE_URL`: your Supabase connection string, from Supabase's
  **Project Settings -> Database -> Connection string -> URI** (the same one you were
  already using on Streamlit Community Cloud).
- `GITHUB_TOKEN`: your GitHub personal access token, for the private forecast repo.

These are stored encrypted on Render and never appear in your repo.

## 4. Deploy

Click **Create Web Service** (or **Apply** if you used the Blueprint). Render builds the
Docker image and deploys it, first build usually takes a couple of minutes. You'll get a
live URL like `https://epf-expert-review.onrender.com`.

## 5. Try it

Open the URL, log in, and click through Review & Adjust, Deterministic Forecast Analysis,
and the admin pages. Page switches and submissions should feel noticeably snappier than
on Streamlit Community Cloud, since this instance never spins down between visits.

## Day-to-day after this

- **Redeploying after a code change**: push to your GitHub repo's main branch, Render
  redeploys automatically. (Configurable in the service's Settings if you'd rather deploy
  manually.)
- **Logs**: visible directly in the Render dashboard, under the service's **Logs** tab.
- **Cost**: Starter instance is a flat ~$7/mo, comfortably in your budget, and predictable
  (unlike usage-based pricing) since it's a fixed monthly rate regardless of traffic.

## What's already been verified

The `Dockerfile` and the exact command it runs (`streamlit run app.py --server.port=8080
--server.address=0.0.0.0 --server.headless=true`) were tested end-to-end in a clean
Python environment against a real Postgres database: dependencies install cleanly from
`requirements.txt`, the app boots, and Streamlit's own health endpoint
(`/_stcore/health`, the same one Render polls to know your app is actually ready) returns
healthy. The one part not testable from this side is the actual Render build/deploy
round-trip, since this sandbox has no network path to GitHub pushes or Render's servers,
that happens for the first time when you follow the steps above.

## If you'd rather not use GitHub

Render also supports deploying directly from a local folder via their CLI, but the
GitHub-connected flow above is simpler and gives you auto-redeploy on every push for
free, worth setting up a repo for even if you haven't already.
