# Runs the EPF Expert Review Streamlit app as a standalone container, no
# platform-specific buildpack needed, so this same image works on Fly.io,
# Render, Cloud Run, or anywhere else that can run a Docker image.
FROM python:3.11-slim

# libpq isn't strictly required (psycopg2-binary bundles its own libpq), but
# curl is handy for the container-level health check below.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Streamlit's own health endpoint, used by Fly.io / most platforms to know
# the app is actually ready, not just that the process started.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
