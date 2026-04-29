# ── Streamlit frontend Docker image ───────────────────────────────────────────
# Build context: project root (so COPY . /app picks up all src/, models/, etc.)
# Heavy data dirs and secrets are excluded via .dockerignore and mounted
# as volumes at runtime — nothing sensitive is baked into the image.

FROM python:3.10-slim

WORKDIR /app

# Minimal system deps (curl for healthcheck)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first — this layer is cached unless requirements change
COPY frontend/requirements-docker.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# Copy application code (data/, models/, oanda.cfg excluded via .dockerignore
# and mounted as volumes in docker-compose.yml)
COPY . /app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "frontend/app.py", \
            "--server.address=0.0.0.0", \
            "--server.port=8501", \
            "--server.fileWatcherType=none", \
            "--server.headless=true"]
