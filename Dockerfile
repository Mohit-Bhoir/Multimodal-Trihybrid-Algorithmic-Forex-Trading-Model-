# ── Airflow image for Astro CLI ──────────────────────────────────────────────
# Astro CLI (astro dev start) builds from this file.
# The Streamlit image is built separately via frontend/Dockerfile.

FROM quay.io/astronomer/astro-runtime:12.0.0

# Install DAG dependencies into the Airflow Python environment
COPY requirements-airflow.txt /tmp/req-airflow.txt
RUN pip install --no-cache-dir -r /tmp/req-airflow.txt
