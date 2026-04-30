# ── Airflow image for Astro CLI ──────────────────────────────────────────────
# Astro CLI (astro dev start) builds from this file.
# Astro automatically installs requirements.txt via ONBUILD — do NOT add
# tensorflow or other heavy packages there; use requirements-ml.txt locally.
#
# The Streamlit image is built separately via frontend/Dockerfile.

FROM quay.io/astronomer/astro-runtime:12.0.0
