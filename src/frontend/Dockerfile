FROM python:3.11.16-slim-bookworm

# Create non-root user
RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash appuser

# Install dependencies (ffmpeg for audio processing)
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends ffmpeg build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Upgrade the system-wide pip/setuptools/wheel as root, before switching to
# the non-root user, so the outdated versions baked into the base image are
# actually replaced rather than shadowed by a separate --user install
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ensurepip bundles its own frozen pip/setuptools wheels for bootstrapping a
# Python install that has neither. This image already has an upgraded pip,
# and nothing here ever re-invokes ensurepip, so the stale bundled wheels
# (and whatever they vendor internally) are unused dead weight.
RUN rm -rf /usr/local/lib/python3.11/ensurepip/_bundled

# Switch to non-root user
USER appuser
WORKDIR /home/appuser/app

# Make sure pip installs to user directory
ENV PATH="/home/appuser/.local/bin:$PATH"

# Copy app and install dependencies
COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser smu_logo.png ./
COPY --chown=appuser:appuser favicon.png ./

# Expose Streamlit at default port
EXPOSE 8501

# Start Streamlit (via python -m to ensure it works)
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableWebsocketCompression=false", \
     "--server.enableXsrfProtection=false", \
     "--server.enableCORS=false", \
     "--server.maxUploadSize=1024"]