FROM python:3.10-slim

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Install dependencies (ffmpeg for audio processing)
RUN apt-get update && apt-get install -y ffmpeg build-essential && apt-get clean

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

# Expose Streamlit default port
EXPOSE 8501

# Start Streamlit (via python -m to ensure it works)
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableWebsocketCompression=false", \
     "--server.enableXsrfProtection=false", \
     "--server.enableCORS=false", \
     "--server.maxUploadSize=1024"]