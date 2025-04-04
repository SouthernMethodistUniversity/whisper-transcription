# Use NVIDIA base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    build-essential \
    && apt-get clean

# Set python3.10 as default
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

USER appuser
WORKDIR /home/appuser/app
ENV PATH="/home/appuser/.local/bin:$PATH"

# Copy and install Python dependencies
COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the app code
COPY --chown=appuser:appuser app.py ./

# Preload the Whisper model using CUDA
RUN mkdir -p /home/appuser/app/models && \
    python3 -c "import whisper; whisper._download(whisper._MODELS['base'], '/home/appuser/app/models/base.pt', False)"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]