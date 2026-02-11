# Use PyTorch base image with Python and CUDA support
FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn9-runtime

# Prevent tzdata and other packages from prompting for input during install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends ffmpeg build-essential git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

USER appuser
WORKDIR /home/appuser/app
ENV PATH="/home/appuser/.local/bin:$PATH"

# Copy and install Python dependencies
COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Get and install diarization code
RUN git clone https://github.com/MahmoudAshraf97/whisper-diarization.git --depth 1
RUN pip install --no-cache-dir --user -c whisper-diarization/constraints.txt -r whisper-diarization/requirements.txt
RUN cp -r whisper-diarization/diarization /home/appuser/app/diarization
RUN rm -r whisper-diarization/

# Copy the app code
COPY --chown=appuser:appuser app.py ./

EXPOSE 8000

# Start the app
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]