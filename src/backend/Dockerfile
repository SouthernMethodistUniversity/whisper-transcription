# Use PyTorch base image with Python and CUDA support
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

# Prevent tzdata and other packages from prompting for input during install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
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

# Copy the app code
COPY --chown=appuser:appuser app.py ./

EXPOSE 8000

# Start the app
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]