# Use PyTorch base image with Python and CUDA support
FROM python:3.11.16-slim-bookworm

# Prevent tzdata and other packages from prompting for input during install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends ffmpeg build-essential git \
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

# Create non-root user
RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash appuser

USER appuser
WORKDIR /home/appuser/app
ENV PATH="/home/appuser/.local/bin:$PATH"

# Copy and install Python dependencies
COPY --chown=appuser:appuser requirements.txt ./
COPY --chown=appuser:appuser constraints.txt ./

# Install Python dependencies first
RUN pip install --no-cache-dir --user -c constraints.txt -r requirements.txt

# Set CUDA/PyTorch library paths before importing torch
ENV LD_LIBRARY_PATH="/home/appuser/.local/lib/python3.11/site-packages/torch/lib:\
/home/appuser/.local/lib/python3.11/site-packages/nvidia/cublas/lib:\
/home/appuser/.local/lib/python3.11/site-packages/nvidia/cudnn/lib:\
/home/appuser/.local/lib/python3.11/site-packages/nvidia/npp/lib:\
/home/appuser/.local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\
/home/appuser/.local/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"

# Optional torch import check
RUN python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Get and install diarization code, pinned to a known commit so upstream
# changes can't silently break the sed patch below on a future rebuild
RUN git init whisper-diarization && \
    cd whisper-diarization && \
    git remote add origin https://github.com/MahmoudAshraf97/whisper-diarization.git && \
    git fetch --depth 1 origin c6614d717764592c4261f2590c1eb29e87948c3e && \
    git checkout FETCH_HEAD
RUN cp -r whisper-diarization/diarization /home/appuser/app/diarization
RUN cp whisper-diarization/helpers.py /home/appuser/app/helpers.py
RUN cp whisper-diarization/diarize.py /home/appuser/app/diarize.py
RUN sed -i 's|temp_path = os.path.join(os.getcwd(), f"temp_outputs_{os.getpid()}")|temp_path = os.path.join("/tmp", f"temp_outputs_{os.getpid()}")|' /home/appuser/app/diarize.py
RUN rm -r whisper-diarization/

# Copy the app code
COPY --chown=appuser:appuser app.py ./

EXPOSE 8000

# Start the app
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]