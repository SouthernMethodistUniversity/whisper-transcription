from huggingface_hub import snapshot_download
import os

MODEL_ID = "Systran/faster-whisper-medium.en"
TARGET_DIR = "/home/appuser/app/models/faster-whisper"

os.makedirs(TARGET_DIR, exist_ok=True)

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=TARGET_DIR,
    local_dir_use_symlinks=False,
)

print(f"Downloaded {MODEL_ID} to {TARGET_DIR}")