from huggingface_hub import snapshot_download
import os

import os
from huggingface_hub import snapshot_download

MODELS = [
    ("Systran/faster-whisper-medium.en", "/home/appuser/app/models/faster-whisper"),
    ("MahmoudAshraf/mms-300m-1130-forced-aligner", "/home/appuser/app/models/mms-300m-1130-forced-aligner"),
]

for repo_id, target_dir in MODELS:
    os.makedirs(target_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )
    print("Downloaded", repo_id, "->", target_dir)