from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
import whisper
import os
import tempfile
import subprocess
import glob
import shutil
from fastapi.middleware.cors import CORSMiddleware
import torch
import logging

app = FastAPI()
MAX_REQUEST_SIZE = 1024 * 1024 * 1024
logging.getLogger('nemo_logging').setLevel(logging.ERROR)

@app.middleware("http")
async def validate_request_size(request: Request, call_next):
    request_body = await request.body()
    if len(request_body) > MAX_REQUEST_SIZE:
        raise HTTPException(status_code=413, detail="Request Entity Too Large")
    return await call_next(request)

@app.on_event("startup")
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    print("Loading Whisper Turbo...")
    app.state.model_turbo = whisper.load_model(
        "turbo",
        download_root="/home/appuser/app/models",
        device=device
    )

    print("Loading Faster-Whisper...")
    import faster_whisper
    mtypes = {"cpu": "int8", "cuda": "float16"}
    whisper_model = faster_whisper.WhisperModel(
        "medium.en", device=device, compute_type=mtypes[device]
    )

    print("Loading Alignment model...")
    from ctc_forced_aligner import load_alignment_model
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    print("Loading Diarizer...")
    from diarization import MSDDDiarizer
    diarizer_model = MSDDDiarizer(device=device)

    print("Loading Punctuation model...")
    from deepmultilingualpunctuation import PunctuationModel
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    print("All models loaded successfully.")


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            file_path = tmp_file.name
            tmp_file.write(await file.read())

        result = app.state.model_turbo.transcribe(file_path)
        text = result["text"]

        os.remove(file_path)

        return JSONResponse({"transcription": text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/diarize/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            file_path = tmp_file.name
            tmp_file.write(await file.read())

        subprocess.run([
            "python",
            "diarize.py",
            "-a", 
            f"{file_path}"
        ])

        txt_path = "/tmp/" + os.path.basename(file_path) + ".txt"
        srt_path = "/tmp/" + os.path.basename(file_path) + ".srt"
        with open(txt_path, 'r', encoding='utf-8') as txt:
            text = txt.read()
        with open(srt_path, 'r', encoding='utf-8') as srt:
            srt_text = srt.read()

        # Cleanup files
        os.remove(file_path)
        os.remove(txt_path)
        os.remove(srt_path)
        #for folder in glob.glob("temp_*"):
        #    if os.path.isdir(folder):
        #        shutil.rmtree(folder)

        return JSONResponse({"transcription": text,
                             "srt": srt_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))