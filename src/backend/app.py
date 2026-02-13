from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
import whisper
import os
import tempfile
import re
import io
import subprocess
import glob
import shutil
from fastapi.middleware.cors import CORSMiddleware

# ---- Add diarization imports ----
#from diarization import MSDDDiarizer
#from deepmultilingualpunctuation import PunctuationModel
#from faster_whisper import WhisperModel
import torch
import torchaudio
import logging
import traceback

# from ctc_forced_aligner import (
#     generate_emissions,
#     get_alignments,
#     get_spans,
#     load_alignment_model,
#     postprocess_results,
#     preprocess_text,
# )

# from helpers import (
#     cleanup,
#     find_numeral_symbol_tokens,
#     get_realigned_ws_mapping_with_punctuation,
#     get_sentences_speaker_mapping,
#     get_speaker_aware_transcript,
#     get_words_speaker_mapping,
#     langs_to_iso,
#     process_language_arg,
#     punct_model_langs,
#     whisper_langs,
#     write_srt,
# )

app = FastAPI()

MAX_REQUEST_SIZE = 1024 * 1024 * 1024

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
    # compute_type = "float16" if device == "cuda" else "int8"
    # app.state.model_fast = WhisperModel(
    #     "medium.en",
    #     #"/home/appuser/app/models/faster-whisper",
    #     device=device,
    #     compute_type=compute_type
    # )
    import faster_whisper
    mtypes = {"cpu": "int8", "cuda": "float16"}
    whisper_model = faster_whisper.WhisperModel(
        "medium.en", device=device, compute_type=mtypes[device]
    )

    print("Loading Alignment model...")
    # app.state.alignment_model, app.state.alignment_tokenizer = load_alignment_model(
    #     device=device,
    #     #"/home/appuser/app/models/mms-300m-1130-forced-aligner",
    #     dtype=torch.float16 if device == "cuda" else torch.float32
    # )
    from ctc_forced_aligner import load_alignment_model
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # print("Loading Diarizer...")
    from diarization import MSDDDiarizer
    diarizer_model = MSDDDiarizer(device=device)

    print("Loading Punctuation model...")
    # app.state.punct_model = PunctuationModel()#model="/home/appuser/app/models/punctuate-all")
    from deepmultilingualpunctuation import PunctuationModel
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    print("All models loaded successfully.")

# Load Whisper model (local)
#model_turbo = whisper.load_model("turbo", download_root="/home/appuser/app/models", device="cuda")

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

        #result = app.state.model_turbo.transcribe(file_path)

        subprocess.run([
            "python",
            "diarize.py",
            "-a", 
            f"{file_path}"
        ])

        #text = result["text"]
        txt_path = "/tmp/" + os.path.basename(file_path) + ".txt"
        srt_path = "/tmp/" + os.path.basename(file_path) + ".srt"
        with open(txt_path, 'r', encoding='utf-8') as txt:
            text = txt.read()

        # Cleanup files
        os.remove(file_path)
        os.remove(txt_path)
        os.remove(srt_path)
        for folder in glob.glob("temp_*"):
            if os.path.isdir(folder):
                shutil.rmtree(folder)

        return JSONResponse({"transcription": text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))