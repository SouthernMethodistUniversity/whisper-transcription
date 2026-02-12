from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
import whisper
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware

# ---- Add diarization imports ----
from diarization import MSDDDiarizer
from ctc_forced_aligner import load_alignment_model
from deepmultilingualpunctuation import PunctuationModel
from faster_whisper import WhisperModel
import torch
import torchaudio
import logging
import traceback

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
    compute_type = "float16" if device == "cuda" else "int8"
    app.state.model_fast = WhisperModel(
        "medium.en",
        #"/home/appuser/app/models/faster-whisper",
        device=device,
        compute_type=compute_type
    )

    print("Loading Alignment model...")
    app.state.alignment_model, app.state.alignment_tokenizer = load_alignment_model(
        device=device,
        #"/home/appuser/app/models/mms-300m-1130-forced-aligner",
        dtype=torch.float16 if device == "cuda" else torch.float32
    )

    print("Loading Diarizer...")
    app.state.diarizer = MSDDDiarizer(device=device)

    print("Loading Punctuation model...")
    app.state.punct_model = PunctuationModel()#model="/home/appuser/app/models/punctuate-all")

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
async def diarize_audio(file: UploadFile = File(...)):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            file_path = tmp_file.name
            tmp_file.write(await file.read())

        # ---- 1. Transcribe ----
        segments, info = app.state.model_fast.transcribe(file_path)
        segments = list(segments)

        wav, sr = torchaudio.load(file_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        wav = wav.to(device)

        # ---- 2. Run diarization ----
        diarized_segments = app.state.diarizer.diarize(file_path)

        # ---- 3. Merge transcript + speakers ----
        merged_segments = []
        previous_speaker = None
        transcription_text = ""

        for segment in segments:
            speaker = "Unknown"

            for d in diarized_segments:
                if segment.start >= d["start"] and segment.end <= d["end"]:
                    speaker = d["speaker"]
                    break

            # Structured output
            merged_segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker,
                "text": segment.text
            })

            # ---- Build formatted transcription string ----
            if speaker != previous_speaker:
                if previous_speaker is not None:
                    transcription_text += "\n\n"  # blank line between speakers
                transcription_text += f"{speaker}: {segment.text.strip()}"
            else:
                transcription_text += f" {segment.text.strip()}"

            previous_speaker = speaker

        os.remove(file_path)

        return JSONResponse({
            "segments": merged_segments,
            "transcription": transcription_text
        })

    except Exception as e:
        logging.exception("ERROR in /diarize")
        raise HTTPException(status_code=500, detail=str(e))