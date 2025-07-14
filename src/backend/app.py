from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
import whisper
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

MAX_REQUEST_SIZE = 1024 * 1024 * 1024

@app.middleware("http")
async def validate_request_size(request: Request, call_next):
    request_body = await request.body()
    if len(request_body) > MAX_REQUEST_SIZE:
        raise HTTPException(status_code=413, detail="Request Entity Too Large")
    return await call_next(request)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load Whisper model (local)
model = whisper.load_model("small", download_root="/home/appuser/app/models", device="cuda")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            file_path = tmp_file.name
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
        
        # Transcribe the audio
        result = model.transcribe(file_path)
        text = result["text"]

        # Clean up temp file
        os.remove(file_path)

        return JSONResponse({"transcription": text})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")

