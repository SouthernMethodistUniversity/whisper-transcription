from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
import whisper
import os
import tempfile
import re
import io
import subprocess
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

    # print("Loading Faster-Whisper...")
    # compute_type = "float16" if device == "cuda" else "int8"
    # app.state.model_fast = WhisperModel(
    #     "medium.en",
    #     #"/home/appuser/app/models/faster-whisper",
    #     device=device,
    #     compute_type=compute_type
    # )

    # print("Loading Alignment model...")
    # app.state.alignment_model, app.state.alignment_tokenizer = load_alignment_model(
    #     device=device,
    #     #"/home/appuser/app/models/mms-300m-1130-forced-aligner",
    #     dtype=torch.float16 if device == "cuda" else torch.float32
    # )

    # print("Loading Diarizer...")
    # app.state.diarizer = MSDDDiarizer(device=device)

    # print("Loading Punctuation model...")
    # app.state.punct_model = PunctuationModel()#model="/home/appuser/app/models/punctuate-all")

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
        out_path = os.path.basename(file_path) + ".txt"
        with open(out_path, 'r', encoding='utf-8') as out:
            text = out.read()

        os.remove(file_path)
        os.remove(out_path)

        return JSONResponse({"transcription": text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/diarize/")
# async def diarize_audio(file: UploadFile = File(...)):
#     file_path = None
#     try:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         batch_size = 8                  # match repo default
#         suppress_numerals = False       # match repo default

#         # ---- Save upload to temp file (needed for faster-whisper transcribe(path)) ----
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
#             file_path = tmp_file.name
#             tmp_file.write(await file.read())

#         # ---- 1) Transcribe (faster-whisper) ----
#         suppress_tokens = (
#             find_numeral_symbol_tokens(app.state.model_fast.hf_tokenizer)
#             if suppress_numerals
#             else [-1]
#         )

#         segments_iter, info = app.state.model_fast.transcribe(
#             file_path,
#             language=None,                 # detect
#             suppress_tokens=suppress_tokens,
#             vad_filter=True,
#         )
#         transcript_segments = list(segments_iter)
#         full_transcript = "".join(seg.text for seg in transcript_segments)

#         # ---- 2) Load audio -> resample/mono @ 16k ----
#         wav, sr = torchaudio.load(file_path)      # (C, T)
#         if sr != 16000:
#             wav = torchaudio.functional.resample(wav, sr, 16000)
#         if wav.shape[0] > 1:
#             wav = wav.mean(dim=0, keepdim=True)   # (1, T)

#         audio_1d = wav.squeeze(0)                 # (T,)

#         # ---- 3) Forced alignment (preloaded) ----
#         # Same as repo: generate_emissions -> preprocess_text -> get_alignments -> get_spans -> postprocess_results
#         with torch.inference_mode():
#             emissions, stride = generate_emissions(
#                 app.state.alignment_model,
#                 audio_1d.to(app.state.alignment_model.dtype).to(app.state.alignment_model.device),
#                 batch_size=batch_size,
#             )

#         lang = getattr(info, "language", None)
#         iso = langs_to_iso.get(lang, "en")

#         tokens_starred, text_starred = preprocess_text(
#             full_transcript,
#             romanize=True,
#             language=iso,
#         )

#         aligned_segments, scores, blank_token = get_alignments(
#             emissions,
#             tokens_starred,
#             app.state.alignment_tokenizer,
#         )

#         spans = get_spans(tokens_starred, aligned_segments, blank_token)
#         word_timestamps = postprocess_results(text_starred, spans, stride, scores)

#         # ---- 4) Diarize (preloaded) ----
#         # IMPORTANT: library uses torchaudio.save() internally -> must pass CPU Tensor, shape (1, T)
#         speaker_ts = app.state.diarizer.diarize(audio_1d.unsqueeze(0).detach().cpu())
#         # speaker_ts is typically List[Tuple[start_ms, end_ms, spk_id]]

#         # ---- 5) Map words->speakers using helpers.py ----
#         wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

#         # ---- 6) Restore punctuation using preloaded punct model (same logic as repo) ----
#         if lang in punct_model_langs:
#             words_list = [x["word"] for x in wsm]
#             labeled_words = app.state.punct_model.predict(words_list, chunk_size=230)

#             ending_puncts = ".?!"
#             model_puncts = ".,;:!?"
#             is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

#             for word_dict, labeled_tuple in zip(wsm, labeled_words):
#                 word = word_dict["word"]
#                 punct = labeled_tuple[1]
#                 if (
#                     word
#                     and punct in ending_puncts
#                     and (word[-1] not in model_puncts or is_acronym(word))
#                 ):
#                     word = word + punct
#                     if word.endswith(".."):
#                         word = word.rstrip(".")
#                     word_dict["word"] = word
#         else:
#             logging.warning(f"Punctuation model not available for language={lang}; leaving punctuation as-is.")

#         # ---- 7) Sentence mapping using helpers.py ----
#         wsm = get_realigned_ws_mapping_with_punctuation(wsm)
#         ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

#         # ---- 8) Produce speaker-aware transcript exactly like repo helper ----
#         buf = io.StringIO()
#         get_speaker_aware_transcript(ssm, buf)
#         speaker_aware_text = buf.getvalue()

#         # Also return sentence segments (ssm is already sentence-level)
#         # We'll pass through what helpers produced, rather than reformatting aggressively.
#         return JSONResponse({
#             "language": lang,
#             "segments": ssm,
#             "transcription": speaker_aware_text,
#         })

#     except Exception as e:
#         logging.exception("ERROR in /diarize")
#         raise HTTPException(status_code=500, detail=str(e))

#     finally:
#         if file_path and os.path.exists(file_path):
#             try:
#                 os.remove(file_path)
#             except Exception:
#                 pass