import os, base64, zipfile, time
import streamlit as st
import requests
import logging, json
from datetime import datetime
import sys

# Force all logs to stdout so Splunk collects them consistently
handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,
    format="%(message)s",  # Keep it JSON-friendly
)

def log_event(event, model=None):
    """Emit a structured JSON log line for Splunk."""
    entry = {
        "event": event,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user": getattr(st.user, "preferred_username", "anonymous"),
    }
    if model:
        entry["model"] = model
    logging.info(json.dumps(entry))

# ─────────── Streamlit server settings ───────────
cfg = os.path.expanduser("~/.streamlit")
os.makedirs(cfg, exist_ok=True)
with open(os.path.join(cfg, "config.toml"), "w") as f:
    f.write("""
[server]
enableWebsocketCompression = false
enableXsrfProtection      = false
headless                  = true
port                      = 8501
baseUrlPath               = ""
enableCORS                = false
            
[theme]
base = "light"
""")

st.set_page_config(page_title="Whisper Transcription",
                   page_icon="favicon.png",
                   layout="centered")

def img64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ─────────── Password protection ───────────
if not st.user.is_logged_in:
    st.markdown(f"""
    <div style="background:transparent;padding:15px;display:flex;align-items:center;justify-content:center">
    <img src="data:image/png;base64,{img64('smu_logo.png')}" width="120" style="margin-right:8px">
    <h1 style="margin:0;font-size:24px">Whisper Transcription</h1>
    </div>""", unsafe_allow_html=True)

    st.button("Sign in with SSO", icon="🔒", on_click=st.login, width="stretch")

    st.warning(
    """**Beta Notice:** This transcription tool is not part of SMU’s standard service offerings. 
    It remains under active development, and improvements or changes may occur. 
    We will do our best to notify users when significant updates are planned."""
    )

    st.stop()
    #log_event("login_success")

#st.sidebar.success(f"{st.user.preferred_username}")
st.sidebar.button("Sign out", on_click=st.logout, width="stretch")

# ─────────── Constants & helpers ───────────
ns = os.getenv("POD_NAMESPACE")
BACKEND_URL = f"http://whisper-backend-service.{ns}.svc.cluster.local:80/transcribe/"

ALLOWED_TYPES = ["mp3", "mp4", "m4a", "wav"]

def hr():
    st.markdown("<hr style='border:1px solid #ddd;'>", unsafe_allow_html=True)

def pretty(sec):
    return f"{int(sec//60)} min {round(sec%60)} sec" if sec>=60 else f"{round(sec)} sec"

def transcribe(fd, model):
    t0 = time.time()
    r  = requests.post(BACKEND_URL,
                       files={"file": (fd["name"], fd["bytes"])},
                       data={"model_size": model})
    if r.status_code == 200:
        txt = r.json().get("transcription", "")
    else:
        txt = f"Error: {r.json().get('detail', r.text)}"
    return fd["name"], txt, round(time.time() - t0, 2)

def zip_it(trs, ext):
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        z = pathlib.Path(tmp) / "transcripts.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for n, t in trs.items():
                p = pathlib.Path(tmp) / f"{n}_transcript{ext}"
                p.write_text(t, encoding="utf-8")
                zf.write(p, p.name)
        return z.read_bytes()

# ─────────── CSS for dual‑layer animated bar ───────────
def bar_html(p):  # p = % complete
    return f"""
<style>
.progress-box{{width:100%;background:#f3f3f3;border-radius:8px;overflow:hidden;height:18px;margin:12px 0;position:relative}}
.bg{{position:absolute;inset:0;background:repeating-linear-gradient(45deg,#d9e6ff 0,#d9e6ff 25%,#e8efff 25%,#e8efff 50%);background-size:40px 40px;animation:move 2s linear infinite}}
.fg{{position:absolute;inset:0;width:{p}%;background:repeating-linear-gradient(45deg,#428bfa 0,#428bfa 25%,#6ba2fb 25%,#6ba2fb 50%);background-size:40px 40px;animation:move 1s linear infinite}}
@keyframes move{{from{{background-position:0 0}}to{{background-position:40px 0}}}}
</style>
<div class="progress-box"><div class="bg"></div><div class="fg"></div></div>
"""

# ─────────── Session‑state scaffolding ───────────
state = st.session_state
for k, v in {"file_keys": (), "uploads": [], "trs": {}, "times": {}, "total": 0.0}.items():
    state.setdefault(k, v)

# ─────────── Header ───────────
st.markdown(f"""
    <div style="background:transparent;padding:15px;display:flex;align-items:center;justify-content:center">
    <img src="data:image/png;base64,{img64('smu_logo.png')}" width="120" style="margin-right:8px">
    <h1 style="margin:0;font-size:24px">Whisper Transcription</h1>
    </div>""", unsafe_allow_html=True)

st.warning(
"""**Beta Notice:** This transcription tool is not part of SMU’s standard service offerings. 
It remains under active development, and improvements or changes may occur. 
We will do our best to notify users when significant updates are planned."""
)

# ─────────── Uploads ───────────
files = st.file_uploader("Upload audio or video files",
                         type=ALLOWED_TYPES, 
                         accept_multiple_files=True)

# reset derived state when list of filenames changes
cur_keys = tuple(sorted(f.name for f in files)) if files else ()
if cur_keys != state.file_keys:
    state.file_keys = cur_keys
    state.uploads   = [{"name": f.name, "bytes": f.getvalue(), "mime": f.type}
                       for f in files] if files else []
    state.trs.clear(); state.times.clear(); state.total = 0.0

if files:
    hr()
    names = [fd["name"] for fd in state.uploads]
    prv   = st.selectbox("Select file for preview", names, key="prev")
    sel   = next(fd for fd in state.uploads if fd["name"] == prv)
    ext   = os.path.splitext(prv)[1].lower()

    # clean preview player (no stray DeltaGenerator output)
    if ext == ".mp4":
        st.video(sel["bytes"])
    else:
        st.audio(
            sel["bytes"],
            format=(
                "audio/mp4" if ext == ".m4a"
                else "audio/wav" if ext == ".wav"
                else "audio/mp3"
            )
        )

    hr()

    model = st.selectbox("Select model size", ["turbo"], index=0)

    # ─────────── Transcribe button ───────────
    if st.button("Transcribe", width="stretch"):
        log_event("transcription_started", model=model)
        state.trs.clear(); state.times.clear(); state.total = 0.0
        n         = len(state.uploads)
        bar_slot  = st.empty(); bar_slot.markdown(bar_html(0), unsafe_allow_html=True)
        msg_slot  = st.empty()
        t0_all    = time.time()

        for i, fd in enumerate(state.uploads, 1):
            msg_slot.info(f"Transcribing *{fd['name']}* ({i}/{n}) …")
            name, txt, secs = transcribe(fd, model)
            if txt.startswith("Error:"):
                st.error(f"{name} → {txt}")
            else:
                state.trs[name]   = txt
                state.times[name] = secs
            bar_slot.markdown(bar_html(int(i / n * 100)), unsafe_allow_html=True)

        bar_slot.empty(); msg_slot.success("All files transcribed!")
        state.total = round(time.time() - t0_all, 2)

# ─────────── Display transcripts ───────────
if state.trs:
    hr(); st.subheader("Transcriptions")
    st.markdown(f"**Total Time Taken:** {pretty(state.total)}")

    view = st.selectbox("Select transcript to view",
                        sorted(state.trs.keys()), key="view")

    st.markdown(
        f"**Time Taken for <span style='color:blue'>{view}</span>: "
        f"{pretty(state.times[view])}**",
        unsafe_allow_html=True,
    )
    st.code(state.trs[view], language="text", wrap_lines=True, height=300)

    hr(); st.subheader("Download Transcripts")
    fmt = st.selectbox("Select file format", [".txt"])
    st.download_button("Download",
                       zip_it(state.trs, fmt),
                       "transcripts.zip",
                       "application/zip",
                       width="stretch")