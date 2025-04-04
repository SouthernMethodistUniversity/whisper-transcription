import os

streamlit_config_path = os.path.expanduser("~/.streamlit")
os.makedirs(streamlit_config_path, exist_ok=True)

with open(os.path.join(streamlit_config_path, "config.toml"), "w") as f:
    f.write("""
[server]
enableWebsocketCompression = false
enableXsrfProtection = false
headless = true
port = 8501
baseUrlPath = ""
enableCORS = false
""")

import streamlit as st
import requests
import tempfile
import time
import base64
import zipfile
import concurrent.futures
from moviepy import VideoFileClip, AudioFileClip

# Backend URL
backend_url = "/transcribe/"

@st.cache_data
def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_base64 = get_image_as_base64("smu_logo.png")

# Header
st.markdown(
    f"""
    <div style="background-color: white; padding: 15px; display: flex; align-items: center; justify-content: center; color: black;">
        <img src="data:image/png;base64,{image_base64}" width="120" style="margin-right: 2px;">
        <h1 style="margin: 0; font-size: 24px; text-align: center;">Whisper Transcription</h1>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)

# Divider Function
def add_divider():
    st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)

# Initialize session state for file and transcription
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
    st.session_state.transcriptions = {}
    st.session_state.trans_times = {}
    st.session_state.selected_file = None
    st.session_state.zip_data = None

# Function to convert video/audio to mp3
def convert_to_mp3(file, file_name):
    if file_name.endswith(".mp4"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(file.read())
            video_path = temp_video.name
        
        audio_path = video_path.replace(".mp4", ".mp3")
        try:
            video_clip = VideoFileClip(video_path)
            video_clip.audio.write_audiofile(audio_path, logger=None)
            video_clip.close()
            os.unlink(video_path)
            return audio_path, video_path
        except Exception as e:
            st.error(f"Error converting video: {e}")
            return None, video_path
            
    elif file_name.endswith(".m4a"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
            temp_audio.write(file.read())
            m4a_path = temp_audio.name
            
        mp3_path = m4a_path.replace(".m4a", ".mp3")
        try:
            audio_clip = AudioFileClip(m4a_path)
            audio_clip.write_audiofile(mp3_path, logger=None)
            audio_clip.close()
            os.unlink(m4a_path)
            return mp3_path, None
        except Exception as e:
            st.error(f"Error converting audio: {e}")
            return None, None
            
    else:  # Already mp3
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(file.read())
            return temp_audio.name, None

# Function to transcribe a single file
def transcribe_file(file_path, file_name, model_size):
    start_time = time.time()
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"model_size": model_size}
            response = requests.post(backend_url, files=files, data=data)
        
        if response.status_code == 200:
            transcription = response.json().get("transcription")
            trans_time = round(time.time() - start_time, 2)
            return file_name, transcription, trans_time
        else:
            error_msg = response.json().get('detail', 'Unknown error')
            return file_name, f"Error: {error_msg}", 0
    except Exception as e:
        return file_name, f"Error: {str(e)}", 0
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.unlink(file_path)

# Generate zip file of transcriptions with given format
@st.cache_data
def generate_zip(transcriptions, file_format):
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "transcripts.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_name, transcript in transcriptions.items():
                transcript_file = os.path.join(temp_dir, f"{file_name}_transcript{file_format}")
                with open(transcript_file, "w") as f:
                    f.write(transcript)
                zipf.write(transcript_file, os.path.basename(transcript_file))
        
        with open(zip_path, "rb") as f:
            return f.read()

# File Uploader
allowed_input_types = ["mp3", "mp4", "m4a"]
uploaded_files = st.file_uploader("Upload audio or video files", type=allowed_input_types, accept_multiple_files=True)

if uploaded_files != st.session_state.uploaded_files:
    st.session_state.transcriptions.clear()
    st.session_state.trans_times.clear()
    st.session_state.selected_file = None
    st.session_state.zip_data = None

if uploaded_files:
    # Store uploaded file in session state
    st.session_state.uploaded_files = uploaded_files
    add_divider()

    file_names = [file.name for file in uploaded_files]
    selected_file = st.selectbox("Select file for playback", file_names)

    if selected_file != st.session_state.selected_file:
        st.session_state.selected_file = selected_file
        st.session_state.show_video = selected_file.endswith(".mp4")

    # Process the selected file for playback
    selected_file_obj = next((f for f in uploaded_files if f.name == selected_file), None)
    if selected_file_obj:
        if selected_file.endswith(".mp4"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(selected_file_obj.getvalue())
                video_path = temp_video.name
                
            if st.session_state.show_video:
                st.video(video_path)
                if os.path.exists(video_path):
                    os.unlink(video_path)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{selected_file.split('.')[-1]}") as temp_audio:
                temp_audio.write(selected_file_obj.getvalue())
                audio_path = temp_audio.name
            
            if selected_file.endswith(".m4a"):
                st.audio(audio_path, format="audio/mp4")
            else:
                st.audio(audio_path, format="audio/mp3")

            if os.path.exists(audio_path):
                os.unlink(audio_path)

    add_divider()

    # Model Selection Dropdown
    model_size = st.selectbox("Select model size", ["tiny", "base", "small", "medium", "large"], index=1)

    # Submit Button for Transcription
    if st.button("Transcribe"):
        # Clear the previous transcription before starting new
        st.session_state.transcriptions.clear()
        st.session_state.trans_times.clear()
        st.session_state.zip_data = None
        
        total_start_time = time.time()
        num_files = len(uploaded_files)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Convert files in parallel
        temp_paths = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(convert_to_mp3, file, file.name): file.name for file in uploaded_files}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                file_name = future_to_file[future]
                progress = (i + 1) / (num_files * 2)  # First half of progress bar for conversion
                progress_bar.progress(progress)
                status_text.info(f"Converted {i+1} of {num_files}: {file_name}")
                
                try:
                    audio_path, _ = future.result()
                    if audio_path:
                        temp_paths[file_name] = audio_path
                except Exception as e:
                    st.error(f"Error processing {file_name}: {e}")
        
        if temp_paths:
            # Transcribe all files in parallel
            status_text.info("Transcribing files...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit all transcription tasks
                future_to_file = {
                    executor.submit(transcribe_file, path, name, model_size): name 
                    for name, path in temp_paths.items()
                }
                
                # Process results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                    file_name = future_to_file[future]
                    progress = 0.5 + (i + 1) / (len(temp_paths) * 2)  # Second half of progress bar for transcription
                    progress_bar.progress(progress)
                    status_text.info(f"Transcribed {i+1} of {len(temp_paths)}: {file_name}")
                    
                    try:
                        name, transcription, trans_time = future.result()
                        if not transcription.startswith("Error:"):
                            st.session_state.transcriptions[name] = transcription
                            st.session_state.trans_times[name] = trans_time
                        else:
                            st.error(f"Error transcribing {name}: {transcription}")
                    except Exception as e:
                        st.error(f"Error processing {file_name}: {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        total_time = round(time.time() - total_start_time, 2)
        st.session_state.total_trans_time = total_time

def format_time(seconds):
    if seconds >= 60:
        minutes = int(seconds // 60)
        remaining_seconds = round(seconds % 60)
        return f"{minutes} min {remaining_seconds} sec"
    return f"{round(seconds)} sec"

# Display transcription if available
if st.session_state.transcriptions:
    add_divider()

    st.subheader("Transcriptions")
    st.markdown(f"**Total Time Taken:** {format_time(st.session_state.total_trans_time)}")

    selected_transcription = st.selectbox("Select transcript to view", list(st.session_state.transcriptions.keys()))
    st.markdown(f"**Time Taken for <span style='color: blue;'>{selected_transcription}</span>:** {format_time(st.session_state.trans_times[selected_transcription])}", unsafe_allow_html=True)
    st.code(st.session_state.transcriptions[selected_transcription], language='text', wrap_lines=True, height=300)

    add_divider()
    
    # Download Section
    allowed_output_types = [".txt"]
    st.subheader("Download Transcripts")
    file_format = st.selectbox("Select file format", allowed_output_types)
    
    transcription_items = tuple(sorted(st.session_state.transcriptions.items()))
    
    # Generate the zip file
    st.download_button(
        "Download", 
        generate_zip(st.session_state.transcriptions, file_format),
        "transcripts.zip", 
        "application/zip"
    )
