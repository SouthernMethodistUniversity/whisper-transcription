FROM python:3.10

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

RUN apt-get update && apt-get install -y ffmpeg

USER appuser
WORKDIR /home/appuser/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableWebsocketCompression=false", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]