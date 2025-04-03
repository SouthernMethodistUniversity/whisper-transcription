FROM python:3.10

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

CMD ["streamlit", "run", "app.py", "--server.port", "80", "--server.address", "0.0.0.0"]
