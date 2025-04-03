FROM python:3.10

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

RUN apt-get update && apt-get install -y ffmpeg

USER appuser
WORKDIR /home/appuser/app

COPY requirements.txt /home/appuser/app
RUN pip install --no-cache-dir -r /home/appuser/app/requirements.txt

COPY app.py /home/appuser/app

EXPOSE 80

COPY run.sh /home/appuser/app
ENTRYPOINT ["./run.sh"]