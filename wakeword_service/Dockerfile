FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_wakeword_docker.txt .
RUN pip install --no-cache-dir -r requirements_wakeword_docker.txt

COPY wyoming_mara_wakeword.py config.yaml ./

ENV WAKE_MODELS_DIR=/data/wakeword_models
ENV WAKE_DETECTION_METHOD=whisper
EXPOSE 10300

CMD ["python", "wyoming_mara_wakeword.py", "--config", "/app/config.yaml"]
