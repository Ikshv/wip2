FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./
COPY static/ ./static/

ENV MODELS_DIR=/data/models
ENV WAKE_MODELS_DIR=/data/wakeword_models
ENV WAKE_LOG_FILE=/data/logs/mara-wakeword.log
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
