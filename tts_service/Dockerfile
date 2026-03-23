FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_wyoming.txt .
RUN pip install --no-cache-dir -r requirements_wyoming.txt

COPY wyoming_mara_tts.py ./
COPY models/ /app/bundled_models/
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENV MODELS_DIR=/data/models
EXPOSE 10200

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "wyoming_mara_tts.py", "--host", "0.0.0.0", "--port", "10200", "--models-dir", "/data/models"]
