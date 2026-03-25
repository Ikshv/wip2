FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_wakeword_docker.txt .
RUN pip install --no-cache-dir -r requirements_wakeword_docker.txt \
    && pip install --no-cache-dir "openwakeword==0.6.0" --no-deps

# PyPI wheel omits bundled ONNX feature models; fetch v0.5.1 assets used by the ONNX preprocessor.
COPY download_openwakeword_onnx_assets.py .
RUN python download_openwakeword_onnx_assets.py && rm download_openwakeword_onnx_assets.py

COPY wyoming_mara_wakeword.py config.yaml ./
COPY models/ /app/bundled_models/
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENV WAKE_MODELS_DIR=/data/wakeword_models
ENV WAKE_DETECTION_METHOD=openwakeword
EXPOSE 10300

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "wyoming_mara_wakeword.py", "--config", "/app/config.yaml"]
