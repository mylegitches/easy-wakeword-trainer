FROM python:3.11-slim-bookworm

# System deps for audio processing and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone openWakeWord (pinned to v0.6.0)
RUN git clone --depth 1 --branch v0.6.0 https://github.com/dscripka/openWakeWord /app/openwakeword

# Clone piper-sample-generator (tag v2.0.0 is stable for piper-phonemize)
RUN git clone --depth 1 --branch v2.0.0 https://github.com/rhasspy/piper-sample-generator /app/piper-sample-generator

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install openWakeWord in editable mode so train.py is on the path
RUN pip install --no-cache-dir -e /app/openwakeword

# Install onnx2tf for TFLite export (separate from openWakeWord deps)
RUN pip install --no-cache-dir onnx2tf

# Copy application code
COPY app/ /app/app/
COPY scripts/ /app/scripts/
COPY make_validation_small.py /app/make_validation_small.py

# Outputs go to /outputs (bind-mounted from host)
# Data files go to /data (bind-mounted from host)
RUN mkdir -p /outputs /data

ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data
ENV OUTPUT_DIR=/outputs
ENV OPENWAKEWORD_DIR=/app/openwakeword
ENV PIPER_GENERATOR_DIR=/app/piper-sample-generator

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
