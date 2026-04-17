FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# system deps + python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# install GPU torch (CUDA 12.4)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt sentencepiece

# copy source code + configs
COPY src/ ./src/
COPY configs/ ./configs/

# model weights are mounted at runtime, not baked into image
# this keeps the image small and allows swapping v1/v2 models without rebuild
VOLUME /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "src.api.main"]
