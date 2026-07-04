FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-distutils python3-pip \
        libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

RUN addgroup --system appgroup && adduser --system --ingroup appgroup --home /app appuser

COPY pyproject.toml .

# Install CUDA-enabled PyTorch (cu121 matches the base image CUDA 12.1)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision

# Copy source and install the project
COPY . .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 .

# Ensure /app is writable by appuser (EasyOCR, matplotlib, HuggingFace caches)
RUN chown -R appuser:appgroup /app

ENV HOME=/app \
    MPLCONFIGDIR=/app/.config/matplotlib \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

USER appuser

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
