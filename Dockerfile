FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

RUN addgroup --system appgroup && adduser --system --group appuser

COPY pyproject.toml .

# Install CPU-only PyTorch first (avoids downloading ~2GB of CUDA packages)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# Copy source and install the project
COPY . .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

USER appuser

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
