FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

LABEL maintainer="ChromoNet Authors"
LABEL description="ChromoNet: Genomic Physical Ordering-Guided Deep Learning for Single-Cell CNV Inference"
LABEL version="1.8.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir jupyter nbformat ipykernel

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY notebooks/ ./notebooks/
COPY test_consistency.py .

RUN mkdir -p data/raw data/processed data/reference results/models

CMD ["bash"]
