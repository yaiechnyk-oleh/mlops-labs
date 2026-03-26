# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Install all heavy Python dependencies here so the final image stays small.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for some packages (scikit-learn, scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
# Lightweight final image — only essentials, no build tools.
FROM python:3.11-slim

WORKDIR /app

# Copy pre-built Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy project source
COPY src/       ./src/
COPY config/    ./config/
COPY dvc.yaml   ./dvc.yaml

# Make user-installed binaries available (mlflow, dvc, etc.)
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Default: run training with standard paths
# Override via: docker run ... python src/train.py <input_dir> <output_dir>
CMD ["python", "src/train.py", "data/prepared", "data/models"]
