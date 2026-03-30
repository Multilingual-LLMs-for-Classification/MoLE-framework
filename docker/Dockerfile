# Base image with CUDA support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# psmisc (killall) and procps (ps) are required by Ray for process management
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    git \
    psmisc \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code and moe_router package
COPY app/ ./app/
COPY moe_router/ ./moe_router/
COPY expert_worker/ ./expert_worker/
COPY ray_cluster/ ./ray_cluster/
COPY tests/ ./tests/

# Set Python path
ENV PYTHONPATH="/app"

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Note: Running as root for now to access GPU and mounted volumes
# In production, configure proper permissions
# USER appuser

# Expose ports
# 8000 — coordinator FastAPI
# 6379 — Ray GCS (head node)
# 8265 — Ray dashboard
# 10001 — Ray client server
EXPOSE 8000 6379 8265 10001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
