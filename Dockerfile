# ── Backend Dockerfile ────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching — only reinstalls if requirements change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Create __init__.py files in case they are missing
RUN touch DL/__init__.py \
    DL/pipeline/__init__.py \
    simulation/__init__.py \
    backend/__init__.py \
    backend/routes/__init__.py \
    backend/services/__init__.py \
    backend/models/__init__.py

# Expose FastAPI port
EXPOSE 8000

# Start the backend
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]