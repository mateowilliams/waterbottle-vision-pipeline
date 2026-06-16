# Image for the inference + monitoring API.
FROM python:3.12-slim

# libgomp1: required by the PyTorch CPU wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (better layer caching).
# The extra index resolves torch/torchvision to their CPU variant (+cpu).
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Code + artifacts (data/ and notebooks/ are excluded via .dockerignore;
# data/ is mounted as a volume in docker-compose).
COPY . .

EXPOSE 8000

# Apply migrations and start the API.
CMD ["sh", "entrypoint.sh"]
