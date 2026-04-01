FROM python:3.11-slim

# System deps: ffmpeg (audio), tesseract (OCR), fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY config/ config/
COPY telegram_bot.py .
COPY ingest.py .
COPY register_patients.py .
COPY migrate_olga.py .

# Data volume — patient data persists outside container
VOLUME /app/data

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from src.patient_manager import _load_registry; _load_registry(); print('ok')" || exit 1

CMD ["python", "-X", "utf8", "telegram_bot.py"]
