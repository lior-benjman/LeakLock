FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/ultralytics
ENV LEAKLOCK_SKIP_STARTUP_WARMUP=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY leaklock ./leaklock
COPY runs_sensitive/yolov86/weights/best.pt ./runs_sensitive/yolov86/weights/best.pt

CMD ["sh", "-c", "python -m uvicorn leaklock.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
