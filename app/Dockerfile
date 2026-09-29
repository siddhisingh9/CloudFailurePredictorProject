# Build from the repo root: docker build -f app/Dockerfile .
FROM python:3.10-slim

WORKDIR /srv

COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY models/failure_model.pkl models/failure_model.pkl
COPY app/ app/

EXPOSE 8000

# Render (and similar hosts) tell the app which port to use via $PORT
CMD ["sh", "-c", "exec uvicorn app.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
