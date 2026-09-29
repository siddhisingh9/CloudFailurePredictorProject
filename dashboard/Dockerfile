# Build from the repo root: docker build -f dashboard/Dockerfile .
FROM python:3.10-slim

WORKDIR /srv

COPY dashboard/requirements.txt dashboard/requirements.txt
RUN pip install --no-cache-dir -r dashboard/requirements.txt

COPY data/processed_gct.csv data/processed_gct.csv
COPY dashboard/ dashboard/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
