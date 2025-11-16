FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8007
CMD ["python","main.py"]

