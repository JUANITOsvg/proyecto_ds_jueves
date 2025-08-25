FROM apache/airflow:3.0.4-python3.11

USER root
RUN apt-get update && apt-get install -y \
    postgresql-client git \
 && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt
