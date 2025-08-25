#!/bin/bash
# Create a virtual environment
python3 -m venv /opt/airflow/venv

# Activate the virtual environment
source /opt/airflow/venv/bin/activate

# Upgrade pip to the latest version
pip install --quiet --upgrade pip

# Install dependencies
pip install --quiet -r /opt/airflow/dags/requirements.txt
