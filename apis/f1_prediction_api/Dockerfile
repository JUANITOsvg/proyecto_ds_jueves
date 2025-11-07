# F1 Race Prediction API Docker Container
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY apis/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and model file
COPY apis/f1_prediction_api/f1_prediction_api.py .
COPY models/f1_race_prediction_model.pkl .

# Create models directory for volume mount
RUN mkdir -p models

# Note: Model file will be mounted via volume from host

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' apiuser && \
    chown -R apiuser:apiuser /app
USER apiuser

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Command to run the application
CMD ["uvicorn", "f1_prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]