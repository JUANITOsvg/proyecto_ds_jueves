#!/bin/bash
# setup_models.sh - Regenerate the ML model

echo "🔄 Setting up F1 Race Prediction Model..."

# Ensure we're in the project root
cd "$(dirname "$0")"

# Create models directory if it doesn't exist
mkdir -p models

# Check if Python environment is active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment detected: $VIRTUAL_ENV"
else
    echo "No virtual environment detected. Activating .venv..."
    source .venv/bin/activate 2>/dev/null || {
        echo "Could not activate virtual environment"
        echo "Please run: source .venv/bin/activate"
        exit 1
    }
fi

# Run the model building script
echo "🏗️  Building F1 race prediction model..."
python scripts/build_model.py

# Check if model was created successfully
if [[ -f "models/f1_race_prediction_model.pkl" ]]; then
    echo "Model created successfully!"
    echo "Model file size: $(ls -lh models/f1_race_prediction_model.pkl | awk '{print $5}')"
else
    echo "Model creation failed!"
    exit 1
fi

echo "🎯 Setup complete! The API can now use the trained model."