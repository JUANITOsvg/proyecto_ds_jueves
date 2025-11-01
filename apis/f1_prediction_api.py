"""
F1 Race Position Prediction API

FastAPI application that serves the trained F1 race prediction model.
Load the pickled model and provide endpoints for race position predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import Optional
import os

# Initialize FastAPI app
app = FastAPI(
    title="F1 Race Position Prediction API",
    description="Predict F1 race finishing positions based on race parameters",
    version="1.0.0"
)

# Global variables for model and components
model = None
encoders = None
feature_columns = None
model_name = None
model_package = None

def load_model():
    """Load the trained model package"""
    global model, encoders, feature_columns, model_name, model_package
    
    model_path = "f1_race_prediction_model.pkl"
    
    # Try different possible locations for the model file
    possible_paths = [
        model_path,
        f"../models/{model_path}",
        f"models/{model_path}",
        f"./models/{model_path}"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    model_package = pickle.load(f)
                
                model = model_package["model"]
                encoders = model_package["label_encoders"]
                feature_columns = model_package["feature_columns"]
                
                print(f"✅ Loaded F1 Race Prediction model from: {path}")
                return True
            except Exception as e:
                print(f"❌ Error loading model from {path}: {e}")
                continue
    
    print("❌ Could not find or load model file")
    return False

# Load model on startup
load_model()

# Pydantic models for request/response
class RacePredictionRequest(BaseModel):
    year: int  # Year of race
    month: int  # Month of race
    round: int  # Round number in season
    grid: int  # Starting grid position (1-20)
    qualifying_position: Optional[int] = 20  # Qualifying position (default: back of grid)
    circuit_name: str  # Circuit name (e.g., "Silverstone Circuit")
    driver_surname: str  # Driver surname (e.g., "Hamilton")  
    constructor_name: str  # Constructor name (e.g., "Mercedes")
    
    class Config:
        schema_extra = {
            "example": {
                "year": 2023,
                "month": 7,
                "round": 10,
                "grid": 3,
                "qualifying_position": 3,
                "circuit_name": "Silverstone Circuit",
                "driver_surname": "Hamilton",
                "constructor_name": "Mercedes"
            }
        }

class RacePredictionResponse(BaseModel):
    predicted_position: int
    confidence: float
    input_features: dict
    warnings: Optional[list] = None

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "F1 Race Position Prediction API",
        "model_loaded": model is not None,
        "model": model_name if model else "No model loaded",
        "endpoints": ["/health", "/predict", "/docs"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_name": model_name if model else None
    }

@app.post("/predict", response_model=RacePredictionResponse)
async def predict_race_position(request: RacePredictionRequest):
    """Predict race finishing position based on input parameters"""
    
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please ensure the model file is available."
        )
    
    try:
        # Prepare input data matching the model's expected features
        input_data = pd.DataFrame([{
            "year": request.year,
            "month": request.month,
            "round": request.round,
            "grid": request.grid,
            "qualifying_position": request.qualifying_position,
            "name": request.circuit_name,  # Circuit name
            "surname": request.driver_surname,  # Driver surname
            "name_constructor": request.constructor_name  # Constructor name
        }])
        
        # Encode categorical features using the trained label encoders
        encoded_warnings = []
        for feature in ["name", "surname", "name_constructor"]:
            if feature in encoders:
                try:
                    # Transform using the trained encoder
                    input_data[feature] = encoders[feature].transform(input_data[feature].astype(str))
                except ValueError as ve:
                    # Handle unseen categories by using the most common category (0)
                    encoded_warnings.append(f"Unknown {feature}: {input_data[feature].iloc[0]}, using default")
                    input_data[feature] = 0
        
        # Ensure features are in the same order as training
        input_data = input_data[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Round prediction to nearest integer (race positions are integers)
        predicted_position = max(1, min(20, round(prediction)))
        
        return RacePredictionResponse(
            predicted_position=predicted_position,
            confidence=float(prediction),  # Raw model output
            input_features=input_data.iloc[0].to_dict(),
            warnings=encoded_warnings if encoded_warnings else None
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
        
        # Round to reasonable position (1-20) and format
        predicted_position = max(1.0, min(20.0, round(prediction, 1)))
        
        # Create confidence note
        confidence_note = "±2-3 positions typical accuracy"
        if encoded_warnings:
            confidence_note += f". Warnings: {', '.join(encoded_warnings)}"
        
        return RacePredictionResponse(
            predicted_position=predicted_position,
            model_used=model_name,
            confidence_note=confidence_note,
            input_summary={
                "grid_position": request.grid,
                "circuit": request.circuitId,
                "year": request.year,
                "round": request.round
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/reload-model")
async def reload_model():
    """Reload the model (useful for updates)"""
    success = load_model()
    
    if success:
        return {"message": "Model reloaded successfully", "model": model_name}
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to reload model"
        )

# For development - can be removed in production
if __name__ == "__main__":
    import uvicorn
    print("🏎️ Starting F1 Race Prediction API...")
    print("📍 API will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)