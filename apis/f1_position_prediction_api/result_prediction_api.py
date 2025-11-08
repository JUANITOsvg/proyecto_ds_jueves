
"""
F1 Position Result Prediction API

FastAPI application that serves the trained F1 position prediction model.
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
	title="F1 Position Result Prediction API",
	description="Predict F1 race results using a trained model.",
	version="1.0.0"
)

# Global variables for model and components
model = None
driver_encoder = None
feature_columns = None
model_package = None

def load_model():
	"""Load the trained model package"""
	global model, driver_encoder, feature_columns, model_package
	model_path = "f1_race_prediction_model.pkl"
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
				driver_encoder = model_package.get("driver_encoder")
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
class PositionPredictionRequest(BaseModel):
	year: int
	month: int
	round: int
	grid: int
	qualifying_position: Optional[int] = 20
	circuit_name: str
	driver_surname: str
	constructor_name: str
	avg_race_pos: float = 0
	avg_sprint_pos: float = 0
	avg_lap_time: float = 0
	points: float = 0
	avg_qual_pos: float = 0
	driver_encoded: int = 0

	class Config:
		schema_extra = {
			"example": {
				"year": 2023,
				"month": 5,
				"round": 1,
				"grid": 1,
				"qualifying_position": 1,
				"circuit_name": "Monaco",
				"driver_surname": "Verstappen",
				"constructor_name": "Red Bull",
				"avg_race_pos": 1.0,
				"avg_sprint_pos": 1.0,
				"avg_lap_time": 80.0,
				"points": 25.0,
				"avg_qual_pos": 1.0,
				"driver_encoded": 0
			}
		}

class PositionPredictionResponse(BaseModel):
	predicted_position: int
	confidence: float
	input_features: dict
	warnings: Optional[list] = None

@app.get("/")
async def root():
	return {
		"message": "F1 Position Result Prediction API",
		"model_loaded": model is not None,
		"endpoints": ["/health", "/predict", "/docs"]
	}

@app.get("/health")
async def health_check():
	return {
		"status": "healthy" if model is not None else "unhealthy",
		"model_loaded": model is not None
	}

@app.post("/predict", response_model=PositionPredictionResponse)
async def predict_position(request: PositionPredictionRequest):
	if model is None:
		raise HTTPException(status_code=500, detail="Model not loaded. Please ensure the model file is available.")
	try:
		# Prepare input data matching the model's expected features
		input_data = pd.DataFrame([{**request.dict()}])
		# Map API field names to model feature names if needed
		if "circuit_name" in input_data:
			input_data["name"] = input_data.pop("circuit_name")
		if "driver_surname" in input_data:
			input_data["surname"] = input_data.pop("driver_surname")
		if "constructor_name" in input_data:
			input_data["name_constructor"] = input_data.pop("constructor_name")
		warnings = []
		# Encode categorical variables
		if driver_encoder and "surname" in input_data:
			try:
				input_data["driver_encoded"] = driver_encoder.transform(input_data["surname"].astype(str))
			except Exception:
				warnings.append(f"Unknown driver_surname: {request.driver_surname}, using default")
				input_data["driver_encoded"] = 0
		# Guarantee all model-required columns exist, fill missing with 0
		input_data = input_data.reindex(columns=feature_columns, fill_value=0)
		input_data = input_data.fillna(0)
		pred = model.predict(input_data)[0]
		predicted_position = max(1, min(20, round(pred)))
		return PositionPredictionResponse(
			predicted_position=predicted_position,
			confidence=float(pred),
			input_features=input_data.iloc[0].to_dict(),
			warnings=warnings if warnings else None
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/reload-model")
async def reload_model():
	success = load_model()
	if success:
		return {"message": "Model reloaded successfully"}
	else:
		raise HTTPException(status_code=500, detail="Failed to reload model")
