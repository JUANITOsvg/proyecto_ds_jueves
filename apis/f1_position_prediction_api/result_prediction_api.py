
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
import random

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
	model_path = "f1_race_position_model.pkl"
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
	avg_race_pos: float
	avg_sprint_pos: float = 0.0  # Optional, default 0.0
	avg_lap_time: float
	points: float
	avg_qual_pos: float
	forename: str
	surname: str

	class Config:
		schema_extra = {
			"example": {
				"avg_race_pos": 5.02,
				"avg_sprint_pos": 6.78,
				"avg_lap_time": 96.75,
				"points": 223.0,
				"avg_qual_pos": 4.07,
				"forename": "Lewis",
				"surname": "Hamilton"
			}
		}

class PositionPredictionResponse(BaseModel):
	win_probability: float
	win: bool
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
		req_dict = request.dict()
		warnings = []
		# Prepare driver_encoded
		driver_name = f"{req_dict['forename']} {req_dict['surname']}"
		if 'driver_encoded' in feature_columns:
			if driver_encoder is not None:
				try:
					driver_encoded = driver_encoder.transform([driver_name])[0]
				except Exception:
					driver_encoded = 0
					warnings.append(f"Unknown driver: {driver_name}, using default encoding 0.")
				req_dict['driver_encoded'] = driver_encoded
			else:
				raise HTTPException(status_code=500, detail="Driver encoder not found in model package.")
		# Prepare input data for prediction, fill NaN with 0
		input_data = pd.DataFrame([{col: req_dict.get(col, 0) for col in feature_columns}])
		input_data = input_data.fillna(0)
		print("Input columns for prediction:", list(input_data.columns))
		print("Input row:", input_data.iloc[0].to_dict())
		# Use model to predict win probability
		if hasattr(model, "predict_proba"):
			win_proba = float(model.predict_proba(input_data)[0][1])
		else:
			win_proba = float(model.predict(input_data)[0])
		win = win_proba > 0.5
		return PositionPredictionResponse(
			win_probability=win_proba,
			win=win,
			input_features=input_data.iloc[0].to_dict(),
			warnings=warnings if warnings else None
		)
	except Exception as e:
		print("Prediction error:", str(e))
		raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/reload-model")
async def reload_model():
	success = load_model()
	if success:
		return {"message": "Model reloaded successfully"}
	else:
		raise HTTPException(status_code=500, detail="Failed to reload model")
