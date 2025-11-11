
"""
Local builder for the F1 position prediction model.

Loads data, performs feature engineering, trains a RandomForestClassifier (for win prediction),
and saves the trained model for use by the API.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

def load_and_prepare_data():
	"""Load and merge F1 datasets into a single DataFrame for position prediction."""
	print("Loading raw F1 data for position prediction...")

	base_dir = Path(__file__).resolve().parent.parent
	data_dir = base_dir / "pipelines" / "data"

	# Load data files
	results = pd.read_csv(data_dir / "results.csv")
	races = pd.read_csv(data_dir / "races.csv")
	drivers = pd.read_csv(data_dir / "drivers.csv")
	qualifying = pd.read_csv(data_dir / "qualifying.csv")

	print(f"Loaded {len(results)} race results from {len(races)} races.")

	# Merge datasets
	df = results.merge(races, on="raceId", how="left")
	df = df.merge(drivers, on="driverId", how="left")

	# Simplify qualifying data (take first recorded position)
	qual = (
		qualifying.groupby(["raceId", "driverId"])["position"]
		.first()
		.reset_index()
		.rename(columns={"position": "qualifying_position"})
	)
	df = df.merge(qual, on=["raceId", "driverId"], how="left")

	print(f"Merged dataset shape: {df.shape}")
	return df

def engineer_features(df):
	"""Feature engineering for position prediction."""
	print("Engineering features for position prediction...")

	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	df["year"] = df["date"].dt.year
	df["month"] = df["date"].dt.month

	# Fill missing qualifying positions with worst grid + 1
	fallback_pos = df["grid"].max() + 1
	df["qualifying_position"] = df["qualifying_position"].fillna(fallback_pos)

	# Create a binary target: win (1 if positionOrder == 1, else 0)
	df["win"] = (df["positionOrder"] == 1).astype(int)

	# Create driver name for encoding
	df["driver"] = df["forename"].astype(str) + " " + df["surname"].astype(str)

	feature_cols = [
		"year", "month", "round", "grid", "qualifying_position",
		"driver"
	]

	df_model = df[feature_cols + ["win"]].dropna(subset=["win"])
	print(f"Prepared {len(df_model)} training samples for position prediction.")
	return df_model, feature_cols

def train_model(df, feature_cols):
	"""Train a Random Forest classifier for win prediction."""
	print("Training RandomForestClassifier for win prediction...")

	X = df[feature_cols].copy()
	y = df["win"]

	# Encode driver name
	driver_encoder = LabelEncoder()
	X["driver_encoded"] = driver_encoder.fit_transform(X["driver"].astype(str))
	X = X.drop(columns=["driver"])
	feature_cols_final = [col for col in X.columns]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	model = RandomForestClassifier(
		n_estimators=120,
		random_state=42,
		n_jobs=-1
	)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	y_proba = model.predict_proba(X_test)[:, 1]
	acc = accuracy_score(y_test, y_pred)
	auc = roc_auc_score(y_test, y_proba)

	print(f"Validation Accuracy: {acc:.3f}")
	print(f"Validation ROC AUC: {auc:.3f}")

	return model, driver_encoder, feature_cols_final

def save_model(model, driver_encoder, feature_cols):
	"""Save model and metadata to disk."""
	print("Saving position prediction model...")

	out_dir = Path(__file__).resolve().parent.parent / "models"
	out_dir.mkdir(exist_ok=True)

	model_bundle = {
		"model": model,
		"driver_encoder": driver_encoder,
		"feature_columns": feature_cols,
	}

	out_path = out_dir / "f1_race_position_model.pkl"
	with open(out_path, "wb") as f:
		pickle.dump(model_bundle, f)

	print(f"Model saved to {out_path}")

def main():
	print("=" * 50)
	print("Building F1 Position Prediction Model")
	print("=" * 50)

	try:
		df = load_and_prepare_data()
		df_model, feat_cols = engineer_features(df)
		model, driver_encoder, feat_cols_final = train_model(df_model, feat_cols)
		save_model(model, driver_encoder, feat_cols_final)

		print("\nPosition prediction model build completed successfully.")
		print("Ready for deployment.\n")
	except Exception as e:
		print(f"Error building position prediction model: {e}")
		raise

if __name__ == "__main__":
	main()