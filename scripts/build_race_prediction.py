#!/usr/bin/env python3
"""
Local builder for the F1 race prediction model.

Loads data, performs basic feature engineering, trains a RandomForestRegressor,
and saves the trained model for use by the API.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score


def load_and_prepare_data():
    """Load and merge F1 datasets into a single DataFrame."""
    print("Loading raw F1 data...")

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "pipelines" / "data"

    # Load data files
    results = pd.read_csv(data_dir / "results.csv")
    races = pd.read_csv(data_dir / "races.csv")
    drivers = pd.read_csv(data_dir / "drivers.csv")
    constructors = pd.read_csv(data_dir / "constructors.csv")
    circuits = pd.read_csv(data_dir / "circuits.csv")
    qualifying = pd.read_csv(data_dir / "qualifying.csv")

    print(f"Loaded {len(results)} race results from {len(races)} races.")

    # Merge datasets
    df = results.merge(races, on="raceId", how="left")
    df = df.merge(drivers, on="driverId", how="left")
    df = df.merge(constructors, on="constructorId", how="left")
    df = df.merge(circuits, on="circuitId", how="left")

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
    """Perform basic feature engineering for the model."""
    print("Engineering features...")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Fill missing qualifying positions with worst grid + 1
    fallback_pos = df["grid"].max() + 1
    df["qualifying_position"] = df["qualifying_position"].fillna(fallback_pos)

    feature_cols = [
        "year", "month", "round", "grid", "qualifying_position",
        "name",  # circuit
        "surname",  # driver
        "name_constructor"  # team
    ]

    df_model = df[feature_cols + ["positionOrder"]].dropna(subset=["positionOrder"])
    df_model["positionOrder"] = pd.to_numeric(df_model["positionOrder"], errors="coerce")

    print(f"Prepared {len(df_model)} training samples.")
    return df_model, feature_cols


def train_model(df, feature_cols):
    """Train a Random Forest model and evaluate it."""
    print("Training RandomForest model...")

    X = df[feature_cols].copy()
    y = df["positionOrder"]

    # Encode categorical variables
    encoders = {}
    for col in ["name", "surname", "name_constructor"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=120,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation R²: {r2:.3f}")

    return model, encoders


def save_model(model, encoders, feature_cols):
    """Save model and metadata to disk."""
    print("Saving model...")

    out_dir = Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(exist_ok=True)

    model_bundle = {
        "model": model,
        "encoders": encoders,
        "features": feature_cols,
    }

    out_path = out_dir / "f1_race_prediction_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"Model saved to {out_path}")


def main():
    print("=" * 50)
    print("Building F1 Race Prediction Model")
    print("=" * 50)

    try:
        df = load_and_prepare_data()
        df_model, feat_cols = engineer_features(df)
        model, encoders = train_model(df_model, feat_cols)
        save_model(model, encoders, feat_cols)

        print("\nModel build completed successfully.")
        print("Ready for deployment.\n")
    except Exception as e:
        print(f"Error building model: {e}")
        raise


if __name__ == "__main__":
    main()
