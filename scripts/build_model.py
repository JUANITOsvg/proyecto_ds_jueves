#!/usr/bin/env python3
"""
Script to build the F1 race prediction model locally.
This script will create the model file that the API needs.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
from pathlib import Path

def load_and_prepare_data():
    """Load and prepare the F1 data for model training"""
    print("🏎️ Loading F1 data...")
    
    # Define data directory
    data_dir = Path(__file__).parent.parent / "pipelines" / "data"
    
    # Load datasets
    results = pd.read_csv(data_dir / "results.csv")
    races = pd.read_csv(data_dir / "races.csv")
    drivers = pd.read_csv(data_dir / "drivers.csv")
    constructors = pd.read_csv(data_dir / "constructors.csv")
    circuits = pd.read_csv(data_dir / "circuits.csv")
    qualifying = pd.read_csv(data_dir / "qualifying.csv")
    
    print(f"📊 Loaded data: {len(results)} results, {len(races)} races")
    
    # Merge datasets with proper suffixes to avoid conflicts
    df = results.merge(races, on='raceId', how='left', suffixes=('', '_race'))
    df = df.merge(drivers, on='driverId', how='left', suffixes=('', '_driver'))
    df = df.merge(constructors, on='constructorId', how='left', suffixes=('', '_constructor'))
    df = df.merge(circuits, on='circuitId', how='left', suffixes=('', '_circuit'))
    
    # Merge qualifying data
    qualifying_simplified = qualifying.groupby(['raceId', 'driverId']).agg({
        'position': 'first'
    }).reset_index()
    qualifying_simplified.rename(columns={'position': 'qualifying_position'}, inplace=True)
    
    df = df.merge(qualifying_simplified, on=['raceId', 'driverId'], how='left')
    
    print(f"📈 Merged dataset shape: {df.shape}")
    return df

def engineer_features(df):
    """Create features for the model"""
    print("🔧 Engineering features...")
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Fill missing qualifying positions with last position + 1
    max_grid = df['grid'].max()
    df['qualifying_position'].fillna(max_grid + 1, inplace=True)
    
    # Select features
    feature_columns = [
        'year', 'month', 'round', 'grid', 'qualifying_position',
        'name',  # Circuit name
        'surname', # Driver surname
        'name_constructor'     # Constructor name
    ]
    
    # Filter data
    df_model = df[feature_columns + ['positionOrder']].copy()
    df_model = df_model.dropna(subset=['positionOrder'])
    
    # Convert positionOrder to numeric
    df_model['positionOrder'] = pd.to_numeric(df_model['positionOrder'], errors='coerce')
    df_model = df_model.dropna(subset=['positionOrder'])
    
    print(f"🎯 Model dataset shape: {df_model.shape}")
    return df_model, feature_columns

def train_model(df_model, feature_columns):
    """Train the Random Forest model"""
    print("🤖 Training model...")
    
    # Prepare features and target
    X = df_model[feature_columns].copy()
    y = df_model['positionOrder']
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['name', 'surname', 'name_constructor']
    
    for col in categorical_columns:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"📊 Model Performance:")
    print(f"   Mean Absolute Error: {mae:.2f}")
    print(f"   R² Score: {r2:.3f}")
    
    return model, label_encoders, feature_columns

def save_model(model, label_encoders, feature_columns):
    """Save the model and associated objects"""
    print("💾 Saving model...")
    
    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Prepare model package
    model_package = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns
    }
    
    # Save to models directory
    model_path = models_dir / "f1_race_prediction_model.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    print(f"✅ Model saved to: {model_path}")

def main():
    """Main function to build the model"""
    print("🏁 Starting F1 Race Prediction Model Build")
    print("=" * 50)
    
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Engineer features
        df_model, feature_columns = engineer_features(df)
        
        # Train model
        model, label_encoders, feature_columns = train_model(df_model, feature_columns)
        
        # Save model
        save_model(model, label_encoders, feature_columns)
        
        print("=" * 50)
        print("🎉 Model build completed successfully!")
        print("🚀 Ready to deploy API with trained model!")
        
    except Exception as e:
        print(f"❌ Error building model: {str(e)}")
        raise

if __name__ == "__main__":
    main()