"""
Simple F1 Race Prediction Pipeline Functions
Boilerplate functions for the Airflow DAG - basic data processing examples.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import logging

# Add the modules directory to the path for database connections
sys.path.append('/opt/airflow/modules')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def data_extraction():
    """
    Load F1 CSV and return DataFrame for next tasks.
    """
    logger.info("Starting data extraction...")
    
    try:
        # Load races CSV once
        races_df = pd.read_csv('/opt/airflow/data/races.csv')
        
        # Basic data info
        logger.info(f"Loaded races data: {len(races_df)} records")
        logger.info(f"Columns: {list(races_df.columns[:5])}...")
        
        # Return DataFrame as JSON for XCom (Airflow will serialize it)
        sample_data = races_df.head(100)  # Take first 100 rows
        logger.info(f"Returning {len(sample_data)} records for next task")
        
        return sample_data.to_dict('records')  # Convert to list of dicts for XCom
        
    except Exception as e:
        logger.error(f"Error in data extraction: {str(e)}")
        raise


def prediction(ti=None):
    """
    Receive data from data_extraction and add predictions.
    """
    logger.info("Starting prediction process...")
    
    try:
        # Get data from previous task via XCom (modern way)
        raw_data = ti.xcom_pull(task_ids='data_extraction')
        
        if not raw_data:
            raise ValueError("No data received from data_extraction task")
        
        # Convert back to DataFrame
        data = pd.DataFrame(raw_data)
        logger.info(f"Received {len(data)} records from data_extraction")
        
        # Select useful columns
        useful_columns = ['raceId', 'year', 'round', 'circuitId', 'name', 'date']
        clean_data = data[useful_columns].copy()
        
        # Add prediction columns
        clean_data['predicted_winner'] = np.random.choice(['Hamilton', 'Verstappen', 'Leclerc'], len(clean_data))
        clean_data['confidence_score'] = np.random.uniform(0.6, 0.95, len(clean_data))
        clean_data['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        clean_data['model_version'] = 'simple_v1.0'
        
        logger.info(f"Generated predictions for {len(clean_data)} races")
        logger.info(f"Reduced columns from {len(data.columns)} to {len(useful_columns)} + 4 prediction columns")
        
        # Return processed data for next task
        return clean_data.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise


def final_table(ti=None):
    """
    Receive predictions from prediction task and store in PostgreSQL warehouse.
    """
    logger.info("Starting final table creation...")
    
    try:
        # Import database connection from modules
        from db_conn_manager import DatabaseManager
        
        # Get processed data from prediction task via XCom (modern way)
        predictions_data = ti.xcom_pull(task_ids='prediction')
        
        if not predictions_data:
            raise ValueError("No predictions data received from prediction task")
        
        # Convert back to DataFrame
        predictions_df = pd.DataFrame(predictions_data)
        logger.info(f"Received {len(predictions_df)} predictions from prediction task")
        
        # Connect to warehouse
        with DatabaseManager() as db:
            
            table_name = 'simple_race_predictions'
            
            # Simple table schema
            table_schema = {
                'id': 'SERIAL PRIMARY KEY',
                'race_id': 'INTEGER',
                'year': 'INTEGER', 
                'round': 'INTEGER',
                'race_name': 'VARCHAR(255)',
                'predicted_winner': 'VARCHAR(100)',
                'confidence_score': 'DECIMAL(3,2)',
                'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
            }
            
            # Create table if needed
            if not db.table_exists(table_name):
                logger.info(f"Creating table: {table_name}")
                db.create_table(table_name, table_schema)
            
            # Prepare data for insertion - take first 10 records
            records = []
            for _, row in predictions_df.head(10).iterrows():
                record = {
                    'race_id': int(row['raceId']),
                    'year': int(row['year']),
                    'round': int(row['round']),
                    'race_name': str(row['name']),
                    'predicted_winner': str(row['predicted_winner']),
                    'confidence_score': float(row['confidence_score'])
                }
                records.append(record)
            
            # Insert into warehouse
            success = db.insert_bulk_data(table_name, records)
            
            if success:
                total_count = db.get_table_count(table_name)
                logger.info(f"✅ Successfully inserted {len(records)} predictions")
                logger.info(f"✅ Total records in {table_name}: {total_count}")
                
                return f"Inserted {len(records)} predictions into warehouse"
            else:
                raise Exception("Failed to insert data")
                
    except Exception as e:
        logger.error(f"Error in final table: {str(e)}")
        raise