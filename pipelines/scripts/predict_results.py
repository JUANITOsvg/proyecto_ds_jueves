
import pandas as pd
import requests
import logging
import sys
import os

# Add the modules directory to the path for database connections
sys.path.append('/opt/airflow/modules')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def data_extraction():
	"""
	Load f1_model_input.csv and return as DataFrame for next tasks.
	"""
	logger.info("Loading f1_model_input.csv...")
	try:
		df = pd.read_csv('/opt/airflow/data/f1_model_input.csv')
		logger.info(f"Loaded {len(df)} rows from f1_model_input.csv")
		return df.to_dict('records')
	except Exception as e:
		logger.error(f"Error loading CSV: {str(e)}")
		raise

def prediction(ti=None):
	"""
	Call the prediction API for each row and add the result.
	"""
	logger.info("Starting prediction for each row via API...")
	try:
		input_data = ti.xcom_pull(task_ids='data_extraction')
		if not input_data:
			raise ValueError("No input data received from data_extraction task")
		df = pd.DataFrame(input_data)
		api_url = os.environ.get('F1_PREDICT_API_URL', 'http://f1-position-result-api:8001/predict')

		def get_prediction(row):
			payload = {
				"avg_race_pos": row["avg_race_pos"],
				"avg_sprint_pos": row["avg_sprint_pos"],
				"avg_lap_time": row["avg_lap_time"],
				"points": row["points"],
				"avg_qual_pos": row["avg_qual_pos"],
				"driver_encoded": row["driver_encoded"]
			}
			try:
				response = requests.post(api_url, json=payload)
				response.raise_for_status()
				return response.json().get("predicted_position")
			except Exception as e:
				logger.error(f"API error for row {row}: {str(e)}")
				return None

		df['predicted_position'] = df.apply(get_prediction, axis=1)
		logger.info(f"Predictions added for {len(df)} rows")
		return df.to_dict('records')
	except Exception as e:
		logger.error(f"Error in prediction: {str(e)}")
		raise

def load_transformed_race_data(ti=None):
	"""
	Load the input data into transformed_race_data table.
	"""
	logger.info("Loading input data into transformed_race_data table...")
	try:
		from db_conn_manager import DatabaseManager
		input_data = ti.xcom_pull(task_ids='data_extraction')
		if not input_data:
			raise ValueError("No input data received from data_extraction task")
		df = pd.DataFrame(input_data)
		with DatabaseManager() as db:
			table_name = 'transformed_race_data'
			db.insert_bulk_data(table_name, df.to_dict('records'))
			logger.info(f"Inserted {len(df)} rows into {table_name}")
		return f"Loaded {len(df)} rows into {table_name}"
	except Exception as e:
		logger.error(f"Error loading transformed_race_data: {str(e)}")
		raise

def load_predicted_race_data(ti=None):
	"""
	Load the prediction results into predicted_race_data table.
	"""
	logger.info("Loading predictions into predicted_race_data table...")
	try:
		from db_conn_manager import DatabaseManager
		pred_data = ti.xcom_pull(task_ids='prediction')
		if not pred_data:
			raise ValueError("No prediction data received from prediction task")
		df = pd.DataFrame(pred_data)
		with DatabaseManager() as db:
			table_name = 'predicted_race_data'
			db.insert_bulk_data(table_name, df.to_dict('records'))
			logger.info(f"Inserted {len(df)} rows into {table_name}")
		return f"Loaded {len(df)} rows into {table_name}"
	except Exception as e:
		logger.error(f"Error loading predicted_race_data: {str(e)}")
		raise