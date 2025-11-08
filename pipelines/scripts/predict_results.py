
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

		# If driver_encoded is missing, fill with 0 or a default encoding
		if 'driver_encoded' not in df.columns:
			df['driver_encoded'] = 0

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
				# Optionally, get win probability if your API returns it
				result = response.json()
				return {
					"predicted_position": result.get("predicted_position"),
					"win_probability": result.get("confidence", 0.0)
				}
			except Exception as e:
				logger.error(f"API error for row {row}: {str(e)}")
				return {"predicted_position": None, "win_probability": 0.0}

		preds = df.apply(get_prediction, axis=1, result_type='expand')
		df['predicted_position'] = preds['predicted_position']
		df['win_probability'] = preds['win_probability']
		# win = True if predicted_position == 1
		df['win'] = df['predicted_position'] == 1
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
		# Only keep required columns and types
		columns = [
			'driverId', 'avg_race_pos', 'avg_sprint_pos', 'avg_lap_time', 'points', 'avg_qual_pos', 'forename', 'surname'
		]
		df = df[columns].copy()
		# Ensure types
		df = df.astype({
			'driverId': int,
			'avg_race_pos': float,
			'avg_sprint_pos': float,
			'avg_lap_time': float,
			'points': float,
			'avg_qual_pos': float,
			'forename': str,
			'surname': str
		})
		with DatabaseManager(schema="dev") as db:
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
		# Only keep required columns and types
		columns = [
			'driverId', 'avg_race_pos', 'avg_sprint_pos', 'avg_lap_time', 'points', 'avg_qual_pos',
			'forename', 'surname', 'driver', 'driver_encoded', 'win', 'win_probability'
		]
		# Add 'driver' column as forename + ' ' + surname if not present
		if 'driver' not in df.columns:
			df['driver'] = df['forename'].astype(str) + ' ' + df['surname'].astype(str)
		# Ensure driver_encoded exists
		if 'driver_encoded' not in df.columns:
			df['driver_encoded'] = 0
		# Ensure win_probability exists
		if 'win_probability' not in df.columns:
			df['win_probability'] = 0.0
		# Ensure win exists
		if 'win' not in df.columns:
			df['win'] = df['predicted_position'] == 1
		# Ensure types
		df = df.astype({
			'driverId': int,
			'avg_race_pos': float,
			'avg_sprint_pos': float,
			'avg_lap_time': float,
			'points': float,
			'avg_qual_pos': float,
			'forename': str,
			'surname': str,
			'driver': str,
			'driver_encoded': int,
			'win': bool,
			'win_probability': float
		})
		df = df[columns].copy()
		with DatabaseManager(schema="warehouse") as db:
			table_name = 'predicted_race_data'
			db.insert_bulk_data(table_name, df.to_dict('records'))
			logger.info(f"Inserted {len(df)} rows into {table_name}")
		return f"Loaded {len(df)} rows into {table_name}"
	except Exception as e:
		logger.error(f"Error loading predicted_race_data: {str(e)}")
		raise