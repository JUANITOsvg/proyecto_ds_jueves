from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

sys.path.append('/opt/airflow/scripts')
from predict_races import data_extraction, prediction, final_table

default_args = {
    "description": "DAG for orchestrating race prediction pipeline",
    "start_date": datetime(2025, 10, 31),
    "catchup": False,
}

dag = DAG(
    dag_id="race_prediction_orchestrator",
    default_args=default_args,
    schedule=timedelta(days=7),
)

with dag:
    
    setup = PythonOperator(
        task_id="data_extraction",
        python_callable=data_extraction,
    )

    prediction = PythonOperator(
        task_id="prediction",
        python_callable=prediction,
    )

    results = PythonOperator(
        task_id="final_table",
        python_callable=final_table,
    )

    setup >> prediction >> results