from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

sys.path.append('/opt/airflow/scripts')
from predict_results import data_extraction, prediction, load_transformed_race_data, load_predicted_race_data

default_args = {
    "description": "DAG for orchestrating predicted race positions",
    "start_date": datetime(2025, 10, 31),
    "catchup": False,
}

dag = DAG(
    dag_id="race_results_predictions",
    default_args=default_args,
    schedule=timedelta(days=7),
)

with dag:
    setup = PythonOperator(
        task_id="data_extraction",
        python_callable=data_extraction,
    )

    predict = PythonOperator(
        task_id="prediction",
        python_callable=prediction,
    )

    load_inputs = PythonOperator(
        task_id="load_transformed_race_data",
        python_callable=load_transformed_race_data,
    )

    load_outputs = PythonOperator(
        task_id="load_predicted_race_data",
        python_callable=load_predicted_race_data,
    )

    setup >> predict >> load_inputs >> load_outputs