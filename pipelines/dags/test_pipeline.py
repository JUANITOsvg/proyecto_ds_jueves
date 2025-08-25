from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
import os

def load_test_df():
    # Load environment variables
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_DB = os.getenv("POSTGRES_DB")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", 5432)

    # Create connection to PostgreSQL
    engine = create_engine(f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")

    # Create a sample DataFrame
    data = {
        'Name': ['Tom', 'Dick', 'Harry'],
        'Age': [22, 21, 24]
    }
    df = pd.DataFrame(data)

    # Write DataFrame to PostgreSQL
    df.to_sql('test_table_1', engine, if_exists='replace', index=False)

    print("✅ test_table_1 loaded to Postgres")

with DAG(
    dag_id="load_test_df_dag",
    start_date=datetime(2025, 8, 23),
    schedule="@daily",
    catchup=False
) as dag:

    load_test_df_task = PythonOperator(
        task_id="load_test_df",
        python_callable=load_test_df
    )
