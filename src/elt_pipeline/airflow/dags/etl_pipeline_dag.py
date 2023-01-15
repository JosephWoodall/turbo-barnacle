from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

# Default arguments for the DAG
default_args = {
    'owner': 'me',
    'start_date': datetime(2022, 1, 1),
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create a DAG instance
dag = DAG(
    'elt_pipeline_dag',
    default_args=default_args,
    schedule_interval='0 0 * * *', # runs every day at 00:00
)

# Define the tasks in the DAG
ingest_data = PythonOperator(
    task_id='ingest_data',
    python_callable='pipeline.ingest_data',
    dag=dag,
)

transform_data = PythonOperator(
    task_id='transform_data',
    python_callable='pipeline.transform_data',
    dag=dag,
)

validate_data = PythonOperator(
    task_id='validate_data',
    python_callable='pipeline.validate_data',
    dag=dag,
)

load_data = PythonOperator(
    task_id='load_data',
    python_callable='pipeline.load_data',
    dag=dag,
)

# Set the order of the tasks
ingest_data >> transform_data >> validate_data >> load_data
