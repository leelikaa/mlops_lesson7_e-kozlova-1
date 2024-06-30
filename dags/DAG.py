from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess
from data_and_models.model_RF import model_experiment


def download_data_from_dvc(file_name):
    subprocess.run(["dvc", "pull", file_name])


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 29),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

file_path_to_model_LC = '/app/data_and_models/model_LC.py'
file_path_to_model_RF = '/app/data_and_models/model_RF.py'
file_path_to_config = '/app/data_and_models/config.py'
file_path_to_data = '/app/data_and_models/winequality-red.csv'

dag = DAG(
    'MLops_7_e-kozlova-1',
    default_args=default_args,
    description='Training model',
    schedule=timedelta(days=1),
)

download_data_task = PythonOperator(
    task_id='download_data_task',
    python_callable=download_data_from_dvc,
    op_args=[file_path_to_data],
    dag=dag, )

train_model = PythonOperator(
    task_id='train_model',
    python_callable=model_experiment,
    op_kwargs={'config_path': file_path_to_model_RF},  # можно заменить на RF
    dag=dag, )

download_data_task >> train_model
