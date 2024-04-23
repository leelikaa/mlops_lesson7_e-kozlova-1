from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from models.data import prepare_data
from models.train_models import train, test
import wandb
import json
import subprocess
from models.config import config
from models.model_LC import model_experiment
import sys

sys.path.append('/app/dags')


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 4, 21),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

file_path_to_model_LC = '/app/models/model_LC.py'
file_path_to_model_RF = '/app/models/model_RF.py'
file_path_to_config = '/app/models/config.py'

dag = DAG(
    'MLops_7_e-kozlova-1',
    default_args=default_args,
    description='Training model',
    schedule=timedelta(days=1),
)

'''
load_config = PythonOperator(
    task_id='prepare_config',
    python_callable=config,
    op_kwargs={'config_path': file_path_to_config},
    dag=dag, )
'''
train_model = PythonOperator(
    task_id='train_model',
    python_callable=model_experiment,
    op_kwargs={'config_path': file_path_to_model_LC},  # можно заменить на RF
    dag=dag, )

#load_config >> train_model

train_model
