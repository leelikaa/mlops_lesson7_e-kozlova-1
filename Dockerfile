FROM apache/airflow:2.8.3
RUN pip install --no-cache-dir pandas sqlalchemy scikit_learn joblib wandb numpy
