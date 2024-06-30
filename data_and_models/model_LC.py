from sklearn.linear_model import LogisticRegression
from data import prepare_data
from train_models import train, test
from config import config
from clearml import Task
import joblib
import subprocess
import json

task = Task.init(
    project_name="Mlops_lesson7",
    task_name="experiment_LC",
)


def download_data_from_dvc(file_name):
    subprocess.run(["dvc", "pull", file_name])


def upload_data_to_dvc(file_name):
    subprocess.run(["dvc", "push", file_name])


def model_experiment(config_LC: dict):
    download_data_from_dvc("../data_dvc/results_LC.json")
    with open("../data_dvc/results_LC.json", "r") as file:
        dvc_results = json.load(file)

    data = prepare_data()
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    model_LC = LogisticRegression(**config_LC['logistic_regression'], random_state=config_LC["random_state"])
    train(model_LC, X_train, y_train)
    results = test(model_LC, X_test, y_test)
    with open('results_LC.json', 'w') as f:
        json.dump(results, f)

    task.connect(config_LC['logistic_regression'])
    logger = task.get_logger()

    logger.report_single_value("Accuracy", value=results['accuracy'])
    logger.report_single_value("MAE", value=results['MAE'])
    logger.report_single_value("F1 score", value=results['f1_score'])
    logger.report_single_value("AUC-ROC", value=results['mean_roc_auc'])

    joblib.dump(model_LC, '../data_dvc/model_LC.pkl', compress=True)
    task.upload_artifact(name='model_LC',artifact_object='model_LC.pkl')

    task.close()

    if dvc_results['MAE'] > results['MAE']:
        upload_data_to_dvc("../data_dvc/results_LC.json")
        upload_data_to_dvc("../data_dvc/model_LC.pkl")


if __name__ == "__main__":
    model_experiment(config_LC=config)

