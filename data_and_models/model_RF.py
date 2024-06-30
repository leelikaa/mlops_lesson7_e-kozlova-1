from sklearn.ensemble import RandomForestClassifier
from data import prepare_data
from train_models import train, test
from models_config import config
from clearml import Task
import joblib
import subprocess
import json

task = Task.init(
    project_name="Mlops_lesson7",
    task_name="experiment_RF",
)


def download_data_from_dvc(file_name):
    subprocess.run(["dvc", "pull", file_name])


def upload_data_to_dvc(file_name):
    subprocess.run(["dvc", "push", file_name])


def model_experiment(config_RF: dict):
    download_data_from_dvc("../data_dvc/results_RF.json")
    with open("../data_dvc/results_RF.json", "r") as file:
        dvc_results = json.load(file)

    data = prepare_data()
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    model_RF = RandomForestClassifier(**config_RF['decision_tree'], random_state=config_RF["random_state"])
    train(model_RF, X_train, y_train)
    results = test(model_RF, X_test, y_test)
    with open('results_RF.json', 'w') as f:
        json.dump(results, f)
    
    task.connect(config_RF['decision_tree'])
    logger = task.get_logger()

    logger.report_single_value("Accuracy", value=results['accuracy'])
    logger.report_single_value("MAE", value=results['MAE'])
    logger.report_single_value("F1 score", value=results['f1_score'])
    logger.report_single_value("AUC-ROC", value=results['mean_roc_auc'])

    joblib.dump(model_RF, '../data_dvc/model_RF.pkl', compress=True)
    task.upload_artifact(name='model_RF', artifact_object='model_RF.pkl')
    
    task.close()

    if dvc_results['MAE'] > results['MAE']:
        upload_data_to_dvc("../data_dvc/results_RF.json")
        upload_data_to_dvc("../data_dvc/model_RF.pkl")


if __name__ == "__main__":
    model_experiment(config_RF=config)
