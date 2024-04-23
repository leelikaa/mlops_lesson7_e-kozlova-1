import joblib
from sklearn.ensemble import RandomForestClassifier
from models.data import prepare_data
from models.train_models import train, test
from models.config import config
import wandb
import json
import subprocess


def download_model_params_from_dvc(file_name):
    subprocess.run(["dvc", "pull", file_name])


def model_experiment(config_RF: dict, run=1):
    download_model_params_from_dvc("../data_dvc/results_RF.json")
    with open("../data_dvc/results_RF.json", "r") as file:
        download_model_results = json.load(file)

    data = prepare_data()
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    model_RF = RandomForestClassifier(**config_RF['decision_tree'], random_state=config_RF["random_state"])
    train(model_RF, X_train, y_train)
    results = test(model_RF, X_test, y_test)

    if download_model_results['MAE'] > results['MAE']:
        json.dump(results, 'results_RF.json')
        joblib.dump(model_RF, 'model_RF.pkl', compress=True)

        wandb.login()

        wandb.init(
            project="Mlops_lesson7_LC",
            name=f'experiment_{run}',
            config=config_RF['decision_tree']
        )
        # model_config = config['logistic_regression']
        # wandb.config.update(model_config)
        wandb.summary["RF Accuracy"] = results['accuracy']
        wandb.summary["RF MAE"] = results['MAE']
        wandb.summary["RF F1 score"] = results['f1_score']
        wandb.summary["RF AUC-ROC"] = results['mean_roc_auc']
        wandb.finish()

    if download_model_results['MAE'] == results['MAE']:
        print("модель актуальная")
        return
    else:
        print("Эксперимент не удался :(")
        return


#model_experiment(config_RF=config)
