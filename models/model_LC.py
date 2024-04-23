import joblib
from sklearn.linear_model import LogisticRegression
from models.data import prepare_data
from models.train_models import train, test
from models.config import config
import wandb
import json
import subprocess


def download_model_params_from_dvc(file_name):
    subprocess.run(["dvc", "pull", file_name])


def model_experiment(config_LC: dict, run=0):
    download_model_params_from_dvc("../data_dvc/results_LC.json")
    with open("../data_dvc/results_LC.json", "r") as file:
        download_model_results = json.load(file)

    data = prepare_data()
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    model_LC = LogisticRegression(**config_LC['logistic_regression'], random_state=config_LC["random_state"])
    train(model_LC, X_train, y_train)
    results = test(model_LC, X_test, y_test)

    if download_model_results['MAE'] > results['MAE']:
        json.dump(results, 'results_LC.json')
        joblib.dump(model_LC, 'model_LC.pkl', compress=True)

        wandb.login()

        wandb.init(
            project="Mlops_lesson7_LC",
            name=f'experiment_{run}',
            config=config_LC['logistic_regression']
        )
        # model_config = config_LC['logistic_regression']
        # wandb.config.update(model_config)
        wandb.summary["LC Accuracy"] = results['accuracy']
        wandb.summary["LC MAE"] = results['MAE']
        wandb.summary["LC F1 score"] = results['f1_score']
        wandb.summary["LC AUC-ROC"] = results['mean_roc_auc']
        wandb.finish()
    if download_model_results['MAE'] == results['MAE']:
        print("модель актуальная")
        return
    else:
        print("Эксперимент не удался :(")
        return


#model_experiment(config_LC=config)
