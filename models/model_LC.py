import joblib
from sklearn.linear_model import LogisticRegression
from data import prepare_data
from train_models import train, test
from config import config
import wandb


def model_experimen(run):
    wandb.login()

    wandb.init(
        project="Mlops_lesson7_LC",
        name=f'experiment_{run}',
        config=config['logistic_regression']
    )
    data = prepare_data()
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    model_LC = LogisticRegression(**config['logistic_regression'], random_state=config["random_state"])
    train(model_LC, X_train, y_train)
    results = test(model_LC, X_test, y_test)

    # model_config = config['logistic_regression']
    # wandb.config.update(model_config)
    wandb.summary["LC Accuracy"] = results['accuracy']
    wandb.summary["LC MAE"] = results['MAE']
    wandb.summary["LC F1 score"] = results['f1_score']
    wandb.summary["LC AUC-ROC"] = results['mean_roc_auc']
    wandb.finish()

    joblib.dump(model_LC, f'model_LC{run}.pkl', compress=True)


model_experimen(0)
