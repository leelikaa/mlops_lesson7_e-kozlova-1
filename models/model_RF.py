import joblib
from sklearn.ensemble import RandomForestClassifier
from data import prepare_data
from train_models import train, test
from config import config
import wandb


def model_experimen(run):
    wandb.login()

    wandb.init(
        project="Mlops_lesson7_LC",
        name=f'experiment_{run}',
        config=config['decision_tree']
    )
    data = prepare_data()
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    model_RF = RandomForestClassifier(**config['decision_tree'], random_state=config["random_state"])
    train(model_RF, X_train, y_train)
    results = test(model_RF, X_test, y_test)

    # model_config = config['logistic_regression']
    # wandb.config.update(model_config)
    wandb.summary["RF Accuracy"] = results['accuracy']
    wandb.summary["RF MAE"] = results['MAE']
    wandb.summary["RF F1 score"] = results['f1_score']
    wandb.summary["RF AUC-ROC"] = results['mean_roc_auc']
    wandb.finish()

    joblib.dump(model_RF, f'model_RF{run}.pkl', compress=True)


model_experimen(1)
