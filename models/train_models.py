import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, roc_auc_score


def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(model, X_test, y_test) -> dict:
    y_predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    MAE = mean_absolute_error(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average='weighted')
    y_predict_proba = model.predict_proba(X_test)
    roc_auc_scores = []
    for i, class_index in enumerate(np.unique(y_test)):
        roc_auc_scores.append(
            roc_auc_score(
                y_test == class_index,
                y_predict_proba[:, i]))
    mean_roc_auc = np.mean(roc_auc_scores)
    results = {
        "accuracy": accuracy,
        "MAE": MAE,
        "f1_score": f1,
        "mean_roc_auc": mean_roc_auc,
    }
    return results
