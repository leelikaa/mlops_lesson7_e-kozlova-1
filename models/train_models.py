import joblib
from data import load_data, split_dataset, scaler

X, y = load_data()
data = split_dataset(X, y)
data = scaler(data)
X_train, y_train = data["X_train"], data["y_train"]


def RandomForestClassifier_model(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model_RF = RandomForestClassifier(random_state=42)
    model_RF.fit(X_train, y_train)
    return joblib.dump(model_RF, 'model_RF.pkl')


def LogisticRegression_model(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    model_LC = LogisticRegression(random_state=42)
    model_LC.fit(X_train, y_train)
    return joblib.dump(model_LC, 'model_LC.pkl')


RandomForestClassifier_model(X_train, y_train)
LogisticRegression_model(X_train, y_train)
