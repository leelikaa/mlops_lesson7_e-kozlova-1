import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_data():
    wine = pd.read_csv('winequality-red.csv')
    X = pd.DataFrame(wine.drop('quality', axis=1))
    y = pd.DataFrame(wine['quality'])
    return X, y


def split_dataset(X, y) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42)
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": np.ravel(y_train),
        "y_test": np.ravel(y_test),
    }
    return data


def scaler(data):
    X_train = data["X_train"]
    X_test = data["X_test"]
    sc = StandardScaler()
    data["X_train"] = sc.fit_transform(X_train)
    data["X_trest"] = sc.transform(X_test)
    return data
