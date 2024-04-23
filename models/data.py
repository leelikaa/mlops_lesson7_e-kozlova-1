import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from models.config import config


def load_data():
    wine = pd.read_csv('winequality-red.csv')
    X = pd.DataFrame(wine.drop('quality', axis=1))
    y = pd.DataFrame(wine['quality'])
    return X, y


def split_dataset(X, y) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["random_state"])
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
    data["X_test"] = sc.transform(X_test)
    return data


def prepare_data():
    X, y = load_data()
    data_origin = split_dataset(X, y)
    data_prepared = scaler(data_origin)
    return data_prepared
