config = {
    "random_state": 42,
    "data": {
        "test_size": 0.2,
    },
    "logistic_regression": {
        "penalty": 'l2',
        "max_iter": 100,
        "multi_class": 'ovr',
        "C": 0.1       
    },
    "decision_tree": {
        "max_depth": 30,
        "min_samples_split": 2,
        "n_estimators": 150,
        "min_samples_leaf": 1
    }
}