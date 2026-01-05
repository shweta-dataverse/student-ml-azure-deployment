import os
import sys
import pickle
import dill

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor

from src.exception_handling import CustomException

# -----------------------------------------------
# wrapper for CatBoost to make it sklearn-compatible
# sklearn utilities like GridSearchCV expect BaseEstimator
# -----------------------------------------------
class SklearnCatBoost(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        depth=6,
        learning_rate=0.1,
        iterations=100,
        loss_function="RMSE",
        verbose=False
    ):
        self.depth = depth
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss_function = loss_function
        self.verbose = verbose

        self.model = CatBoostRegressor(
            depth=self.depth,
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            loss_function=self.loss_function,
            verbose=self.verbose
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

# -----------------------------------------------
# save any Python object (like model, preprocessor) as pickle
# this allows reuse later without retraining or recalculating
# -----------------------------------------------
def save_object(file_path, obj):
    try:
        # ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # write object to file in binary mode
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            trained_models[model_name] = best_model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)