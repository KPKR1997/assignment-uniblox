import sys
import os

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException  
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            param = params.get(model_name, {})
            gs = GridSearchCV(
                estimator=model,
                param_grid=param,
                cv=3,
                scoring="f1",
                n_jobs=-1
            )
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)
            model_score = f1_score(y_test, y_pred)
            report[model_name] = model_score
            models[model_name] = best_model
        return report

    except Exception as e:
        raise CustomException(e, sys)



def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)