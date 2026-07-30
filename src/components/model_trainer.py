import os
import sys
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix

#We have 62:38 proportion distribution in target feature. And enrollment is a binary classification. So I'm prioritizing Classifiers and trees here

)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Training Started")
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            #Above step need assurance that last col is target

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Extra Trees": ExtraTreesClassifier(),
                "XGBoost": XGBClassifier(
                    eval_metric="logloss"
                ),
                "CatBoost": CatBoostClassifier(
                    verbose=0
                )
            }
            #Defining parameters for tuning each models for GridsearchCV
            params = {

                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10],
                    "penalty": ["l2"],
                    "solver": ["lbfgs"]
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200],
                    "subsample": [0.8, 1.0]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1]
                },
                "Extra Trees": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20]
                },
                "XGBoost": {
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5, 7],
                    "n_estimators": [100, 200]
                },
                "CatBoost": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [100, 200]
                }
            }

            #The hyperparameter tuning here is done using Gridsearch CV, implemented in 'evaluate_models' method in 'utils.py'
            #Also prioritizing F1 Score for this data in model evaluation to get best model
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            print(model_report)
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            best_model_score = model_report[best_model_name]

            logging.info(
                f"Best Model : {best_model_name} | F1 Score : {best_model_score:.4f}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            return {
                "Best Model": best_model_name,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1 Score": round(f1, 4),
                "Confusion Matrix": cm
            }

        except Exception as e:
            raise CustomException(e, sys)
        
