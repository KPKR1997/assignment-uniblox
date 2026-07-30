import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        age: int,
        gender: str,
        marital_status: str,
        salary: float,
        employment_type: str,
        region: str,
        has_dependents: str,
        tenure_years: float
    ):

        self.age = age
        self.gender = gender
        self.marital_status = marital_status
        self.salary = salary
        self.employment_type = employment_type
        self.region = region
        self.has_dependents = has_dependents
        self.tenure_years = tenure_years

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "gender": [self.gender],
                "marital_status": [self.marital_status],
                "salary": [self.salary],
                "employment_type": [self.employment_type],
                "region": [self.region],
                "has_dependents": [self.has_dependents],
                "tenure_years": [self.tenure_years]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)