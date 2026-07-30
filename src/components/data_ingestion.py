import sys
import os

import yaml
import pandas as pd
import numpy as np

from src.exception import CustomException   
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() #This sometimes I prefer to give in yaml file for customization

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            with open("config/config.yaml") as file:
                config = yaml.safe_load(file)
            df = pd.read_csv(config["data"]["raw_data_path"])
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            #Removing employee id col. This also can be included in yaml.
            #Usually I used to do a yaml having cols to remove, prefered focus metrics, prefered models and cutoff r2 scores etc for highly reusable pipelines. 
            # We used to do all configurations for analytics using yaml here
            df = df.drop(columns=['employee_id'])

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            #Train and test data will be stored in Artifacts folder/All pipeline objects created will be in Artifacts
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

#Trial
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

    