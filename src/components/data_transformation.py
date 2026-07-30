import sys
import os
import pandas as pd
import numpy as np

from src.exception import CustomException   
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
        # for saving the pkl file
        preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl') 

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        # For numerical columns using imputer and standard scaler
        # For categorical columns using most frequent imputer, one hot encoder and standard scaler
        try:
            numerical_columns = ['age', 'salary', 'tenure_years']
            categorical_columns = [
                'gender',
                'marital_status',
                'employment_type',
                'region',
                'has_dependents',
            ]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),   
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),      
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            return preprocessor
        #preprocessor will be saved in Artifacts as obj

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed succesfully')

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'enrolled'
            numerical_columns = ['age', 'salary', 'tenure_years']

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            #Applying preprocessing object on training and testing datasets

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #Append  target column to the transformed feature arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f'Saved preprocessing object.')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
                        raise CustomException(e, sys)   