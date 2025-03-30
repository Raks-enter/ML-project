import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['StudentID', 'Age', 'Gender', 'Absences', 'Tutoring', 'GPA', 'GradeClass']
            categorical_columns = []

            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")), #this will be responsible for handling missing values
                    ("scaler",StandardScaler())
                ]

            )
            cat_pipeline= Pipeline(

                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("scaler",StandardScaler()),
                    ("One_hot_encoder",OneHotEncoder())
                ]
            )

            logging.info("categorical columns standard scaling cokmpleted")

            logging.info("numerical columns standard scaling cokmpleted")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor



        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object ")

            preprocessing_object=self.get_data_transformer_object()

            target_column_name="GPA"
            numerical_columns= ['StudentID', 'Age', 'Gender', 'Absences', 'Tutoring', 'GPA', 'GradeClass']

            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_features_train_arr=preprocessing_object.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessing_object.transform(input_features_test_df)

            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_features_test_arr,np.array(target_feature_test_df)]

            logging.info(f"saved preprocessing object")

            save_object(

                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )

            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
            