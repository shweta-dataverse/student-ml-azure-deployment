import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception_handling import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # path where the preprocessor object will be saved for later use
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        # initialize the config object to access paths and settings
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            # define numerical columns which need scaling and missing value handling
            numerical_columns = ["writing_score", "reading_score"]
            # define categorical columns which need encoding and imputation
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # pipeline for numerical columns: fill missing values with median and scale
            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")), # median handles outliers better than mean
                    ("scaler",StandardScaler()) # standardize numerical features for ML algorithms
                ]
            )

            # pipeline for categorical columns: fill missing with mode, one-hot encode, then scale
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")), # most frequent value for categorical missing
                    ("one_hot_encoder",OneHotEncoder()), # convert categorical to numerical
                    ("scaler",StandardScaler(with_mean=False)) # scale after encoding, mean=False to avoid errors with sparse data
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # combine numerical and categorical pipelines into one preprocessor object
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            # raise custom exception with system info for debugging
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            # read train and test CSV files into pandas dataframes
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # get the combined preprocessing pipeline
            preprocessing_obj=self.get_data_transformer_object()

            # target variable for prediction
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # separate input features and target from train dataframe
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            # separate input features and target from test dataframe
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training and testing dataframe."
            )

            # fit preprocessor on training data and transform it
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            # only transform test data (do not fit again)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # combine processed input features and target into single arrays for training/testing
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # save preprocessor object for reuse during model training or inference
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # return train array, test array, and preprocessor path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            # catch any exception and raise custom exception for better debugging
            raise CustomException(e,sys)