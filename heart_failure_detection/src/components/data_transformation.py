import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import get_logger
from src.exception import HFException
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.utils.utils import save_object, read_yaml_file

logger = get_logger(__name__)

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            logger.info(f"{'='*20}Data Transformation log started.{'='*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise HFException(e, sys)

    def get_data_transformer_object(self):
        """
        Get the data transformer object
        """
        try:
            # This method would typically create a preprocessing pipeline
            # For now, we'll just return a simple pipeline that drops the StudyID column
            preprocessing_pipeline = Pipeline(
                steps=[
                    ('drop_columns', DropColumns(columns=self.data_transformation_config.columns_to_drop))
                ]
            )

            return preprocessing_pipeline
        except Exception as e:
            raise HFException(e, sys)

    def initiate_data_transformation(self):
        """
        Initiate the data transformation process
        """
        try:
            logger.info("Starting data transformation")

            # Get the training and testing file paths
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read the training and testing data
            logger.info("Reading training and testing data")
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)

            logger.info(f"Training dataframe head:\n{train_df.head().to_string()}")
            logger.info(f"Training dataframe shape: {train_df.shape}")
            logger.info(f"Testing dataframe head:\n{test_df.head().to_string()}")
            logger.info(f"Testing dataframe shape: {test_df.shape}")

            # Get the preprocessing object
            logger.info("Getting preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Create directories for transformed data
            logger.info("Creating directories for transformed data")
            os.makedirs(self.data_transformation_config.transformed_train_dir, exist_ok=True)
            os.makedirs(self.data_transformation_config.transformed_test_dir, exist_ok=True)
            os.makedirs(self.data_transformation_config.preprocessing_dir, exist_ok=True)

            # Drop the StudyID column
            logger.info("Dropping unnecessary columns")
            train_df = train_df.drop(columns=self.data_transformation_config.columns_to_drop)
            test_df = test_df.drop(columns=self.data_transformation_config.columns_to_drop)

            logger.info(f"Columns dropped: {self.data_transformation_config.columns_to_drop}")
            logger.info(f"Training dataframe shape after dropping columns: {train_df.shape}")
            logger.info(f"Testing dataframe shape after dropping columns: {test_df.shape}")

            # Verify target column exists
            target_column = self.data_transformation_config.target_column
            logger.info(f"Target column: {target_column}")

            if target_column not in train_df.columns or target_column not in test_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in the dataset!")

            logger.info(f"Target column values in training data: {train_df[target_column].value_counts().to_dict()}")
            logger.info(f"Target column values in testing data: {test_df[target_column].value_counts().to_dict()}")

            # Save the transformed data
            logger.info("Saving transformed data")
            train_df.to_csv(self.data_transformation_config.transformed_train_file_path, index=False)
            test_df.to_csv(self.data_transformation_config.transformed_test_file_path, index=False)

            # Save the preprocessing object
            logger.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessed_object_file_path,
                obj=preprocessing_obj
            )

            # Create and return the data transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessed_object_file_path=self.data_transformation_config.preprocessed_object_file_path
            )

            logger.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise HFException(e, sys)


class DropColumns:
    """
    Custom transformer to drop specified columns
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')
