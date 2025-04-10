import os
import sys
import pandas as pd
from HF.logger import logging
from HF.exception import HFException
from HF.entity.config_entity import DataIngestionConfig
from HF.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise HFException(e, sys)

    def load_sample_data(self) -> pd.DataFrame:
        try:
            logging.info("Loading sample data from existing train.csv")
            # Use the existing train.csv file as a sample data source
            sample_data_path = "/Users/khalid/Desktop/ML-Model-Deployment/artifact/04_07_2025_15_36_55/data_ingestion/data_ingestion/train.csv"
            if not os.path.exists(sample_data_path):
                raise FileNotFoundError(f"Sample data file not found: {sample_data_path}")

            dataframe = pd.read_csv(sample_data_path)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            return dataframe
        except Exception as e:
            raise HFException(f"Error loading sample data: {e}", sys)

    def create_directories(self):
        """Ensure all required directories exist"""
        # Create directory for feature store
        feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
        if not os.path.exists(feature_store_dir):
            logging.info(f"Creating directory: {feature_store_dir}")
            os.makedirs(feature_store_dir, exist_ok=True)

        # Create directory for training and testing files
        train_test_dir = os.path.dirname(self.data_ingestion_config.training_file_path)
        if not os.path.exists(train_test_dir):
            logging.info(f"Creating directory: {train_test_dir}")
            os.makedirs(train_test_dir, exist_ok=True)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            logging.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False)
            logging.info(f"Train and test data saved at: {self.data_ingestion_config.training_file_path}, {self.data_ingestion_config.testing_file_path}")
        except Exception as e:
            raise HFException(f"Error splitting data: {e}", sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Initiating data ingestion")

            # Ensure directories exist
            self.create_directories()

            # Load sample data from existing train.csv file
            logging.info("Loading sample data")
            df = self.load_sample_data()

            # Save the data to feature store for reference
            df.to_csv(self.data_ingestion_config.feature_store_file_path, index=False)
            logging.info(f"Data saved to feature store at: {self.data_ingestion_config.feature_store_file_path}")

            # Split the data into train and test sets
            self.split_data_as_train_test(df)

            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
        except Exception as e:
            raise HFException(f"Error in data ingestion: {e}", sys)
