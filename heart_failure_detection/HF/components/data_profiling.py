import os
import sys
import pandas as pd
from ydata_profiling import ProfileReport

from HF.logger import logging
from HF.exception import HFException
from HF.entity.config_entity import DataProfilingConfig
from HF.entity.artifact_entity import DataIngestionArtifact, DataProfilingArtifact

class DataProfiling:
    def __init__(self, data_profiling_config: DataProfilingConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(f"{'='*20}Data Profiling log started.{'='*20}")
            self.data_profiling_config = data_profiling_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise HFException(e, sys)
    
    def generate_profile_report(self, data: pd.DataFrame, report_file_path: str, title: str):
        """
        Generate a profile report for the given data
        """
        try:
            logging.info(f"Generating profile report: {title}")
            profile = ProfileReport(data, title=title, explorative=True)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
            
            # Save the report
            profile.to_file(report_file_path)
            logging.info(f"Profile report saved at: {report_file_path}")
            
            return profile
        except Exception as e:
            raise HFException(f"Error generating profile report: {e}", sys)
    
    def initiate_data_profiling(self) -> DataProfilingArtifact:
        """
        Initiate the data profiling process
        """
        try:
            logging.info("Starting data profiling")
            
            # Read the training and testing data
            logging.info("Reading training and testing data")
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            # Combine train and test data for full profile
            full_df = pd.concat([train_df, test_df], axis=0)
            
            # Generate profile reports
            logging.info("Generating profile reports")
            
            # Train data profile
            self.generate_profile_report(
                data=train_df,
                report_file_path=self.data_profiling_config.train_profile_report_file_path,
                title="Training Dataset Profiling Report"
            )
            
            # Test data profile
            self.generate_profile_report(
                data=test_df,
                report_file_path=self.data_profiling_config.test_profile_report_file_path,
                title="Testing Dataset Profiling Report"
            )
            
            # Full data profile
            self.generate_profile_report(
                data=full_df,
                report_file_path=self.data_profiling_config.full_profile_report_file_path,
                title="Full Dataset Profiling Report"
            )
            
            # Create and return the data profiling artifact
            data_profiling_artifact = DataProfilingArtifact(
                train_profile_report_file_path=self.data_profiling_config.train_profile_report_file_path,
                test_profile_report_file_path=self.data_profiling_config.test_profile_report_file_path,
                full_profile_report_file_path=self.data_profiling_config.full_profile_report_file_path
            )
            
            logging.info(f"Data profiling artifact: {data_profiling_artifact}")
            return data_profiling_artifact
        
        except Exception as e:
            raise HFException(f"Error in data profiling: {e}", sys)
