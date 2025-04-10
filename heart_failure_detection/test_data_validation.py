import os
import sys
from HF.logger import logging
from HF.exception import HFException
from HF.entity.config_entity import DataIngestionConfig, DataValidationConfig
from HF.entity.artifact_entity import DataIngestionArtifact
from HF.components.data_ingestion import DataIngestion
from HF.components.data_validation import DataValidation

def test_data_validation():
    try:
        # First, run data ingestion to get the data
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        # Now, run data validation
        data_validation_config = DataValidationConfig()
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        
        logging.info(f"Data validation completed with status: {data_validation_artifact.validation_status}")
        logging.info(f"Data validation message: {data_validation_artifact.message}")
        logging.info(f"Drift report file: {data_validation_artifact.drift_report_file_path}")
        
        return data_validation_artifact
    
    except Exception as e:
        raise HFException(e, sys)

if __name__ == "__main__":
    try:
        test_data_validation()
    except Exception as e:
        logging.error(f"Error in test_data_validation: {e}")
        print(f"Error: {e}")
