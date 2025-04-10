import os
import sys
from HF.logger import logging
from HF.exception import HFException
from HF.entity.config_entity import DataIngestionConfig, DataTransformationConfig
from HF.entity.artifact_entity import DataIngestionArtifact
from HF.components.data_ingestion import DataIngestion
from HF.components.data_transformation import DataTransformation

def test_data_transformation():
    try:
        # First, run data ingestion to get the data
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        # Now, run data transformation
        data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation(
            data_transformation_config=data_transformation_config,
            data_ingestion_artifact=data_ingestion_artifact
        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        
        logging.info(f"Data transformation completed successfully")
        logging.info(f"Transformed train file: {data_transformation_artifact.transformed_train_file_path}")
        logging.info(f"Transformed test file: {data_transformation_artifact.transformed_test_file_path}")
        logging.info(f"Preprocessed object file: {data_transformation_artifact.preprocessed_object_file_path}")
        
        return data_transformation_artifact
    
    except Exception as e:
        raise HFException(e, sys)

if __name__ == "__main__":
    try:
        test_data_transformation()
    except Exception as e:
        logging.error(f"Error in test_data_transformation: {e}")
        print(f"Error: {e}")
