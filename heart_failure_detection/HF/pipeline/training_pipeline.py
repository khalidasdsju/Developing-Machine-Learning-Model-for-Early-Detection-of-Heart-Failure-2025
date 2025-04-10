import os
import sys
from HF.logger import logging
from HF.exception import HFException
from HF.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, DataProfilingConfig
from HF.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, DataProfilingArtifact
from HF.components.data_ingestion import DataIngestion
from HF.components.data_validation import DataValidation
from HF.components.data_transformation import DataTransformation
from HF.components.data_profiling import DataProfiling

class TrainingPipeline:
    def __init__(self):
        try:
            logging.info(f"{'='*20}Training Pipeline started.{'='*20}")
        except Exception as e:
            raise HFException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            data_ingestion_config = DataIngestionConfig()
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise HFException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            data_validation_config = DataValidationConfig()
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed and artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise HFException(e, sys)

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            data_transformation_config = DataTransformationConfig()
            data_transformation = DataTransformation(
                data_transformation_config=data_transformation_config,
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed and artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise HFException(e, sys)

    def start_data_profiling(self, data_ingestion_artifact: DataIngestionArtifact) -> DataProfilingArtifact:
        try:
            logging.info("Starting data profiling")
            data_profiling_config = DataProfilingConfig()
            data_profiling = DataProfiling(
                data_profiling_config=data_profiling_config,
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_profiling_artifact = data_profiling.initiate_data_profiling()
            logging.info(f"Data profiling completed and artifact: {data_profiling_artifact}")
            return data_profiling_artifact
        except Exception as e:
            raise HFException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline")

            # Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()

            # Data Validation
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)

            # Data Profiling
            data_profiling_artifact = self.start_data_profiling(data_ingestion_artifact)
            logging.info(f"Data profiling completed. Reports available at: {data_profiling_artifact.full_profile_report_file_path}")

            # If validation is successful, proceed with transformation
            if data_validation_artifact.validation_status:
                # Data Transformation
                data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact)
                logging.info("Training pipeline completed successfully")
                return data_transformation_artifact, data_profiling_artifact
            else:
                logging.error("Data validation failed. Stopping the pipeline.")
                raise Exception("Data validation failed. Check the validation artifact for details.")

        except Exception as e:
            raise HFException(e, sys)
