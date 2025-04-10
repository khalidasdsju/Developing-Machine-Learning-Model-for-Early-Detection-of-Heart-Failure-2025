import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, ModelTrainerArtifact

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config,
                data_ingestion_artifact=data_ingestion_artifact
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model trainer")
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact
            )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Pipeline started")

            # Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()

            # Data Transformation
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            # Model Trainer
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            logging.info("Pipeline completed")

        except Exception as e:
            raise CustomException(e, sys)
