import os
import sys
import json
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.logger import get_logger
from src.exception import HFException
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

logger = get_logger(__name__)

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            logger.info(f"{'='*20}Data Validation log started.{'='*20}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            # Define a simple schema config for validation
            # This would typically be loaded from a schema file
            self._schema_config = {
                "columns": [
                    "Age", "Sex", "BMI", "NYHA", "HR", "HTN", "DM", "Smoker",
                    "DL", "BA", "RBS", "HbA1C", "Creatinine", "Na", "K", "Cl", "Hb",
                    "TropI", "CXR", "ECG", "LVIDd", "FS", "LVIDs", "LVEF", "RWMA",
                    "LAV", "MI", "ACS", "Wall", "Thrombolysis", "ICT", "IRT", "MR",
                    "EA", "DT", "MPI", "RR", "Chest_pain", "TC", "LDLc", "HDLc", "TG",
                    "BNP", "HF"
                ]
            }
        except Exception as e:
            raise HFException(f"Error in DataValidation initialization: {e}", sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logger.info(f"Validation status of columns: {status}")
            return status
        except Exception as e:
            raise HFException(f"Error validating columns: {e}", sys)

    def detect_data_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_df, current_data=current_df)
            drift_status = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
            return drift_status
        except Exception as e:
            raise HFException(f"Error detecting data drift: {e}", sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation")

            # Read the training and testing data
            logger.info("Reading training and testing data")
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Validate the number of columns
            logger.info("Validating number of columns")
            validation_status = self.validate_number_of_columns(train_df) and self.validate_number_of_columns(test_df)

            # Create directory for drift report
            logger.info("Creating directory for drift report")
            drift_report_dir = os.path.dirname(self.data_validation_config.drift_report_file_path)
            os.makedirs(drift_report_dir, exist_ok=True)

            # Detect data drift
            logger.info("Detecting data drift")
            drift_status = self.detect_data_drift(train_df, test_df)
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # Save drift report
            logger.info("Saving drift report")
            with open(drift_report_file_path, 'w') as f:
                json.dump({"drift_detected": drift_status}, f)

            logger.info(f"Data drift status: {drift_status}")
            logger.info(f"Drift report saved at: {drift_report_file_path}")

            # Create and return the data validation artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message="Validation successful" if validation_status else "Validation failed",
                drift_report_file_path=drift_report_file_path
            )

            logger.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise HFException(f"Error during data validation: {e}", sys)
