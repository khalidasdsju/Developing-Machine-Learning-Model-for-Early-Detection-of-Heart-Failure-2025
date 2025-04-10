import os
import sys
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion
    """
    def __init__(self,
                 dataset_download_url: str = None,
                 raw_data_dir: str = None,
                 feature_store_dir: str = None,
                 ingested_train_dir: str = None,
                 ingested_test_dir: str = None,
                 train_test_split_ratio: float = 0.2):

        # Set default paths if not provided
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        artifact_dir = os.path.join("artifacts", timestamp)

        self.dataset_download_url = dataset_download_url

        # Raw data directory
        self.raw_data_dir = raw_data_dir or os.path.join(
            artifact_dir, "data_ingestion", "raw_data"
        )

        # Feature store directory and file path
        self.feature_store_dir = feature_store_dir or os.path.join(
            artifact_dir, "data_ingestion", "feature_store"
        )
        self.feature_store_file_path = os.path.join(
            self.feature_store_dir, "heart_failure_data.csv"
        )

        # Train and test directories and file paths
        self.ingested_train_dir = ingested_train_dir or os.path.join(
            artifact_dir, "data_ingestion", "dataset"
        )
        self.training_file_path = os.path.join(
            self.ingested_train_dir, "train.csv"
        )

        self.ingested_test_dir = ingested_test_dir or os.path.join(
            artifact_dir, "data_ingestion", "dataset"
        )
        self.testing_file_path = os.path.join(
            self.ingested_test_dir, "test.csv"
        )

        # Train-test split ratio
        self.train_test_split_ratio = train_test_split_ratio


@dataclass
class DataValidationConfig:
    """
    Configuration for data validation
    """
    def __init__(self,
                 schema_file_path: str = None,
                 report_file_path: str = None,
                 drift_report_file_path: str = None):

        # Set default paths if not provided
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        artifact_dir = os.path.join("artifacts", timestamp)

        # Schema file path
        self.schema_file_path = schema_file_path or os.path.join(
            "config", "schema.yaml"
        )

        # Report directory and file paths
        report_dir = os.path.join(artifact_dir, "data_validation")

        self.report_file_path = report_file_path or os.path.join(
            report_dir, "report.json"
        )

        self.drift_report_file_path = drift_report_file_path or os.path.join(
            report_dir, "drift_report.json"
        )


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation
    """
    def __init__(self,
                 transformed_train_dir: str = None,
                 transformed_test_dir: str = None,
                 preprocessing_dir: str = None,
                 preprocessed_object_file_path: str = None):

        # Set default paths if not provided
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        artifact_dir = os.path.join("artifacts", timestamp)

        # Transformed data directories and file paths
        self.transformed_train_dir = transformed_train_dir or os.path.join(
            artifact_dir, "data_transformation", "transformed", "train"
        )
        self.transformed_train_file_path = os.path.join(
            self.transformed_train_dir, "transformed_train.csv"
        )
        self.transformed_train_array_file_path = os.path.join(
            self.transformed_train_dir, "transformed_train.npz"
        )

        self.transformed_test_dir = transformed_test_dir or os.path.join(
            artifact_dir, "data_transformation", "transformed", "test"
        )
        self.transformed_test_file_path = os.path.join(
            self.transformed_test_dir, "transformed_test.csv"
        )
        self.transformed_test_array_file_path = os.path.join(
            self.transformed_test_dir, "transformed_test.npz"
        )

        # Preprocessing directory and file path
        self.preprocessing_dir = preprocessing_dir or os.path.join(
            artifact_dir, "data_transformation", "preprocessed"
        )
        self.preprocessed_object_file_path = preprocessed_object_file_path or os.path.join(
            self.preprocessing_dir, "preprocessed.pkl"
        )
        self.transformed_object_file_path = os.path.join(
            self.preprocessing_dir, "transformer.pkl"
        )

        # Columns to drop
        self.drop_columns = ["StudyID"]

        # Feature categories
        self.num_features = [
            "Age", "BMI", "HR", "RBS", "HbA1C", "Creatinine", "Na", "K", "Cl", "Hb",
            "TropI", "LVIDd", "FS", "LVIDs", "LVEF", "LAV", "ICT", "IRT", "EA", "DT",
            "MPI", "RR", "TC", "LDLc", "HDLc", "TG", "BNP"
        ]

        self.or_columns = [
            "Sex", "NYHA", "HTN", "DM", "Smoker", "DL", "BA", "CXR", "RWMA", "MI", "Chest_pain"
        ]

        self.oh_columns = [
            "ECG", "ACS", "Wall", "MR", "Thrombolysis"
        ]

        # Target column
        self.target_column = "HF"

        # Available columns (will be set dynamically during transformation)
        self.available_columns = []


@dataclass
class DataProfilingConfig:
    """
    Configuration for data profiling
    """
    def __init__(self,
                 profile_report_dir: str = None,
                 train_profile_report_file_path: str = None,
                 test_profile_report_file_path: str = None,
                 full_profile_report_file_path: str = None):

        # Set default paths if not provided
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        artifact_dir = os.path.join("artifacts", timestamp)

        # Profile report directory
        self.profile_report_dir = profile_report_dir or os.path.join(
            artifact_dir, "data_profiling"
        )

        # Profile report file paths
        self.train_profile_report_file_path = train_profile_report_file_path or os.path.join(
            self.profile_report_dir, "train_profile_report.html"
        )

        self.test_profile_report_file_path = test_profile_report_file_path or os.path.join(
            self.profile_report_dir, "test_profile_report.html"
        )

        self.full_profile_report_file_path = full_profile_report_file_path or os.path.join(
            self.profile_report_dir, "full_profile_report.html"
        )


@dataclass
class ModelTrainerConfig:
    """
    Configuration for model training
    """
    def __init__(self,
                 trained_model_dir: str = None,
                 trained_model_file_path: str = None,
                 model_config_file_path: str = None):

        # Set default paths if not provided
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        artifact_dir = os.path.join("artifacts", timestamp)

        # Trained model directory and file path
        self.trained_model_dir = trained_model_dir or os.path.join(
            artifact_dir, "model_trainer"
        )
        self.trained_model_file_path = trained_model_file_path or os.path.join(
            self.trained_model_dir, "model.pkl"
        )

        # Model config file path
        self.model_config_file_path = model_config_file_path or os.path.join(
            "config", "model.yaml"
        )

        # Model parameters
        self.base_accuracy = 0.3
