from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    Artifact produced by data ingestion component
    """
    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    """
    Artifact produced by data validation component
    """
    validation_status: bool
    message: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    """
    Artifact produced by data transformation component
    """
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessed_object_file_path: str
    transformed_object_file_path: str = None
    transformed_train_array_file_path: str = None
    transformed_test_array_file_path: str = None


@dataclass
class DataProfilingArtifact:
    """
    Artifact produced by data profiling component
    """
    train_profile_report_file_path: str
    test_profile_report_file_path: str
    full_profile_report_file_path: str


@dataclass
class ModelTrainerArtifact:
    """
    Artifact produced by model trainer component
    """
    trained_model_file_path: str
    train_accuracy: float
    test_accuracy: float
