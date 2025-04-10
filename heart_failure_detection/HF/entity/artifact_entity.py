from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    Artifact produced by data ingestion component
    """
    trained_file_path: str
    test_file_path: str


@dataclass
class DataTransformationArtifact:
    """
    Artifact produced by data transformation component
    """
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessed_object_file_path: str
