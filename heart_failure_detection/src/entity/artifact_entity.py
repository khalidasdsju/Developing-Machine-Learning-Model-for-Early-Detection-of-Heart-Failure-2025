from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str
    train_file_path: str
    test_file_path: str

@dataclass
class DataTransformationArtifact:
    feature_engineering_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_train_array_file_path: str
    transformed_test_array_file_path: str
    feature_names: list

@dataclass
class ModelTrainerArtifact:
    model_file_path: str
    accuracy_score: float
    precision_score: float
    recall_score: float
    f1_score: float
    model_comparison_path: str

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    best_model_metrics_path: str

@dataclass
class ModelPusherArtifact:
    pusher_model_dir: str
    saved_model_dir: str
