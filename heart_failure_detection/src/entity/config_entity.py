import os
import sys
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.ingestion_dir: str = os.path.join("artifacts", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "data_ingestion")
        self.feature_store_file_path: str = os.path.join(self.ingestion_dir, "feature_store", "heart_failure_data.csv")
        self.train_file_path: str = os.path.join(self.ingestion_dir, "dataset", "train.csv")
        self.test_file_path: str = os.path.join(self.ingestion_dir, "dataset", "test.csv")
        self.test_size: float = 0.2

@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.data_ingestion_dir: str = os.path.join("artifacts", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "data_ingestion")
        self.data_transformation_dir: str = os.path.join("artifacts", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "data_transformation")
        self.feature_engineering_object_file_path: str = os.path.join(self.data_transformation_dir, "preprocessed", "preprocessed.pkl")
        self.transformed_train_file_path: str = os.path.join(self.data_transformation_dir, "transformed", "train", "transformed_train.csv")
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir, "transformed", "test", "transformed_test.csv")
        self.transformed_train_array_file_path: str = os.path.join(self.data_transformation_dir, "transformed", "train", "transformed_train.npz")
        self.transformed_test_array_file_path: str = os.path.join(self.data_transformation_dir, "transformed", "test", "transformed_test.npz")
        self.analysis_dir: str = os.path.join(self.data_transformation_dir, "transformed", "train", "analysis")
        self.columns_to_drop: list = ["StudyID"]
        self.target_column: str = "HF"

@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.data_transformation_dir: str = os.path.join("artifacts", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "data_transformation")
        self.model_trainer_dir: str = os.path.join("artifacts", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "model_trainer")
        self.model_file_path: str = os.path.join(self.model_trainer_dir, "model.pkl")
        self.expected_accuracy: float = 0.7
        self.seed: int = 42
        self.model_comparison_dir: str = os.path.join(self.model_trainer_dir, "model_comparison")

@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.model_evaluation_dir: str = os.path.join("artifacts", "model_evaluation")
        self.model_comparison_dir: str = os.path.join("artifacts", "model_comparison")
        self.feature_importance_dir: str = os.path.join("artifacts", "feature_importance")
        self.threshold: float = 0.7

@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.model_pusher_dir: str = os.path.join("artifacts", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "model_pusher")
        self.saved_model_dir: str = os.path.join("saved_models")
        self.pusher_model_dir: str = os.path.join(self.model_pusher_dir, "saved_models")
        self.pusher_model_path: str = os.path.join(self.pusher_model_dir, "model.pkl")
        self.pusher_transformer_path: str = os.path.join(self.pusher_model_dir, "transformer.pkl")
