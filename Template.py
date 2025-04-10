import os

def create_project_structure(base_path):
    directories = [
        ".github/workflows",
        "artifacts/data_ingestion/raw_data",
        "artifacts/data_ingestion/feature_store",
        "artifacts/data_ingestion/dataset",
        "config",
        "data/raw",
        "data/processed",
        "data/external",
        "docs/data_dictionaries",
        "docs/references",
        "docs/reports",
        "logs",
        "models/trained",
        "models/registry",
        "notebooks",
        "src/api",
        "src/components",
        "src/config",
        "src/constants",
        "src/data",
        "src/entity",
        "src/features",
        "src/models",
        "src/monitoring",
        "src/pipeline",
        "src/utils",
        "src/visualization",
        "tests",
    ]
    
    files = [
        ".github/workflows/ci-cd.yml",
        "config/config.yaml",
        "src/api/__init__.py",
        "src/api/main.py",
        "src/components/__init__.py",
        "src/components/data_ingestion.py",
        "src/components/data_validation.py",
        "src/components/data_transformation.py",
        "src/components/model_trainer.py",
        "src/components/model_evaluation.py",
        "src/components/model_pusher.py",
        "src/config/__init__.py",
        "src/config/configuration.py",
        "src/constants/__init__.py",
        "src/data/__init__.py",
        "src/data/make_dataset.py",
        "src/entity/__init__.py",
        "src/entity/config_entity.py",
        "src/entity/artifact_entity.py",
        "src/features/__init__.py",
        "src/features/build_features.py",
        "src/models/__init__.py",
        "src/models/train_model.py",
        "src/models/predict_model.py",
        "src/models/evaluate_model.py",
        "src/monitoring/__init__.py",
        "src/monitoring/monitor.py",
        "src/pipeline/__init__.py",
        "src/pipeline/data_ingestion_pipeline.py",
        "src/pipeline/training_pipeline.py",
        "src/pipeline/prediction_pipeline.py",
        "src/utils/__init__.py",
        "src/utils/utils.py",
        "src/visualization/__init__.py",
        "src/visualization/visualize.py",
        "src/exception.py",
        "src/logger.py",
        "src/__init__.py",
        "tests/__init__.py",
        "tests/test_data.py",
        "tests/test_models.py",
        ".env",
        ".gitignore",
        "Dockerfile",
        "Makefile",
        "README.md",
        "requirements.txt",
        "setup.py",
        "main.py"
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)
    
    for file in files:
        file_path = os.path.join(base_path, file)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("")  # Create empty file

if __name__ == "__main__":
    base_directory = "heart_failure_detection"
    create_project_structure(base_directory)
    print(f"Project structure created at {base_directory}")
