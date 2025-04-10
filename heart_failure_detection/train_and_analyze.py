import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from HF.logger import logging
from HF.exception import HFException
from HF.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig
from HF.components.data_ingestion import DataIngestion
from HF.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from HF.utils import load_numpy_array_data

def train_and_analyze():
    """
    Function to train the model and analyze performance
    """
    try:
        # Data Ingestion
        logging.info("Starting data ingestion")
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        logging.info("Starting data transformation")
        data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation(data_transformation_config, data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()

        # Model Training
        logging.info("Starting model training")
        model_trainer_config = ModelTrainerConfig()
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        # SHAP Analysis removed

        # Load transformed data
        train_arr = load_numpy_array_data(data_transformation_artifact.transformed_train_array_file_path)
        test_arr = load_numpy_array_data(data_transformation_artifact.transformed_test_array_file_path)

        # Split the data into features and target
        X_train = train_arr[:, :-1]
        y_train = train_arr[:, -1]
        X_test = test_arr[:, :-1]
        y_test = test_arr[:, -1]

        # Convert labels to numeric if they are not already
        if not np.issubdtype(y_train.dtype, np.number):
            logging.info(f"Converting training labels to numeric. Current dtype: {y_train.dtype}")
            # Create a label encoder
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            logging.info(f"Unique labels: {le.classes_}")
            logging.info(f"Encoded labels: {np.unique(y_train)}")

        # Load the trained model
        model = model_trainer.model

        # Evaluate the model
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Calculate accuracy manually
        train_score = np.mean(train_pred == y_train)
        test_score = np.mean(test_pred == y_test)

        logging.info(f"Model accuracy - Train: {train_score:.4f}, Test: {test_score:.4f}")

        # Create a directory for analysis plots
        analysis_dir = os.path.join(os.path.dirname(data_transformation_artifact.transformed_train_file_path), "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        # Convert predictions to match label type if needed
        if isinstance(y_test[0], str) and isinstance(test_pred[0], (int, float, np.integer, np.floating)):
            logging.info("Converting numeric predictions to string labels for visualization")
            # Map 0 to 'No HF' and 1 to 'HF'
            test_pred_str = np.array(['HF' if pred == 1 else 'No HF' for pred in test_pred])

            # Create and save confusion matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test, test_pred_str)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            confusion_matrix_path = os.path.join(analysis_dir, "confusion_matrix.png")
            plt.savefig(confusion_matrix_path)
            plt.close()

            # Create and save classification report
            report = classification_report(y_test, test_pred_str, output_dict=True)
            plt.figure(figsize=(10, 8))
            sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='Blues')
            plt.title('Classification Report')
            classification_report_path = os.path.join(analysis_dir, "classification_report.png")
            plt.savefig(classification_report_path)
            plt.close()
        else:
            # Create and save confusion matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test, test_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            confusion_matrix_path = os.path.join(analysis_dir, "confusion_matrix.png")
            plt.savefig(confusion_matrix_path)
            plt.close()

            # Create and save classification report
            report = classification_report(y_test, test_pred, output_dict=True)
            plt.figure(figsize=(10, 8))
            sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='Blues')
            plt.title('Classification Report')
            classification_report_path = os.path.join(analysis_dir, "classification_report.png")
            plt.savefig(classification_report_path)
            plt.close()

        # SHAP analysis removed
        logging.info("SHAP analysis has been removed")

        # Get feature names for reference
        feature_names = data_transformation_artifact.feature_names
        if feature_names is None or len(feature_names) == 0:
            # If feature names are not available, create generic names
            feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]

        logging.info(f"Using feature names: {feature_names}")

        logging.info("Model training and analysis completed successfully")
        logging.info(f"Analysis plots saved to: {analysis_dir}")

        print("="*80)
        print("Model training and analysis completed successfully!")
        print("Analysis plots are available at:")
        print(f"  - Confusion Matrix: {confusion_matrix_path}")
        print(f"  - Classification Report: {classification_report_path}")
        print("="*80)

        return model_trainer_artifact

    except Exception as e:
        logging.error(f"Error in train_and_analyze: {e}")
        raise HFException(e, sys)

if __name__ == "__main__":
    import pandas as pd
    train_and_analyze()
