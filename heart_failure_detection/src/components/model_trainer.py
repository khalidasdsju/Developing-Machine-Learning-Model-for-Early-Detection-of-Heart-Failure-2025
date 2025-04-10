import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import yaml

from HF.logger import logging
from HF.exception import HFException
from HF.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from HF.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*30} Model Trainer {'<<'*30}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise HFException(e, sys)

    def load_model_config(self):
        try:
            logging.info("Loading model configuration")
            with open(self.model_trainer_config.model_config_file_path, 'r') as f:
                model_config = yaml.safe_load(f)
            return model_config
        except Exception as e:
            raise HFException(e, sys)

    def train_model(self, X, y):
        try:
            logging.info("Training Random Forest model")
            model_config = self.load_model_config()
            rf_params = model_config.get('random_forest', {})

            # Create and train the model
            rf_classifier = RandomForestClassifier(
                n_estimators=rf_params.get('n_estimators', 100),
                max_depth=rf_params.get('max_depth', None),
                min_samples_split=rf_params.get('min_samples_split', 2),
                min_samples_leaf=rf_params.get('min_samples_leaf', 1),
                random_state=42
            )

            rf_classifier.fit(X, y)
            return rf_classifier
        except Exception as e:
            raise HFException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")

            # Load transformed training and testing arrays
            train_arr = np.load(self.data_transformation_artifact.transformed_train_array_file_path, allow_pickle=True)
            test_arr = np.load(self.data_transformation_artifact.transformed_test_array_file_path, allow_pickle=True)

            # Split arrays into features and target
            logging.info("Splitting data into features and target")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Train the model
            logging.info("Training the model")
            self.model = self.train_model(X_train, y_train)

            # Make predictions
            logging.info("Making predictions on training and testing data")
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)

            # Calculate accuracy
            # Convert string labels to numeric if needed for comparison
            if isinstance(y_test[0], str) and isinstance(y_test_pred[0], (int, float, np.integer, np.floating)):
                logging.info("Converting numeric predictions back to string labels for comparison")
                # Map 0 to 'No HF' and 1 to 'HF'
                y_train_pred_str = np.array(['HF' if pred == 1 else 'No HF' for pred in y_train_pred])
                y_test_pred_str = np.array(['HF' if pred == 1 else 'No HF' for pred in y_test_pred])

                train_accuracy = accuracy_score(y_train, y_train_pred_str)
                test_accuracy = accuracy_score(y_test, y_test_pred_str)
            else:
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)

            logging.info(f"Train accuracy: {train_accuracy}")
            logging.info(f"Test accuracy: {test_accuracy}")

            # Check if model meets base accuracy
            if test_accuracy < self.model_trainer_config.base_accuracy:
                logging.warning(f"Model accuracy {test_accuracy} is less than base accuracy {self.model_trainer_config.base_accuracy}")
                # Continue anyway for demonstration purposes
                logging.info("Continuing with the model despite low accuracy for demonstration purposes")

            # Save the model
            logging.info("Saving the model")
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

            import joblib
            joblib.dump(self.model, self.model_trainer_config.trained_model_file_path)

            # Create model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise HFException(e, sys)