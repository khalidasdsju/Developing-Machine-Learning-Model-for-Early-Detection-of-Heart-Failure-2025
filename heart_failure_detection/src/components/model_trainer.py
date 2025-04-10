import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.exception import CustomException
from src.logger import logging
from src.utils.utils import save_object, load_numpy_array_data, get_model_metrics
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'='*20} Model Training {'='*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.SEED = self.model_trainer_config.seed
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed training and testing data")
            train_array = load_numpy_array_data(self.data_transformation_artifact.transformed_train_array_file_path)
            test_array = load_numpy_array_data(self.data_transformation_artifact.transformed_test_array_file_path)

            logging.info("Splitting training and testing data into features and target")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Log information about the target variable
            logging.info(f"Target variable (HF) shape - Train: {y_train.shape}, Test: {y_test.shape}")

            # Check unique values in target
            unique_train = np.unique(y_train)
            unique_test = np.unique(y_test)
            logging.info(f"Unique values in target - Train: {unique_train}, Test: {unique_test}")

            # Count occurrences of each class
            train_counts = {val: np.sum(y_train == val) for val in unique_train}
            test_counts = {val: np.sum(y_test == val) for val in unique_test}
            logging.info(f"Class distribution in train set: {train_counts}")
            logging.info(f"Class distribution in test set: {test_counts}")

            # Define models
            models = {
                # 🔹 Logistic Regression: Strong regularization for better generalization
                'Logistic Regression': LogisticRegression(
                    C=0.3, solver='liblinear', penalty='l1', class_weight='balanced', random_state=self.SEED
                ),

                # 🔹 K-Nearest Neighbors: Reducing overfitting with distance-based weighting
                'K-Nearest Neighbors': KNeighborsClassifier(
                    n_neighbors=5, weights='distance', algorithm='ball_tree', leaf_size=20, p=2
                ),

                # 🔹 Naive Bayes: Adjusted for better probability estimation
                'Naive Bayes': GaussianNB(var_smoothing=1e-8),

                # 🔹 Decision Tree: More depth and splitting to capture complex patterns
                'Decision Tree': DecisionTreeClassifier(
                    criterion='entropy', max_depth=20, min_samples_split=3, min_samples_leaf=2, random_state=self.SEED
                ),

                # 🔹 Random Forest: More trees & depth for higher accuracy
                'Random Forest': RandomForestClassifier(
                    n_estimators=300, max_depth=15, min_samples_split=3, min_samples_leaf=1,
                    class_weight='balanced', random_state=self.SEED
                ),

                # 🔹 Support Vector Machine: Higher C, balanced class weights
                'Support Vector Machine': SVC(
                    C=5.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=self.SEED
                ),

                # 🔹 Ridge Classifier: Optimized regularization for stability
                'Ridge Classifier': RidgeClassifier(alpha=0.3, class_weight='balanced'),

                # 🔹 Linear Discriminant Analysis: Optimized shrinkage
                'Linear Discriminant Analysis': LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),

                # 🔹 AdaBoost: More estimators & lower learning rate
                'AdaBoost': AdaBoostClassifier(
                    n_estimators=200, learning_rate=0.05, random_state=self.SEED
                ),

                # 🔹 Gradient Boosting: Lower learning rate, more estimators
                'Gradient Boosting': GradientBoostingClassifier(
                    learning_rate=0.03, n_estimators=250, max_depth=10, min_samples_split=3,
                    min_samples_leaf=2, subsample=0.85, random_state=self.SEED
                ),

                # 🔹 Extra Trees Classifier: Higher estimators for robustness
                'Extra Trees Classifier': ExtraTreesClassifier(
                    n_estimators=300, max_depth=12, min_samples_split=4, random_state=self.SEED
                ),

                # 🔹 LightGBM: Optimized for best recall & precision
                'LightGBM': LGBMClassifier(
                    colsample_bytree=0.9, learning_rate=0.03, max_depth=20, min_child_samples=5,
                    n_estimators=250, num_leaves=90, subsample=0.85, class_weight='balanced', verbose=-1, random_state=self.SEED
                ),

                # 🔹 Multi-Layer Perceptron (MLP): More layers & epochs for better feature learning
                'Multi-Layer Perceptron (MLP)': MLPClassifier(
                    hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', alpha=0.0001,
                    batch_size=16, learning_rate='adaptive', max_iter=500, random_state=self.SEED
                ),

                # 🔹 XGBoost: Tuned for high performance
                'XGBoost': XGBClassifier(
                    colsample_bytree=0.8, learning_rate=0.03, max_depth=12, min_child_weight=4,
                    n_estimators=250, subsample=0.85, gamma=0.1, reg_alpha=0.1, reg_lambda=0.1,
                    scale_pos_weight=1, random_state=self.SEED
                ),

                # 🔹 CatBoost: Tuned for medical applications
                'CatBoost': CatBoostClassifier(
                    iterations=250, learning_rate=0.03, depth=14, min_data_in_leaf=5,
                    subsample=0.85, l2_leaf_reg=3, class_weights=[1, 4], random_state=self.SEED, verbose=0
                )
            }

            # Evaluate models and store results in a DataFrame
            logging.info("Evaluating models")
            results = []
            for name, model in models.items():
                logging.info(f"Training {name}")
                # Fit the model to the training data
                model.fit(X_train, y_train)

                # Make predictions on the test data
                y_pred = model.predict(X_test)

                # Convert y_test and y_pred to integers before calculating accuracy
                accuracy = accuracy_score(y_test.astype(int), y_pred.astype(int))

                # Get classification report for multi-class classification
                report = classification_report(y_test.astype(int), y_pred.astype(int), output_dict=True)

                # Collect results dynamically for each class
                model_result = {
                    'Model': name,
                    'Accuracy': accuracy,
                }

                # Add Precision, Recall, F1 score for each class dynamically
                for label in report.keys():
                    if label != 'accuracy':  # Exclude the accuracy key
                        model_result[f'Precision (Class {label})'] = report[label]['precision']
                        model_result[f'Recall (Class {label})'] = report[label]['recall']
                        model_result[f'F1 Score (Class {label})'] = report[label]['f1-score']

                results.append(model_result)

            # Create a DataFrame from results
            results_df = pd.DataFrame(results)

            # Sort results by accuracy
            sorted_results_df = results_df.sort_values(by='Accuracy', ascending=False)

            # Display the sorted results
            logging.info("\nSorted Results by Accuracy:")
            logging.info(sorted_results_df)

            # Print the best three models
            logging.info("\nTop 5 Best Models Based on Overall Performance:\n")
            logging.info(sorted_results_df.head(5))

            # Store best models
            best_models = sorted_results_df.head(5)

            # Create model comparison directory
            os.makedirs(self.model_trainer_config.model_comparison_dir, exist_ok=True)

            # Save model comparison results
            model_comparison_path = os.path.join(self.model_trainer_config.model_comparison_dir, "model_comparison.csv")
            sorted_results_df.to_csv(model_comparison_path, index=False)

            # Function to create a comparison plot with line graphs for all metrics in one plot
            def plot_model_comparison_all_metrics(sorted_results_df):
                # Set the plot style
                sns.set(style="whitegrid")

                # Prepare the data for plotting
                metrics = [
                    'Accuracy',
                    'Precision (Class macro avg)',
                    'Recall (Class macro avg)',
                    'F1 Score (Class macro avg)',
                    'Precision (Class weighted avg)',
                    'Recall (Class weighted avg)',
                    'F1 Score (Class weighted avg)'
                ]

                # Create the figure and axis for plotting
                plt.figure(figsize=(14, 8))  # Adjusted plot size

                # Define a color palette for distinct line colors
                color_palette = sns.color_palette("Set2", len(metrics))  # Choose a different color for each metric

                # Iterate through each metric and plot its value
                for i, metric in enumerate(metrics):
                    if metric in sorted_results_df.columns:
                        sns.lineplot(x='Model', y=metric, data=sorted_results_df, label=metric,
                                    marker='o', linewidth=2, color=color_palette[i])

                # Set the title and labels with larger fonts
                plt.title('Comparison of Metrics for Each Model (Before Cross-Validation and Hyperamiter, With Feature Selection)', fontsize=18)
                plt.xlabel('Model', fontsize=16)
                plt.ylabel('Score', fontsize=16)
                plt.xticks(rotation=90, fontsize=14)
                plt.yticks(fontsize=14)

                # Add legend
                plt.legend(title='Metrics', fontsize=12)

                # Save the plot
                plt.tight_layout()
                plot_path = os.path.join(self.model_trainer_config.model_comparison_dir, "model_comparison_plot.png")
                plt.savefig(plot_path)
                plt.close()
                return plot_path

            # Create model comparison plot
            logging.info("Creating model comparison plot")
            plot_path = plot_model_comparison_all_metrics(sorted_results_df)

            # Get the best model
            best_model_name = sorted_results_df.iloc[0]['Model']
            best_model = models[best_model_name]

            # Save the best model
            logging.info(f"Saving the best model: {best_model_name}")
            save_object(self.model_trainer_config.model_file_path, best_model)

            # Get model metrics
            y_pred = best_model.predict(X_test)
            metrics = get_model_metrics(y_test, y_pred)

            # Create model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                model_file_path=self.model_trainer_config.model_file_path,
                accuracy_score=metrics['accuracy'],
                precision_score=metrics['precision'],
                recall_score=metrics['recall'],
                f1_score=metrics['f1_score'],
                model_comparison_path=model_comparison_path
            )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
