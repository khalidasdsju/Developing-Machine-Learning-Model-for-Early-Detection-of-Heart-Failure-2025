import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
# SHAP analysis removed
import joblib
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

from HF.logger import logging
from HF.exception import HFException

def load_data_file(file_path):
    """Load data from various file formats (npy, npz, csv)"""
    try:
        if file_path.endswith('.npz'):
            # Load npz file
            data = np.load(file_path, allow_pickle=True)
            # For npz files, try different approaches
            try:
                # First try to access as a dictionary
                if isinstance(data, np.lib.npyio.NpzFile):
                    if 'arr_0' in data:
                        return data['arr_0']
                    else:
                        # Try to get the first array in the file
                        for key in data.keys():
                            return data[key]
                else:
                    # If it's already an array, return it
                    return data
            except:
                # If all else fails, just return the data
                return data
        elif file_path.endswith('.npy'):
            # Load npy file
            return np.load(file_path, allow_pickle=True)
        elif file_path.endswith('.csv'):
            # Load csv file
            df = pd.read_csv(file_path)
            return df.values
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        # Try a simpler approach as a fallback
        try:
            return np.load(file_path, allow_pickle=True)
        except:
            raise e

def load_model_config():
    """Load model configuration from YAML file"""
    try:
        config_path = os.path.join("config", "model.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise Exception(f"Error loading model config: {e}")

def initialize_models(config):
    """Initialize all models from configuration"""
    models = {}

    # Get model configurations
    model_configs = config.get('models', {})

    # Initialize each model based on its configuration
    for model_key, model_config in model_configs.items():
        model_name = model_config.get('name')
        model_params = model_config.get('params', {})

        try:
            if model_key == 'logistic_regression':
                models[model_name] = LogisticRegression(**model_params)
            elif model_key == 'knn':
                models[model_name] = KNeighborsClassifier(**model_params)
            elif model_key == 'naive_bayes':
                models[model_name] = GaussianNB(**model_params)
            elif model_key == 'decision_tree':
                models[model_name] = DecisionTreeClassifier(**model_params)
            elif model_key == 'random_forest':
                models[model_name] = RandomForestClassifier(**model_params)
            elif model_key == 'svm':
                models[model_name] = SVC(**model_params)
            elif model_key == 'ridge':
                models[model_name] = RidgeClassifier(**model_params)
            elif model_key == 'lda':
                models[model_name] = LinearDiscriminantAnalysis(**model_params)
            elif model_key == 'adaboost':
                models[model_name] = AdaBoostClassifier(**model_params)
            elif model_key == 'gradient_boosting':
                models[model_name] = GradientBoostingClassifier(**model_params)
            elif model_key == 'extra_trees':
                models[model_name] = ExtraTreesClassifier(**model_params)
            elif model_key == 'lightgbm':
                models[model_name] = lgb.LGBMClassifier(**model_params)
            elif model_key == 'mlp':
                models[model_name] = MLPClassifier(**model_params)
            elif model_key == 'xgboost':
                models[model_name] = xgb.XGBClassifier(**model_params)
            elif model_key == 'catboost':
                models[model_name] = CatBoostClassifier(**model_params)
            else:
                logging.warning(f"Unknown model type: {model_key}, skipping")
                continue

            logging.info(f"Initialized model: {model_name}")
        except Exception as e:
            logging.error(f"Error initializing model {model_name}: {e}")
            continue

    return models

def evaluate_models(models, X_train, X_test, y_train, y_test, feature_names=None):
    """Evaluate all models and return results DataFrame"""
    results = []
    trained_models = {}

    for name, model in models.items():
        try:
            logging.info(f"Training and evaluating model: {name}")

            # Fit the model to the training data
            model.fit(X_train, y_train)
            trained_models[name] = model

            # Make predictions on the test data
            y_pred = model.predict(X_test)

            # Convert predictions to the same type as y_test if needed
            if isinstance(y_test[0], str) and not isinstance(y_pred[0], str):
                y_pred = np.array(['HF' if pred == 1 else 'No HF' for pred in y_pred])

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Get classification report
            report = classification_report(y_test, y_pred, output_dict=True)

            # Collect results
            model_result = {
                'Model': name,
                'Accuracy': accuracy,
            }

            # Add metrics for each class and averages
            for label, metrics in report.items():
                if isinstance(metrics, dict):  # Skip non-dict items
                    for metric_name, value in metrics.items():
                        if metric_name in ['precision', 'recall', 'f1-score']:
                            model_result[f'{metric_name.capitalize()} (Class {label})'] = value

            results.append(model_result)
            logging.info(f"Model {name} - Accuracy: {accuracy:.4f}")

        except Exception as e:
            logging.error(f"Error evaluating model {name}: {e}")
            continue

    # Create a DataFrame from results
    results_df = pd.DataFrame(results)

    # Save the trained models
    os.makedirs("artifacts/model_evaluation", exist_ok=True)
    for name, model in trained_models.items():
        try:
            model_path = os.path.join("artifacts/model_evaluation", f"{name.replace(' ', '_').lower()}.pkl")
            joblib.dump(model, model_path)
            logging.info(f"Saved model {name} to {model_path}")
        except Exception as e:
            logging.error(f"Error saving model {name}: {e}")

    return results_df, trained_models

def plot_model_comparison(results_df, output_dir="artifacts/model_evaluation"):
    """Create visualization plots for model comparison"""
    os.makedirs(output_dir, exist_ok=True)

    # Sort results by accuracy
    sorted_results = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

    # 1. Bar plot for accuracy
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Model', y='Accuracy', data=sorted_results, palette='viridis')
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_accuracy_comparison.png'))
    plt.close()

    # 2. Heatmap for all metrics
    # Get all metric columns (excluding 'Model')
    metric_columns = [col for col in sorted_results.columns if col != 'Model']

    # Create a heatmap
    plt.figure(figsize=(16, 10))
    heatmap_data = sorted_results.set_index('Model')[metric_columns]
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
    plt.title('Model Performance Metrics Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_metrics_heatmap.png'))
    plt.close()

    # 3. Line plot for all metrics
    # Prepare data for line plot
    metrics_to_plot = [col for col in sorted_results.columns if col != 'Model' and 'macro avg' in col or col == 'Accuracy']

    if metrics_to_plot:
        plt.figure(figsize=(14, 8))
        for metric in metrics_to_plot:
            sns.lineplot(x='Model', y=metric, data=sorted_results, marker='o', label=metric)

        plt.title('Model Performance Metrics Comparison', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.legend(title='Metrics', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_metrics_comparison.png'))
        plt.close()

    return sorted_results

# SHAP analysis function removed

def main():
    try:
        # Use a specific artifact directory that we know has the data
        artifacts_dir = "artifacts"
        latest_dir = os.path.join(artifacts_dir, "2025-04-10_21-33-48")

        # Load transformed data
        transformed_train_path = os.path.join(latest_dir, "data_transformation", "transformed", "train", "transformed_train.npz")
        transformed_test_path = os.path.join(latest_dir, "data_transformation", "transformed", "test", "transformed_test.npz")

        # Check if files exist
        if not os.path.exists(transformed_train_path) or not os.path.exists(transformed_test_path):
            # Try to find array files
            transformed_train_path = os.path.join(latest_dir, "data_transformation", "transformed_train_array.npy")
            transformed_test_path = os.path.join(latest_dir, "data_transformation", "transformed_test_array.npy")

        # Load the data
        train_arr = load_data_file(transformed_train_path)
        test_arr = load_data_file(transformed_test_path)

        # Split into features and target
        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        # Try to load feature names from preprocessor
        preprocessor_path = os.path.join(latest_dir, "data_transformation", "preprocessed", "preprocessed.pkl")
        feature_names = None

        if os.path.exists(preprocessor_path):
            try:
                preprocessor = joblib.load(preprocessor_path)
                # Try to get feature names from preprocessor
                if hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out()
                else:
                    # Create generic feature names
                    feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            except:
                # Create generic feature names
                feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        else:
            # Create generic feature names
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        # Load model configuration
        config = load_model_config()

        # Initialize models
        models = initialize_models(config)

        # Evaluate models
        results_df, trained_models = evaluate_models(models, X_train, X_test, y_train, y_test, feature_names)

        # Plot model comparison
        sorted_results = plot_model_comparison(results_df)

        # Print top models
        print("\nTop 5 Models Based on Accuracy:")
        print(sorted_results.head(5))

        # SHAP analysis removed
        best_model_name = sorted_results.iloc[0]['Model']
        print(f"\nBest model: {best_model_name}")

        print("\nModel evaluation completed successfully!")
        print("Results and visualizations saved to artifacts/model_evaluation/")

    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise HFException(e, sys)

if __name__ == "__main__":
    main()
