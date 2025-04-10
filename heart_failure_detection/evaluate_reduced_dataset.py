import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
# SHAP analysis removed
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
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

def preprocess_data(X_train, X_test, scaler_type='robust'):
    """Preprocess the data using the specified scaler"""
    try:
        # Convert categorical features to numeric using one-hot encoding
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()

        # Identify categorical columns
        categorical_cols = []
        for col in X_train.columns:
            # Check if column contains non-numeric data
            if X_train[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X_train[col]):
                categorical_cols.append(col)

        # Apply one-hot encoding to categorical columns
        if categorical_cols:
            logging.info(f"Applying one-hot encoding to {len(categorical_cols)} categorical columns")
            X_train_dummies = pd.get_dummies(X_train[categorical_cols], drop_first=True)
            X_test_dummies = pd.get_dummies(X_test[categorical_cols], drop_first=True)

            # Ensure test set has same columns as train set
            for col in X_train_dummies.columns:
                if col not in X_test_dummies.columns:
                    X_test_dummies[col] = 0

            # Keep only columns that are in the training data
            X_test_dummies = X_test_dummies[X_train_dummies.columns]

            # Drop original categorical columns
            X_train_processed = X_train_processed.drop(columns=categorical_cols)
            X_test_processed = X_test_processed.drop(columns=categorical_cols)

            # Concatenate with one-hot encoded columns
            X_train_processed = pd.concat([X_train_processed, X_train_dummies], axis=1)
            X_test_processed = pd.concat([X_test_processed, X_test_dummies], axis=1)

        # Convert remaining columns to numeric
        for col in X_train_processed.columns:
            X_train_processed[col] = pd.to_numeric(X_train_processed[col], errors='coerce')
            X_test_processed[col] = pd.to_numeric(X_test_processed[col], errors='coerce')

        # Handle missing values
        X_train_processed = X_train_processed.fillna(X_train_processed.median())
        X_test_processed = X_test_processed.fillna(X_test_processed.median())

        # Apply scaling
        if scaler_type.lower() == 'standard':
            scaler = StandardScaler()
        else:  # default to RobustScaler
            scaler = RobustScaler()

        X_train_scaled = scaler.fit_transform(X_train_processed)
        X_test_scaled = scaler.transform(X_test_processed)

        # Store feature names for SHAP analysis
        feature_names = X_train_processed.columns.tolist()

        return X_train_scaled, X_test_scaled, scaler, feature_names
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise e

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
            if isinstance(y_test.iloc[0], str) and not isinstance(y_pred[0], str):
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
    os.makedirs("artifacts/reduced_model_evaluation", exist_ok=True)
    for name, model in trained_models.items():
        try:
            model_path = os.path.join("artifacts/reduced_model_evaluation", f"{name.replace(' ', '_').lower()}.pkl")
            joblib.dump(model, model_path)
            logging.info(f"Saved model {name} to {model_path}")
        except Exception as e:
            logging.error(f"Error saving model {name}: {e}")

    return results_df, trained_models

def plot_model_comparison(results_df, output_dir="artifacts/reduced_model_evaluation"):
    """Create visualization plots for model comparison"""
    os.makedirs(output_dir, exist_ok=True)

    # Sort results by accuracy
    sorted_results = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

    # 1. Bar plot for accuracy
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Model', y='Accuracy', data=sorted_results, palette='viridis')
    plt.title('Model Accuracy Comparison (Reduced Dataset)', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reduced_model_accuracy_comparison.png'))
    plt.close()

    # 2. Heatmap for all metrics
    # Get all metric columns (excluding 'Model')
    metric_columns = [col for col in sorted_results.columns if col != 'Model']

    # Create a heatmap
    plt.figure(figsize=(16, 10))
    heatmap_data = sorted_results.set_index('Model')[metric_columns]
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
    plt.title('Model Performance Metrics Heatmap (Reduced Dataset)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reduced_model_metrics_heatmap.png'))
    plt.close()

    # 3. Line plot for all metrics
    # Prepare data for line plot
    metrics_to_plot = [col for col in sorted_results.columns if col != 'Model' and 'macro avg' in col or col == 'Accuracy']

    if metrics_to_plot:
        plt.figure(figsize=(14, 8))
        for metric in metrics_to_plot:
            sns.lineplot(x='Model', y=metric, data=sorted_results, marker='o', label=metric)

        plt.title('Model Performance Metrics Comparison (Reduced Dataset)', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.legend(title='Metrics', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reduced_model_metrics_comparison.png'))
        plt.close()

    return sorted_results

# SHAP analysis function removed

def compare_original_vs_reduced(original_results_path, reduced_results_df, output_dir="artifacts/reduced_model_evaluation"):
    """Compare the performance of models on original vs reduced dataset"""
    try:
        # Load original results
        original_results_df = pd.read_csv(original_results_path)

        # Ensure both dataframes have the same models
        common_models = list(set(original_results_df['Model']).intersection(set(reduced_results_df['Model'])))

        if not common_models:
            logging.warning("No common models found between original and reduced results")
            return

        # Filter to common models
        original_filtered = original_results_df[original_results_df['Model'].isin(common_models)]
        reduced_filtered = reduced_results_df[reduced_results_df['Model'].isin(common_models)]

        # Sort both by model name for consistency
        original_filtered = original_filtered.sort_values('Model').reset_index(drop=True)
        reduced_filtered = reduced_filtered.sort_values('Model').reset_index(drop=True)

        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': original_filtered['Model'],
            'Original Accuracy': original_filtered['Accuracy'],
            'Reduced Accuracy': reduced_filtered['Accuracy'],
            'Accuracy Difference': reduced_filtered['Accuracy'] - original_filtered['Accuracy']
        })

        # Save comparison to CSV
        comparison_path = os.path.join(output_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)

        # Create comparison bar chart
        plt.figure(figsize=(14, 8))

        # Set width of bars
        barWidth = 0.3

        # Set positions of bars on X axis
        r1 = np.arange(len(comparison_df))
        r2 = [x + barWidth for x in r1]

        # Create bars
        plt.bar(r1, comparison_df['Original Accuracy'], width=barWidth, label='Original Dataset', color='blue', alpha=0.7)
        plt.bar(r2, comparison_df['Reduced Accuracy'], width=barWidth, label='Reduced Dataset', color='green', alpha=0.7)

        # Add labels and title
        plt.xlabel('Models', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title('Model Accuracy: Original vs. Reduced Dataset', fontsize=16)
        plt.xticks([r + barWidth/2 for r in range(len(comparison_df))], comparison_df['Model'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        # Save the plot
        comparison_plot_path = os.path.join(output_dir, "accuracy_comparison.png")
        plt.savefig(comparison_plot_path)
        plt.close()

        # Create difference plot
        plt.figure(figsize=(14, 8))
        colors = ['green' if x >= 0 else 'red' for x in comparison_df['Accuracy Difference']]
        sns.barplot(x='Model', y='Accuracy Difference', data=comparison_df, palette=colors)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Accuracy Difference: Reduced - Original Dataset', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Accuracy Difference', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the plot
        difference_plot_path = os.path.join(output_dir, "accuracy_difference.png")
        plt.savefig(difference_plot_path)
        plt.close()

        logging.info(f"Comparison analysis completed and saved to {output_dir}")
        return comparison_df

    except Exception as e:
        logging.error(f"Error in comparison analysis: {e}")
        return None

def main():
    try:
        # Check if reduced dataset exists
        reduced_train_path = "artifacts/reduced_dataset/reduced_train.csv"
        reduced_test_path = "artifacts/reduced_dataset/reduced_test.csv"

        if not os.path.exists(reduced_train_path) or not os.path.exists(reduced_test_path):
            logging.error("Reduced dataset not found. Please run create_reduced_dataset.py first.")
            print("Reduced dataset not found. Please run create_reduced_dataset.py first.")
            return

        # Load reduced dataset
        train_df = pd.read_csv(reduced_train_path)
        test_df = pd.read_csv(reduced_test_path)

        # Split into features and target
        X_train = train_df.drop(columns=['HF'])
        y_train = train_df['HF']
        X_test = test_df.drop(columns=['HF'])
        y_test = test_df['HF']

        # Preprocess the data
        X_train_scaled, X_test_scaled, _, feature_names = preprocess_data(X_train, X_test)

        # Load model configuration
        config = load_model_config()

        # Initialize models
        models = initialize_models(config)

        # Evaluate models
        results_df, trained_models = evaluate_models(models, X_train_scaled, X_test_scaled, y_train, y_test, feature_names)

        # Plot model comparison
        sorted_results = plot_model_comparison(results_df)

        # Print top models
        print("\nTop 5 Models Based on Accuracy (Reduced Dataset):")
        print(sorted_results.head(5))

        # SHAP analysis removed
        best_model_name = sorted_results.iloc[0]['Model']
        print(f"\nBest model: {best_model_name}")

        # Compare with original results if available
        original_results_path = "artifacts/model_evaluation/model_results.csv"
        if os.path.exists(original_results_path):
            # Save current results first
            results_path = os.path.join("artifacts/reduced_model_evaluation", "reduced_model_results.csv")
            results_df.to_csv(results_path, index=False)

            print("\nComparing results with original dataset...")
            comparison_df = compare_original_vs_reduced(original_results_path, results_df)

            if comparison_df is not None:
                print("\nComparison of Original vs. Reduced Dataset:")
                print(comparison_df)
        else:
            # Just save the current results
            results_path = os.path.join("artifacts/reduced_model_evaluation", "reduced_model_results.csv")
            results_df.to_csv(results_path, index=False)
            logging.warning("Original results not found, skipping comparison")

        print("\nModel evaluation on reduced dataset completed successfully!")
        print("Results and visualizations saved to artifacts/reduced_model_evaluation/")

    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise HFException(e, sys)

if __name__ == "__main__":
    main()
