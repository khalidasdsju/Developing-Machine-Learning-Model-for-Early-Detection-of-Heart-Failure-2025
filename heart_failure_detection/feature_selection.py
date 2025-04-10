import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from HF.logger import logging
from HF.exception import HFException
from HF.utils import load_numpy_array_data

# Domain expert features to drop
DOMAIN_EXPERT_FEATURES_TO_DROP = ["BA", "HbA1C", "Na", "K", "Cl", "Hb", "MPI", "HDLc"]

def load_dataset(data_path):
    """Load the original dataset"""
    try:
        logging.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        logging.info(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise e

def preprocess_data(df):
    """Preprocess the data for feature selection"""
    try:
        # Handle missing values
        df_processed = df.copy()

        # Print column info for debugging
        logging.info(f"Original columns: {df_processed.columns.tolist()}")
        logging.info(f"Original data types: {df_processed.dtypes}")

        # Remove StudyID column if it exists
        if 'StudyID' in df_processed.columns:
            df_processed = df_processed.drop(columns=['StudyID'])
            logging.info("Removed StudyID column")

        # Fill missing values with median for numeric columns
        for col in df_processed.columns:
            if df_processed[col].dtype != 'object' and df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                logging.info(f"Filled missing values in {col} with median")

        # Handle categorical features
        categorical_columns = []
        for col in df_processed.columns:
            if col != 'HF' and (df_processed[col].dtype == 'object' or df_processed[col].nunique() < 10):
                categorical_columns.append(col)

        logging.info(f"Categorical columns: {categorical_columns}")

        # One-hot encode categorical features
        if categorical_columns:
            df_processed = pd.get_dummies(df_processed, columns=categorical_columns, drop_first=True)
            logging.info(f"One-hot encoded categorical columns: {categorical_columns}")

        # Convert target to numeric if needed
        if df_processed['HF'].dtype == 'object':
            df_processed['HF'] = df_processed['HF'].map({'HF': 1, 'No HF': 0})
            logging.info("Converted target to numeric")

        logging.info(f"Data preprocessed successfully with shape: {df_processed.shape}")
        logging.info(f"Final columns: {df_processed.columns.tolist()}")
        return df_processed
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise e

def apply_lasso_feature_selection(X, y, alpha=0.01):
    """Apply Lasso regularization for feature selection"""
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply Lasso
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_scaled, y)

        # Get feature importance
        feature_importance = np.abs(lasso.coef_)

        # Create DataFrame with feature names and importance
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        })

        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

        # Select features with non-zero coefficients
        selected_features = feature_importance_df[feature_importance_df['Importance'] > 0]['Feature'].tolist()

        logging.info(f"Lasso feature selection completed. Selected {len(selected_features)} features.")
        return selected_features, feature_importance_df
    except Exception as e:
        logging.error(f"Error in Lasso feature selection: {e}")
        raise e

def apply_tree_based_feature_selection(X, y, threshold=0.01):
    """Apply tree-based feature selection using Random Forest"""
    try:
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Get feature importance
        feature_importance = rf.feature_importances_

        # Create DataFrame with feature names and importance
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        })

        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

        # Select features with importance above threshold
        selected_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature'].tolist()

        logging.info(f"Tree-based feature selection completed. Selected {len(selected_features)} features.")
        return selected_features, feature_importance_df
    except Exception as e:
        logging.error(f"Error in tree-based feature selection: {e}")
        raise e

def apply_logistic_regression_feature_selection(X, y, C=0.1):
    """Apply Logistic Regression with L1 regularization for feature selection"""
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply Logistic Regression with L1 regularization
        lr = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=42)
        lr.fit(X_scaled, y)

        # Get feature importance
        feature_importance = np.abs(lr.coef_[0])

        # Create DataFrame with feature names and importance
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        })

        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

        # Select features with non-zero coefficients
        selected_features = feature_importance_df[feature_importance_df['Importance'] > 0]['Feature'].tolist()

        logging.info(f"Logistic Regression feature selection completed. Selected {len(selected_features)} features.")
        return selected_features, feature_importance_df
    except Exception as e:
        logging.error(f"Error in Logistic Regression feature selection: {e}")
        raise e

def get_domain_expert_features(df, features_to_drop):
    """Get features selected by domain experts (all features except those to drop)"""
    try:
        # Get all feature names except the target
        all_features = [col for col in df.columns if col != 'HF']

        # Filter out features to drop that exist in the dataset
        existing_features_to_drop = [col for col in features_to_drop if col in all_features]
        logging.info(f"Found {len(existing_features_to_drop)} out of {len(features_to_drop)} features to drop in the dataset")

        # Filter out features to drop
        selected_features = [col for col in all_features if col not in existing_features_to_drop]

        logging.info(f"Domain expert feature selection completed. Selected {len(selected_features)} features.")
        return selected_features
    except Exception as e:
        logging.error(f"Error in domain expert feature selection: {e}")
        raise e

def compare_feature_selection_methods(lasso_features, tree_features, lr_features, domain_expert_features):
    """Compare different feature selection methods"""
    try:
        # Create sets for comparison
        lasso_set = set(lasso_features)
        tree_set = set(tree_features)
        lr_set = set(lr_features)
        domain_set = set(domain_expert_features)

        # Print feature sets for debugging
        logging.info(f"Lasso features: {len(lasso_set)} features")
        logging.info(f"Tree features: {len(tree_set)} features")
        logging.info(f"LR features: {len(lr_set)} features")
        logging.info(f"Domain features: {len(domain_set)} features")

        # Find common features across all methods
        common_features = lasso_set.intersection(tree_set).intersection(lr_set).intersection(domain_set)

        # Find features selected by at least 2 methods
        features_2_methods = set()
        for feature in set.union(lasso_set, tree_set, lr_set, domain_set):
            count = sum([
                feature in lasso_set,
                feature in tree_set,
                feature in lr_set,
                feature in domain_set
            ])
            if count >= 2:
                features_2_methods.add(feature)

        # Find features selected by at least 3 methods
        features_3_methods = set()
        for feature in set.union(lasso_set, tree_set, lr_set, domain_set):
            count = sum([
                feature in lasso_set,
                feature in tree_set,
                feature in lr_set,
                feature in domain_set
            ])
            if count >= 3:
                features_3_methods.add(feature)

        # Create comparison DataFrame
        all_features = list(set.union(lasso_set, tree_set, lr_set, domain_set))
        comparison_df = pd.DataFrame({
            'Feature': all_features,
            'Lasso': [feature in lasso_set for feature in all_features],
            'Random Forest': [feature in tree_set for feature in all_features],
            'Logistic Regression': [feature in lr_set for feature in all_features],
            'Domain Expert': [feature in domain_set for feature in all_features]
        })

        # Add count of methods
        comparison_df['Methods Count'] = comparison_df.iloc[:, 1:5].sum(axis=1)

        # Sort by count of methods
        comparison_df = comparison_df.sort_values('Methods Count', ascending=False)

        logging.info(f"Feature selection comparison completed. Found {len(common_features)} common features across all methods.")
        return comparison_df, list(common_features), list(features_2_methods), list(features_3_methods)
    except Exception as e:
        logging.error(f"Error in feature selection comparison: {e}")
        raise e

def create_dataset_with_selected_features(df, selected_features, output_dir, test_size=0.2, random_state=42):
    """Create a new dataset with selected features"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Ensure all selected features exist in the dataset
        existing_features = [f for f in selected_features if f in df.columns]
        if len(existing_features) < len(selected_features):
            missing_features = [f for f in selected_features if f not in df.columns]
            logging.warning(f"Some selected features are not in the dataset: {missing_features}")
            logging.info(f"Using {len(existing_features)} out of {len(selected_features)} selected features")

        # Select features and target
        df_selected = df[existing_features + ['HF']]

        # Save the full dataset
        full_path = os.path.join(output_dir, "selected_features_dataset.csv")
        df_selected.to_csv(full_path, index=False)
        logging.info(f"Selected features dataset saved to {full_path} with shape: {df_selected.shape}")

        # Split the dataset
        X = df_selected.drop(columns=['HF'])
        y = df_selected['HF']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Create train and test dataframes
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        # Save train and test datasets
        train_path = os.path.join(output_dir, "selected_features_train.csv")
        test_path = os.path.join(output_dir, "selected_features_test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info(f"Train dataset saved to {train_path} with shape: {train_df.shape}")
        logging.info(f"Test dataset saved to {test_path} with shape: {test_df.shape}")

        return train_path, test_path
    except Exception as e:
        logging.error(f"Error creating dataset with selected features: {e}")
        raise e

def plot_feature_importance(feature_importance_df, method_name, output_dir):
    """Plot feature importance"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Plot top 20 features
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
        plt.title(f'Top 20 Features by {method_name}', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, f"{method_name.lower().replace(' ', '_')}_feature_importance.png")
        plt.savefig(plot_path)
        plt.close()

        logging.info(f"{method_name} feature importance plot saved to {plot_path}")
        return plot_path
    except Exception as e:
        logging.error(f"Error plotting feature importance: {e}")
        raise e

def plot_feature_selection_comparison(comparison_df, output_dir):
    """Plot feature selection comparison"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Plot heatmap of feature selection methods
        plt.figure(figsize=(14, 10))

        # Get top 30 features by methods count
        top_features = comparison_df.head(30)

        # Create heatmap data
        heatmap_data = top_features.set_index('Feature').iloc[:, :4]

        # Plot heatmap
        sns.heatmap(heatmap_data, cmap='viridis', cbar=False, linewidths=0.5)
        plt.title('Feature Selection Methods Comparison (Top 30 Features)', fontsize=16)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, "feature_selection_comparison.png")
        plt.savefig(plot_path)
        plt.close()

        # Plot Venn diagram-like visualization
        plt.figure(figsize=(12, 10))

        # Count features by method
        method_counts = comparison_df.iloc[:, 1:5].sum()

        # Plot bar chart
        sns.barplot(x=method_counts.index, y=method_counts.values)
        plt.title('Number of Features Selected by Each Method', fontsize=16)
        plt.ylabel('Number of Features', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save plot
        count_plot_path = os.path.join(output_dir, "feature_selection_counts.png")
        plt.savefig(count_plot_path)
        plt.close()

        logging.info(f"Feature selection comparison plots saved to {output_dir}")
        return plot_path, count_plot_path
    except Exception as e:
        logging.error(f"Error plotting feature selection comparison: {e}")
        raise e

def main():
    """Main function to perform feature selection"""
    try:
        # Use a specific dataset file that we know exists
        data_path = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/2025-04-10_21-33-48/data_ingestion/feature_store/heart_failure_data.csv"

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found at {data_path}")

        # Load the dataset
        df = load_dataset(data_path)

        # Preprocess the data
        df_processed = preprocess_data(df)

        # Split into features and target
        X = df_processed.drop(columns=['HF'])
        y = df_processed['HF']

        # Create output directory
        output_dir = os.path.join("artifacts", "feature_selection")
        os.makedirs(output_dir, exist_ok=True)

        # Apply Lasso feature selection
        lasso_features, lasso_importance = apply_lasso_feature_selection(X, y)

        # Apply tree-based feature selection
        tree_features, tree_importance = apply_tree_based_feature_selection(X, y)

        # Apply Logistic Regression feature selection
        lr_features, lr_importance = apply_logistic_regression_feature_selection(X, y)

        # Get domain expert features
        # First, check which of the domain expert features exist in the original dataset
        existing_domain_features = [f for f in DOMAIN_EXPERT_FEATURES_TO_DROP if f in df.columns]
        logging.info(f"Found {len(existing_domain_features)} domain expert features in the dataset: {existing_domain_features}")

        # Get domain expert features (all features except those to drop)
        domain_expert_features = get_domain_expert_features(df, existing_domain_features)

        # Ensure domain expert features exist in the processed dataset
        domain_expert_features = [f for f in domain_expert_features if f in X.columns]
        logging.info(f"Final domain expert features: {len(domain_expert_features)} features")

        # Compare feature selection methods
        comparison_df, common_features, features_2_methods, features_3_methods = compare_feature_selection_methods(
            lasso_features, tree_features, lr_features, domain_expert_features
        )

        # Save comparison results
        comparison_path = os.path.join(output_dir, "feature_selection_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)

        # Plot feature importance
        plot_dir = os.path.join(output_dir, "plots")
        lasso_plot = plot_feature_importance(lasso_importance, "Lasso", plot_dir)
        tree_plot = plot_feature_importance(tree_importance, "Random Forest", plot_dir)
        lr_plot = plot_feature_importance(lr_importance, "Logistic Regression", plot_dir)

        # Plot feature selection comparison
        comparison_plot, count_plot = plot_feature_selection_comparison(comparison_df, plot_dir)

        # Create datasets with selected features
        # 1. Features selected by all methods
        if len(common_features) > 0:
            common_dir = os.path.join(output_dir, "common_features")
            _ = create_dataset_with_selected_features(
                df_processed, common_features, common_dir
            )

        # 2. Features selected by at least 3 methods
        if len(features_3_methods) > 0:
            methods_3_dir = os.path.join(output_dir, "features_3_methods")
            _ = create_dataset_with_selected_features(
                df_processed, features_3_methods, methods_3_dir
            )

        # 3. Features selected by at least 2 methods
        if len(features_2_methods) > 0:
            methods_2_dir = os.path.join(output_dir, "features_2_methods")
            _ = create_dataset_with_selected_features(
                df_processed, features_2_methods, methods_2_dir
            )

        # 4. Domain expert features
        domain_dir = os.path.join(output_dir, "domain_expert_features")
        _ = create_dataset_with_selected_features(
            df_processed, domain_expert_features, domain_dir
        )

        # Print summary
        print("\n" + "="*80)
        print("Feature Selection Completed Successfully!")
        print("-"*80)
        print(f"Original dataset shape: {df.shape}")
        print(f"Preprocessed dataset shape: {df_processed.shape}")
        print("-"*80)
        print("Features selected by each method:")
        print(f"  - Lasso: {len(lasso_features)} features")
        print(f"  - Random Forest: {len(tree_features)} features")
        print(f"  - Logistic Regression: {len(lr_features)} features")
        print(f"  - Domain Expert: {len(domain_expert_features)} features")
        print("-"*80)
        print("Common features across methods:")
        print(f"  - All methods: {len(common_features)} features")
        print(f"  - At least 3 methods: {len(features_3_methods)} features")
        print(f"  - At least 2 methods: {len(features_2_methods)} features")
        print("-"*80)
        print("Datasets created:")

        if len(common_features) > 0:
            print(f"  - Common features: {common_dir}")

        if len(features_3_methods) > 0:
            print(f"  - Features selected by at least 3 methods: {methods_3_dir}")

        if len(features_2_methods) > 0:
            print(f"  - Features selected by at least 2 methods: {methods_2_dir}")

        print(f"  - Domain expert features: {domain_dir}")
        print("-"*80)
        print("Plots created:")
        print(f"  - Lasso feature importance: {lasso_plot}")
        print(f"  - Random Forest feature importance: {tree_plot}")
        print(f"  - Logistic Regression feature importance: {lr_plot}")
        print(f"  - Feature selection comparison: {comparison_plot}")
        print(f"  - Feature selection counts: {count_plot}")
        print("="*80)

        return True
    except Exception as e:
        logging.error(f"Error in feature selection: {e}")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    main()
