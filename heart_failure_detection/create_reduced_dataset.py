import os
import sys
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Features to drop
FEATURES_TO_DROP = ["BA", "HbA1C", "Na", "K", "Cl", "Hb", "MPI", "HDLc"]

def load_dataset(data_path):
    """Load the original dataset"""
    try:
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise e

def create_reduced_dataset(df, features_to_drop):
    """Create a new dataset by dropping specified features"""
    try:
        logger.info(f"Creating reduced dataset by dropping features: {features_to_drop}")

        # Check which features actually exist in the dataset
        existing_features = [f for f in features_to_drop if f in df.columns]
        non_existing_features = [f for f in features_to_drop if f not in df.columns]

        if non_existing_features:
            logger.warning(f"The following features do not exist in the dataset: {non_existing_features}")

        # Drop the existing features
        reduced_df = df.drop(columns=existing_features, errors='ignore')

        logger.info(f"Reduced dataset created with shape: {reduced_df.shape}")
        return reduced_df
    except Exception as e:
        logger.error(f"Error creating reduced dataset: {e}")
        raise e

def split_and_save_dataset(df, output_dir, test_size=0.2, random_state=42):
    """Split the dataset into train and test sets and save them"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the full dataset
        full_path = os.path.join(output_dir, "reduced_dataset.csv")
        df.to_csv(full_path, index=False)
        logger.info(f"Full reduced dataset saved to {full_path}")

        # Split the dataset
        logger.info(f"Splitting dataset with test_size={test_size}, random_state={random_state}")
        X = df.drop(columns=['HF'])  # Assuming 'HF' is the target column
        y = df['HF']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Create train and test dataframes
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        # Save train and test datasets
        train_path = os.path.join(output_dir, "reduced_train.csv")
        test_path = os.path.join(output_dir, "reduced_test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Train dataset saved to {train_path} with shape: {train_df.shape}")
        logger.info(f"Test dataset saved to {test_path} with shape: {test_df.shape}")

        return train_path, test_path
    except Exception as e:
        logger.error(f"Error splitting and saving dataset: {e}")
        raise e

def analyze_reduced_dataset(df, output_dir):
    """Perform basic analysis on the reduced dataset"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Convert all columns to numeric where possible
        numeric_df = df.copy()
        for col in numeric_df.columns:
            if col != 'HF':  # Skip the target column
                try:
                    numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
                except:
                    logger.warning(f"Could not convert column {col} to numeric")

        # Fill NaN values with column median
        numeric_df = numeric_df.fillna(numeric_df.median())

        # 1. Feature correlation heatmap (only for numeric columns)
        numeric_cols = numeric_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:  # Need at least 2 columns for correlation
            plt.figure(figsize=(14, 12))
            correlation = numeric_df[numeric_cols].corr()
            mask = np.triu(correlation)
            sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, mask=mask)
            plt.title('Feature Correlation Heatmap (Reduced Dataset)')
            plt.tight_layout()
            corr_path = os.path.join(output_dir, "reduced_correlation_heatmap.png")
            plt.savefig(corr_path)
            plt.close()
            logger.info(f"Correlation heatmap saved to {corr_path}")

        # 2. Class distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='HF', data=df)
        plt.title('Class Distribution (Reduced Dataset)')
        plt.xlabel('Heart Failure')
        plt.ylabel('Count')
        class_dist_path = os.path.join(output_dir, "reduced_class_distribution.png")
        plt.savefig(class_dist_path)
        plt.close()
        logger.info(f"Class distribution plot saved to {class_dist_path}")

        # 3. Feature distributions by class (only for numeric features)
        numeric_features = [col for col in numeric_cols if col != 'HF']
        for feature in numeric_features[:min(10, len(numeric_features))]:  # Limit to 10 features
            plt.figure(figsize=(10, 6))
            sns.histplot(data=numeric_df, x=feature, hue='HF', kde=True, element="step")
            plt.title(f'{feature} Distribution by Class (Reduced Dataset)')
            feature_dist_path = os.path.join(output_dir, f"reduced_{feature}_distribution.png")
            plt.savefig(feature_dist_path)
            plt.close()
        logger.info(f"Feature distribution plots saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error analyzing reduced dataset: {e}")
        # Don't raise the exception, just log it and continue
        return False

    return True

def main():
    """Main function to create and analyze the reduced dataset"""
    try:
        # Use a specific dataset file that we know exists
        train_file_path = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/2025-04-10_21-33-48/data_ingestion/feature_store/heart_failure_data.csv"

        if not os.path.exists(train_file_path):
            raise FileNotFoundError(f"Dataset file not found at {train_file_path}")

        # Load the dataset
        df = load_dataset(train_file_path)

        # Create reduced dataset
        reduced_df = create_reduced_dataset(df, FEATURES_TO_DROP)

        # Create output directory
        output_dir = os.path.join("artifacts", "reduced_dataset")
        os.makedirs(output_dir, exist_ok=True)

        # Split and save the dataset
        try:
            train_path, test_path = split_and_save_dataset(reduced_df, output_dir)
            logger.info(f"Dataset split and saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error splitting and saving dataset: {e}")
            # Continue with the rest of the process

        # Analyze the reduced dataset
        try:
            analyze_success = analyze_reduced_dataset(reduced_df, os.path.join(output_dir, "analysis"))
            if analyze_success:
                logger.info("Dataset analysis completed successfully")
            else:
                logger.warning("Dataset analysis completed with warnings")
        except Exception as e:
            logger.error(f"Error analyzing dataset: {e}")
            # Continue with the rest of the process

        logger.info("Reduced dataset created successfully")

        print("\n" + "="*80)
        print("Reduced dataset created successfully!")
        print(f"Original dataset shape: {df.shape}")
        print(f"Reduced dataset shape: {reduced_df.shape}")
        print(f"Features removed: {[f for f in FEATURES_TO_DROP if f in df.columns]}")
        print(f"Reduced dataset saved to: {output_dir}")
        print("="*80)

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")
        return False

    return True

if __name__ == "__main__":
    main()
