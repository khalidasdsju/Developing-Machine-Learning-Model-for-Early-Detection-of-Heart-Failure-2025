import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    """Create a test dataset for model deployment testing"""
    # Set paths
    dataset_path = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/feature_selection/features_2_methods/selected_features_dataset.csv"
    output_dir = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/feature_selection/features_2_methods"
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Split dataset into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['HF'])
    
    # Save test dataset
    test_path = os.path.join(output_dir, "selected_features_test.csv")
    test_df.to_csv(test_path, index=False)
    print(f"Test dataset saved to {test_path} with shape {test_df.shape}")
    
    # Print test dataset distribution
    print(f"Test dataset distribution:")
    print(test_df['HF'].value_counts(normalize=True))

if __name__ == "__main__":
    main()
