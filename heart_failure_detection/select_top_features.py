import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

def load_dataset(file_path):
    """Load dataset from file"""
    print(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def prepare_data(df, target_col='HF'):
    """Prepare data for modeling"""
    # Convert target to numeric if needed
    if df[target_col].dtype == 'object':
        # Map 'HF' to 1 and 'No HF' to 0
        df[target_col] = df[target_col].map({'HF': 1, 'No HF': 0})
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Target distribution: {y.value_counts(normalize=True)}")
    
    return X, y

def get_feature_importance(X, y, n_estimators=300, min_features=25):
    """Get feature importance using multiple methods and select top features"""
    feature_names = X.columns
    
    # Initialize models
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=SEED)
    lgbm = LGBMClassifier(n_estimators=n_estimators, random_state=SEED)
    
    # Train models
    print("Training Random Forest model...")
    rf.fit(X, y)
    print("Training LightGBM model...")
    lgbm.fit(X, y)
    
    # Get feature importance from Random Forest
    rf_importance = pd.DataFrame({
        'Feature': feature_names,
        'RF_Importance': rf.feature_importances_
    })
    
    # Get feature importance from LightGBM
    lgbm_importance = pd.DataFrame({
        'Feature': feature_names,
        'LGBM_Importance': lgbm.feature_importances_
    })
    
    # Merge feature importance
    feature_importance = pd.merge(rf_importance, lgbm_importance, on='Feature')
    
    # Calculate average importance
    feature_importance['Average_Importance'] = (feature_importance['RF_Importance'] + feature_importance['LGBM_Importance']) / 2
    
    # Sort by average importance
    feature_importance = feature_importance.sort_values('Average_Importance', ascending=False)
    
    # Select top features (at least min_features)
    top_features = feature_importance.head(max(min_features, int(len(feature_names) * 0.25)))
    
    return feature_importance, top_features

def plot_feature_importance(feature_importance, output_path=None):
    """Plot feature importance"""
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create bar plot
    sns.barplot(x='Average_Importance', y='Feature', data=feature_importance.head(30), palette='viridis')
    
    # Add labels and title
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title('Top 30 Features by Importance', fontsize=16)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {output_path}")
    
    plt.close()

def create_dataset_with_top_features(df, top_features, output_path=None):
    """Create dataset with only top features"""
    # Get list of top feature names
    top_feature_names = top_features['Feature'].tolist()
    
    # Add target column if it exists
    if 'HF' in df.columns:
        top_feature_names.append('HF')
    
    # Create dataset with top features
    df_top = df[top_feature_names]
    
    # Save dataset if output path is provided
    if output_path:
        df_top.to_csv(output_path, index=False)
        print(f"Dataset with top features saved to {output_path}")
    
    return df_top

def main():
    """Main function to select top features"""
    # Set paths
    dataset_path = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/feature_selection/features_2_methods/selected_features_dataset.csv"
    
    # Set output directory
    output_dir = os.path.join("/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts", "feature_selection", "top_25_features")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    df = load_dataset(dataset_path)
    
    # Prepare data
    X, y = prepare_data(df, target_col='HF')
    
    # Get feature importance
    feature_importance, top_features = get_feature_importance(X, y, n_estimators=300, min_features=25)
    
    # Plot feature importance
    plot_path = os.path.join(output_dir, "top_features_importance.png")
    plot_feature_importance(feature_importance, plot_path)
    
    # Save feature importance to CSV
    importance_path = os.path.join(output_dir, "feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")
    
    # Save top features to CSV
    top_features_path = os.path.join(output_dir, "top_features.csv")
    top_features.to_csv(top_features_path, index=False)
    print(f"Top features saved to {top_features_path}")
    
    # Create dataset with top features
    dataset_path = os.path.join(output_dir, "top_features_dataset.csv")
    df_top = create_dataset_with_top_features(df, top_features, dataset_path)
    
    print("\nTop 25 Features:")
    for i, feature in enumerate(top_features['Feature'].head(25)):
        print(f"{i+1}. {feature}")
    
    print(f"\nTotal selected features: {len(top_features)}")
    print(f"Dataset with top features shape: {df_top.shape}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
