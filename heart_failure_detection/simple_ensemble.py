import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
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

def create_ensemble():
    """Create a voting ensemble of top models"""
    # Initialize models with optimized parameters
    lightgbm = LGBMClassifier(
        boosting_type='dart',
        num_leaves=45,
        max_depth=13,
        learning_rate=0.21,
        n_estimators=431,
        min_child_samples=16,
        subsample=0.72,
        colsample_bytree=0.83,
        reg_alpha=1.84e-08,
        reg_lambda=0.0007,
        min_split_gain=2.12e-07,
        class_weight='balanced',
        random_state=SEED
    )
    
    xgboost = XGBClassifier(
        booster='gbtree',
        max_depth=8,
        learning_rate=0.05,
        n_estimators=300,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.01,
        reg_lambda=0.01,
        scale_pos_weight=1.2,
        random_state=SEED
    )
    
    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=SEED
    )
    
    gradient_boosting = GradientBoostingClassifier(
        learning_rate=0.03,
        n_estimators=250,
        max_depth=10,
        min_samples_split=3,
        min_samples_leaf=2,
        subsample=0.85,
        random_state=SEED
    )
    
    extra_trees = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        random_state=SEED
    )
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('lightgbm', lightgbm),
            ('xgboost', xgboost),
            ('random_forest', random_forest),
            ('gradient_boosting', gradient_boosting),
            ('extra_trees', extra_trees)
        ],
        voting='soft'
    )
    
    return ensemble

def main():
    """Main function to create and evaluate ensemble model"""
    # Set output directory
    output_dir = os.path.join("/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts", "model_optimization", "ensemble")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset_path = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/feature_selection/features_2_methods/selected_features_dataset.csv"
    df = load_dataset(dataset_path)
    
    # Prepare data
    X, y = prepare_data(df, target_col='HF')
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    
    # Create ensemble
    ensemble = create_ensemble()
    
    # Train ensemble
    print("Training ensemble model...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate ensemble
    y_pred = ensemble.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate ROC AUC if possible
    try:
        y_prob = ensemble.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    except:
        roc_auc = None
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate classification report
    cr = classification_report(y_test, y_pred)
    
    # Print results
    print("\n" + "="*80)
    print("Ensemble Model Evaluation")
    print("="*80)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    print("-"*80)
    print("Confusion Matrix:")
    print(cm)
    print("-"*80)
    print("Classification Report:")
    print(cr)
    print("="*80)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "ensemble_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # Save ensemble model
    model_path = os.path.join(output_dir, "ensemble_model.pkl")
    joblib.dump(ensemble, model_path)
    print(f"Ensemble model saved to {model_path}")
    
    # Save results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': cr
    }
    
    results_path = os.path.join(output_dir, "ensemble_results.txt")
    with open(results_path, 'w') as f:
        f.write("Ensemble Model Evaluation\n")
        f.write("="*80 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        if roc_auc is not None:
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write("-"*80 + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
        f.write("-"*80 + "\n")
        f.write("Classification Report:\n")
        f.write(cr + "\n")
        f.write("="*80 + "\n")
    
    print(f"Results saved to {results_path}")
    
    # Perform cross-validation
    print("\nPerforming 10-fold cross-validation...")
    cv_scores = cross_val_score(ensemble, X, y, cv=10, scoring='accuracy')
    
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Save cross-validation results
    cv_results_path = os.path.join(output_dir, "ensemble_cv_results.txt")
    with open(cv_results_path, 'w') as f:
        f.write("Ensemble Model Cross-Validation\n")
        f.write("="*80 + "\n")
        f.write(f"10-fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
        f.write("Individual fold scores:\n")
        for i, score in enumerate(cv_scores):
            f.write(f"Fold {i+1}: {score:.4f}\n")
        f.write("="*80 + "\n")
    
    print(f"Cross-validation results saved to {cv_results_path}")

if __name__ == "__main__":
    main()
