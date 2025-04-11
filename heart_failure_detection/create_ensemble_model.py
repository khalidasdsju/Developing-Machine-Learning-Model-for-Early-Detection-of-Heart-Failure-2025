import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
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

def initialize_models():
    """Initialize all models with optimized parameters"""
    models = {
        # LightGBM with optimized parameters from Optuna
        'LightGBM': LGBMClassifier(
            boosting_type='dart',
            num_leaves=45,
            max_depth=13,
            learning_rate=0.20877688906853645,
            n_estimators=431,
            min_child_samples=16,
            subsample=0.7248601499640451,
            colsample_bytree=0.8317048891519834,
            reg_alpha=1.8413701553749397e-08,
            reg_lambda=0.0007154009278400164,
            min_split_gain=2.1212034815716092e-07,
            class_weight='balanced',
            random_state=SEED
        ),
        
        # XGBoost with manually optimized parameters
        'XGBoost': XGBClassifier(
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
        ),
        
        # Random Forest with optimized parameters
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=SEED
        ),
        
        # Gradient Boosting with optimized parameters
        'Gradient Boosting': GradientBoostingClassifier(
            learning_rate=0.03,
            n_estimators=250,
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=2,
            subsample=0.85,
            random_state=SEED
        ),
        
        # Extra Trees Classifier with optimized parameters
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            random_state=SEED
        )
    }
    
    return models

def create_voting_ensemble(models, voting='soft'):
    """Create a voting ensemble from the given models"""
    estimators = [(name, model) for name, model in models.items()]
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting=voting,
        n_jobs=-1
    )
    
    return ensemble

def create_stacking_ensemble(models, meta_model=None):
    """Create a stacking ensemble from the given models"""
    estimators = [(name, model) for name, model in models.items()]
    
    if meta_model is None:
        meta_model = LogisticRegression(C=1.0, class_weight='balanced', random_state=SEED)
    
    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    return ensemble

def evaluate_model(model, X, y, n_folds=10):
    """Evaluate a model using cross-validation"""
    # Create cross-validation folds
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Calculate cross-validation scores
    accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    precision = cross_val_score(model, X, y, cv=cv, scoring='precision_weighted')
    recall = cross_val_score(model, X, y, cv=cv, scoring='recall_weighted')
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
    roc_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    # Get predictions for confusion matrix
    y_pred = cross_val_predict(model, X, y, cv=cv)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Calculate classification report
    cr = classification_report(y, y_pred, output_dict=True)
    
    # Return evaluation metrics
    return {
        'accuracy': (accuracy.mean(), accuracy.std()),
        'precision': (precision.mean(), precision.std()),
        'recall': (recall.mean(), recall.std()),
        'f1': (f1.mean(), f1.std()),
        'roc_auc': (roc_auc.mean(), roc_auc.std()),
        'confusion_matrix': cm,
        'classification_report': cr
    }

def plot_confusion_matrix(cm, output_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_model_comparison(results, output_path):
    """Plot model comparison"""
    # Prepare data for plotting
    models = list(results.keys())
    accuracy = [results[model]['accuracy'][0] for model in models]
    precision = [results[model]['precision'][0] for model in models]
    recall = [results[model]['recall'][0] for model in models]
    f1 = [results[model]['f1'][0] for model in models]
    roc_auc = [results[model]['roc_auc'][0] for model in models]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })
    
    # Sort by accuracy
    df = df.sort_values('Accuracy', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.15
    
    # Set positions of bars on X axis
    r = np.arange(len(df))
    
    # Plot bars
    plt.bar(r, df['Accuracy'], width=bar_width, label='Accuracy', color='#1f77b4')
    plt.bar(r + bar_width, df['Precision'], width=bar_width, label='Precision', color='#ff7f0e')
    plt.bar(r + 2*bar_width, df['Recall'], width=bar_width, label='Recall', color='#2ca02c')
    plt.bar(r + 3*bar_width, df['F1 Score'], width=bar_width, label='F1 Score', color='#d62728')
    plt.bar(r + 4*bar_width, df['ROC AUC'], width=bar_width, label='ROC AUC', color='#9467bd')
    
    # Add labels and title
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Model Comparison', fontsize=16)
    plt.xticks(r + 2*bar_width, df['Model'], rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return df

def main():
    """Main function to create and evaluate ensemble models"""
    # Set output directory
    output_dir = os.path.join("/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts", "model_optimization", "ensemble")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset_path = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/feature_selection/features_2_methods/selected_features_dataset.csv"
    df = load_dataset(dataset_path)
    
    # Prepare data
    X, y = prepare_data(df, target_col='HF')
    
    # Initialize models
    models = initialize_models()
    
    # Create ensembles
    voting_soft = create_voting_ensemble(models, voting='soft')
    voting_hard = create_voting_ensemble(models, voting='hard')
    stacking_lr = create_stacking_ensemble(models, meta_model=LogisticRegression(C=1.0, class_weight='balanced', random_state=SEED))
    stacking_rf = create_stacking_ensemble(models, meta_model=RandomForestClassifier(n_estimators=100, random_state=SEED))
    
    # Combine all models for evaluation
    all_models = {
        **models,
        'Voting (Soft)': voting_soft,
        'Voting (Hard)': voting_hard,
        'Stacking (LR)': stacking_lr,
        'Stacking (RF)': stacking_rf
    }
    
    # Evaluate all models
    results = {}
    for name, model in all_models.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_model(model, X, y, n_folds=10)
        
        # Print results
        print(f"  Accuracy: {results[name]['accuracy'][0]:.4f} ± {results[name]['accuracy'][1]:.4f}")
        print(f"  Precision: {results[name]['precision'][0]:.4f} ± {results[name]['precision'][1]:.4f}")
        print(f"  Recall: {results[name]['recall'][0]:.4f} ± {results[name]['recall'][1]:.4f}")
        print(f"  F1 Score: {results[name]['f1'][0]:.4f} ± {results[name]['f1'][1]:.4f}")
        print(f"  ROC AUC: {results[name]['roc_auc'][0]:.4f} ± {results[name]['roc_auc'][1]:.4f}")
        
        # Plot confusion matrix
        cm_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_confusion_matrix.png")
        plot_confusion_matrix(results[name]['confusion_matrix'], cm_path)
    
    # Plot model comparison
    comparison_path = os.path.join(output_dir, "model_comparison.png")
    comparison_df = plot_model_comparison(results, comparison_path)
    
    # Save comparison results
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'][0])
    best_model = all_models[best_model_name]
    
    # Train best model on full dataset
    best_model.fit(X, y)
    
    # Save best model
    best_model_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    
    # Save all models
    for name, model in all_models.items():
        model_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, model_path)
    
    # Print summary
    print("\n" + "="*80)
    print("Ensemble Model Creation and Evaluation Completed Successfully!")
    print("="*80)
    print(f"Best model: {best_model_name}")
    print(f"  Accuracy: {results[best_model_name]['accuracy'][0]:.4f} ± {results[best_model_name]['accuracy'][1]:.4f}")
    print(f"  Precision: {results[best_model_name]['precision'][0]:.4f} ± {results[best_model_name]['precision'][1]:.4f}")
    print(f"  Recall: {results[best_model_name]['recall'][0]:.4f} ± {results[best_model_name]['recall'][1]:.4f}")
    print(f"  F1 Score: {results[best_model_name]['f1'][0]:.4f} ± {results[best_model_name]['f1'][1]:.4f}")
    print(f"  ROC AUC: {results[best_model_name]['roc_auc'][0]:.4f} ± {results[best_model_name]['roc_auc'][1]:.4f}")
    print("-"*80)
    print("All models saved to:")
    for name in all_models.keys():
        model_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.pkl")
        print(f"  - {name}: {model_path}")
    print("-"*80)
    print(f"Results saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
