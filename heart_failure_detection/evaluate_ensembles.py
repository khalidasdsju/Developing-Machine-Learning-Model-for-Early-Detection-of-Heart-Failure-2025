import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
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

def initialize_base_models():
    """Initialize base models with optimized parameters"""
    models = {
        # LightGBM with optimized parameters
        'LightGBM': LGBMClassifier(
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
        ),
        
        # XGBoost with optimized parameters
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
        
        # Extra Trees with optimized parameters
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            random_state=SEED
        )
    }
    
    return models

def create_ensemble_models(base_models):
    """Create ensemble models"""
    # Convert base_models dictionary to list of tuples
    estimators = [(name, model) for name, model in base_models.items()]
    
    # Create ensemble models
    ensemble_models = {
        'Voting (Soft)': VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        ),
        
        'Voting (Hard)': VotingClassifier(
            estimators=estimators,
            voting='hard',
            n_jobs=-1
        ),
        
        'Stacking (LR)': StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=1.0, class_weight='balanced', random_state=SEED),
            cv=5,
            n_jobs=-1
        ),
        
        'Stacking (RF)': StackingClassifier(
            estimators=estimators,
            final_estimator=RandomForestClassifier(n_estimators=100, random_state=SEED),
            cv=5,
            n_jobs=-1
        )
    }
    
    return ensemble_models

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate a model on train and test data"""
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate ROC AUC if possible
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = None
    except:
        roc_auc = None
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate classification report
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    # Return evaluation metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': cr,
        'y_pred': y_pred,
        'y_prob': y_prob if 'y_prob' in locals() else None
    }

def cross_validate_model(model, X, y, cv=10):
    """Perform cross-validation on a model"""
    # Create cross-validation folds
    cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    
    # Calculate cross-validation scores
    cv_accuracy = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
    cv_precision = cross_val_score(model, X, y, cv=cv_folds, scoring='precision_weighted')
    cv_recall = cross_val_score(model, X, y, cv=cv_folds, scoring='recall_weighted')
    cv_f1 = cross_val_score(model, X, y, cv=cv_folds, scoring='f1_weighted')
    
    # Calculate ROC AUC if possible
    try:
        if hasattr(model, "predict_proba"):
            cv_roc_auc = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc')
        else:
            cv_roc_auc = None
    except:
        cv_roc_auc = None
    
    # Return cross-validation metrics
    return {
        'accuracy': (cv_accuracy.mean(), cv_accuracy.std()),
        'precision': (cv_precision.mean(), cv_precision.std()),
        'recall': (cv_recall.mean(), cv_recall.std()),
        'f1': (cv_f1.mean(), cv_f1.std()),
        'roc_auc': (cv_roc_auc.mean(), cv_roc_auc.std()) if cv_roc_auc is not None else None
    }

def plot_confusion_matrix(cm, model_name, output_dir):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def plot_roc_curve(y_test, y_probs, model_names, output_dir):
    """Plot ROC curves for multiple models"""
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each model
    for i, (model_name, y_prob) in enumerate(zip(model_names, y_probs)):
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves for Ensemble Models', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "ensemble_roc_curves.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def plot_model_comparison(results, output_dir):
    """Plot model comparison"""
    # Prepare data for plotting
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Create DataFrame for plotting
    data = []
    for model_name in model_names:
        row = {'Model': model_name}
        for metric in metrics:
            if metric == 'roc_auc' and results[model_name]['test'][metric] is None:
                continue
            row[metric.capitalize()] = results[model_name]['test'][metric]
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Sort by accuracy
    df = df.sort_values('Accuracy', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.15
    
    # Set positions of bars on X axis
    r = np.arange(len(df))
    
    # Plot bars for each metric
    for i, metric in enumerate([m.capitalize() for m in metrics if m != 'roc_auc' or all(m.capitalize() in df.columns for m in metrics)]):
        if metric in df.columns:
            plt.bar(r + i*bar_width, df[metric], width=bar_width, label=metric)
    
    # Add labels and title
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Model Comparison', fontsize=16)
    plt.xticks(r + bar_width*2, df['Model'], rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return df, output_path

def save_results(results, output_dir):
    """Save evaluation results to files"""
    # Create summary DataFrame
    summary_data = []
    for model_name, result in results.items():
        # Test results
        test_row = {
            'Model': model_name,
            'Type': 'Test',
            'Accuracy': result['test']['accuracy'],
            'Precision': result['test']['precision'],
            'Recall': result['test']['recall'],
            'F1 Score': result['test']['f1']
        }
        if result['test']['roc_auc'] is not None:
            test_row['ROC AUC'] = result['test']['roc_auc']
        
        # CV results
        cv_row = {
            'Model': model_name,
            'Type': 'CV',
            'Accuracy': f"{result['cv']['accuracy'][0]:.4f} ± {result['cv']['accuracy'][1]:.4f}",
            'Precision': f"{result['cv']['precision'][0]:.4f} ± {result['cv']['precision'][1]:.4f}",
            'Recall': f"{result['cv']['recall'][0]:.4f} ± {result['cv']['recall'][1]:.4f}",
            'F1 Score': f"{result['cv']['f1'][0]:.4f} ± {result['cv']['f1'][1]:.4f}"
        }
        if result['cv']['roc_auc'] is not None:
            cv_row['ROC AUC'] = f"{result['cv']['roc_auc'][0]:.4f} ± {result['cv']['roc_auc'][1]:.4f}"
        
        summary_data.extend([test_row, cv_row])
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to CSV
    summary_path = os.path.join(output_dir, "ensemble_results_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Save detailed results for each model
    for model_name, result in results.items():
        # Create model directory
        model_dir = os.path.join(output_dir, model_name.replace(' ', '_').lower())
        os.makedirs(model_dir, exist_ok=True)
        
        # Save classification report
        cr_df = pd.DataFrame(result['test']['classification_report']).transpose()
        cr_path = os.path.join(model_dir, "classification_report.csv")
        cr_df.to_csv(cr_path)
        
        # Save confusion matrix
        cm_df = pd.DataFrame(result['test']['confusion_matrix'])
        cm_path = os.path.join(model_dir, "confusion_matrix.csv")
        cm_df.to_csv(cm_path, index=False)
        
        # Save detailed results as text
        with open(os.path.join(model_dir, "results.txt"), 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write("="*50 + "\n\n")
            
            f.write("Test Results:\n")
            f.write("-"*50 + "\n")
            f.write(f"Accuracy: {result['test']['accuracy']:.4f}\n")
            f.write(f"Precision: {result['test']['precision']:.4f}\n")
            f.write(f"Recall: {result['test']['recall']:.4f}\n")
            f.write(f"F1 Score: {result['test']['f1']:.4f}\n")
            if result['test']['roc_auc'] is not None:
                f.write(f"ROC AUC: {result['test']['roc_auc']:.4f}\n")
            f.write("\n")
            
            f.write("Cross-Validation Results:\n")
            f.write("-"*50 + "\n")
            f.write(f"Accuracy: {result['cv']['accuracy'][0]:.4f} ± {result['cv']['accuracy'][1]:.4f}\n")
            f.write(f"Precision: {result['cv']['precision'][0]:.4f} ± {result['cv']['precision'][1]:.4f}\n")
            f.write(f"Recall: {result['cv']['recall'][0]:.4f} ± {result['cv']['recall'][1]:.4f}\n")
            f.write(f"F1 Score: {result['cv']['f1'][0]:.4f} ± {result['cv']['f1'][1]:.4f}\n")
            if result['cv']['roc_auc'] is not None:
                f.write(f"ROC AUC: {result['cv']['roc_auc'][0]:.4f} ± {result['cv']['roc_auc'][1]:.4f}\n")
            f.write("\n")
            
            f.write("Classification Report:\n")
            f.write("-"*50 + "\n")
            f.write(str(cr_df) + "\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write("-"*50 + "\n")
            f.write(str(cm_df) + "\n")
    
    # Create comprehensive results text file
    with open(os.path.join(output_dir, "ensemble_evaluation_results.txt"), 'w') as f:
        f.write("HEART FAILURE DETECTION - ENSEMBLE MODEL EVALUATION\n")
        f.write("="*80 + "\n\n")
        
        f.write("SUMMARY OF RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['test']['accuracy'])
        
        f.write(f"Best Model: {best_model}\n")
        f.write(f"Accuracy: {results[best_model]['test']['accuracy']:.4f}\n")
        f.write(f"Precision: {results[best_model]['test']['precision']:.4f}\n")
        f.write(f"Recall: {results[best_model]['test']['recall']:.4f}\n")
        f.write(f"F1 Score: {results[best_model]['test']['f1']:.4f}\n")
        if results[best_model]['test']['roc_auc'] is not None:
            f.write(f"ROC AUC: {results[best_model]['test']['roc_auc']:.4f}\n")
        f.write("\n")
        
        f.write("MODEL COMPARISON\n")
        f.write("-"*80 + "\n\n")
        
        # Sort models by accuracy
        sorted_models = sorted(results.keys(), key=lambda x: results[x]['test']['accuracy'], reverse=True)
        
        for model_name in sorted_models:
            f.write(f"{model_name}:\n")
            f.write(f"  Accuracy: {results[model_name]['test']['accuracy']:.4f}\n")
            f.write(f"  Precision: {results[model_name]['test']['precision']:.4f}\n")
            f.write(f"  Recall: {results[model_name]['test']['recall']:.4f}\n")
            f.write(f"  F1 Score: {results[model_name]['test']['f1']:.4f}\n")
            if results[model_name]['test']['roc_auc'] is not None:
                f.write(f"  ROC AUC: {results[model_name]['test']['roc_auc']:.4f}\n")
            f.write("\n")
        
        f.write("CROSS-VALIDATION RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        for model_name in sorted_models:
            f.write(f"{model_name}:\n")
            f.write(f"  Accuracy: {results[model_name]['cv']['accuracy'][0]:.4f} ± {results[model_name]['cv']['accuracy'][1]:.4f}\n")
            f.write(f"  Precision: {results[model_name]['cv']['precision'][0]:.4f} ± {results[model_name]['cv']['precision'][1]:.4f}\n")
            f.write(f"  Recall: {results[model_name]['cv']['recall'][0]:.4f} ± {results[model_name]['cv']['recall'][1]:.4f}\n")
            f.write(f"  F1 Score: {results[model_name]['cv']['f1'][0]:.4f} ± {results[model_name]['cv']['f1'][1]:.4f}\n")
            if results[model_name]['cv']['roc_auc'] is not None:
                f.write(f"  ROC AUC: {results[model_name]['cv']['roc_auc'][0]:.4f} ± {results[model_name]['cv']['roc_auc'][1]:.4f}\n")
            f.write("\n")
        
        f.write("CONCLUSION\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"The {best_model} model achieved the highest accuracy of {results[best_model]['test']['accuracy']:.4f} ")
        f.write(f"on the test set and {results[best_model]['cv']['accuracy'][0]:.4f} ± {results[best_model]['cv']['accuracy'][1]:.4f} ")
        f.write("in 10-fold cross-validation.\n\n")
        
        f.write("This model is recommended for deployment in the heart failure detection system.\n")
    
    return summary_df

def main():
    """Main function to evaluate ensemble models"""
    # Set output directory
    output_dir = os.path.join("/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts", "model_optimization", "ensemble_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset_path = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/feature_selection/features_2_methods/selected_features_dataset.csv"
    df = load_dataset(dataset_path)
    
    # Prepare data
    X, y = prepare_data(df, target_col='HF')
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    
    # Initialize base models
    base_models = initialize_base_models()
    
    # Create ensemble models
    ensemble_models = create_ensemble_models(base_models)
    
    # Combine all models
    all_models = {**base_models, **ensemble_models}
    
    # Evaluate all models
    results = {}
    y_probs = []
    model_names = []
    
    for model_name, model in all_models.items():
        print(f"Evaluating {model_name}...")
        
        # Evaluate on test set
        test_result = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Perform cross-validation
        cv_result = cross_validate_model(model, X, y, cv=10)
        
        # Store results
        results[model_name] = {
            'test': test_result,
            'cv': cv_result
        }
        
        # Store probabilities for ROC curve
        if test_result['y_prob'] is not None:
            y_probs.append(test_result['y_prob'])
            model_names.append(model_name)
        
        # Plot confusion matrix
        plot_confusion_matrix(test_result['confusion_matrix'], model_name, output_dir)
        
        # Print results
        print(f"  Test Accuracy: {test_result['accuracy']:.4f}")
        print(f"  CV Accuracy: {cv_result['accuracy'][0]:.4f} ± {cv_result['accuracy'][1]:.4f}")
    
    # Plot ROC curves
    plot_roc_curve(y_test, y_probs, model_names, output_dir)
    
    # Plot model comparison
    comparison_df, _ = plot_model_comparison(results, output_dir)
    
    # Save results
    summary_df = save_results(results, output_dir)
    
    # Find best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = all_models[best_model_name]
    
    # Train best model on full dataset
    best_model.fit(X, y)
    
    # Save best model
    best_model_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"Best model ({best_model_name}) saved to {best_model_path}")
    
    # Save all models
    for name, model in all_models.items():
        model_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, model_path)
    
    # Print summary
    print("\n" + "="*80)
    print("Ensemble Model Evaluation Completed Successfully!")
    print("="*80)
    print(f"Best model: {best_model_name}")
    print(f"  Accuracy: {results[best_model_name]['test']['accuracy']:.4f}")
    print(f"  Precision: {results[best_model_name]['test']['precision']:.4f}")
    print(f"  Recall: {results[best_model_name]['test']['recall']:.4f}")
    print(f"  F1 Score: {results[best_model_name]['test']['f1']:.4f}")
    if results[best_model_name]['test']['roc_auc'] is not None:
        print(f"  ROC AUC: {results[best_model_name]['test']['roc_auc']:.4f}")
    print("-"*80)
    print(f"Results saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
