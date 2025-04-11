import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
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
        # 🔹 Logistic Regression: Strong regularization for better generalization
        'Logistic Regression': LogisticRegression(
            C=0.3, solver='liblinear', penalty='l1', class_weight='balanced', random_state=SEED
        ),

        # 🔹 K-Nearest Neighbors: Reducing overfitting with distance-based weighting
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5, weights='distance', algorithm='ball_tree', leaf_size=20, p=2
        ),

        # 🔹 Naive Bayes: Adjusted for better probability estimation
        'Naive Bayes': GaussianNB(var_smoothing=1e-8),

        # 🔹 Decision Tree: More depth and splitting to capture complex patterns
        'Decision Tree': DecisionTreeClassifier(
            criterion='entropy', max_depth=20, min_samples_split=3, min_samples_leaf=2, random_state=SEED
        ),

        # 🔹 Random Forest: More trees & depth for higher accuracy
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=3, min_samples_leaf=1,
            class_weight='balanced', random_state=SEED
        ),

        # 🔹 Support Vector Machine: Higher C, balanced class weights
        'Support Vector Machine': SVC(
            C=5.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=SEED
        ),

        # 🔹 Ridge Classifier: Optimized regularization for stability
        'Ridge Classifier': RidgeClassifier(alpha=0.3, class_weight='balanced'),

        # 🔹 Linear Discriminant Analysis: Optimized shrinkage
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),

        # 🔹 AdaBoost: More estimators & lower learning rate
        'AdaBoost': AdaBoostClassifier(
            n_estimators=200, learning_rate=0.05, random_state=SEED
        ),

        # 🔹 Gradient Boosting: Lower learning rate, more estimators
        'Gradient Boosting': GradientBoostingClassifier(
            learning_rate=0.03, n_estimators=250, max_depth=10, min_samples_split=3,
            min_samples_leaf=2, subsample=0.85, random_state=SEED
        ),

        # 🔹 Extra Trees Classifier: Higher estimators for robustness
        'Extra Trees Classifier': ExtraTreesClassifier(
            n_estimators=300, max_depth=12, min_samples_split=4, random_state=SEED
        ),

        # 🔹 LightGBM: Optimized for best recall & precision
        'LightGBM': LGBMClassifier(
            colsample_bytree=0.9, learning_rate=0.03, max_depth=20, min_child_samples=5,
            n_estimators=250, num_leaves=90, subsample=0.85, class_weight='balanced', verbose=-1, random_state=SEED
        ),

        # 🔹 Multi-Layer Perceptron (MLP): More layers & epochs for better feature learning
        'Multi-Layer Perceptron (MLP)': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', alpha=0.0001,
            batch_size=16, learning_rate='adaptive', max_iter=500, random_state=SEED
        ),

        # 🔹 XGBoost: Tuned for high performance
        'XGBoost': XGBClassifier(
            colsample_bytree=0.8, learning_rate=0.03, max_depth=12, min_child_weight=4,
            n_estimators=250, subsample=0.85, gamma=0.1, reg_alpha=0.1, reg_lambda=0.1,
            scale_pos_weight=1, random_state=SEED
        ),

        # 🔹 CatBoost: Tuned for medical applications
        'CatBoost': CatBoostClassifier(
            iterations=250, learning_rate=0.03, depth=14, min_data_in_leaf=5,
            subsample=0.85, l2_leaf_reg=3, class_weights=[1, 4], random_state=SEED, verbose=0
        )
    }
    
    return models

def perform_cross_validation(X, y, models, n_folds=10):
    """Perform cross-validation with multiple metrics"""
    print(f"Performing {n_folds}-fold cross-validation")
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted',
        'roc_auc': 'roc_auc',
        'neg_log_loss': 'neg_log_loss'
    }
    
    # Initialize results dictionary
    cv_results = {}
    
    # Perform cross-validation for each model
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Skip Ridge Classifier as it doesn't support predict_proba
        if name == 'Ridge Classifier':
            continue
            
        try:
            # Perform cross-validation with multiple metrics
            cv_result = cross_validate(
                model, X, y, 
                cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED),
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Store results
            cv_results[name] = {
                'accuracy': cv_result['test_accuracy'].mean(),
                'precision': cv_result['test_precision'].mean(),
                'recall': cv_result['test_recall'].mean(),
                'f1': cv_result['test_f1'].mean(),
                'roc_auc': cv_result['test_roc_auc'].mean(),
                'log_loss': -cv_result['test_neg_log_loss'].mean(),
                'accuracy_std': cv_result['test_accuracy'].std(),
                'precision_std': cv_result['test_precision'].std(),
                'recall_std': cv_result['test_recall'].std(),
                'f1_std': cv_result['test_f1'].std(),
                'roc_auc_std': cv_result['test_roc_auc'].std(),
                'log_loss_std': cv_result['test_neg_log_loss'].std()
            }
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(cv_results, orient='index')
    
    # Sort by ROC AUC (primary) and log loss (secondary)
    sorted_results = results_df.sort_values(by=['roc_auc', 'log_loss'], ascending=[False, True])
    
    return sorted_results

def plot_roc_curves(X, y, models, n_folds=10, output_dir='results'):
    """Plot ROC curves for the top models"""
    print("Generating ROC curves...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top 5 models based on ROC AUC
    top_models = list(models.keys())[:5]
    
    # Set up the plot
    plt.figure(figsize=(12, 10))
    
    # Define colors for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Initialize stratified k-fold
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Plot ROC curve for each model
    for i, model_name in enumerate(top_models):
        model = models[model_name]
        
        # Initialize arrays to store TPR and FPR for each fold
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict probabilities
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            
            # Interpolate TPR at fixed FPR points
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            
            # Compute AUC
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        
        # Compute mean TPR and AUC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        # Plot mean ROC curve
        plt.plot(
            mean_fpr, mean_tpr, 
            color=colors[i], 
            label=f'{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
            lw=2, alpha=0.8
        )
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.8, label='Random')
    
    # Set plot properties
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves for Top 5 Models (10-fold CV)', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    roc_plot_path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves saved to {roc_plot_path}")
    return roc_plot_path

def plot_precision_recall_curves(X, y, models, n_folds=10, output_dir='results'):
    """Plot Precision-Recall curves for the top models"""
    print("Generating Precision-Recall curves...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top 5 models based on ROC AUC
    top_models = list(models.keys())[:5]
    
    # Set up the plot
    plt.figure(figsize=(12, 10))
    
    # Define colors for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Initialize stratified k-fold
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Plot Precision-Recall curve for each model
    for i, model_name in enumerate(top_models):
        model = models[model_name]
        
        # Initialize arrays to store precision and recall for each fold
        precisions = []
        recalls = []
        aps = []
        
        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict probabilities
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)
            
            # Compute Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            precisions.append(precision)
            recalls.append(recall)
            
            # Compute Average Precision
            ap = average_precision_score(y_test, y_prob)
            aps.append(ap)
        
        # Compute mean Average Precision
        mean_ap = np.mean(aps)
        std_ap = np.std(aps)
        
        # Plot Precision-Recall curve for the last fold (for simplicity)
        plt.plot(
            recalls[-1], precisions[-1], 
            color=colors[i], 
            label=f'{model_name} (AP = {mean_ap:.3f} ± {std_ap:.3f})',
            lw=2, alpha=0.8
        )
    
    # Set plot properties
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves for Top 5 Models (10-fold CV)', fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Save plot
    pr_plot_path = os.path.join(output_dir, 'precision_recall_curves.png')
    plt.savefig(pr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Precision-Recall curves saved to {pr_plot_path}")
    return pr_plot_path

def plot_log_loss_comparison(results_df, output_dir='results'):
    """Plot log loss comparison for all models"""
    print("Generating log loss comparison plot...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by log loss (ascending)
    sorted_results = results_df.sort_values(by='log_loss')
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    bars = plt.barh(
        sorted_results.index, 
        sorted_results['log_loss'],
        xerr=sorted_results['log_loss_std'],
        color='skyblue',
        alpha=0.7,
        capsize=5
    )
    
    # Add values to bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{sorted_results['log_loss'].iloc[i]:.4f} ± {sorted_results['log_loss_std'].iloc[i]:.4f}",
            va='center',
            fontsize=10
        )
    
    # Set plot properties
    plt.xlabel('Log Loss (lower is better)', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    plt.title('Log Loss Comparison (10-fold CV)', fontsize=16)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save plot
    log_loss_plot_path = os.path.join(output_dir, 'log_loss_comparison.png')
    plt.savefig(log_loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Log loss comparison plot saved to {log_loss_plot_path}")
    return log_loss_plot_path

def plot_metrics_comparison(results_df, output_dir='results'):
    """Plot metrics comparison for top 5 models"""
    print("Generating metrics comparison plot...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top 5 models
    top_models = results_df.head(5)
    
    # Metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Set up the plot
    plt.figure(figsize=(14, 10))
    
    # Set width of bars
    bar_width = 0.15
    
    # Set positions of bars on X axis
    r = np.arange(len(top_models))
    
    # Define colors for different metrics
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        plt.bar(
            r + i * bar_width, 
            top_models[metric], 
            width=bar_width, 
            label=metric.replace('_', ' ').title(),
            color=colors[i],
            yerr=top_models[f'{metric}_std'],
            capsize=3,
            alpha=0.8
        )
    
    # Add labels and title
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Performance Metrics Comparison for Top 5 Models (10-fold CV)', fontsize=16)
    plt.xticks(r + bar_width * 2, top_models.index, rotation=45, ha='right')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0.7, 1.0)  # Adjust as needed
    plt.tight_layout()
    
    # Save plot
    metrics_plot_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics comparison plot saved to {metrics_plot_path}")
    return metrics_plot_path

def generate_summary_report(results_df, output_dir='results'):
    """Generate a summary report of the model evaluation"""
    print("Generating summary report...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top 5 models
    top_models = results_df.head(5)
    
    # Create summary report
    report = "# Heart Failure Detection Model Evaluation\n\n"
    report += "## 10-Fold Cross-Validation Results\n\n"
    
    # Add table of top 5 models
    report += "### Top 5 Models\n\n"
    report += "| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Log Loss |\n"
    report += "|-------|----------|-----------|--------|----------|---------|----------|\n"
    
    for model, row in top_models.iterrows():
        report += f"| {model} | {row['accuracy']:.4f} ± {row['accuracy_std']:.4f} | "
        report += f"{row['precision']:.4f} ± {row['precision_std']:.4f} | "
        report += f"{row['recall']:.4f} ± {row['recall_std']:.4f} | "
        report += f"{row['f1']:.4f} ± {row['f1_std']:.4f} | "
        report += f"{row['roc_auc']:.4f} ± {row['roc_auc_std']:.4f} | "
        report += f"{row['log_loss']:.4f} ± {row['log_loss_std']:.4f} |\n"
    
    report += "\n### All Models (Sorted by ROC AUC)\n\n"
    
    # Add table of all models
    report += "| Model | ROC AUC | Log Loss | Accuracy | F1 Score |\n"
    report += "|-------|---------|----------|----------|----------|\n"
    
    for model, row in results_df.iterrows():
        report += f"| {model} | {row['roc_auc']:.4f} | {row['log_loss']:.4f} | "
        report += f"{row['accuracy']:.4f} | {row['f1']:.4f} |\n"
    
    # Add analysis and recommendations
    report += "\n## Analysis and Recommendations\n\n"
    
    # Best model
    best_model = results_df.index[0]
    report += f"### Best Overall Model: {best_model}\n\n"
    report += f"- ROC AUC: {results_df.loc[best_model, 'roc_auc']:.4f} ± {results_df.loc[best_model, 'roc_auc_std']:.4f}\n"
    report += f"- Log Loss: {results_df.loc[best_model, 'log_loss']:.4f} ± {results_df.loc[best_model, 'log_loss_std']:.4f}\n"
    report += f"- Accuracy: {results_df.loc[best_model, 'accuracy']:.4f} ± {results_df.loc[best_model, 'accuracy_std']:.4f}\n"
    report += f"- F1 Score: {results_df.loc[best_model, 'f1']:.4f} ± {results_df.loc[best_model, 'f1_std']:.4f}\n\n"
    
    # Model with lowest log loss
    min_log_loss_model = results_df['log_loss'].idxmin()
    report += f"### Model with Lowest Log Loss: {min_log_loss_model}\n\n"
    report += f"- Log Loss: {results_df.loc[min_log_loss_model, 'log_loss']:.4f} ± {results_df.loc[min_log_loss_model, 'log_loss_std']:.4f}\n"
    report += f"- ROC AUC: {results_df.loc[min_log_loss_model, 'roc_auc']:.4f} ± {results_df.loc[min_log_loss_model, 'roc_auc_std']:.4f}\n\n"
    
    # Recommendations
    report += "### Recommendations\n\n"
    report += f"1. **Primary Model**: {best_model} - Best overall performance with highest ROC AUC.\n"
    report += f"2. **Alternative Model**: {min_log_loss_model} - Lowest log loss, indicating good probability calibration.\n"
    report += "3. **Ensemble Approach**: Consider an ensemble of the top 3-5 models for potentially improved performance.\n"
    report += "4. **Model Deployment**: Deploy the primary model with careful monitoring of performance metrics.\n"
    report += "5. **Further Optimization**: Fine-tune hyperparameters of the top models for potential performance improvements.\n"
    
    # Save report
    report_path = os.path.join(output_dir, 'model_evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Summary report saved to {report_path}")
    return report_path

def main():
    """Main function to run the model evaluation"""
    # Set output directory
    output_dir = os.path.join("/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts", "model_evaluation_cv")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset_path = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/feature_selection/features_2_methods/selected_features_dataset.csv"
    df = load_dataset(dataset_path)
    
    # Prepare data
    X, y = prepare_data(df, target_col='HF')
    
    # Initialize models
    models = initialize_models()
    
    # Perform cross-validation
    results_df = perform_cross_validation(X, y, models, n_folds=10)
    
    # Save results
    results_path = os.path.join(output_dir, 'cv_results.csv')
    results_df.to_csv(results_path)
    print(f"Cross-validation results saved to {results_path}")
    
    # Get top 5 models
    top_5_models = {name: models[name] for name in results_df.head(5).index}
    
    # Plot ROC curves
    roc_plot_path = plot_roc_curves(X, y, top_5_models, n_folds=10, output_dir=output_dir)
    
    # Plot Precision-Recall curves
    pr_plot_path = plot_precision_recall_curves(X, y, top_5_models, n_folds=10, output_dir=output_dir)
    
    # Plot log loss comparison
    log_loss_plot_path = plot_log_loss_comparison(results_df, output_dir=output_dir)
    
    # Plot metrics comparison
    metrics_plot_path = plot_metrics_comparison(results_df, output_dir=output_dir)
    
    # Generate summary report
    report_path = generate_summary_report(results_df, output_dir=output_dir)
    
    print("\n" + "="*80)
    print("Model Evaluation Completed Successfully!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"Top 5 models:")
    for i, model in enumerate(results_df.head(5).index):
        print(f"{i+1}. {model} - ROC AUC: {results_df.loc[model, 'roc_auc']:.4f}, Log Loss: {results_df.loc[model, 'log_loss']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
