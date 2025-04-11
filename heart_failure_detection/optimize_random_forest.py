import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
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

def objective_rf(trial, X, y, n_folds=5):
    """Optuna objective function for Random Forest optimization"""
    # Define the hyperparameter search space
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': 'balanced',
        'random_state': SEED
    }
    
    # Create cross-validation folds
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Initialize scores list
    cv_scores = []
    
    # Perform cross-validation
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create and train model
        model = RandomForestClassifier(**param)
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        cv_scores.append(score)
    
    # Return mean accuracy across folds
    return np.mean(cv_scores)

def optimize_random_forest(X, y, n_trials=100, n_folds=5, output_dir='results'):
    """Optimize Random Forest hyperparameters using Optuna"""
    print(f"Starting Random Forest optimization with {n_trials} trials...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Optuna study
    study = optuna.create_study(direction='maximize', study_name='random_forest_optimization')
    
    # Optimize
    study.optimize(lambda trial: objective_rf(trial, X, y, n_folds), n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"Best accuracy: {best_score:.4f}")
    print("Best hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Train model with best parameters
    best_model = RandomForestClassifier(
        class_weight='balanced',
        random_state=SEED,
        **best_params
    )
    
    best_model.fit(X, y)
    
    # Save best model
    model_path = os.path.join(output_dir, "optimized_random_forest.pkl")
    joblib.dump(best_model, model_path)
    print(f"Optimized Random Forest model saved to {model_path}")
    
    # Save optimization history
    history_df = pd.DataFrame({
        'trial': range(len(study.trials)),
        'accuracy': [trial.value for trial in study.trials]
    })
    history_path = os.path.join(output_dir, "random_forest_optimization_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Optimization history saved to {history_path}")
    
    # Save best parameters
    params_df = pd.DataFrame([best_params])
    params_path = os.path.join(output_dir, "random_forest_best_params.csv")
    params_df.to_csv(params_path, index=False)
    print(f"Best parameters saved to {params_path}")
    
    # Plot optimization history
    plt.figure(figsize=(12, 8))
    plt.plot(history_df['trial'], history_df['accuracy'], 'b-', alpha=0.3)
    plt.plot(history_df['trial'], history_df['accuracy'].cummax(), 'r-')
    plt.xlabel('Trial', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Random Forest Hyperparameter Optimization History', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "random_forest_optimization_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Optimization plot saved to {plot_path}")
    
    # Plot parameter importances
    param_importances = optuna.importance.get_param_importances(study)
    importance_df = pd.DataFrame({
        'Parameter': list(param_importances.keys()),
        'Importance': list(param_importances.values())
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Parameter', data=importance_df)
    plt.title('Hyperparameter Importance for Random Forest', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Parameter', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    importance_path = os.path.join(output_dir, "random_forest_param_importance.png")
    plt.savefig(importance_path, dpi=300)
    plt.close()
    print(f"Parameter importance plot saved to {importance_path}")
    
    # Evaluate final model
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')
    
    print(f"Final {n_folds}-fold CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    return best_model, best_params, best_score

def main():
    """Main function to optimize Random Forest model"""
    # Set paths
    dataset_path = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/feature_selection/top_25_features/top_features_dataset.csv"
    
    # Set output directory
    output_dir = os.path.join("/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts", "model_optimization", "top_25_features", "random_forest")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    df = load_dataset(dataset_path)
    
    # Prepare data
    X, y = prepare_data(df, target_col='HF')
    
    # Optimize Random Forest
    rf_model, rf_params, rf_score = optimize_random_forest(X, y, n_trials=100, n_folds=5, output_dir=output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("Random Forest Optimization Completed Successfully!")
    print("="*80)
    print("Performance:")
    print(f"  Accuracy: {rf_score:.4f}")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
