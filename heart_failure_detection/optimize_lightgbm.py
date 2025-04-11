import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

def objective(trial, X, y, n_folds=5):
    """Optuna objective function for LightGBM optimization"""
    # Define the hyperparameter search space
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 0.5, log=True),
        'random_state': SEED
    }

    # Add class weight for imbalanced data
    param['class_weight'] = 'balanced'

    # Create cross-validation folds
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # Initialize scores list
    cv_scores = []

    # Perform cross-validation
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create and train model
        model = LGBMClassifier(**param)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        cv_scores.append(score)

    # Return mean accuracy across folds
    return np.mean(cv_scores)

def optimize_lightgbm(X, y, n_trials=500, n_folds=5, output_dir='results'):
    """Optimize LightGBM hyperparameters using Optuna"""
    print(f"Starting LightGBM optimization with {n_trials} trials...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create Optuna study
    study = optuna.create_study(direction='maximize', study_name='lightgbm_optimization')

    # Optimize
    study.optimize(lambda trial: objective(trial, X, y, n_folds), n_trials=n_trials)

    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value

    print(f"Best accuracy: {best_score:.4f}")
    print("Best hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Train model with best parameters
    best_model = LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        verbosity=-1,
        class_weight='balanced',
        random_state=SEED,
        **best_params
    )

    best_model.fit(X, y)

    # Save best model
    model_path = os.path.join(output_dir, 'optimized_lightgbm.pkl')
    joblib.dump(best_model, model_path)
    print(f"Optimized LightGBM model saved to {model_path}")

    # Save optimization history
    history = {
        'value': [trial.value for trial in study.trials],
        'params': [trial.params for trial in study.trials]
    }
    history_df = pd.DataFrame({
        'trial': range(len(study.trials)),
        'accuracy': [trial.value for trial in study.trials]
    })
    history_path = os.path.join(output_dir, 'lightgbm_optimization_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"Optimization history saved to {history_path}")

    # Save best parameters
    params_df = pd.DataFrame([best_params])
    params_path = os.path.join(output_dir, 'lightgbm_best_params.csv')
    params_df.to_csv(params_path, index=False)
    print(f"Best parameters saved to {params_path}")

    # Plot optimization history
    plt.figure(figsize=(12, 8))
    plt.plot(history_df['trial'], history_df['accuracy'], 'b-', alpha=0.3)
    plt.plot(history_df['trial'], history_df['accuracy'].cummax(), 'r-')
    plt.xlabel('Trial', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('LightGBM Hyperparameter Optimization History', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'lightgbm_optimization_plot.png')
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
    plt.title('Hyperparameter Importance for LightGBM', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Parameter', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    importance_path = os.path.join(output_dir, 'lightgbm_param_importance.png')
    plt.savefig(importance_path, dpi=300)
    plt.close()
    print(f"Parameter importance plot saved to {importance_path}")

    # Evaluate final model
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')

    print(f"Final {n_folds}-fold CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    return best_model, best_params, best_score

def main():
    """Main function to run LightGBM optimization"""
    # Set output directory
    output_dir = os.path.join("/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts", "model_optimization", "lightgbm")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset_path = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/feature_selection/features_2_methods/selected_features_dataset.csv"
    df = load_dataset(dataset_path)

    # Prepare data
    X, y = prepare_data(df, target_col='HF')

    # Optimize LightGBM
    best_model, best_params, best_score = optimize_lightgbm(X, y, n_trials=500, n_folds=5, output_dir=output_dir)

    print("\n" + "="*80)
    print("LightGBM Optimization Completed Successfully!")
    print("="*80)
    print(f"Best accuracy: {best_score:.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
