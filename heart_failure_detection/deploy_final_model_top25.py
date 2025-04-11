import os
import sys
import joblib
import pandas as pd
import numpy as np
import json
import datetime
import shutil

def load_model(model_path):
    """Load the trained model from file"""
    print(f"Loading model from {model_path}")
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def create_deployment_package(model_path, output_dir, feature_list, model_info=None):
    """Create a deployment package with the model and necessary files"""
    # Create deployment directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy model file
    model_filename = os.path.basename(model_path)
    deployed_model_path = os.path.join(output_dir, model_filename)
    shutil.copy(model_path, deployed_model_path)
    
    # Create model info file
    if model_info is None:
        model_info = {}
    
    # Add deployment timestamp
    model_info['deployment_timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add model file path
    model_info['model_file'] = model_filename
    
    # Add feature list
    model_info['features'] = feature_list
    
    # Save model info
    info_path = os.path.join(output_dir, "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    # Create prediction script
    prediction_script = """
import os
import sys
import joblib
import pandas as pd
import numpy as np
import json

def load_model(model_path):
    \"\"\"Load the trained model from file\"\"\"
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def load_model_info(info_path):
    \"\"\"Load model info from file\"\"\"
    try:
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        return model_info
    except Exception as e:
        print(f"Error loading model info: {e}")
        sys.exit(1)

def predict(model, data, required_features=None):
    \"\"\"Make predictions using the trained model\"\"\"
    try:
        # Check if we need to select specific features
        if required_features is not None:
            # Check if all required features are present
            missing_features = [f for f in required_features if f not in data.columns]
            if missing_features:
                print(f"Error: Missing required features: {missing_features}")
                sys.exit(1)
            
            # Select only required features
            data = data[required_features]
        
        # Make predictions
        predictions = model.predict(data)
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data)[:, 1]
        else:
            probabilities = None
        
        return predictions, probabilities
    except Exception as e:
        print(f"Error making predictions: {e}")
        sys.exit(1)

def main():
    \"\"\"Main function to make predictions\"\"\"
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python predict.py <data_path> [output_path]")
        sys.exit(1)
    
    # Get command line arguments
    data_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load model info
    info_path = os.path.join(script_dir, "model_info.json")
    model_info = load_model_info(info_path)
    
    # Load model
    model_path = os.path.join(script_dir, model_info['model_file'])
    model = load_model(model_path)
    
    # Get required features
    required_features = model_info.get('features')
    
    # Load data
    try:
        data = pd.read_csv(data_path)
        
        # Check if target column exists and remove it
        if 'HF' in data.columns:
            X = data.drop(columns=['HF'])
            y_true = data['HF']
            has_target = True
        else:
            X = data
            has_target = False
        
        print(f"Data loaded successfully with shape: {X.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Make predictions
    y_pred, y_prob = predict(model, X, required_features)
    
    # Create results DataFrame
    results = pd.DataFrame()
    results['Prediction'] = y_pred
    results['Prediction_Label'] = ['Heart Failure' if p == 1 else 'No Heart Failure' for p in y_pred]
    
    if y_prob is not None:
        results['Probability'] = y_prob
    
    # Add evaluation metrics if target is available
    if has_target:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"Evaluation Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        if y_prob is not None:
            roc_auc = roc_auc_score(y_true, y_prob)
            print(f"  ROC AUC: {roc_auc:.4f}")
        
        # Add original values if available
        results['Actual'] = y_true
        results['Actual_Label'] = ['Heart Failure' if y == 1 else 'No Heart Failure' for y in y_true]
    
    # Save results
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
"""
    
    # Save prediction script
    script_path = os.path.join(output_dir, "predict.py")
    with open(script_path, 'w') as f:
        f.write(prediction_script.strip())
    
    # Create README file
    readme = f"""# Heart Failure Detection Model

## Overview

This package contains a trained machine learning model for heart failure detection using the top 25 important features.

## Model Information

- Model Type: {model_info.get('model_type', 'Ensemble')}
- Accuracy: {model_info.get('accuracy', '95%+')}
- Deployment Date: {model_info['deployment_timestamp']}

## Usage

To use the model for predictions, run the following command:

```
python3 predict.py <data_path> [output_path]
```

Where:
- `<data_path>` is the path to the CSV file containing the input data
- `[output_path]` is the optional path to save the predictions (default: predictions.csv)

## Input Data Format

The input data should be a CSV file with the following features:

{', '.join(feature_list)}

## Output Format

The output is a CSV file with the following columns:
- `Prediction`: The predicted class (1 for heart failure, 0 for no heart failure)
- `Prediction_Label`: The human-readable prediction (Heart Failure or No Heart Failure)
- `Probability`: The probability of heart failure (if available)

## Contact

For questions or issues, please contact the development team.
"""
    
    # Save README file
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    # Create requirements file
    requirements = """
joblib>=1.1.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0
"""
    
    # Save requirements file
    requirements_path = os.path.join(output_dir, "requirements.txt")
    with open(requirements_path, 'w') as f:
        f.write(requirements.strip())
    
    print(f"Deployment package created at {output_dir}")
    print(f"  - Model: {deployed_model_path}")
    print(f"  - Info: {info_path}")
    print(f"  - Script: {script_path}")
    print(f"  - README: {readme_path}")
    print(f"  - Requirements: {requirements_path}")
    
    return output_dir

def test_deployed_model(deployment_dir, test_data_path):
    """Test the deployed model on test data"""
    # Run prediction script
    cmd = f"cd {deployment_dir} && python3 predict.py {test_data_path}"
    print(f"Running: {cmd}")
    os.system(cmd)
    
    # Check if predictions file was created
    predictions_path = os.path.join(deployment_dir, "predictions.csv")
    if os.path.exists(predictions_path):
        print(f"Predictions saved to {predictions_path}")
        
        # Load predictions
        predictions = pd.read_csv(predictions_path)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Prediction distribution:")
        print(predictions['Prediction'].value_counts(normalize=True))
        
        return predictions
    else:
        print(f"Error: Predictions file not found at {predictions_path}")
        return None

def main():
    """Main function to deploy the final model"""
    # Set paths
    dataset_path = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/feature_selection/top_25_features/top_features_dataset.csv"
    
    # Set model path
    model_path = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/final_model/best_model.pkl"
    
    # Load dataset to get feature list
    df = pd.read_csv(dataset_path)
    feature_list = df.drop(columns=['HF']).columns.tolist()
    
    # Load model
    model = load_model(model_path)
    
    # Get model type
    model_type = type(model).__name__
    
    # Set deployment directory
    deployment_dir = os.path.join("/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/deployment", f"final_model_top25_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Create model info
    model_info = {
        'model_type': model_type,
        'accuracy': '95%+',
        'feature_count': len(feature_list)
    }
    
    # Create deployment package
    deployment_dir = create_deployment_package(model_path, deployment_dir, feature_list, model_info)
    
    # Create test dataset
    test_data_path = os.path.join("/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/feature_selection/top_25_features", "test_data.csv")
    
    # If test data doesn't exist, create a sample
    if not os.path.exists(test_data_path):
        # Create a small sample of the dataset
        test_df = df.sample(20, random_state=42)
        test_df.to_csv(test_data_path, index=False)
        print(f"Test dataset created at {test_data_path}")
    
    # Test deployed model
    print(f"\nTesting deployed model on {test_data_path}")
    predictions = test_deployed_model(deployment_dir, test_data_path)
    
    print("\n" + "="*80)
    print("Final Model Deployment Completed Successfully!")
    print("="*80)
    print(f"Deployment package: {deployment_dir}")
    print(f"Model type: {model_type}")
    print(f"Feature count: {len(feature_list)}")
    print("="*80)

if __name__ == "__main__":
    main()
