import os
import sys
import joblib
import pandas as pd
import numpy as np
import json

def load_model(model_path):
    """Load the trained model from file"""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def load_model_info(info_path):
    """Load model info from file"""
    try:
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        return model_info
    except Exception as e:
        print(f"Error loading model info: {e}")
        sys.exit(1)

def predict(model, data, required_features=None):
    """Make predictions using the trained model"""
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
    """Main function to make predictions"""
    # Check command line arguments
    if len(sys.argv) < 3:
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