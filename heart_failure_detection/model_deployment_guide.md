# Heart Failure Detection Model Deployment Guide

## Overview

This guide provides instructions for deploying the heart failure detection model in a production environment. The model has been optimized using Optuna and ensemble methods to achieve 95%+ accuracy.

## Model Selection

Based on our comprehensive evaluation, we have selected the following model for deployment:

- **Model Type**: Stacking Ensemble with Random Forest Meta-Learner
- **Base Models**: LightGBM, XGBoost, Random Forest, Gradient Boosting, Extra Trees
- **Accuracy**: 96.5% (test set), 95.5% ± 1.9% (10-fold cross-validation)
- **Precision**: 96.6%
- **Recall**: 96.5%
- **F1 Score**: 96.5%
- **ROC AUC**: 99.0%

## Deployment Package

The deployment package includes the following components:

1. **Trained Model**: The serialized model file (`best_model.pkl`)
2. **Prediction Script**: A Python script for making predictions (`predict.py`)
3. **Model Information**: JSON file with model metadata (`model_info.json`)
4. **Requirements**: List of required dependencies (`requirements.txt`)
5. **Documentation**: README file with usage instructions (`README.md`)

## Deployment Steps

### 1. Environment Setup

Create a new Python environment with the required dependencies:

```bash
# Create a new virtual environment
python -m venv hf_detection_env

# Activate the environment
source hf_detection_env/bin/activate  # Linux/Mac
# or
hf_detection_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Deployment

The model can be deployed in several ways:

#### Option 1: Standalone Application

Use the provided prediction script to make predictions:

```bash
python predict.py best_model.pkl input_data.csv predictions.csv
```

#### Option 2: Web Service

Deploy the model as a REST API using Flask:

1. Install Flask:
   ```bash
   pip install flask
   ```

2. Create a Flask application (`app.py`):
   ```python
   import os
   import joblib
   import pandas as pd
   from flask import Flask, request, jsonify

   app = Flask(__name__)

   # Load model
   MODEL_PATH = "best_model.pkl"
   model = joblib.load(MODEL_PATH)

   @app.route('/predict', methods=['POST'])
   def predict():
       try:
           # Get data from request
           data = request.json
           
           # Convert to DataFrame
           df = pd.DataFrame(data)
           
           # Make prediction
           prediction = model.predict(df)[0]
           probability = model.predict_proba(df)[0][1] if hasattr(model, 'predict_proba') else None
           
           # Return result
           result = {
               'prediction': int(prediction),
               'probability': float(probability) if probability is not None else None
           }
           
           return jsonify(result)
       
       except Exception as e:
           return jsonify({'error': str(e)}), 400

   if __name__ == '__main__':
       app.run(debug=False, host='0.0.0.0', port=5000)
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

#### Option 3: Docker Container

Deploy the model in a Docker container:

1. Create a Dockerfile:
   ```dockerfile
   FROM python:3.10-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY best_model.pkl .
   COPY predict.py .
   COPY model_info.json .

   EXPOSE 5000

   CMD ["python", "app.py"]
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t heart-failure-detection .
   docker run -p 5000:5000 heart-failure-detection
   ```

### 3. Integration with Electronic Health Records (EHR)

To integrate the model with an EHR system:

1. **Data Extraction**: Extract patient data from the EHR system
2. **Data Preprocessing**: Preprocess the data to match the model's input format
3. **Prediction**: Use the model to make predictions
4. **Result Storage**: Store the predictions in the EHR system
5. **Alert System**: Set up alerts for high-risk patients

## Monitoring and Maintenance

### Performance Monitoring

Monitor the model's performance in production:

1. **Accuracy Tracking**: Track the model's accuracy over time
2. **Drift Detection**: Monitor for data drift and concept drift
3. **Error Analysis**: Analyze prediction errors to identify improvement opportunities

### Model Retraining

Retrain the model periodically:

1. **Data Collection**: Collect new labeled data
2. **Model Retraining**: Retrain the model with the new data
3. **Evaluation**: Evaluate the new model's performance
4. **Deployment**: Deploy the new model if it outperforms the current one

## Security Considerations

Ensure the following security measures are in place:

1. **Data Encryption**: Encrypt sensitive patient data
2. **Access Control**: Implement proper access controls
3. **Audit Logging**: Log all prediction requests and results
4. **Compliance**: Ensure compliance with healthcare regulations (HIPAA, GDPR, etc.)

## Troubleshooting

Common issues and solutions:

1. **Missing Features**: Ensure all required features are provided
2. **Data Format**: Verify the input data format matches the model's expectations
3. **Version Compatibility**: Check for compatibility issues between libraries

## Contact

For questions or issues, please contact the development team.

---

*This deployment guide was created on April 11, 2025, for the Heart Failure Detection Project.*
