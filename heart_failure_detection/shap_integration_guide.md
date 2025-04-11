# SHAP Integration for Model Interpretability

## Overview

This document provides an overview of the SHAP (SHapley Additive exPlanations) integration for the Heart Failure Detection model. SHAP values help explain individual predictions by showing how each feature contributes to pushing the model output from a baseline value to the final prediction.

## Implementation Details

### 1. SHAP Explainer Initialization

The SHAP explainer is initialized when the model is loaded:

```python
# Initialize SHAP explainer for tree-based models
if hasattr(model, 'predict') and any(model_type in type(model).__name__.lower() 
                                    for model_type in ['lightgbm', 'xgboost', 'randomforest', 
                                                      'extratrees', 'gradientboosting']):
    # Create a small background dataset for SHAP
    sample_data = pd.read_csv(sample_data_path)
    if 'HF' in sample_data.columns:
        sample_data = sample_data.drop(columns=['HF'])
    background_data = sample_data.sample(min(50, len(sample_data)), random_state=42)
    
    # Create the SHAP explainer
    explainer = shap.TreeExplainer(model)
```

### 2. SHAP Explanation Endpoint

A new `/explain` endpoint has been added to the Flask application to generate SHAP explanations for individual predictions:

```python
@app.route('/explain', methods=['POST'])
def explain():
    """Generate SHAP explanations for a prediction"""
    # Check if model and explainer are loaded
    if model is None or explainer is None:
        return jsonify({'error': 'Model or explainer not available'}), 500
    
    # Get input data
    data = request.json
    df = pd.DataFrame(data)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(df)
    
    # Generate visualizations
    summary_plot = generate_summary_plot(shap_values, df)
    force_plot = generate_force_plot(explainer.expected_value, shap_values, df)
    waterfall_plot = generate_waterfall_plot(shap_values, df)
    
    # Return results
    return jsonify({
        'prediction': prediction,
        'feature_importance': feature_importance,
        'summary_plot': summary_plot,
        'force_plot': force_plot,
        'waterfall_plot': waterfall_plot
    })
```

### 3. SHAP Visualizations

The implementation includes four types of SHAP visualizations:

1. **Feature Importance Table**: Shows the impact of each feature on the prediction, sorted by absolute SHAP value.

2. **Waterfall Plot**: Shows how each feature pushes the prediction higher or lower from the base value.

3. **Summary Plot**: Shows the distribution of SHAP values across features, helping to understand the global impact of features.

4. **Force Plot**: Shows how each feature contributes to pushing the model output from the base value to the final prediction.

### 4. Web Interface Integration

The web interface has been enhanced with a tabbed interface for SHAP explanations:

- An "Explain Prediction" button appears after making a prediction
- Clicking the button sends the input data to the `/explain` endpoint
- The results are displayed in a tabbed interface with different visualization types
- Users can switch between tabs to explore different aspects of the explanation

## Using SHAP Explanations

### For Individual Predictions

1. Enter patient data in the form
2. Click "Predict" to get the prediction
3. Click "Explain Prediction" to see the SHAP explanation
4. Explore the different tabs to understand the prediction:
   - **Feature Importance**: See which features had the most impact
   - **Waterfall Plot**: Understand how each feature contributed to the final prediction
   - **Summary Plot**: See the distribution of feature impacts
   - **Force Plot**: Visualize how features push the prediction from the base value

### Interpreting SHAP Values

- **Positive SHAP values** (red) push the prediction higher (toward heart failure)
- **Negative SHAP values** (blue) push the prediction lower (away from heart failure)
- The **magnitude** of the SHAP value indicates the strength of the impact
- The **base value** represents the average model output over the training dataset

## Benefits of SHAP Integration

1. **Transparency**: Provides clear explanations for model predictions
2. **Trust**: Helps healthcare professionals understand and trust the model's decisions
3. **Insights**: Reveals which features are most important for individual predictions
4. **Clinical Relevance**: Connects model predictions to clinical factors that doctors understand

## Technical Implementation

The SHAP integration uses:

- **shap** library for calculating SHAP values
- **matplotlib** for generating static plots
- **plotly** for interactive waterfall plots
- **base64** encoding for transferring images to the web interface

## Future Enhancements

1. **Interactive Force Plots**: Replace static force plots with interactive JavaScript-based visualizations
2. **Cohort Analysis**: Add ability to compare SHAP values across patient groups
3. **Feature Dependence Plots**: Add visualizations showing how features interact
4. **Global Explanations**: Add model-level explanations to understand overall feature importance

## References

- [SHAP GitHub Repository](https://github.com/slundberg/shap)
- [SHAP Paper: A Unified Approach to Interpreting Model Predictions](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
- [Interpretable Machine Learning with SHAP](https://christophm.github.io/interpretable-ml-book/shap.html)
