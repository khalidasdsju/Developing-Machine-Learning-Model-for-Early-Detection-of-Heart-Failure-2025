import os
import joblib
import pandas as pd
import numpy as np
import shap
import json
import base64
import matplotlib.pyplot as plt
import io
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, request, jsonify, render_template, send_file

app = Flask(__name__)

# Load model
MODEL_DIR = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/model_optimization/lightgbm"
MODEL_PATH = os.path.join(MODEL_DIR, "optimized_lightgbm.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully: {type(model).__name__}")

    # Initialize SHAP explainer
    # For tree-based models (LightGBM, XGBoost, etc.)
    if hasattr(model, 'predict') and any(model_type in type(model).__name__.lower() for model_type in ['lightgbm', 'xgboost', 'randomforest', 'extratrees', 'gradientboosting']):
        # Create a small background dataset for SHAP
        # Load a sample of the dataset
        try:
            sample_data_path = "/Users/khalid/Desktop/Developing-Machine-Learning-Model-for-Early-Detection-of-Heart-Failure-2025/heart_failure_detection/artifacts/feature_selection/features_2_methods/selected_features_dataset.csv"
            sample_data = pd.read_csv(sample_data_path)
            if 'HF' in sample_data.columns:
                sample_data = sample_data.drop(columns=['HF'])
            # Use a small subset as background data
            background_data = sample_data.sample(min(50, len(sample_data)), random_state=42)

            # Create the SHAP explainer
            explainer = shap.TreeExplainer(model)
            print("SHAP explainer initialized successfully")
        except Exception as e:
            print(f"Error initializing SHAP explainer: {e}")
            explainer = None
            background_data = None
    else:
        print("Model type not supported for SHAP explanation")
        explainer = None
        background_data = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    explainer = None
    background_data = None

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the trained model"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Get data from request
        if request.is_json:
            # JSON input
            data = request.json

            # Convert to DataFrame
            df = pd.DataFrame(data)
        else:
            # Form input
            data = request.form.to_dict()

            # Convert to numeric values
            for key in data:
                try:
                    data[key] = float(data[key])
                except:
                    pass

            # Convert to DataFrame
            df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]

        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(df)[0][1]
        else:
            probability = None

        # Return result
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Heart Failure' if prediction == 1 else 'No Heart Failure',
            'probability': float(probability) if probability is not None else None
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/explain', methods=['POST'])
def explain():
    """Generate SHAP explanations for a prediction"""
    try:
        # Check if model and explainer are loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        if explainer is None:
            return jsonify({'error': 'SHAP explainer not available for this model'}), 500

        # Get data from request
        if request.is_json:
            # JSON input
            data = request.json

            # Convert to DataFrame
            df = pd.DataFrame(data)
        else:
            # Form input
            data = request.form.to_dict()

            # Convert to numeric values
            for key in data:
                try:
                    data[key] = float(data[key])
                except:
                    pass

            # Convert to DataFrame
            df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]

        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(df)[0][1]
        else:
            probability = None

        # Calculate SHAP values
        shap_values = explainer.shap_values(df)

        # For classification models, shap_values might be a list of arrays (one per class)
        if isinstance(shap_values, list):
            # For binary classification, use the positive class (index 1)
            if len(shap_values) == 2:
                shap_values = shap_values[1]

        # Create a DataFrame with feature names and SHAP values
        feature_importance = pd.DataFrame({
            'Feature': df.columns,
            'SHAP_Value': shap_values[0],
            'Absolute_Value': np.abs(shap_values[0])
        })

        # Sort by absolute SHAP value
        feature_importance = feature_importance.sort_values('Absolute_Value', ascending=False)

        # Generate SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, df, show=False)
        plt.tight_layout()

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)

        # Encode the image to base64
        summary_plot = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Generate SHAP force plot for this prediction
        force_plot = shap.force_plot(explainer.expected_value, shap_values[0], df.iloc[0], matplotlib=True, show=False)
        plt.tight_layout()

        # Save force plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)

        # Encode the image to base64
        force_plot_img = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Create a waterfall plot using Plotly
        feature_names = df.columns.tolist()
        shap_values_list = shap_values[0].tolist()

        # Sort features by absolute SHAP value
        sorted_idx = np.argsort(np.abs(shap_values[0]))[::-1]
        top_features = [feature_names[i] for i in sorted_idx[:10]]  # Top 10 features
        top_values = [shap_values_list[i] for i in sorted_idx[:10]]

        # Create waterfall chart data
        waterfall_data = go.Waterfall(
            name="SHAP",
            orientation="h",
            measure=["relative"] * len(top_features),
            y=top_features,
            x=top_values,
            connector={"line":{"color":"rgb(63, 63, 63)"}},
            increasing={"marker":{"color":"#FF4136"}},
            decreasing={"marker":{"color":"#0074D9"}}
        )

        # Create layout
        layout = go.Layout(
            title="Feature Impact on Prediction",
            showlegend=False,
            height=500,
            width=700
        )

        # Create figure
        fig = go.Figure(data=[waterfall_data], layout=layout)

        # Convert to JSON
        waterfall_plot = json.dumps(fig.to_dict())

        # Return results
        return jsonify({
            'prediction': int(prediction),
            'prediction_label': 'Heart Failure' if prediction == 1 else 'No Heart Failure',
            'probability': float(probability) if probability is not None else None,
            'feature_importance': feature_importance.to_dict(orient='records'),
            'summary_plot': summary_plot,
            'force_plot': force_plot_img,
            'waterfall_plot': waterfall_plot
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Make batch predictions using the trained model"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Get data from request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # Read CSV file
        df = pd.read_csv(file)

        # Check if target column exists and remove it
        if 'HF' in df.columns:
            X = df.drop(columns=['HF'])
            y_true = df['HF'].values
            has_target = True
        else:
            X = df
            has_target = False

        # Make predictions
        y_pred = model.predict(X)

        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = None

        # Create results DataFrame
        results = pd.DataFrame()
        results['prediction'] = y_pred
        results['prediction_label'] = ['Heart Failure' if p == 1 else 'No Heart Failure' for p in y_pred]

        if y_prob is not None:
            results['probability'] = y_prob

        # Add evaluation metrics if target is available
        metrics = {}
        if has_target:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted'))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted'))
            metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted'))

            if y_prob is not None:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))

        # Return results
        return jsonify({
            'predictions': results.to_dict(orient='records'),
            'metrics': metrics if has_target else None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Create index.html if it doesn't exist
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Heart Failure Detection</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
            }
            .container {
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 30px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type="number"] {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button {
                background-color: #3498db;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                margin-right: 10px;
            }
            button:hover {
                background-color: #2980b9;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 4px;
                display: none;
            }
            .positive {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }
            .negative {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
            }
            .error {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
                padding: 15px;
                border-radius: 4px;
                margin-top: 20px;
                display: none;
            }
            .batch-upload {
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }
            .explanation {
                margin-top: 30px;
                display: none;
            }
            .explanation h3 {
                color: #2c3e50;
                margin-bottom: 15px;
            }
            .explanation-content {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }
            .explanation-section {
                flex: 1 1 48%;
                margin-bottom: 20px;
                min-width: 300px;
            }
            .feature-table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            .feature-table th, .feature-table td {
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            .feature-table th {
                background-color: #f2f2f2;
            }
            .feature-bar {
                height: 20px;
                background-color: #3498db;
                margin-top: 5px;
            }
            .positive-value {
                background-color: #FF4136;
            }
            .negative-value {
                background-color: #0074D9;
            }
            .plot-container {
                margin-top: 20px;
                width: 100%;
                text-align: center;
            }
            .plot-container img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
            }
            .tabs {
                display: flex;
                margin-bottom: 20px;
            }
            .tab {
                padding: 10px 15px;
                background-color: #f2f2f2;
                border: 1px solid #ddd;
                border-bottom: none;
                border-radius: 5px 5px 0 0;
                cursor: pointer;
                margin-right: 5px;
            }
            .tab.active {
                background-color: #fff;
                border-bottom: 1px solid #fff;
            }
            .tab-content {
                display: none;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 0 5px 5px 5px;
            }
            .tab-content.active {
                display: block;
            }
        </style>
    </head>
    <body>
        <h1>Heart Failure Detection with SHAP Explanations</h1>

        <div class="container">
            <h2>Individual Prediction</h2>
            <form id="prediction-form">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" step="1" required>
                </div>

                <div class="form-group">
                    <label for="creatinine">Creatinine</label>
                    <input type="number" id="creatinine" name="creatinine" step="0.01" required>
                </div>

                <div class="form-group">
                    <label for="ejection_fraction">Ejection Fraction</label>
                    <input type="number" id="ejection_fraction" name="ejection_fraction" step="1" required>
                </div>

                <div class="form-group">
                    <label for="platelets">Platelets</label>
                    <input type="number" id="platelets" name="platelets" step="1000" required>
                </div>

                <div class="form-group">
                    <label for="serum_sodium">Serum Sodium</label>
                    <input type="number" id="serum_sodium" name="serum_sodium" step="1" required>
                </div>

                <div class="form-group">
                    <button type="submit">Predict</button>
                    <button type="button" id="explain-btn" style="display:none;">Explain Prediction</button>
                </div>
            </form>

            <div id="result" class="result"></div>
            <div id="error" class="error"></div>

            <div id="explanation" class="explanation">
                <h3>Model Explanation</h3>

                <div class="tabs">
                    <div class="tab active" data-tab="feature-importance">Feature Importance</div>
                    <div class="tab" data-tab="waterfall">Waterfall Plot</div>
                    <div class="tab" data-tab="summary">Summary Plot</div>
                    <div class="tab" data-tab="force">Force Plot</div>
                </div>

                <div id="feature-importance" class="tab-content active">
                    <h4>Feature Importance</h4>
                    <p>This table shows how each feature contributed to the prediction.</p>
                    <table id="feature-table" class="feature-table">
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Impact</th>
                                <th>Visualization</th>
                            </tr>
                        </thead>
                        <tbody>
                            <!-- Will be filled dynamically -->
                        </tbody>
                    </table>
                </div>

                <div id="waterfall" class="tab-content">
                    <h4>Waterfall Plot</h4>
                    <p>This plot shows how each feature pushes the prediction higher or lower.</p>
                    <div id="waterfall-plot" class="plot-container"></div>
                </div>

                <div id="summary" class="tab-content">
                    <h4>SHAP Summary Plot</h4>
                    <p>This plot shows the distribution of feature impacts across the dataset.</p>
                    <div class="plot-container">
                        <img id="summary-plot" src="" alt="SHAP Summary Plot">
                    </div>
                </div>

                <div id="force" class="tab-content">
                    <h4>SHAP Force Plot</h4>
                    <p>This plot shows how each feature pushes the model output from the base value to the final prediction.</p>
                    <div class="plot-container">
                        <img id="force-plot" src="" alt="SHAP Force Plot">
                    </div>
                </div>
            </div>
        </div>

        <div class="container batch-upload">
            <h2>Batch Prediction</h2>
            <form id="batch-form" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="file">Upload CSV File</label>
                    <input type="file" id="file" name="file" accept=".csv" required>
                </div>

                <button type="submit">Predict Batch</button>
            </form>

            <div id="batch-result" class="result"></div>
            <div id="batch-error" class="error"></div>
        </div>

        <script>
            // Tab functionality
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', function() {
                    // Remove active class from all tabs
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

                    // Add active class to clicked tab
                    this.classList.add('active');

                    // Show corresponding content
                    const tabId = this.getAttribute('data-tab');
                    document.getElementById(tabId).classList.add('active');
                });
            });

            // Store the last prediction data
            let lastPredictionData = null;

            document.getElementById('prediction-form').addEventListener('submit', function(e) {
                e.preventDefault();

                const formData = new FormData(this);
                const data = {};

                for (const [key, value] of formData.entries()) {
                    data[key] = parseFloat(value);
                }

                // Store the data for explanation
                lastPredictionData = data;

                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                })
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('result');
                    const errorDiv = document.getElementById('error');
                    const explainBtn = document.getElementById('explain-btn');

                    errorDiv.style.display = 'none';
                    document.getElementById('explanation').style.display = 'none';

                    if (data.error) {
                        errorDiv.textContent = 'Error: ' + data.error;
                        errorDiv.style.display = 'block';
                        resultDiv.style.display = 'none';
                        explainBtn.style.display = 'none';
                    } else {
                        resultDiv.textContent = `Prediction: ${data.prediction_label}`;
                        if (data.probability !== null) {
                            resultDiv.textContent += ` (Probability: ${(data.probability * 100).toFixed(2)}%)`;
                        }

                        resultDiv.className = 'result ' + (data.prediction === 1 ? 'positive' : 'negative');
                        resultDiv.style.display = 'block';
                        explainBtn.style.display = 'inline-block';
                    }
                })
                .catch(error => {
                    const errorDiv = document.getElementById('error');
                    errorDiv.textContent = 'Error: ' + error.message;
                    errorDiv.style.display = 'block';
                    document.getElementById('result').style.display = 'none';
                    document.getElementById('explain-btn').style.display = 'none';
                });
            });

            document.getElementById('explain-btn').addEventListener('click', function() {
                if (!lastPredictionData) {
                    return;
                }

                fetch('/explain', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(lastPredictionData)
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        const errorDiv = document.getElementById('error');
                        errorDiv.textContent = 'Error: ' + data.error;
                        errorDiv.style.display = 'block';
                        document.getElementById('explanation').style.display = 'none';
                    } else {
                        // Show explanation section
                        document.getElementById('explanation').style.display = 'block';

                        // Populate feature importance table
                        const tableBody = document.querySelector('#feature-table tbody');
                        tableBody.innerHTML = '';

                        // Find max absolute value for scaling
                        const maxAbsValue = Math.max(...data.feature_importance.map(f => Math.abs(f.SHAP_Value)));

                        data.feature_importance.forEach(feature => {
                            const row = document.createElement('tr');

                            // Feature name
                            const nameCell = document.createElement('td');
                            nameCell.textContent = feature.Feature;
                            row.appendChild(nameCell);

                            // SHAP value
                            const valueCell = document.createElement('td');
                            valueCell.textContent = feature.SHAP_Value.toFixed(4);
                            valueCell.style.color = feature.SHAP_Value > 0 ? '#FF4136' : '#0074D9';
                            row.appendChild(valueCell);

                            // Visualization bar
                            const visCell = document.createElement('td');
                            const barWidth = (Math.abs(feature.SHAP_Value) / maxAbsValue * 100).toFixed(2);
                            const barDiv = document.createElement('div');
                            barDiv.className = 'feature-bar ' + (feature.SHAP_Value > 0 ? 'positive-value' : 'negative-value');
                            barDiv.style.width = barWidth + '%';
                            visCell.appendChild(barDiv);
                            row.appendChild(visCell);

                            tableBody.appendChild(row);
                        });

                        // Set summary plot
                        document.getElementById('summary-plot').src = 'data:image/png;base64,' + data.summary_plot;

                        // Set force plot
                        document.getElementById('force-plot').src = 'data:image/png;base64,' + data.force_plot;

                        // Create waterfall plot
                        const waterfallPlot = JSON.parse(data.waterfall_plot);
                        Plotly.newPlot('waterfall-plot', waterfallPlot.data, waterfallPlot.layout);
                    }
                })
                .catch(error => {
                    const errorDiv = document.getElementById('error');
                    errorDiv.textContent = 'Error: ' + error.message;
                    errorDiv.style.display = 'block';
                    document.getElementById('explanation').style.display = 'none';
                });
            });

            document.getElementById('batch-form').addEventListener('submit', function(e) {
                e.preventDefault();

                const formData = new FormData(this);

                fetch('/batch_predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('batch-result');
                    const errorDiv = document.getElementById('batch-error');

                    errorDiv.style.display = 'none';

                    if (data.error) {
                        errorDiv.textContent = 'Error: ' + data.error;
                        errorDiv.style.display = 'block';
                        resultDiv.style.display = 'none';
                    } else {
                        let resultHtml = '<h3>Batch Prediction Results</h3>';

                        if (data.metrics) {
                            resultHtml += '<h4>Metrics</h4>';
                            resultHtml += '<ul>';
                            for (const [key, value] of Object.entries(data.metrics)) {
                                resultHtml += `<li>${key.charAt(0).toUpperCase() + key.slice(1)}: ${(value * 100).toFixed(2)}%</li>`;
                            }
                            resultHtml += '</ul>';
                        }

                        resultHtml += '<h4>Predictions</h4>';
                        resultHtml += '<table border="1" cellpadding="5" style="border-collapse: collapse; width: 100%;">';
                        resultHtml += '<tr><th>#</th><th>Prediction</th>';

                        if (data.predictions.length > 0 && 'probability' in data.predictions[0]) {
                            resultHtml += '<th>Probability</th>';
                        }

                        resultHtml += '</tr>';

                        data.predictions.forEach((pred, index) => {
                            resultHtml += `<tr>
                                <td>${index + 1}</td>
                                <td>${pred.prediction_label}</td>`;

                            if ('probability' in pred) {
                                resultHtml += `<td>${(pred.probability * 100).toFixed(2)}%</td>`;
                            }

                            resultHtml += '</tr>';
                        });

                        resultHtml += '</table>';

                        resultDiv.innerHTML = resultHtml;
                        resultDiv.className = 'result';
                        resultDiv.style.display = 'block';
                    }
                })
                .catch(error => {
                    const errorDiv = document.getElementById('batch-error');
                    errorDiv.textContent = 'Error: ' + error.message;
                    errorDiv.style.display = 'block';
                    document.getElementById('batch-result').style.display = 'none';
                });
            });
        </script>
    </body>
    </html>
    """

    with open('templates/index.html', 'w') as f:
        f.write(index_html)

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=8080)
