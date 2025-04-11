# Heart Failure Detection Model Evaluation

## 10-Fold Cross-Validation Results

### Top 5 Models

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Log Loss |
|-------|----------|-----------|--------|----------|---------|----------|
| LightGBM | 0.8700 ± 0.0589 | 0.8791 ± 0.0549 | 0.8700 ± 0.0589 | 0.8680 ± 0.0611 | 0.9343 ± 0.0339 | 0.4179 ± 0.1813 |
| Extra Trees Classifier | 0.8600 ± 0.0550 | 0.8650 ± 0.0547 | 0.8600 ± 0.0550 | 0.8591 ± 0.0557 | 0.9325 ± 0.0346 | 0.3438 ± 0.0512 |
| Random Forest | 0.8525 ± 0.0506 | 0.8597 ± 0.0494 | 0.8525 ± 0.0506 | 0.8507 ± 0.0521 | 0.9322 ± 0.0291 | 0.3586 ± 0.0462 |
| XGBoost | 0.8400 ± 0.0436 | 0.8446 ± 0.0432 | 0.8400 ± 0.0436 | 0.8390 ± 0.0438 | 0.9310 ± 0.0236 | 0.3261 ± 0.0540 |
| Gradient Boosting | 0.8575 ± 0.0404 | 0.8654 ± 0.0360 | 0.8575 ± 0.0404 | 0.8554 ± 0.0428 | 0.9290 ± 0.0256 | 0.4484 ± 0.1533 |

### All Models (Sorted by ROC AUC)

| Model | ROC AUC | Log Loss | Accuracy | F1 Score |
|-------|---------|----------|----------|----------|
| LightGBM | 0.9343 | 0.4179 | 0.8700 | 0.8680 |
| Extra Trees Classifier | 0.9325 | 0.3438 | 0.8600 | 0.8591 |
| Random Forest | 0.9322 | 0.3586 | 0.8525 | 0.8507 |
| XGBoost | 0.9310 | 0.3261 | 0.8400 | 0.8390 |
| Gradient Boosting | 0.9290 | 0.4484 | 0.8575 | 0.8554 |
| Linear Discriminant Analysis | 0.9161 | 0.4004 | 0.8375 | 0.8368 |
| AdaBoost | 0.9106 | 0.4637 | 0.8200 | 0.8183 |
| Logistic Regression | 0.9086 | 0.3932 | 0.8300 | 0.8289 |
| CatBoost | 0.9015 | 0.4202 | 0.8300 | 0.8231 |
| Naive Bayes | 0.8990 | 1.8830 | 0.8250 | 0.8247 |
| Multi-Layer Perceptron (MLP) | 0.8640 | 0.7123 | 0.7725 | 0.7692 |
| Support Vector Machine | 0.8416 | 0.4868 | 0.7550 | 0.7539 |
| Decision Tree | 0.8183 | 6.3381 | 0.8000 | 0.8001 |
| K-Nearest Neighbors | 0.7977 | 2.1240 | 0.7525 | 0.7508 |

## Analysis and Recommendations

### Best Overall Model: LightGBM

- ROC AUC: 0.9343 ± 0.0339
- Log Loss: 0.4179 ± 0.1813
- Accuracy: 0.8700 ± 0.0589
- F1 Score: 0.8680 ± 0.0611

### Model with Lowest Log Loss: XGBoost

- Log Loss: 0.3261 ± 0.0540
- ROC AUC: 0.9310 ± 0.0236

### Recommendations

1. **Primary Model**: LightGBM - Best overall performance with highest ROC AUC.
2. **Alternative Model**: XGBoost - Lowest log loss, indicating good probability calibration.
3. **Ensemble Approach**: Consider an ensemble of the top 3-5 models for potentially improved performance.
4. **Model Deployment**: Deploy the primary model with careful monitoring of performance metrics.
5. **Further Optimization**: Fine-tune hyperparameters of the top models for potential performance improvements.
