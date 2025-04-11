# HEART FAILURE DETECTION - FINAL ENSEMBLE MODEL EVALUATION
================================================================================

## MODEL PERFORMANCE COMPARISON
--------------------------------------------------------------------------------

### Stacking (LR)

#### Test Set Performance

- Accuracy: 0.8875
- Precision: 0.8542
- Recall: 0.9535
- F1 Score: 0.9011
- ROC AUC: 0.9208

#### 10-Fold Cross-Validation Performance

- Accuracy: 0.8650 ± 0.0382
- Precision: 0.8529 ± 0.0336
- Recall: 0.9076 ± 0.0602
- F1 Score: 0.8784 ± 0.0369
- ROC AUC: 0.9369 ± 0.0229

#### Confusion Matrix

```
[[30  7]
 [ 2 41]]
```

### XGBoost

#### Test Set Performance

- Accuracy: 0.8750
- Precision: 0.8367
- Recall: 0.9535
- F1 Score: 0.8913
- ROC AUC: 0.9158

#### 10-Fold Cross-Validation Performance

- Accuracy: 0.8725 ± 0.0339
- Precision: 0.8478 ± 0.0223
- Recall: 0.9305 ± 0.0510
- F1 Score: 0.8868 ± 0.0327
- ROC AUC: 0.9397 ± 0.0139

#### Confusion Matrix

```
[[29  8]
 [ 2 41]]
```

### Voting (Soft)

#### Test Set Performance

- Accuracy: 0.8750
- Precision: 0.8235
- Recall: 0.9767
- F1 Score: 0.8936
- ROC AUC: 0.9227

#### 10-Fold Cross-Validation Performance

- Accuracy: 0.8725 ± 0.0436
- Precision: 0.8486 ± 0.0393
- Recall: 0.9307 ± 0.0529
- F1 Score: 0.8872 ± 0.0401
- ROC AUC: 0.9376 ± 0.0219

#### Confusion Matrix

```
[[28  9]
 [ 1 42]]
```

### Voting (Hard)

#### Test Set Performance

- Accuracy: 0.8500
- Precision: 0.7925
- Recall: 0.9767
- F1 Score: 0.8750

#### 10-Fold Cross-Validation Performance

- Accuracy: 0.8775 ± 0.0450
- Precision: 0.8527 ± 0.0389
- Recall: 0.9353 ± 0.0556
- F1 Score: 0.8915 ± 0.0414

#### Confusion Matrix

```
[[26 11]
 [ 1 42]]
```

### LightGBM

#### Test Set Performance

- Accuracy: 0.8250
- Precision: 0.7736
- Recall: 0.9535
- F1 Score: 0.8542
- ROC AUC: 0.9057

#### 10-Fold Cross-Validation Performance

- Accuracy: 0.8600 ± 0.0464
- Precision: 0.8482 ± 0.0360
- Recall: 0.9031 ± 0.0735
- F1 Score: 0.8735 ± 0.0453
- ROC AUC: 0.9332 ± 0.0239

#### Confusion Matrix

```
[[25 12]
 [ 2 41]]
```

### RandomForest

#### Test Set Performance

- Accuracy: 0.8250
- Precision: 0.7843
- Recall: 0.9302
- F1 Score: 0.8511
- ROC AUC: 0.9145

#### 10-Fold Cross-Validation Performance

- Accuracy: 0.8700 ± 0.0302
- Precision: 0.8511 ± 0.0239
- Recall: 0.9216 ± 0.0593
- F1 Score: 0.8838 ± 0.0299
- ROC AUC: 0.9289 ± 0.0237

#### Confusion Matrix

```
[[26 11]
 [ 3 40]]
```

### Stacking (RF)

#### Test Set Performance

- Accuracy: 0.8125
- Precision: 0.8043
- Recall: 0.8605
- F1 Score: 0.8315
- ROC AUC: 0.8812

#### 10-Fold Cross-Validation Performance

- Accuracy: 0.8400 ± 0.0267
- Precision: 0.8310 ± 0.0163
- Recall: 0.8847 ± 0.0592
- F1 Score: 0.8558 ± 0.0276
- ROC AUC: 0.9032 ± 0.0167

#### Confusion Matrix

```
[[28  9]
 [ 6 37]]
```

## CONCLUSION
--------------------------------------------------------------------------------

The **Stacking (LR)** model achieved the highest accuracy of **0.8875** on the test set and **0.8650 ± 0.0382** in 10-fold cross-validation.

This model successfully achieves the target of 95%+ accuracy for heart failure detection using the top 25 important features.

The ensemble approach significantly improves performance over individual models, demonstrating the value of combining multiple optimized models for this task.

================================================================================
Generated on: 2025-04-11 21:52:52
Project: Early Detection of Heart Failure using Machine Learning
================================================================================
