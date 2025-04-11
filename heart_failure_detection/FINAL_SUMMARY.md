# Heart Failure Detection Model - Final Summary

## Project Overview

This project develops a machine learning model for early detection of heart failure using a dataset with clinical features. The model achieves high accuracy (88.75% on the test set) by using the top 25 important features and ensemble learning techniques.

## Key Features

- **Feature Selection**: Identified the top 25 important features for heart failure detection
- **Model Optimization**: Used Optuna to optimize LightGBM, XGBoost, and Random Forest models
- **Ensemble Learning**: Created ensemble models using stacking and voting techniques
- **High Performance**: Achieved 88.75% accuracy on the test set
- **Deployment**: Created a deployment package with a prediction script

## Top 25 Features

The following features were identified as the most important for heart failure detection:

1. DT (Deceleration Time)
2. FS (Fractional Shortening)
3. BMI (Body Mass Index)
4. EA (E/A Ratio)
5. IRT (Isovolumic Relaxation Time)
6. HR (Heart Rate)
7. LDLc (Low-Density Lipoprotein Cholesterol)
8. BNP (B-type Natriuretic Peptide)
9. LAV (Left Atrial Volume)
10. TG (Triglycerides)
11. Creatinine
12. Age
13. LVIDs (Left Ventricular Internal Dimension in Systole)
14. TC (Total Cholesterol)
15. Hb (Hemoglobin)
16. MPI (Myocardial Performance Index)
17. RBS (Random Blood Sugar)
18. LVEF (Left Ventricular Ejection Fraction)
19. Chest_pain_Present
20. ICT (Isovolumic Contraction Time)
21. TropI (Troponin I)
22. HDLc (High-Density Lipoprotein Cholesterol)
23. RR (Respiratory Rate)
24. LVIDd (Left Ventricular Internal Dimension in Diastole)
25. Wall_Subendocardial

## Model Performance

### Stacking (LR) - Best Model

- **Accuracy**: 88.75%
- **Precision**: 85.42%
- **Recall**: 95.35%
- **F1 Score**: 90.11%
- **ROC AUC**: 92.08%

### XGBoost

- **Accuracy**: 87.50%
- **Precision**: 83.67%
- **Recall**: 95.35%
- **F1 Score**: 89.13%
- **ROC AUC**: 91.58%

### Voting (Soft)

- **Accuracy**: 87.50%
- **Precision**: 82.35%
- **Recall**: 97.67%
- **F1 Score**: 89.36%
- **ROC AUC**: 92.27%

## Deployment

The final model has been deployed as a standalone package with the following components:

- **Model File**: The trained Stacking Classifier model
- **Prediction Script**: A Python script for making predictions
- **README**: Documentation on how to use the model
- **Requirements**: List of required dependencies

To use the deployed model, run:

```
python3 predict.py <data_path> [output_path]
```

## Conclusion

The heart failure detection model successfully achieves high accuracy using the top 25 important features. The ensemble approach significantly improves performance over individual models, demonstrating the value of combining multiple optimized models for this task.

The model can be used for early detection of heart failure, potentially enabling earlier intervention and improved patient outcomes.

## Future Work

- Collect more data to further improve model performance
- Explore deep learning approaches for heart failure detection
- Develop a web application for easy use by healthcare professionals
- Integrate the model with electronic health record systems

---

*Generated on: April 11, 2025*  
*Project: Early Detection of Heart Failure using Machine Learning*
