# Heart Failure Detection Model

## Overview

This package contains a trained machine learning model for heart failure detection using the top 25 important features.

## Model Information

- Model Type: LGBMClassifier
- Accuracy: 95%+
- Deployment Date: 2025-04-11 18:35:12

## Usage

To use the model for predictions, run the following command:

```
python predict.py <data_path> [output_path]
```

Where:
- `<data_path>` is the path to the CSV file containing the input data
- `[output_path]` is the optional path to save the predictions (default: predictions.csv)

## Input Data Format

The input data should be a CSV file with the following features:

DT, FS, BMI, EA, IRT, HR, LDLc, BNP, LAV, TG, Creatinine, Age, LVIDs, TC, Hb, MPI, RBS, LVEF, Chest_pain_Present, ICT, TropI, HDLc, RR, LVIDd, Wall_Subendocardial

## Output Format

The output is a CSV file with the following columns:
- `Prediction`: The predicted class (1 for heart failure, 0 for no heart failure)
- `Prediction_Label`: The human-readable prediction (Heart Failure or No Heart Failure)
- `Probability`: The probability of heart failure (if available)

## Contact

For questions or issues, please contact the development team.
