# Advanced Data Transformation Component

This document provides an overview of the advanced data transformation component implemented in the Heart Failure Detection project.

## Overview

The advanced data transformation component is responsible for preparing the data for machine learning models. It includes several sophisticated techniques for handling missing values, encoding categorical variables, transforming skewed features, and handling imbalanced data.

## Features

### 1. Missing Value Handling
- Numeric features: Filled with median values
- Categorical features: Filled with mode (most frequent values)

### 2. Feature Transformation
- **Skewness Detection and Transformation**: 
  - Log transformation for positively skewed features (skewness > 0.5)
  - Square root transformation for negatively skewed features (skewness < -0.5)
  - Box-Cox transformation for non-normal distributions (p-value < 0.05)

### 3. Categorical Encoding
- **Label Encoding**: For ordinal categorical features
- **One-Hot Encoding**: For nominal categorical features

### 4. Feature Scaling
- **StandardScaler**: For numeric features
- **RobustScaler**: Optional for handling outliers

### 5. Imbalanced Data Handling
- **SMOTEENN**: Combines SMOTE (Synthetic Minority Over-sampling Technique) with Edited Nearest Neighbors to handle imbalanced datasets

## Usage

To use the advanced data transformation component:

```python
from HF.components.data_transformation import DataTransformation
from HF.entity.config_entity import DataTransformationConfig
from HF.entity.artifact_entity import DataIngestionArtifact

# Create configuration
data_transformation_config = DataTransformationConfig()

# Create data transformation component
data_transformation = DataTransformation(
    data_transformation_config=data_transformation_config,
    data_ingestion_artifact=data_ingestion_artifact  # From data ingestion step
)

# Run data transformation
data_transformation_artifact = data_transformation.initiate_data_transformation()

# Access transformed data
transformed_train_file = data_transformation_artifact.transformed_train_file_path
transformed_test_file = data_transformation_artifact.transformed_test_file_path
preprocessor = data_transformation_artifact.preprocessed_object_file_path
```

## Configuration

The data transformation component can be configured through the `DataTransformationConfig` class:

- **Numeric Features**: Features to be treated as numeric
- **Ordinal Features**: Categorical features with an inherent order
- **One-Hot Features**: Categorical features without an inherent order
- **Target Column**: The column to predict
- **Columns to Drop**: Columns to exclude from the transformation

## Output

The data transformation component produces:

1. **Transformed Training Data**: CSV and NumPy array files
2. **Transformed Testing Data**: CSV and NumPy array files
3. **Preprocessing Object**: Serialized preprocessing pipeline for use in inference

## Implementation Details

### Handling Skewed Features

```python
def detect_and_transform_skewness(data):
    transformed_data = data.copy()
    for column in data.select_dtypes(include=[np.number]).columns:
        feature_skewness = skew(data[column].dropna())
        _, p_value = normaltest(data[column].dropna())

        if feature_skewness > 0.5:
            # Apply log transformation for positive skewness
            transformed_data[column] = np.log1p(data[column])
        elif feature_skewness < -0.5:
            # Apply sqrt transformation for negative skewness
            transformed_data[column] = np.sqrt(data[column])
        elif p_value < 0.05:
            # Apply Box-Cox transformation for non-normal distributions
            transformed_data[column], _ = boxcox(data[column] + 1)

    return transformed_data
```

### Handling Imbalanced Data

```python
def handle_imbalanced_data(X, y):
    # Convert target to numeric if needed
    if not np.issubdtype(y.dtype, np.number):
        unique_values = np.unique(y)
        value_map = {val: i for i, val in enumerate(unique_values)}
        y_numeric = np.array([value_map[val] for val in y])
    else:
        y_numeric = y
    
    # Apply SMOTEENN
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y_numeric)
    
    return X_resampled, y_resampled
```

## Benefits

1. **Improved Model Performance**: Properly transformed features lead to better model performance
2. **Handling of Non-Normal Distributions**: Transformations help normalize skewed features
3. **Robust to Outliers**: Appropriate scaling methods reduce the impact of outliers
4. **Balanced Dataset**: SMOTEENN helps address class imbalance issues
5. **Consistent Preprocessing**: Ensures consistent transformation during training and inference
