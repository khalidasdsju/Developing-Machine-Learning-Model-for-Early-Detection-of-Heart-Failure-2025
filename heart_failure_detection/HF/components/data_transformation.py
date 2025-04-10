import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN
from scipy.stats import skew, normaltest, boxcox
import matplotlib.pyplot as plt

from HF.logger import logging
from HF.exception import HFException
from HF.entity.config_entity import DataTransformationConfig
from HF.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from HF.utils import save_object, save_numpy_array_data

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(f"{'='*20}Data Transformation log started.{'='*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise HFException(e, sys)

    def get_data_transformer_object(self):
        """
        Get the data transformer object
        """
        try:
            logging.info("Creating data transformer object")

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Filter numeric features that are actually in the dataset
            available_num_features = [col for col in self.data_transformation_config.num_features
                                     if col in self.data_transformation_config.available_columns]

            # Create a list of transformers
            transformers = []

            # Add numeric pipeline if there are numeric features
            if available_num_features:
                transformers.append(("num_pipeline", num_pipeline, available_num_features))
                logging.info(f"Added numeric pipeline for columns: {available_num_features}")

            # Add categorical pipelines if there are categorical features
            if self.data_transformation_config.or_columns:
                transformers.append(("cat_pipeline", cat_pipeline, self.data_transformation_config.or_columns))
                logging.info(f"Added categorical pipeline for columns: {self.data_transformation_config.or_columns}")

            if self.data_transformation_config.oh_columns:
                transformers.append(("oh_pipeline", cat_pipeline, self.data_transformation_config.oh_columns))
                logging.info(f"Added one-hot pipeline for columns: {self.data_transformation_config.oh_columns}")

            # Create the column transformer
            preprocessor = ColumnTransformer(transformers, remainder='passthrough')

            return preprocessor
        except Exception as e:
            raise HFException(e, sys)

    def detect_and_transform_skewness(self, data):
        """
        Detect and transform skewed features
        """
        try:
            logging.info("Detecting and transforming skewed features")
            transformed_data = data.copy()

            for column in data.select_dtypes(include=[np.number]).columns:
                if column == self.data_transformation_config.target_column:
                    continue

                feature_skewness = skew(data[column].dropna())

                # Check if there are enough samples for normaltest (at least 8)
                if len(data[column].dropna()) >= 8:
                    _, p_value = normaltest(data[column].dropna())
                else:
                    p_value = 0.05  # Default to not significant if not enough samples

                if feature_skewness > 0.5:
                    logging.info(f"Applying log transformation to {column} (skewness: {feature_skewness:.2f})")
                    # Add a small constant to avoid log(0)
                    min_val = data[column].min()
                    if min_val <= 0:
                        transformed_data[column] = np.log1p(data[column] - min_val + 1)
                    else:
                        transformed_data[column] = np.log1p(data[column])
                elif feature_skewness < -0.5:
                    logging.info(f"Applying sqrt transformation to {column} (skewness: {feature_skewness:.2f})")
                    # Ensure data is positive for sqrt
                    min_val = data[column].min()
                    if min_val < 0:
                        transformed_data[column] = np.sqrt(data[column] - min_val + 1)
                    else:
                        transformed_data[column] = np.sqrt(data[column])
                elif p_value < 0.05:
                    logging.info(f"Applying Box-Cox transformation to {column} (p-value: {p_value:.4f})")
                    # Ensure data is positive for Box-Cox
                    min_val = data[column].min()
                    if min_val <= 0:
                        shifted_data = data[column] - min_val + 1
                        try:
                            transformed_data[column], _ = boxcox(shifted_data)
                        except:
                            logging.warning(f"Box-Cox transformation failed for {column}, using original data")
                    else:
                        try:
                            transformed_data[column], _ = boxcox(data[column])
                        except:
                            logging.warning(f"Box-Cox transformation failed for {column}, using original data")

            return transformed_data
        except Exception as e:
            raise HFException(f"Error in detect_and_transform_skewness: {e}", sys)

    def encode_categorical_columns(self, train_df, test_df):
        """
        Encode categorical columns
        """
        try:
            logging.info("Encoding categorical columns")

            # Label encoding for ordinal columns
            le = LabelEncoder()
            for col in self.data_transformation_config.or_columns:
                if col in train_df.columns and col in test_df.columns:
                    # Combine train and test data for fitting to ensure consistent encoding
                    combined_data = pd.concat([train_df[col], test_df[col]])
                    le.fit(combined_data)

                    train_df[col] = le.transform(train_df[col])
                    test_df[col] = le.transform(test_df[col])

                    logging.info(f"Label encoded column: {col}")

            # One-hot encoding for nominal columns
            for col in self.data_transformation_config.oh_columns:
                if col in train_df.columns and col in test_df.columns:
                    # Get all unique categories from both train and test
                    all_categories = pd.concat([train_df[col], test_df[col]]).unique()

                    # Create dummy variables for train and test
                    train_dummies = pd.get_dummies(train_df[col], prefix=col, drop_first=True)
                    test_dummies = pd.get_dummies(test_df[col], prefix=col, drop_first=True)

                    # Ensure both have the same columns
                    for category in all_categories[1:]:  # Skip the first category (drop_first=True)
                        dummy_col = f"{col}_{category}"
                        if dummy_col not in train_dummies.columns:
                            train_dummies[dummy_col] = 0
                        if dummy_col not in test_dummies.columns:
                            test_dummies[dummy_col] = 0

                    # Drop original column and add dummy variables
                    train_df = train_df.drop(columns=[col])
                    test_df = test_df.drop(columns=[col])

                    train_df = pd.concat([train_df, train_dummies], axis=1)
                    test_df = pd.concat([test_df, test_dummies], axis=1)

                    logging.info(f"One-hot encoded column: {col}")

            return train_df, test_df
        except Exception as e:
            raise HFException(f"Error in encode_categorical_columns: {e}", sys)

    def scale_features(self, X_train, X_test):
        """
        Scale features using RobustScaler
        """
        try:
            logging.info("Scaling features using RobustScaler")
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            return X_train_scaled, X_test_scaled, scaler
        except Exception as e:
            raise HFException(f"Error in scale_features: {e}", sys)

    def handle_imbalanced_data(self, X, y):
        """
        Handle imbalanced data using SMOTEENN
        """
        try:
            logging.info("Handling imbalanced data using SMOTEENN")

            # Convert y to numeric if it's not already
            if not np.issubdtype(y.dtype, np.number):
                logging.info(f"Converting target to numeric. Current dtype: {y.dtype}")
                # If y is categorical, convert to numeric
                if y.dtype == np.dtype('O'):
                    # Map unique values to integers
                    unique_values = np.unique(y)
                    value_map = {val: i for i, val in enumerate(unique_values)}
                    y_numeric = np.array([value_map[val] for val in y])
                    logging.info(f"Mapped values: {value_map}")
                else:
                    y_numeric = y.astype(int)
            else:
                y_numeric = y

            # Apply SMOTEENN
            try:
                smote_enn = SMOTEENN(random_state=42)
                X_resampled, y_resampled = smote_enn.fit_resample(X, y_numeric)

                logging.info(f"Original class distribution: {np.bincount(y_numeric)}")
                logging.info(f"Resampled class distribution: {np.bincount(y_resampled)}")

                return X_resampled, y_resampled
            except Exception as e:
                logging.warning(f"SMOTEENN failed: {e}. Returning original data.")
                return X, y_numeric
        except Exception as e:
            logging.error(f"Error in handle_imbalanced_data: {e}")
            # Return original data if there's an error
            return X, y

    def initiate_data_transformation(self):
        """
        Initiate the data transformation process
        """
        try:
            logging.info("Starting data transformation")

            # Create directories for transformed data
            logging.info("Creating directories for transformed data")
            os.makedirs(self.data_transformation_config.transformed_train_dir, exist_ok=True)
            os.makedirs(self.data_transformation_config.transformed_test_dir, exist_ok=True)
            os.makedirs(self.data_transformation_config.preprocessing_dir, exist_ok=True)

            # Get the training and testing file paths
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read the training and testing data
            logging.info("Reading training and testing data")
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)

            # Clean column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            logging.info(f"Training dataframe shape: {train_df.shape}")
            logging.info(f"Testing dataframe shape: {test_df.shape}")

            # Drop unnecessary columns
            logging.info("Dropping unnecessary columns")
            train_df = train_df.drop(columns=self.data_transformation_config.drop_columns, errors='ignore')
            test_df = test_df.drop(columns=self.data_transformation_config.drop_columns, errors='ignore')

            logging.info(f"Columns dropped: {self.data_transformation_config.drop_columns}")

            # Handle missing values
            logging.info("Handling missing values")

            # Handle numeric columns with median
            numeric_columns = train_df.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                train_df[col].fillna(train_df[col].median(), inplace=True)
                test_df[col].fillna(test_df[col].median(), inplace=True)

            # Handle categorical columns with mode
            categorical_columns = train_df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                train_df[col].fillna(train_df[col].mode()[0], inplace=True)
                test_df[col].fillna(test_df[col].mode()[0], inplace=True)

            # Remove duplicate columns
            train_df = train_df.loc[:, ~train_df.columns.duplicated()]
            test_df = test_df.loc[:, ~test_df.columns.duplicated()]

            # Remove duplicate rows
            train_df = train_df.drop_duplicates()
            test_df = test_df.drop_duplicates()

            # Check if target column exists
            target_column = self.data_transformation_config.target_column
            if target_column not in train_df.columns or target_column not in test_df.columns:
                raise ValueError(f"Target column '{target_column}' is missing! Available columns: {train_df.columns.tolist()}")

            # Separate features and target
            logging.info("Separating features and target")
            target_feature_train_df = train_df[target_column]
            target_feature_test_df = test_df[target_column]

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)

            # Apply skewness transformation
            logging.info("Applying skewness transformation")
            input_feature_train_df = self.detect_and_transform_skewness(input_feature_train_df)
            input_feature_test_df = self.detect_and_transform_skewness(input_feature_test_df)

            # Check available columns
            available_columns = input_feature_train_df.columns.tolist()
            logging.info(f"Available columns in train_df: {available_columns}")
            logging.info(f"Available columns in test_df: {input_feature_test_df.columns.tolist()}")

            # Set available columns in the configuration
            self.data_transformation_config.available_columns = available_columns

            # Filter columns based on what's available in the dataset
            available_or_columns = [col for col in self.data_transformation_config.or_columns if col in available_columns]
            available_oh_columns = [col for col in self.data_transformation_config.oh_columns if col in available_columns]

            logging.info(f"Available ordinal columns: {available_or_columns}")
            logging.info(f"Available one-hot columns: {available_oh_columns}")

            # Update the configuration
            self.data_transformation_config.or_columns = available_or_columns
            self.data_transformation_config.oh_columns = available_oh_columns

            # Encode categorical columns if available
            if available_or_columns or available_oh_columns:
                logging.info("Encoding categorical columns")
                input_feature_train_df, input_feature_test_df = self.encode_categorical_columns(
                    input_feature_train_df, input_feature_test_df
                )
            else:
                logging.warning("No categorical columns found for encoding")

            # Get preprocessing object
            logging.info("Getting preprocessing object")

            # Check if there are any columns to transform
            if not self.data_transformation_config.available_columns:
                logging.warning("No columns available for transformation, using identity transformer")
                # Use identity transformer
                from sklearn.preprocessing import FunctionTransformer
                preprocessor = FunctionTransformer(lambda X: X, validate=False)
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df.values)
                input_feature_test_arr = preprocessor.transform(input_feature_test_df.values)
            else:
                # Use the regular preprocessor
                preprocessor = self.get_data_transformer_object()

                # Apply preprocessing
                logging.info("Applying preprocessing")
                try:
                    input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                    input_feature_test_arr = preprocessor.transform(input_feature_test_df)
                except Exception as e:
                    logging.error(f"Error applying preprocessing: {e}")
                    logging.info("Falling back to identity transformer")
                    # Use identity transformer as fallback
                    from sklearn.preprocessing import FunctionTransformer
                    preprocessor = FunctionTransformer(lambda X: X, validate=False)
                    input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df.values)
                    input_feature_test_arr = preprocessor.transform(input_feature_test_df.values)

            # Handle imbalanced data
            logging.info("Handling imbalanced data")
            input_feature_train_arr, target_feature_train_arr = self.handle_imbalanced_data(
                input_feature_train_arr, np.array(target_feature_train_df)
            )

            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save transformed data
            logging.info("Saving transformed data")
            save_numpy_array_data(self.data_transformation_config.transformed_train_array_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_array_file_path, test_arr)

            # Save preprocessing objects
            logging.info("Saving preprocessing objects")
            save_object(self.data_transformation_config.preprocessed_object_file_path, preprocessor)

            # Save transformed data as CSV for easier inspection
            pd.DataFrame(train_arr).to_csv(self.data_transformation_config.transformed_train_file_path, index=False)
            pd.DataFrame(test_arr).to_csv(self.data_transformation_config.transformed_test_file_path, index=False)

            # Create and return the data transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessed_object_file_path=self.data_transformation_config.preprocessed_object_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_array_file_path=self.data_transformation_config.transformed_train_array_file_path,
                transformed_test_array_file_path=self.data_transformation_config.transformed_test_array_file_path
            )

            logging.info(f"Data transformation completed successfully")
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")

            return data_transformation_artifact

        except Exception as e:
            raise HFException(e, sys)


class DropColumns:
    """
    Custom transformer to drop specified columns
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')
