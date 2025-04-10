import os
import sys
from HF.logger import logging
from HF.exception import HFException
from HF.entity.config_entity import DataIngestionConfig
from HF.components.data_ingestion import DataIngestion

def test_data_ingestion():
    try:
        # Create a data ingestion config
        data_ingestion_config = DataIngestionConfig()
        
        # Create a data ingestion object
        data_ingestion = DataIngestion(data_ingestion_config)
        
        # Check if the sample data file exists
        sample_data_path = "/Users/khalid/Desktop/ML-Model-Deployment/artifact/04_07_2025_15_36_55/data_ingestion/data_ingestion/train.csv"
        if not os.path.exists(sample_data_path):
            logging.warning(f"Sample data file not found: {sample_data_path}")
            logging.info("Creating a dummy sample data file for testing")
            
            # Create a dummy directory
            os.makedirs(os.path.dirname(sample_data_path), exist_ok=True)
            
            # Create a dummy CSV file with some heart failure data
            import pandas as pd
            import numpy as np
            
            # Create dummy data
            np.random.seed(42)
            n_samples = 100
            
            data = {
                'age': np.random.randint(40, 90, n_samples),
                'sex': np.random.randint(0, 2, n_samples),
                'cp': np.random.randint(0, 4, n_samples),
                'trestbps': np.random.randint(90, 200, n_samples),
                'chol': np.random.randint(120, 400, n_samples),
                'fbs': np.random.randint(0, 2, n_samples),
                'restecg': np.random.randint(0, 3, n_samples),
                'thalach': np.random.randint(70, 200, n_samples),
                'exang': np.random.randint(0, 2, n_samples),
                'oldpeak': np.round(np.random.uniform(0, 6, n_samples), 1),
                'slope': np.random.randint(0, 3, n_samples),
                'ca': np.random.randint(0, 4, n_samples),
                'thal': np.random.randint(0, 3, n_samples),
                'target': np.random.randint(0, 2, n_samples)
            }
            
            df = pd.DataFrame(data)
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(sample_data_path), exist_ok=True)
            
            # Save the dummy data
            df.to_csv(sample_data_path, index=False)
            logging.info(f"Dummy data saved to {sample_data_path}")
        
        # Initiate data ingestion
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        logging.info(f"Data ingestion completed successfully")
        logging.info(f"Train file: {data_ingestion_artifact.trained_file_path}")
        logging.info(f"Test file: {data_ingestion_artifact.test_file_path}")
        
        return data_ingestion_artifact
    
    except Exception as e:
        raise HFException(e, sys)

if __name__ == "__main__":
    try:
        test_data_ingestion()
    except Exception as e:
        logging.error(f"Error in test_data_ingestion: {e}")
        print(f"Error: {e}")
