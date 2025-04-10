import os
import sys
from HF.logger import logging
from HF.exception import HFException
from HF.entity.config_entity import DataIngestionConfig, DataProfilingConfig
from HF.entity.artifact_entity import DataIngestionArtifact
from HF.components.data_ingestion import DataIngestion
from HF.components.data_profiling import DataProfiling

def main():
    try:
        # First, run data ingestion to get the data
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        # Now, run data profiling
        data_profiling_config = DataProfilingConfig()
        data_profiling = DataProfiling(
            data_profiling_config=data_profiling_config,
            data_ingestion_artifact=data_ingestion_artifact
        )
        data_profiling_artifact = data_profiling.initiate_data_profiling()
        
        logging.info(f"Data profiling completed successfully")
        logging.info(f"Train profile report: {data_profiling_artifact.train_profile_report_file_path}")
        logging.info(f"Test profile report: {data_profiling_artifact.test_profile_report_file_path}")
        logging.info(f"Full profile report: {data_profiling_artifact.full_profile_report_file_path}")
        
        # Print instructions for viewing the reports
        print("\n" + "="*80)
        print("Data profiling completed successfully!")
        print("Profile reports are available at:")
        print(f"  - Train profile report: {data_profiling_artifact.train_profile_report_file_path}")
        print(f"  - Test profile report: {data_profiling_artifact.test_profile_report_file_path}")
        print(f"  - Full profile report: {data_profiling_artifact.full_profile_report_file_path}")
        print("\nTo view the reports, open the HTML files in your web browser.")
        print("="*80 + "\n")
        
        return data_profiling_artifact
    
    except Exception as e:
        logging.error(f"Error in data profiling: {e}")
        raise HFException(e, sys)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error in main: {e}")
        print(f"Error: {e}")
