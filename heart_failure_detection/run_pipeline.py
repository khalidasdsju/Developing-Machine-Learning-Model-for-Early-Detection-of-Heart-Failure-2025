import os
import sys
from HF.logger import logging
from HF.exception import HFException
from HF.pipeline.training_pipeline import TrainingPipeline

def main():
    try:
        # Create and run the training pipeline
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"Error in pipeline: {e}")
        raise HFException(e, sys)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error in main: {e}")
        print(f"Error: {e}")
