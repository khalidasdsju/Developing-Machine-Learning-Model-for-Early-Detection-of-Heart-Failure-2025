from src.pipeline.training_pipeline import TrainingPipeline
from src.logger import logging
from src.exception import CustomException
import sys

def main():
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
