import os
import sys
from src.logger import get_logger
from src.exception import CustomException
from src.utils.utils import (
    get_current_time_stamp,
    create_log_file_name,
    get_log_file_path,
    get_all_log_files
)

# Create a logger
logger = get_logger("test_logging")

def test_logging():
    """
    Test the logging functionality
    """
    try:
        logger.info("Starting the test_logging function")
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")
        
        # Get all log files
        log_files = get_all_log_files()
        logger.info(f"Log files in the logs directory: {log_files}")
        
        # Create a custom log file name
        custom_log_name = create_log_file_name(prefix="custom_test")
        logger.info(f"Created custom log file name: {custom_log_name}")
        
        # Get the path to the custom log file
        custom_log_path = get_log_file_path(custom_log_name)
        logger.info(f"Path to custom log file: {custom_log_path}")
        
        # Test exception handling
        # Uncomment the following line to test exception handling
        # raise Exception("Test exception")
        
        logger.info("Completed the test_logging function")
        return True
    except Exception as e:
        logger.error(f"Error in test_logging function: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        test_logging()
        print("Logging test completed successfully!")
        print(f"Log files are stored in the 'logs' directory: {os.path.join(os.getcwd(), 'logs')}")
    except Exception as e:
        print(f"Error: {e}")
