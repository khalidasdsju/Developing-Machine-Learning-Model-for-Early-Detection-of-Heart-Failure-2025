import logging
import os
import sys
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(os.getcwd(), LOG_DIR, LOG_FILE)

os.makedirs(os.path.join(os.getcwd(), LOG_DIR), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def get_logger(name=__name__):
    """
    Returns a logger object with the given name
    """
    logger = logging.getLogger(name)

    # Add console handler to show logs in console as well
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

    return logger
