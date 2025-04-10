import os
import sys
import yaml
import json
import dill
import numpy as np
import pandas as pd
from datetime import datetime

from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    """
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        logger.error(f"Error reading YAML file: {file_path}")
        raise CustomException(e, sys)

def write_yaml_file(file_path: str, content: dict):
    """
    Writes a dictionary to a YAML file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file, default_flow_style=False)
        logger.info(f"Successfully written YAML file: {file_path}")
    except Exception as e:
        logger.error(f"Error writing YAML file: {file_path}")
        raise CustomException(e, sys)

def save_object(file_path: str, obj):
    """
    Saves an object to a file using dill.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logger.info(f"Successfully saved object to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving object to: {file_path}")
        raise CustomException(e, sys)

def load_object(file_path: str):
    """
    Loads an object from a file using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logger.error(f"Error loading object from: {file_path}")
        raise CustomException(e, sys)

def get_current_time_stamp():
    """
    Returns the current timestamp as a string.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def create_log_file_name(prefix="log"):
    """
    Creates a log file name with the given prefix and current timestamp.
    """
    timestamp = get_current_time_stamp()
    return f"{prefix}_{timestamp}.log"

def get_log_file_path(log_file_name, log_dir="logs"):
    """
    Returns the full path to a log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(os.getcwd(), log_dir, log_file_name)

def get_all_log_files(log_dir="logs"):
    """
    Returns a list of all log files in the log directory.
    """
    log_dir_path = os.path.join(os.getcwd(), log_dir)
    os.makedirs(log_dir_path, exist_ok=True)
    return [f for f in os.listdir(log_dir_path) if f.endswith(".log")]
