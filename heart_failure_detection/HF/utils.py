import os
import sys
import dill
import yaml
import pandas as pd
from HF.exception import HFException
from HF.logger import logging

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    """
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        logging.error(f"Error reading YAML file: {file_path}")
        raise HFException(e, sys)

def write_yaml_file(file_path: str, content: dict):
    """
    Writes a dictionary to a YAML file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file, default_flow_style=False)
        logging.info(f"Successfully written YAML file: {file_path}")
    except Exception as e:
        logging.error(f"Error writing YAML file: {file_path}")
        raise HFException(e, sys)

def save_object(file_path: str, obj):
    """
    Saves an object to a file using dill.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Successfully saved object to: {file_path}")
    except Exception as e:
        logging.error(f"Error saving object to: {file_path}")
        raise HFException(e, sys)

def load_object(file_path: str):
    """
    Loads an object from a file using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.error(f"Error loading object from: {file_path}")
        raise HFException(e, sys)
