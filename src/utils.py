import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
import tensorflow as tf

from src.exception import CustomException, logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def save_model(file_path, obj):
    try:
        if os.path.exists(file_path):
            logging.info(f"Model directory '{file_path}' already exists. Removing it to allow overwriting.")
            shutil.rmtree(file_path)  # Remove the existing directory
        
        # Ensure the parent directory exists
        parent_dir = os.path.dirname(file_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            logging.info(f"Created parent directory: {parent_dir}")
        
        # Save the model in TensorFlow 'SavedModel' format
        obj.save(file_path, save_format='tf')
        logging.info(f"Model saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)