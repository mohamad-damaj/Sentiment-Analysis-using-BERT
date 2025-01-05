import os
import sys
import pandas as pd
from src.exception import CustomException, logging
import tensorflow_text as text
import tensorflow as tf
import keras_nlp

class PredictPipeline:
    def __init__(self, model_dir="artifact/model_tf", class_labels=None):
        """
        Initializes the prediction pipeline.

        :param model_dir: Directory where the SavedModel is stored.
        :param class_labels: Dictionary mapping class indices to labels.
        """
        self.model_dir = model_dir
        self.class_labels = class_labels
        self.model = self.load_model()

    def load_model(self):
        """
        Loads the TensorFlow SavedModel.

        :return: Loaded Keras model.
        """
        try:
            # Define the custom objects dictionary
            custom_objects = {
                "BertPreprocessor": keras_nlp.models.BertPreprocessor,
                "BertBackbone": keras_nlp.models.BertBackbone  # Include if used
            }

            # Verify that the model directory exists
            if not os.path.exists(self.model_dir):
                raise CustomException(f"Model directory '{self.model_dir}' does not exist.", sys)

            # Load the model with custom_objects
            logging.info(f"Loading model from: {self.model_dir}")
            model = tf.keras.models.load_model(
                self.model_dir,
                custom_objects=custom_objects,
                compile=False  # Set to True if you need to compile the model upon loading
            )
            logging.info("Model loaded successfully.")
            return model

        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise CustomException(e, sys)

    def predict(self, texts):
        """
        Makes predictions on the provided texts.

        :param texts: List of text inputs.
        :return: List of predicted class labels or indices.
        """
        try:
            # Validate input
            logging.info(f"Input is {texts}")
            if not isinstance(texts, list):
                raise CustomException("Input must be a list of strings.", sys)

            if any(not isinstance(text, str) for text in texts):
                raise CustomException("All items in the input list must be strings.", sys)

            # Ensure input is not empty
            if len(texts) == 0:
                raise CustomException("Input list is empty.", sys)

            logging.info(f"Input texts for prediction: {texts}")

            # Make predictions
            logging.info(f"Input to predict method: {texts}")
            preds = self.model.predict(texts)
            logging.info(f"Raw model predictions: {preds}")

            logging.info(f"Raw predictions: {preds}")

            # Convert predictions to class indices
            predicted_indices = tf.argmax(preds, axis=1).numpy()
            logging.info(f"Predicted class indices: {predicted_indices}")

            # Map indices to labels if provided
            if self.class_labels:
                predicted_labels = [self.class_labels.get(idx, "Unknown") for idx in predicted_indices]
                logging.info(f"Predicted class labels: {predicted_labels}")
                return predicted_labels
            else:
                return predicted_indices

        except Exception as e:
            logging.error(f"Error during prediction: {e}: The input text is {texts}")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, text: str):
        self.text = text

    def get_data_as_list(self):
        try:
            return [self.text]
        except Exception as e:
            raise CustomException(e, sys)
