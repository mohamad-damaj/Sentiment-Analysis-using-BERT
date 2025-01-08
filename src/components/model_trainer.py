import os
import sys
import tensorflow as tf
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, save_model
from src.components.model import model_build


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model_tf")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def train_model(self, train_data, test_data, epochs=2, batch_size=32):
        try:

            # Unpack the training and test data
            X_train, y_train = train_data
            X_test, y_test = test_data

            logging.info("Building the model.")

            model_builder = model_build()
            model = model_builder.build_model()

            logging.info("Starting model training.")
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )

            logging.info("Evaluating the model.")
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

            logging.info(f"Model Evaluation: Accuracy = {test_accuracy:.4f}")

            if test_accuracy < 0.6:
                raise CustomException("Model accuracy is too low. Training failed.")

            logging.info("Saving the trained model.")
            save_model(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info("Model saved successfully.")
            return test_accuracy
        

        except Exception as e:
            raise CustomException(e, sys)
