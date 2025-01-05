import os
import sys
import tensorflow as tf
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, save_model
from keras_nlp.models import BertBackbone, BertPreprocessor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model_tf")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def build_model(self, num_classes=6):
        try:
            # Define input layers for preprocessed inputs
            text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text_input")

            preprocessor = BertPreprocessor.from_preset(
                "bert_base_en_uncased",
                sequence_length=128,  # Adjust if you need longer sequences
                # Typically not trainable, but you can set it if you want
                trainable=False
            )

            encoder_inputs = preprocessor(text_input)

            encoder = BertBackbone.from_preset(
                "bert_base_en_uncased",
                load_weights=True,
                trainable=True
            )

            # Extract the BERT outputs
            outputs = encoder(encoder_inputs)
            pooled_output = outputs["pooled_output"]

            # Add dropout and classification layers
            dropout = tf.keras.layers.Dropout(0.1)(pooled_output)
            classifier = tf.keras.layers.Dense(num_classes, activation="softmax")(dropout)

            # Define the model
            model = tf.keras.Model(
                inputs=text_input,
                outputs=classifier,
            )

            # Compile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            logging.info("Model built successfully.")

            return model

        except Exception as e:
            raise CustomException(e, sys)


    def train_model(self, train_data, test_data, epochs=2, batch_size=32):
        try:

            # Unpack the training and test data
            X_train, y_train = train_data
            X_test, y_test = test_data



            X_train_small = X_train[:100]
            y_train_small = y_train[:100]
            X_test_small = X_test[:20]
            y_test_small = y_test[:20]



            logging.info("Building the model.")
            model = self.build_model()

            logging.info("Starting model training.")
            history = model.fit(
                X_train_small,
                y_train_small,
                validation_data=(X_test_small, y_test_small),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )

            logging.info("Evaluating the model.")
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

            logging.info(f"Model Evaluation: Accuracy = {test_accuracy:.4f}")

            #if test_accuracy < 0.6:
                #raise CustomException("Model accuracy is too low. Training failed.")

            logging.info("Saving the trained model.")
            save_model(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info("Model saved successfully.")
            return test_accuracy
        

        except Exception as e:
            raise CustomException(e, sys)
