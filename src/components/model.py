import os
import sys
import tensorflow as tf
from dataclasses import dataclass, field

from src.exception import CustomException
from src.logger import logging
from keras_nlp.models import BertBackbone, BertPreprocessor
 


@dataclass
class ModelConfig:
    """
    Configuration dataclass for building the classification model.
    """
    preset_name: str = "bert_base_en_uncased"
    sequence_length: int = 128
    train_encoder: bool = True
    train_preprocessor: bool = False
    load_weights: bool = True

    dropout_rate: float = 0.1
    learning_rate: float = 1e-5
    num_classes: int = 6



class model_builder:

    def __init__(self, config: ModelConfig):
        """
        Initialize the ModelBuilder with a given configuration.

        :param config: ModelConfig object containing all relevant hyperparameters.
        """
        self.config = config

    def build_model(self):
        try:
            logging.info("Starting to build the BERT classification model...")


            text_input = tf.keras.layers.Input(
                shape=(), 
                dtype=tf.string, 
                name="text_input"
            )


            logging.info(
                f"Using BertPreprocessor with preset={self.config.preset_name}, "
                f"sequence_length={self.config.sequence_length}, "
                f"trainable={self.config.train_preprocessor}"
            )
            preprocessor = BertPreprocessor.from_preset(
                self.config.preset_name,
                sequence_length=self.config.sequence_length,
                trainable=self.config.train_preprocessor
            )

            encoder_inputs = preprocessor(text_input)

            logging.info(
                f"Using BertBackbone with preset={self.config.preset_name}, "
                f"load_weights={self.config.load_weights}, "
                f"trainable={self.config.train_encoder}"
            )
            encoder = BertBackbone.from_preset(
                self.config.preset_name,
                load_weights=self.config.load_weights,
            )
            encoder.trainable = self.config.train_encoder

            outputs = encoder(encoder_inputs)
            pooled_output = outputs["pooled_output"] 

            dropout = tf.keras.layers.Dropout(self.config.dropout_rate)(pooled_output)
            classifier = tf.keras.layers.Dense(
                self.config.num_classes, 
                activation="softmax", 
                name="classifier"
            )(dropout)

            model = tf.keras.Model(inputs=text_input, outputs=classifier)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.config.learning_rate
                ),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            logging.info("BERT classification model built and compiled successfully.")
            logging.debug(f"Model summary:\n{model.summary()}")
            return model

        except Exception as e:
            logging.error("An error occurred while building the model.", exc_info=True)

            raise CustomException(e, sys)