import os
import sys
import tensorflow as tf
from src.exception import CustomException
from src.logger import logging
from keras_nlp.models import BertBackbone, BertPreprocessor


class model_build():

    def build_model(self, num_classes=6):
        try:
            # Define input layers for preprocessed inputs
            text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text_input")

            preprocessor = BertPreprocessor.from_preset(
                "bert_base_en_uncased",
                sequence_length=128, 
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

            # Defining the model
            model = tf.keras.Model(
                inputs=text_input,
                outputs=classifier,
            )

            # Compiling the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            logging.info("Model built successfully.")

            return model

        except Exception as e:
            raise CustomException(e, sys)