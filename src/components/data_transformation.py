import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from keras_nlp.models import BertPreprocessor
from src.exception import CustomException
from src.logger import logging
import tensorflow_text
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    

    def get_data_transformer_object(self):
        '''
        This function is responsible for text preprocessing for BERT-based models.
        '''
        try:
            # Setting up a the BERT preprocessor
            bert_preprocessor = BertPreprocessor.from_preset(
                "bert_base_en_uncased",
                sequence_length=128,
                trainable=False
            )
            
            logging.info("Created BERT Preprocessor")

            return bert_preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Loading training and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Balancing and preprocessing the data")

            train_grouped = train_df.groupby("label")
            min_count = train_grouped.size().min()

            balanced_train_df = train_grouped.apply(lambda x: x.sample(min_count, random_state=42))
            balanced_train_df = balanced_train_df.reset_index(drop=True)

            le = LabelEncoder()
            balanced_train_df["label"] = le.fit_transform(balanced_train_df["label"])
            test_df["label"] = le.transform(test_df["label"])  

            logging.info(f"Label Encoder Classes: {le.classes_}")

            X_train_text = balanced_train_df["text"].astype(str).to_numpy()
            y_train = balanced_train_df["label"].to_numpy()

            X_test_text = test_df["text"].astype(str).to_numpy()
            y_test = test_df["label"].to_numpy()

            logging.info("Preparing preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Preprocessing text data")

            X_train = X_train_text
            X_test = X_test_text

            train_arr = (X_train, y_train)
            test_arr = (X_test, y_test)



            return (
                train_arr,
                test_arr,
            )
        except Exception as e:
            raise CustomException(e, sys)