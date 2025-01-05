import tensorflow as tf
import keras_nlp as kn
import keras
import tensorflow_text as tft
import tensorflow_hub as tfh

print(keras.__version__)
print(kn.__version__)
print(tf.__version__)
print(tft.__version__)
print(tfh.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
