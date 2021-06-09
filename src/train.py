"""
Responsible for training model on google server
"""
import os
import sys

import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

PATH_TO_SRC = os.path.join(os.path.dirname(__file__))
sys.path.append(PATH_TO_SRC)

from meta.paths import PATH_TO_TRAIN_PAIRS
from preprocessing.data_generator import CustomGen

BATCH_SIZE = 32
IMAGE_SIZE = (100, 100)
EPOCHS = 100

if __name__ == "__main__":
    train_pairs_df = pd.read_csv(PATH_TO_TRAIN_PAIRS)
    train_generator = CustomGen(train_pairs_df,
                                shuffle=True,
                                batch_size=BATCH_SIZE,
                                image_size=IMAGE_SIZE)

    checkpoint_file_path = "model.h5"

    model = keras.models.load_model(checkpoint_file_path)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file_path,
        save_weights_only=True,
        mode='max',
        save_best_only=True,
        save_freq="epoch")

    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(checkpoint_file_path)
    history = model.fit(train_generator,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=[model_checkpoint_callback])
    model.save("model.h5")
