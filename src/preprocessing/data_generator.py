"""
The following module is responsible for generating image pairs and their associated class labels.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from meta.paths import PATH_TO_TRAINING_IMAGES

BATCH_SIZE = 32
SEED = 0
IMAGE_SIZE = (256, 256)

"""
DATA CONSTANTS
"""
Y_COL = "class_label"


def build_generator():
    """
    Creates ImageDataGenerator performing randome the data augmentations.
    """
    return ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=15,
        width_shift_range=5.0,
        height_shift_range=5.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        dtype=None
    )


def create_df_generator(df,
                        col_name,
                        batch_size,
                        image_size):
    """
    Creates DataFrameIterator that returns images from the column specified.
    : param: df - DataFrame containing image names
    : param: col_name - either [source_image] or [target_image]
    : returns: DataFrameIterator
    """
    df_generator = build_generator()
    df[Y_COL] = df[Y_COL].astype(str)
    return df_generator.flow_from_dataframe(
        df,
        directory=PATH_TO_TRAINING_IMAGES,
        x_col=col_name,
        y_col=Y_COL,
        target_size=image_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
        seed=SEED,
        subset="training",
        interpolation="nearest",
        validate_filenames=True)


class CustomGen(tf.keras.utils.Sequence):
    """
    The following class is responsible for generator joint source and target images alongside
    their corresponding label (1 if related otherwise 0).
    TODO: Implement slicing
    """

    def __init__(self, df,
                 shuffle=False,
                 batch_size=BATCH_SIZE,
                 image_size=IMAGE_SIZE):
        if shuffle:
            df = df.sample(frac=1)
        self.source_gen = create_df_generator(df, "source_image", batch_size, image_size)
        self.target_gen = create_df_generator(df, "target_image", batch_size, image_size)
        assert len(self.source_gen) == len(self.target_gen)

    def __len__(self):
        return len(self.source_gen)

    def __getitem__(self, i):
        if isinstance(i, int) and i > len(self.source_gen):
            raise ValueError("Reached end of custom data generator.")
        x1, y1 = self.source_gen[i]
        x2, y2 = self.target_gen[i]
        return [x1, x2], y1

    def on_epoch_end(self):
        self.source_gen.on_epoch_end()
        self.target_gen.on_epoch_end()
        self.target_gen.index_array = self.source_gen.index_array
