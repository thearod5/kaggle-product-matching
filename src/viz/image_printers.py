import os

import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img
from mpl_toolkits.axes_grid1 import ImageGrid

from meta.paths import PATH_TO_TRAIN, PATH_TO_TRAINING_IMAGES


def get_images_for(df, image_type: str):
    query = df[df["type"] == image_type]
    entry = query.sample().iloc[0]
    return entry["source_image"], entry["target_image"]


def print_image_pair(image_pair):
    source_name, target_name = image_pair

    path_to_source = os.path.join(PATH_TO_TRAINING_IMAGES, source_name)
    path_to_target = os.path.join(PATH_TO_TRAINING_IMAGES, target_name)

    source_image = img_to_array(load_img(path_to_source))
    target_image = img_to_array(load_img(path_to_target))

    print_image_matrices((source_image, target_image))


def print_image_matrices(matrices):
    source_matrix, target_matrix = matrices
    fig = plt.figure(figsize=(8., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, 2),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    grid[0].imshow(source_matrix.astype(int))
    grid[1].imshow(target_matrix.astype(int))


def get_index_of_image_type(df, image_type: str):
    image_type_index = df[df["type"] == image_type].sample(n=1).index[0]

    batch_index = image_type_index // 32
    sample_index = image_type_index % 32
    return batch_index, sample_index


def print_image_type(df, data_generator, image_type: str):
    batch_index, sample_index = get_index_of_image_type(df, image_type)
    [test_source, test_target], test_label = data_generator[batch_index]
    print_image_matrices((test_source[sample_index], test_target[sample_index]))
    return batch_index, sample_index


original_train = pd.read_csv(PATH_TO_TRAIN).set_index("image")


def get_image_title(image_name: str):
    query = original_train.loc[image_name]["title"]
    if isinstance(query, str):
        return query
    return query[0]


def print_titles(train_df, indices):
    batch_index, sample_index = indices
    entry = train_df.iloc[batch_index * 32 + sample_index]
    source_title = get_image_title(entry['source_image'])
    target_title = get_image_title(entry['target_image'])
    return source_title, target_title
