"""
The following module is responsible for creating training image pairs.
For more information about the methodology and reasoning of why and how this
is done see notebook `1. Create Training Data`
"""

import numpy as np
import pandas as pd

from text.title_similarity import calculate_title_similarities

MIN_SIMILARITY_THRESHOLD = 1 / 3.0
N_MOST_SIMILAR_PAIRS = 2


def create_negative_pairs(image_df: pd.DataFrame,
                          x_col: str = "image",
                          y_col: str = "label_group",
                          title_col: str = 'title',
                          n_top_examples: int = N_MOST_SIMILAR_PAIRS,
                          min_similarity_threshold: float = MIN_SIMILARITY_THRESHOLD):
    """
    For each image in df constructs image pairs with images the n most similar titles.
    :param image_df: DataFrame containing images (x_col) with post titles (title_col)
    :param x_col: the name of the column containing the image file names
    :param y_col: the name of the column containing image class label
    :param title_col: the name of the columns containing the title of the image posts.
    :param n_top_examples: the number of most similar examples to include in image pairs
    :param min_similarity_threshold: the minimum similarity that must image pairs must exhibit
    (except for dissimilar case)
    :return: DataFrame containing [source_image, target_image, type, class_label]
    where type is [UNRELATED_SIMILAR or [UNRELATED_NOTSIMILAR]
    """
    image_df = image_df.reset_index(drop=True)
    title2title_similarities = calculate_title_similarities(image_df[title_col], load_previous=False)

    negative_examples_df = pd.DataFrame()
    processed_tuples = []
    for source_index in range(len(image_df)):
        source_example = image_df.iloc[source_index]
        source_label = source_example[y_col]

        negative_examples = image_df[image_df[y_col] != source_label]
        negative_index = negative_examples.index
        negative_similarities = title2title_similarities[source_index, negative_index]
        sorted_negative_indices = np.argsort(negative_similarities)[::-1]

        # add n most similar negative examples
        n_examples = 0
        for similar_negative_index in sorted_negative_indices:
            if negative_similarities[similar_negative_index] < min_similarity_threshold:
                break
            similar_negative_example = image_df.iloc[similar_negative_index]
            image_pair = (source_example[x_col], similar_negative_example[x_col])
            image_pair_r = (image_pair[1], image_pair[0])
            if image_pair in processed_tuples or image_pair_r in processed_tuples:
                continue

            # example is similar enough and not previously done
            negative_examples_df = negative_examples_df.append({
                "source_image": image_pair[0],
                "target_image": image_pair[1],
                "type": "UNRELATED_SIMILAR",
                "class_label": 0
            }, ignore_index=True)
            n_examples += 1

            if n_examples >= n_top_examples:
                break

        # add a completely unrelated image
        least_similar_negative_index = negative_index[np.argmin(negative_similarities)]
        least_similar_negative = image_df.iloc[least_similar_negative_index]
        negative_examples_df = negative_examples_df.append({
            "source_image": source_example["image"],
            "target_image": least_similar_negative["image"],
            "type": "UNRELATED_NOTSIMILAR",
            "class_label": 0
        }, ignore_index=True)
    return negative_examples_df


def create_positive_pairs(image_df: pd.DataFrame,
                          x_col: str = "image",
                          y_col: str = "label_group"):
    """
    For each image in DataFrame finds all images in same label_group and
    :param image_df: DataFrame containing x_col and y_col for identifying images and their classes.
    :param x_col: the name of the column containing the image file names
    :param y_col: the name of the column containing image class label
    :return:
    """
    image_df = image_df.reset_index(drop=True)
    positive_examples_df = pd.DataFrame()
    labels_processed = []
    for source_index in range(len(image_df)):
        source_example = image_df.iloc[source_index]
        source_label = source_example[y_col]
        if source_label in labels_processed:
            continue
        labels_processed.append(source_label)
        positive_examples_image = image_df[image_df[y_col] == source_label][x_col]

        for p_image in positive_examples_image:
            positive_examples_df = positive_examples_df.append({
                "source_image": source_example[x_col],
                "target_image": p_image,
                "type": "RELATED",
                "class_label": 1
            }, ignore_index=True)
    return positive_examples_df


def train_test_split_no_stratification(x, y, percent):
    df = x.groupby(['label_group'])['label_group'].count().reset_index(name="label_group_count")

    groups = []
    sum = 0
    for i in df.sample(frac=1).iterrows():
        if (sum + i[1]['label_group_count']) < (x.shape[0] * percent):
            sum += i[1]['label_group_count']
            groups.append(i[1]['label_group'])
        if sum >= (x.shape[0] * percent):
            break

    return x[x['label_group'].isin(groups)], x[~x['label_group'].isin(groups)], \
           y[y.isin(groups)], y[~y.isin(groups)]
