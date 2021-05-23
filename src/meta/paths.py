"""
The following module is responsible for linking all paths in a cohesive way
so that they can be easily imported anywhere
"""
import os

PATH_TO_DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

"""
DATA FILES
"""
PATH_TO_TRAIN = os.path.join(PATH_TO_DATA, "train.csv")
PATH_TO_TRAIN_PAIRS = os.path.join(PATH_TO_DATA, "training_image_pairs.csv")
PATH_TO_TEST_PAIRS = os.path.join(PATH_TO_DATA, "testing_image_pairs.csv")
PATH_TO_TRAINING_IMAGES = os.path.join(PATH_TO_DATA, "train_images")
PATH_TO_VECTORIZER = os.path.join(PATH_TO_DATA, "tfidf_vectorizer.pickle")
