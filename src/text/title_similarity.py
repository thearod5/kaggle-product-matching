import pickle
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from meta.paths import PATH_TO_VECTORIZER


def calculate_title_similarities(image_titles: List[str], load_previous: bool = False):
    """
     Given N titles, creates N x N matrix containing the cosine similarity of the
    Tfidf vectors between all titles. The vectorizer is trained on all the image titles.
    : image_titles: titles in the order that rows and columns are presented.
    """

    if load_previous:
        vectorizer = load_vectorizer()
        title_vectors = vectorizer.transform(image_titles)
    else:
        vectorizer = TfidfVectorizer(stop_words="english", strip_accents="unicode")
        title_vectors = vectorizer.fit_transform(image_titles)
        save_vectorizer(vectorizer)

    title2title_similarities = cosine_similarity(title_vectors, title_vectors)
    return title2title_similarities


def save_vectorizer(vectorizer):
    pickle.dump(vectorizer, open(PATH_TO_VECTORIZER, "wb"))


def load_vectorizer():
    return pickle.load(open(PATH_TO_VECTORIZER, "rb"))
