import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

def get_topn_words(article: str, n: int = 2) -> List[str]:
    """
    Extracts the top N words from an article based on their TF-IDF scores.

    Args:
        article (str): The input article text from which to extract top words.
        n (int, optional): The number of top words to extract. Defaults to 2.

    Returns:
        List[str]: A list of the top N words with the highest TF-IDF scores.
    """
    vectorizer = TfidfVectorizer(stop_words='english')

    X = vectorizer.fit_transform([article])

    feature_names = vectorizer.get_feature_names_out()

    feature_names = np.array(vectorizer.get_feature_names_out())
    
    tfidf_scores = X.toarray().flatten()

    sorted_indices = np.argsort(tfidf_scores)[::-1]

    topn = feature_names[sorted_indices][:n]

    return topn.tolist()
