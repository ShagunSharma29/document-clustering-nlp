from sklearn.feature_extraction.text import CountVectorizer


def create_bow_vectors(documents):
    vectorizer = CountVectorizer()

    document_vectors = vectorizer.fit_transform(documents)

    return document_vectors.toarray(), vectorizer


def get_bow_features(vectorizer):
    return vectorizer.get_feature_names_out()
