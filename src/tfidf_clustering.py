from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectors(documents):
    vectorizer = TfidfVectorizer()

    document_vectors = vectorizer.fit_transform(documents)

    return document_vectors.toarray(), vectorizer


def get_tfidf_features(vectorizer):
    return vectorizer.get_feature_names_out()
