import numpy as np
from gensim.models import Word2Vec


def train_word2vec(tokenized_documents, vector_size=100):
    model = Word2Vec(
        sentences=tokenized_documents,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=1,
        seed=42
    )

    return model


def create_document_vector(document, model):
    word_vectors = []

    for word in document:
        if word in model.wv:
            word_vectors.append(model.wv[word])

    if not word_vectors:
        return np.zeros(model.vector_size)

    return np.mean(word_vectors, axis=0)


def create_word2vec_vectors(tokenized_documents, vector_size=100):
    model = train_word2vec(
        tokenized_documents,
        vector_size=vector_size
    )

    document_vectors = np.array([
        create_document_vector(document, model)
        for document in tokenized_documents
    ])

    return document_vectors, model
