import numpy as np
import gensim.downloader as api


def load_glove_model():
    model = api.load("glove-wiki-gigaword-100")
    return model


def create_document_vector(document, model):
    word_vectors = []

    for word in document:
        if word in model:
            word_vectors.append(model[word])

    if not word_vectors:
        return np.zeros(model.vector_size)

    return np.mean(word_vectors, axis=0)


def create_glove_vectors(tokenized_documents):
    model = load_glove_model()

    document_vectors = np.array([
        create_document_vector(document, model)
        for document in tokenized_documents
    ])

    return document_vectors, model
