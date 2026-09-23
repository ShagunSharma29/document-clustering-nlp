from collections import defaultdict
import numpy as np


def build_vocabulary(tokenized_documents):
    vocabulary = sorted(
        set(word for document in tokenized_documents for word in document)
    )

    return vocabulary


def create_cooccurrence_matrix(tokenized_documents, window_size=2):
    vocabulary = build_vocabulary(tokenized_documents)
    word_to_index = {word: index for index, word in enumerate(vocabulary)}

    matrix = np.zeros((len(vocabulary), len(vocabulary)))

    for document in tokenized_documents:
        for i, word in enumerate(document):
            start = max(0, i - window_size)
            end = min(len(document), i + window_size + 1)

            for j in range(start, end):
                if i == j:
                    continue

                neighbour = document[j]

                row = word_to_index[word]
                column = word_to_index[neighbour]

                matrix[row, column] += 1

    return matrix, vocabulary


def create_distance_weighted_matrix(tokenized_documents, window_size=2):
    vocabulary = build_vocabulary(tokenized_documents)
    word_to_index = {word: index for index, word in enumerate(vocabulary)}

    matrix = np.zeros((len(vocabulary), len(vocabulary)))

    for document in tokenized_documents:
        for i, word in enumerate(document):
            start = max(0, i - window_size)
            end = min(len(document), i + window_size + 1)

            for j in range(start, end):
                if i == j:
                    continue

                neighbour = document[j]
                distance = abs(i - j)

                weight = 1.0 / distance

                row = word_to_index[word]
                column = word_to_index[neighbour]

                matrix[row, column] += weight

    return matrix, vocabulary


def highest_cooccurring_pairs(matrix, vocabulary, top_n=10):
    pairs = []

    for i in range(len(vocabulary)):
        for j in range(i + 1, len(vocabulary)):
            score = matrix[i, j] + matrix[j, i]

            if score > 0:
                pairs.append(
                    (vocabulary[i], vocabulary[j], score)
                )

    pairs.sort(key=lambda item: item[2], reverse=True)

    return pairs[:top_n]


def find_singleton_words(matrix, vocabulary):
    singleton_words = []

    for index, word in enumerate(vocabulary):
        if np.sum(matrix[index]) == 0:
            singleton_words.append(word)

    return singleton_words
