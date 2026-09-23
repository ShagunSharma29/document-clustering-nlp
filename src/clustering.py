import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def apply_kmeans(document_vectors, n_clusters=3):
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(document_vectors)

    return labels, kmeans


def calculate_elbow_values(document_vectors, max_clusters=4):
    inertias = []

    max_k = min(max_clusters, len(document_vectors))

    for k in range(1, max_k + 1):
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        kmeans.fit(document_vectors)
        inertias.append(kmeans.inertia_)

    return list(range(1, max_k + 1)), inertias


def reduce_dimensions(document_vectors, n_components=2):
    vectors = np.asarray(document_vectors)

    max_components = min(
        n_components,
        vectors.shape[0],
        vectors.shape[1]
    )

    pca = PCA(n_components=max_components)

    reduced_vectors = pca.fit_transform(vectors)

    return reduced_vectors, pca
