import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score


def evaluate_clustering(document_vectors, labels):
    vectors = np.asarray(document_vectors)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return {
            "silhouette_score": None,
            "davies_bouldin_index": None
        }

    if len(unique_labels) >= len(vectors):
        return {
            "silhouette_score": None,
            "davies_bouldin_index": None
        }

    silhouette = silhouette_score(vectors, labels)
    davies_bouldin = davies_bouldin_score(vectors, labels)

    return {
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies_bouldin
    }


def print_evaluation(method_name, metrics):
    print(f"\n{method_name}")
    print("-" * len(method_name))

    silhouette = metrics["silhouette_score"]
    davies_bouldin = metrics["davies_bouldin_index"]

    if silhouette is None:
        print("Silhouette Score: Not available")
    else:
        print(f"Silhouette Score: {silhouette:.4f}")

    if davies_bouldin is None:
        print("Davies-Bouldin Index: Not available")
    else:
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
