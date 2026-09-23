from pathlib import Path

import matplotlib.pyplot as plt

from src.clustering import calculate_elbow_values, reduce_dimensions


RESULTS_DIRECTORY = Path("results/figures")


def ensure_results_directory():
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)


def plot_elbow(document_vectors, method_name):
    ensure_results_directory()

    k_values, inertias = calculate_elbow_values(
        document_vectors,
        max_clusters=4
    )

    plt.figure(figsize=(7, 5))
    plt.plot(k_values, inertias, marker="o")

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title(f"Elbow Method - {method_name}")
    plt.xticks(k_values)
    plt.tight_layout()

    filename = (
        method_name.lower()
        .replace(" ", "_")
        .replace("-", "")
        + "_elbow.png"
    )

    output_path = RESULTS_DIRECTORY / filename

    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_clusters(document_vectors, labels, document_names, method_name):
    ensure_results_directory()

    reduced_vectors, _ = reduce_dimensions(
        document_vectors,
        n_components=2
    )

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        reduced_vectors[:, 0],
        reduced_vectors[:, 1],
        c=labels,
        s=100
    )

    for index, document_name in enumerate(document_names):
        plt.annotate(
            document_name,
            (
                reduced_vectors[index, 0],
                reduced_vectors[index, 1]
            ),
            xytext=(6, 6),
            textcoords="offset points"
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"PCA Document Clusters - {method_name}")

    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()

    filename = (
        method_name.lower()
        .replace(" ", "_")
        .replace("-", "")
        + "_clusters.png"
    )

    output_path = RESULTS_DIRECTORY / filename

    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path
