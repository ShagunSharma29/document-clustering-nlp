from pathlib import Path

from src.preprocessing import (
    download_nltk_resources,
    preprocess_text,
    preprocess_document
)

from src.bow_clustering import create_bow_vectors
from src.tfidf_clustering import create_tfidf_vectors
from src.word2vec_clustering import create_word2vec_vectors
from src.glove_clustering import create_glove_vectors
from src.clustering import apply_kmeans
from src.evaluation import evaluate_clustering, print_evaluation
from src.visualization import plot_elbow, plot_clusters

DATA_DIRECTORY = Path("data/sample_documents")


def load_documents():
    file_paths = sorted(DATA_DIRECTORY.glob("*.txt"))

    documents = []
    document_names = []

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            documents.append(file.read())

        document_names.append(file_path.name)

    return documents, document_names

def run_experiment(method_name, vectors, document_names, n_clusters):
    labels, _ = apply_kmeans(
        vectors,
        n_clusters=n_clusters
    )

    metrics = evaluate_clustering(vectors, labels)

    print_evaluation(method_name, metrics)

    print("Cluster assignments:")

    for document_name, label in zip(document_names, labels):
        print(f"  {document_name} -> Cluster {label}")

    elbow_path = plot_elbow(
        vectors,
        method_name
    )

    cluster_path = plot_clusters(
        vectors,
        labels,
        document_names,
        method_name
    )

    print(f"Elbow plot saved to: {elbow_path}")
    print(f"Cluster plot saved to: {cluster_path}")

    return metrics

def main():
    download_nltk_resources()

    documents, document_names = load_documents()

    if len(documents) < 3:
        raise ValueError(
            "At least three documents are required for this experiment."
        )

    print(f"Loaded {len(documents)} documents.")

    processed_documents = [
        preprocess_document(document)
        for document in documents
    ]

    tokenized_documents = [
        preprocess_text(document)
        for document in documents
    ]

    n_clusters = min(3, len(documents) - 1)

    bow_vectors, _ = create_bow_vectors(
        processed_documents
    )

    run_experiment(
        "Bag of Words",
        bow_vectors,
        document_names,
        n_clusters
    )

    tfidf_vectors, _ = create_tfidf_vectors(
        processed_documents
    )

    run_experiment(
        "TF-IDF",
        tfidf_vectors,
        document_names,
        n_clusters
    )

    word2vec_vectors, _ = create_word2vec_vectors(
        tokenized_documents
    )

    run_experiment(
        "Word2Vec",
        word2vec_vectors,
        document_names,
        n_clusters
    )

    print("\nLoading GloVe model...")

    glove_vectors, _ = create_glove_vectors(
        tokenized_documents
    )

    run_experiment(
        "GloVe",
        glove_vectors,
        document_names,
        n_clusters
    )


if __name__ == "__main__":
    main()
