import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


def download_nltk_resources():
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
    ]

    for resource in resources:
        nltk.download(resource, quiet=True)


def preprocess_text(text):
    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    processed_tokens = []

    for token in tokens:
        lemma = lemmatizer.lemmatize(token)
        stemmed_word = stemmer.stem(lemma)
        processed_tokens.append(stemmed_word)

    return processed_tokens


def preprocess_document(text):
    tokens = preprocess_text(text)
    return " ".join(tokens)


if __name__ == "__main__":
    download_nltk_resources()

    sample_text = """
    Natural Language Processing enables computers to understand
    and analyze human languages.
    """

    print("Original Text:")
    print(sample_text)

    print("\nProcessed Text:")
    print(preprocess_document(sample_text))
