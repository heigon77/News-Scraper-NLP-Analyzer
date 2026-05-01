import spacy
import numpy as np
from transformers import pipeline


# ─────────────────────────────────────────────
# ⚙️ MODELS
# ─────────────────────────────────────────────

# spaCy (NER)
nlp = spacy.load("en_core_web_sm")

# Sentiment
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0  # use -1 se não tiver GPU
)

# Embeddings
embedding_model = pipeline(
    "feature-extraction",
    model="sentence-transformers/all-MiniLM-L6-v2",
    device=0
)


# ─────────────────────────────────────────────
# 🧠 SENTIMENT (BATCH)
# ─────────────────────────────────────────────
def batch_sentiment(texts, batch_size=16):
    try:
        return sentiment_model(
            texts,
            batch_size=batch_size,
            truncation=True
        )
    except Exception as e:
        print("Sentiment error:", e)
        return [{"label": "NEUTRAL", "score": 0.0} for _ in texts]


# ─────────────────────────────────────────────
# 🌍 NER (BATCH via spaCy)
# ─────────────────────────────────────────────
def batch_entities(texts, batch_size=32):
    results = []

    try:
        docs = nlp.pipe(
            (t[:1000] for t in texts),
            batch_size=batch_size
        )

        for doc in docs:
            ents = [(ent.text, ent.label_) for ent in doc.ents]
            results.append(ents)

    except Exception as e:
        print("NER error:", e)
        return [[] for _ in texts]

    return results


# ─────────────────────────────────────────────
# 🔗 EMBEDDINGS (BATCH)
# ─────────────────────────────────────────────
def batch_embeddings(texts, batch_size=16):
    try:
        outputs = embedding_model(
            [t[:512] for t in texts],
            batch_size=batch_size,
            truncation=True
        )

        # mean pooling
        embeddings = []
        for e in outputs:
            if isinstance(e, list) and len(e) > 0:
                emb = np.mean(e, axis=0).tolist()
            else:
                emb = []
            embeddings.append(emb)

        return embeddings

    except Exception as e:
        print("Embedding error:", e)
        return [[] for _ in texts]


# ─────────────────────────────────────────────
# ⚠️ LEGACY (opcional)
# ─────────────────────────────────────────────
def extract_entities(text):
    """Legacy single-text NER (evitar usar em produção)"""
    doc = nlp(text[:1000])
    return [(ent.text, ent.label_) for ent in doc.ents]


def get_embedding(text):
    """Legacy single-text embedding (evitar usar)"""
    try:
        emb = embedding_model(text[:512])[0]
        return np.mean(emb, axis=0).tolist()
    except:
        return []
    

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "and", "for", "is", "are",
    "with", "on", "we", "our", "that", "this", "by", "from", "which",
    "as", "at", "be", "have", "it", "can", "also", "using", "based",
    "show", "propose", "paper", "model", "models", "method", "approach",
    "results", "data", "set", "new", "used", "use", "two", "these",
    "their", "show", "shows", "learn", "learning"
}


def extract_tfidf_keywords(texts: list[str], top_n: int = 30) -> list[tuple[str, float]]:
    """
    Extracts top keywords from a list of texts using TF-IDF.

    Args:
        texts: List of cleaned text strings
        top_n: Number of top keywords to return

    Returns:
        List of (keyword, score) tuples sorted by score descending
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.mean(axis=0).A1

    keyword_scores = sorted(
        zip(feature_names, scores), key=lambda x: x[1], reverse=True
    )

    # Filter generic stopwords
    filtered = [(kw, sc) for kw, sc in keyword_scores if kw not in STOPWORDS]
    return filtered[:top_n]


def count_keywords_over_time(df: pd.DataFrame, keywords: list[str], text_col: str = "clean_abstract") -> pd.DataFrame:
    """
    Counts keyword occurrences per year_month.

    Args:
        df: DataFrame with text_col and year_month columns
        keywords: List of keywords to track
        text_col: Column containing cleaned text

    Returns:
        DataFrame with year_month as index and keywords as columns
    """
    records = []
    for _, row in df.iterrows():
        text = row[text_col]
        period = row["year_month"]
        counts = {kw: int(kw in text) for kw in keywords}
        counts["year_month"] = period
        records.append(counts)

    trend_df = pd.DataFrame(records)
    return trend_df.groupby("year_month")[keywords].sum().reset_index()


def top_authors(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Returns the most prolific authors in the dataset."""
    all_authors = [author for authors in df["authors"] for author in authors]
    counts = Counter(all_authors)
    return pd.DataFrame(counts.most_common(top_n), columns=["author", "paper_count"])


def category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Returns paper count per primary category."""
    return (
        df["primary_category"]
        .value_counts()
        .rename_axis("primary_category")
        .reset_index(name="count")
    )
