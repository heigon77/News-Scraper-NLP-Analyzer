import os
import logging
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from scraper import fetch_all_news
from content_extractor.content_extractor import extract_full_text

from processing import (
    batch_sentiment,
    batch_embeddings,
    batch_entities
)

from processing import extract_tfidf_keywords

from viz.viz import (
    plot_top_keywords,
    plot_sentiment,
    plot_source_distribution,
    plot_top_entities,
    plot_top_locations,
    plot_papers_per_month,
    save_all_charts
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 🚀 PIPELINE
# ─────────────────────────────────────────────
def run_pipeline():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/charts", exist_ok=True)

    # ── 1. Scrape ─────────────────────
    logger.info("Fetching news...")
    articles = fetch_all_news()

    if not articles:
        logger.error("No articles found.")
        return

    # ── 2. Extract (PARALELO) ─────────
    logger.info("Extracting full text (parallel)...")

    def extract_wrapper(a):
        a["content"] = extract_full_text(a["link"])
        return a

    with ThreadPoolExecutor(max_workers=8) as executor:
        articles = list(executor.map(extract_wrapper, articles))

    # ── 3. NLP ────────────────────────
    logger.info("Running NLP (batched)...")

    texts = [
        (a.get("content") or a["title"])[:512]
        for a in articles
    ]

    # ✅ Sentiment
    sentiment_results = batch_sentiment(texts)

    for a, r in zip(articles, sentiment_results):
        a["sentiment_label"] = r["label"]
        a["sentiment_score"] = r["score"]

    # ✅ NER
    entities_batch = batch_entities(texts)

    for a, ents in zip(articles, entities_batch):
        a["entities"] = ents

    # ✅ Embeddings
    embeddings = batch_embeddings(texts)

    for a, emb in zip(articles, embeddings):
        a["embedding"] = emb

    # ── 5. DataFrame ──────────────────
    df = pd.DataFrame(articles)

    df["clean_text"] = df["content"].fillna(df["title"])

    # 📅 datas
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # salvar dados
    df.to_json("data/raw/news_full.json", orient="records", indent=2)
    df.to_csv("data/raw/news.csv", index=False)

    # ── 6. Keywords ───────────────────
    keywords = extract_tfidf_keywords(
        df["clean_text"].tolist(),
        top_n=30
    )

    # ── 7. Análises extras ────────────
    monthly_df = pd.DataFrame()

    if "year_month" in df.columns:
        monthly_df = (
            df.groupby("year_month")
            .size()
            .reset_index(name="count")
        )

    # ── 8. Charts ─────────────────────
    figures = {
        "top_keywords": plot_top_keywords(keywords),
        "sentiment": plot_sentiment(df),
        "source_distribution": plot_source_distribution(df),
        "entities": plot_top_entities(df),
        "locations": plot_top_locations(df),
    }

    if not monthly_df.empty:
        figures["articles_per_month"] = plot_papers_per_month(monthly_df)

    save_all_charts(figures)

    print("\nPipeline finished successfully 🚀")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()