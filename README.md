
# 📰 News Scraper & NLP Analyzer

A high-performance Python pipeline that scrapes international news from sources like BBC News, The Guardian and NPR, extracts full article text, and applies advanced NLP techniques (transformers, embeddings, clustering) to generate interactive insights.

---

## 🚀 Features

### 🌐 Data Collection
- Multi-source scraping (RSS + HTML fallback)
- Sources:
  - BBC
  - Guardian
  - NPR
- Automatic deduplication by URL
- Full article extraction (not just headlines)

---

### 🧠 NLP & ML Pipeline
- Named Entity Recognition (NER) via `spaCy`
- Sentiment Analysis via `transformers`
- Embeddings via `sentence-transformers`
- Topic Clustering (KMeans)
- TF-IDF keyword extraction

---

### ⚡ Performance Optimizations
- Batch inference on GPU (no sequential bottlenecks)
- Parallel article content extraction
- Efficient text truncation (512 tokens)
- Scalable to hundreds of articles

---

### 📊 Visualization
Interactive charts with Plotly:

- 🔑 Top keywords
- 🧠 Sentiment distribution
- 🧩 Topic clusters
- 📰 Source distribution
- 🌍 Named entities (NER)
- 🗺️ Location analysis

All charts are exported as interactive HTML dashboards

---

## 📁 Project Structure

```

news-analyzer/
├── scraper/
│   └── fetcher.py        # RSS + HTML scraping
├── processing/
│   ├── cleaner.py            # DataFrame builder
│   └── nlp.py       # Sentiment, embeddings, NER
├── content_extractor/
│   ├── content_extractor.py      # Full article extraction
├── viz/
│   ├── viz.py                    # Plotly visualizations
├── data/
│   ├── raw/
│   │   └── news_full.json
│   └── charts/
├── notebooks/
│   └── analysis.ipynb
├── main.py
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
````

---

## ▶️ Usage

```bash
python main.py
```

---

## 📊 Output

```
data/
├── raw/
│   ├── news_full.json
│   └── news.csv
└── charts/
    ├── top_keywords.html
    ├── sentiment.html
    ├── source_distribution.html
    ├── entities.html
    └── locations.html
```

Open `.html` files in your browser.

---

## 🧠 NLP Pipeline

### Sentiment

* Model: `distilbert-base-uncased-finetuned-sst-2-english`

### Embeddings

* Model: `sentence-transformers/all-MiniLM-L6-v2`

### NER

* Model: `en_core_web_sm`

---

## ⚡ Performance

* Batch inference enabled
* GPU acceleration supported
* Parallel scraping
* No sequential transformer bottlenecks

---

## 📈 Example Insights

* Source sentiment comparison
* Topic clustering of global news
* Geographic entity distribution
* Keyword trends

---

## ⚠️ Responsible Scraping

* Uses RSS feeds when possible
* Avoids aggressive crawling
* Includes request headers and timeouts

---

## 🛠️ Stack

| Layer         | Tools                   |
| ------------- | ----------------------- |
| Scraping      | requests, BeautifulSoup |
| NLP           | spaCy, transformers     |
| Embeddings    | sentence-transformers   |
| Data          | pandas, numpy           |
| ML            | scikit-learn            |
| Visualization | plotly                  |

---

## 🚀 Roadmap

* [ ] Streamlit dashboard
* [ ] Semantic search
* [ ] Topic labeling (LLM)
* [ ] Bias comparison between sources

---

## 📝 License

MIT

```
```
