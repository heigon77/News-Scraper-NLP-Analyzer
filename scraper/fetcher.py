import requests
from bs4 import BeautifulSoup
import time
import random

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# ─────────────────────────────────────────────
# BBC
# ─────────────────────────────────────────────
BBC_URLS = [
    "https://www.bbc.com/news",
    "https://www.bbc.com/news/world",
    "https://www.bbc.com/news/technology",
    "https://www.bbc.com/news/business",
]

def fetch_bbc():
    articles = []

    for url in BBC_URLS:
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            for a in soup.select("a"):
                title = a.get_text(strip=True)
                link = a.get("href", "")

                if not title or len(title) < 25:
                    continue

                if link.startswith("/"):
                    link = "https://www.bbc.com" + link

                if "/news/" not in link:
                    continue

                articles.append({
                    "title": title,
                    "link": link,
                    "source": "bbc"
                })

        except Exception as e:
            print("BBC error:", e)

    return articles


# ─────────────────────────────────────────────
# Guardian (muito bom pra scraping)
# ─────────────────────────────────────────────
def fetch_guardian_rss():
    import xml.etree.ElementTree as ET

    url = "https://www.theguardian.com/world/rss"
    articles = []

    try:
        r = requests.get(url, headers=HEADERS, timeout=10)

        root = ET.fromstring(r.content)

        for item in root.iter("item"):
            title = item.find("title")
            link = item.find("link")

            if title is None or link is None:
                continue

            articles.append({
                "title": title.text,
                "link": link.text,
                "source": "guardian"
            })

    except Exception as e:
        print("Guardian RSS error:", e)

    return articles


# ─────────────────────────────────────────────
# NPR (super estável)
# ─────────────────────────────────────────────
def fetch_npr():
    url = "https://www.npr.org/sections/news/"
    r = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    articles = []

    for a in soup.select("h2 a"):
        title = a.get_text(strip=True)
        link = a.get("href", "")

        if not title or len(title) < 20:
            continue

        articles.append({
            "title": title,
            "link": link,
            "source": "npr"
        })

    return articles


# ─────────────────────────────────────────────
# Unified
# ─────────────────────────────────────────────
def fetch_all_news():
    all_articles = []

    bbc = fetch_bbc()
    print(f"BBC: {len(bbc)}")
    all_articles.extend(bbc)

    guardian = fetch_guardian_rss()
    print(f"Guardian: {len(guardian)}")
    all_articles.extend(guardian)

    npr = fetch_npr()
    print(f"NPR: {len(npr)}")
    all_articles.extend(npr)

    print(f"TOTAL BEFORE DEDUP: {len(all_articles)}")

    # deduplicação correta
    seen = set()
    unique = []

    for a in all_articles:
        key = a["link"]  # 🔥 chave correta

        if key not in seen:
            seen.add(key)
            unique.append(a)

    print(f"TOTAL AFTER DEDUP: {len(unique)}")

    return unique