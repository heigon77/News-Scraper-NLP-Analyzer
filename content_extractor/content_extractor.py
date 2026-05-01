import requests
from bs4 import BeautifulSoup

BAD_WORDS = ["cookie", "privacy", "advertisement", "subscribe"]

def extract_full_text(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        paragraphs = soup.find_all("p")

        text = " ".join(
            p.get_text(strip=True)
            for p in paragraphs
            if not any(b in p.get_text().lower() for b in BAD_WORDS)
        )

        # filtro mínimo de qualidade
        if len(text.split()) < 100:
            return ""

        return text

    except Exception:
        return ""