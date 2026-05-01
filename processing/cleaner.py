import re
import pandas as pd


def clean_text(text: str) -> str:
    """Lowercase, remove special chars, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_date(date_str: str) -> pd.Timestamp | None:
    """
    Tries to parse arXiv submission dates like '3 November, 2024'.
    Returns pandas Timestamp or NaT.
    """
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except Exception:
        return pd.NaT


def papers_to_dataframe(papers: list[dict]) -> pd.DataFrame:
    """
    Converts list of paper dicts into a clean pandas DataFrame.

    Adds:
    - clean_title, clean_abstract: lowercased/normalized text
    - submitted_date: parsed datetime
    - year, month: extracted from submitted_date
    - num_authors: author count
    - primary_category: first category tag
    """
    df = pd.DataFrame(papers)

    # Clean text
    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_abstract"] = df["abstract"].apply(clean_text)

    # Parse dates
    df["submitted_date"] = df["submitted"].apply(parse_date)
    df["year"] = df["submitted_date"].dt.year
    df["month"] = df["submitted_date"].dt.month
    df["year_month"] = df["submitted_date"].dt.to_period("M").astype(str)

    # Author count
    df["num_authors"] = df["authors"].apply(len)

    # Primary category
    df["primary_category"] = df["categories"].apply(
        lambda cats: cats[0] if cats else "unknown"
    )

    return df
