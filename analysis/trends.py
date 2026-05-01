import pandas as pd


def papers_per_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts number of papers submitted per year_month.

    Returns:
        DataFrame with columns: year_month, count
    """
    counts = df.groupby("year_month").size().reset_index(name="count")
    counts = counts.sort_values("year_month")
    return counts


def avg_authors_per_month(df: pd.DataFrame) -> pd.DataFrame:
    """Average number of authors per paper, grouped by month."""
    return df.groupby("year_month")["num_authors"].mean().reset_index(name="avg_authors")


def top_papers_by_title_length(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Returns papers with unusually long or short titles — often interesting outliers."""
    df = df.copy()
    df["title_len"] = df["title"].str.len()
    return df.nlargest(top_n, "title_len")[["title", "authors", "submitted_date", "url"]]


def category_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shows how many papers per category appear each month.
    Useful for spotting emerging research areas.
    """
    return df.groupby(["year_month", "primary_category"]).size().reset_index(name="count")


def growth_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes month-over-month growth rate in paper submissions.

    Returns:
        DataFrame with year_month, count, and growth_pct columns
    """
    monthly = papers_per_month(df)
    monthly["growth_pct"] = monthly["count"].pct_change() * 100
    return monthly
