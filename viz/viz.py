import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ─────────────────────────────────────────────
# 📈 Papers / Articles per Month
# ─────────────────────────────────────────────
def plot_papers_per_month(monthly_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        monthly_df,
        x="year_month",
        y="count",
        title="📈 Articles Over Time",
        markers=True,
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


# ─────────────────────────────────────────────
# 🔑 Keywords
# ─────────────────────────────────────────────
def plot_top_keywords(keywords: list[tuple[str, float]], top_n: int = 20) -> go.Figure:
    df = pd.DataFrame(keywords[:top_n], columns=["keyword", "score"])
    df = df.sort_values("score")

    fig = px.bar(
        df,
        x="score",
        y="keyword",
        orientation="h",
        title="🔑 Top Keywords",
        color="score",
        color_continuous_scale="Blues",
    )
    return fig


# ─────────────────────────────────────────────
# 📂 Category Distribution (legacy / optional)
# ─────────────────────────────────────────────
def plot_category_distribution(cat_df: pd.DataFrame) -> go.Figure:
    cat_df = cat_df.copy()

    # evita erro de coluna duplicada
    cat_df = cat_df.loc[:, ~cat_df.columns.duplicated()]

    fig = px.pie(
        cat_df,
        names=cat_df.columns[0],
        values=cat_df.columns[1],
        title="📂 Category Distribution",
        hole=0.3,
    )
    return fig



# ─────────────────────────────────────────────
# 🧠 Sentiment
# ─────────────────────────────────────────────
def plot_sentiment(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="sentiment_label",
        color="sentiment_label",
        title="🧠 Sentiment Distribution",
    )
    return fig


# ─────────────────────────────────────────────
# 📰 Source Distribution
# ─────────────────────────────────────────────
def plot_source_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        df,
        names="source",
        title="📰 Source Distribution",
        hole=0.3,
    )
    return fig


# ─────────────────────────────────────────────
# 🌍 Entities (NER)
# ─────────────────────────────────────────────
def plot_top_entities(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    from collections import Counter

    all_entities = []

    for ents in df.get("entities", []):
        if isinstance(ents, list):
            for e in ents:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    all_entities.append((e[0], e[1]))

    counter = Counter(all_entities)
    top = counter.most_common(top_n)

    if not top:
        return go.Figure()

    ent_df = pd.DataFrame(top, columns=["entity", "count"])
    ent_df["text"] = ent_df["entity"].apply(lambda x: x[0])
    ent_df["label"] = ent_df["entity"].apply(lambda x: x[1])

    fig = px.bar(
        ent_df,
        x="count",
        y="text",
        color="label",
        orientation="h",
        title="🌍 Top Entities",
    )
    return fig


# ─────────────────────────────────────────────
# 🗺️ Locations (GPE/LOC)
# ─────────────────────────────────────────────
def plot_top_locations(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    from collections import Counter

    locations = []

    for ents in df.get("entities", []):
        if isinstance(ents, list):
            for text, label in ents:
                if label in ["GPE", "LOC"]:
                    locations.append(text)

    counter = Counter(locations)
    top = counter.most_common(top_n)

    if not top:
        return go.Figure()

    loc_df = pd.DataFrame(top, columns=["location", "count"])

    fig = px.bar(
        loc_df,
        x="count",
        y="location",
        orientation="h",
        title="🗺️ Top Locations",
    )
    return fig


# ─────────────────────────────────────────────
# 💾 Save all charts
# ─────────────────────────────────────────────
def save_all_charts(figures: dict[str, go.Figure], output_dir: str = "data/charts") -> None:
    import os

    os.makedirs(output_dir, exist_ok=True)

    for name, fig in figures.items():
        path = os.path.join(output_dir, f"{name}.html")
        fig.write_html(path)
        print(f"Saved: {path}")