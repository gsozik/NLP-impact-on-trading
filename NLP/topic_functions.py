import ast
import re

import numpy as np
import pandas as pd


def compute_topic_scores(
    df: pd.DataFrame,
    topic_groups: dict[str, list[str]],
    news_col: str = "news",
) -> pd.DataFrame:
    df = df.copy()
    texts = df[news_col].apply(_join_news)

    for topic, words in topic_groups.items():
        score_col = f"daily_topic_{topic}_score"
        flag_col = f"daily_topic_{topic}_flag"

        df[score_col] = texts.apply(lambda text: _count_words(text, words))
        df[flag_col] = (df[score_col] > 0).astype(int)

    score_cols = [f"daily_topic_{topic}_score" for topic in topic_groups]
    flag_cols = [f"daily_topic_{topic}_flag" for topic in topic_groups]

    df["daily_total_topic_score"] = df[score_cols].sum(axis=1)
    df["daily_active_topics_count"] = df[flag_cols].sum(axis=1)
    df["daily_topic_entropy"] = df[score_cols].apply(_entropy, axis=1)
    df["daily_dominant_topic"] = df[score_cols].apply(
        lambda row: _dominant_topic(row, topic_groups),
        axis=1,
    )

    return df


def add_topic_share(
    df: pd.DataFrame,
    topic: str,
    window: int,
    date_col: str = "timestamp",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)

    topic_col = f"daily_topic_{topic}_score"
    output_col = f"topic_{topic}_share_{window}d"

    topic_sum = df[topic_col].shift(1).rolling(window, min_periods=1).sum()
    total_sum = df["daily_total_topic_score"].shift(1).rolling(window, min_periods=1).sum()

    df[output_col] = topic_sum / total_sum.replace(0, np.nan)

    return df


def add_topic_intensity(
    df: pd.DataFrame,
    topic: str,
    window: int,
    date_col: str = "timestamp",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)

    source_col = f"daily_topic_{topic}_score"
    output_col = f"topic_{topic}_intensity_{window}d"

    df[output_col] = (
        df[source_col]
        .shift(1)
        .rolling(window, min_periods=1)
        .sum()
    )

    return df


def add_topic_entropy(
    df: pd.DataFrame,
    window: int,
    date_col: str = "timestamp",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)

    df[f"topic_entropy_{window}d"] = (
        df["daily_topic_entropy"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
    )

    return df


def add_active_topics_count(
    df: pd.DataFrame,
    window: int,
    date_col: str = "timestamp",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)

    df[f"active_topics_count_{window}d"] = (
        df["daily_active_topics_count"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
    )

    return df


def add_dominant_topic(
    df: pd.DataFrame,
    topic_groups: dict[str, list[str]],
    window: int,
    date_col: str = "timestamp",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)

    topics = list(topic_groups.keys())
    score_cols = [f"daily_topic_{topic}_score" for topic in topics]

    dominant_topics = []

    for i in range(len(df)):
        start = max(0, i - window)
        window_scores = df.loc[start:i - 1, score_cols].sum()

        if window_scores.sum() == 0:
            dominant_topics.append("no_topic")
        else:
            dominant_topics.append(
                window_scores.idxmax()
                .replace("daily_topic_", "")
                .replace("_score", "")
            )

    df[f"dominant_topic_{window}d"] = dominant_topics

    return df


def add_dominant_topic_id(
    df: pd.DataFrame,
    topic_groups: dict[str, list[str]],
    window: int,
    date_col: str = "timestamp",
) -> pd.DataFrame:
    df = add_dominant_topic(df, topic_groups, window, date_col)

    mapping = {"no_topic": 0}
    mapping.update({topic: idx + 1 for idx, topic in enumerate(topic_groups)})

    df[f"dominant_topic_id_{window}d"] = (
        df[f"dominant_topic_{window}d"]
        .map(mapping)
        .astype(int)
    )

    return df


def _count_words(text: str, words: list[str]) -> int:
    text = str(text).lower()
    count = 0

    for word in words:
        pattern = rf"(?<!\w){re.escape(word.lower())}(?!\w)"
        count += len(re.findall(pattern, text))

    return count


def _entropy(row: pd.Series) -> float:
    values = row.to_numpy(dtype=float)

    if values.sum() == 0:
        return 0.0

    probs = values / values.sum()
    probs = probs[probs > 0]

    return float(-(probs * np.log(probs)).sum())


def _dominant_topic(row: pd.Series, topic_groups: dict[str, list[str]]) -> str:
    if row.sum() == 0:
        return "no_topic"

    return (
        row.idxmax()
        .replace("daily_topic_", "")
        .replace("_score", "")
    )


def _sort_by_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    return df.sort_values(date_col).reset_index(drop=True)


def _join_news(value) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if str(item).strip())

    if isinstance(value, str):
        value = value.strip()

        if value == "[]":
            return ""

        if value.startswith("[") and value.endswith("]"):
            parsed = ast.literal_eval(value)
            return " ".join(str(item) for item in parsed if str(item).strip())

        return value

    if pd.isna(value):
        return ""

    return str(value)