import ast
import re

import pandas as pd


def compute_entity_mentions(
    df: pd.DataFrame,
    entity_groups: dict[str, list[str]],
    news_col: str = "news",
) -> pd.DataFrame:
    df = df.copy()
    texts = df[news_col].apply(_join_news)

    for group, entities in entity_groups.items():
        col = f"daily_{group}_mentions"
        df[col] = texts.apply(lambda text: _count_entities(text, entities))

        flag_col = f"daily_{group}_flag"
        df[flag_col] = (df[col] > 0).astype(int)

    mention_cols = [f"daily_{group}_mentions" for group in entity_groups]

    df["daily_total_entity_mentions"] = df[mention_cols].sum(axis=1)
    df["daily_entity_groups_count"] = df[
        [f"daily_{group}_flag" for group in entity_groups]
    ].sum(axis=1)

    return df


def add_entity_mentions(
    df: pd.DataFrame,
    group: str,
    window: int,
    date_col: str = "timestamp",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)

    source_col = f"daily_{group}_mentions"
    output_col = f"{group}_mentions_{window}d"

    df[output_col] = (
        df[source_col]
        .shift(1)
        .rolling(window, min_periods=1)
        .sum()
    )

    return df


def add_entity_share(
    df: pd.DataFrame,
    group: str,
    window: int,
    date_col: str = "timestamp",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)

    source_col = f"daily_{group}_flag"
    output_col = f"{group}_share_{window}d"

    df[output_col] = (
        df[source_col]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
    )

    return df


def add_total_entity_mentions(
    df: pd.DataFrame,
    window: int,
    date_col: str = "timestamp",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)

    df[f"total_entity_mentions_{window}d"] = (
        df["daily_total_entity_mentions"]
        .shift(1)
        .rolling(window, min_periods=1)
        .sum()
    )

    return df


def add_entity_groups_count(
    df: pd.DataFrame,
    window: int,
    date_col: str = "timestamp",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)

    df[f"entity_groups_count_{window}d"] = (
        df["daily_entity_groups_count"]
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
    )

    return df


def add_co_mention(
    df: pd.DataFrame,
    group_a: str,
    group_b: str,
    window: int,
    date_col: str = "timestamp",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)

    col_a = f"daily_{group_a}_flag"
    col_b = f"daily_{group_b}_flag"
    daily_col = f"daily_{group_a}_{group_b}_co_mention"

    df[daily_col] = ((df[col_a] == 1) & (df[col_b] == 1)).astype(int)

    df[f"{group_a}_{group_b}_co_mention_{window}d"] = (
        df[daily_col]
        .shift(1)
        .rolling(window, min_periods=1)
        .sum()
    )

    return df


def _count_entities(text: str, entities: list[str]) -> int:
    text = str(text).lower()
    count = 0

    for entity in entities:
        pattern = rf"(?<!\w){re.escape(entity.lower())}(?!\w)"
        count += len(re.findall(pattern, text))

    return count


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