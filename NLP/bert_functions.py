import ast

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "cointegrated/rubert-tiny-sentiment-balanced"


def compute_bert_sentiment(
    df: pd.DataFrame,
    news_col: str = "news",
    model_name: str = MODEL_NAME,
    batch_size: int = 16,
    max_length: int = 512,
    score_col: str = "bert_sentiment_score",
    threshold: float = 0.15,
) -> pd.DataFrame:
    df = df.copy()

    texts = df[news_col].apply(_join_news)
    has_news = texts.str.len() > 0

    df["has_news"] = has_news.astype(int)
    df["news_count"] = df[news_col].apply(_count_news)

    scores = np.full(len(df), np.nan)

    non_empty_texts = texts[has_news].tolist()

    if len(non_empty_texts) > 0:
        predicted_scores = _predict_scores(
            texts=non_empty_texts,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
        )

        scores[has_news.values] = predicted_scores

    df[score_col] = scores

    df["bert_is_positive"] = np.where(
        df[score_col].notna(),
        (df[score_col] > threshold).astype(int),
        np.nan,
    )

    df["bert_is_negative"] = np.where(
        df[score_col].notna(),
        (df[score_col] < -threshold).astype(int),
        np.nan,
    )

    return df


def add_avg_sentiment(
    df: pd.DataFrame,
    window: int,
    date_col: str = "timestamp",
    score_col: str = "bert_sentiment_score",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)
    df[f"avg_sentiment_{window}d"] = (
        df[score_col].shift(1).rolling(window, min_periods=1).mean()
    )
    return df


def add_sentiment_std(
    df: pd.DataFrame,
    window: int,
    date_col: str = "timestamp",
    score_col: str = "bert_sentiment_score",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)
    df[f"sentiment_std_{window}d"] = (
        df[score_col].shift(1).rolling(window, min_periods=2).std()
    )
    return df


def add_pct_negative(
    df: pd.DataFrame,
    window: int,
    date_col: str = "timestamp",
    negative_col: str = "bert_is_negative",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)
    df[f"pct_negative_{window}d"] = (
        df[negative_col].shift(1).rolling(window, min_periods=1).mean()
    )
    return df


def add_pct_positive(
    df: pd.DataFrame,
    window: int,
    date_col: str = "timestamp",
    positive_col: str = "bert_is_positive",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)
    df[f"pct_positive_{window}d"] = (
        df[positive_col].shift(1).rolling(window, min_periods=1).mean()
    )
    return df


def add_sentiment_change(
    df: pd.DataFrame,
    window: int,
    date_col: str = "timestamp",
    score_col: str = "bert_sentiment_score",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)

    current = df[score_col].shift(1).rolling(window, min_periods=1).mean()
    previous = df[score_col].shift(window + 1).rolling(window, min_periods=1).mean()

    df[f"sentiment_change_{window}d"] = current - previous

    return df


def add_news_count(
    df: pd.DataFrame,
    window: int,
    date_col: str = "timestamp",
    news_count_col: str = "news_count",
) -> pd.DataFrame:
    df = _sort_by_date(df, date_col)
    df[f"news_count_{window}d"] = (
        df[news_count_col].shift(1).rolling(window, min_periods=1).sum()
    )
    return df


def _predict_scores(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
) -> list[float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits

        probs = torch.softmax(logits, dim=1).cpu()

        for prob in probs:
            scores.append(_sentiment_score(prob, model.config.id2label))

    return scores


def _sentiment_score(prob: torch.Tensor, id2label: dict) -> float:
    labels = {int(k): str(v).lower() for k, v in id2label.items()}

    positive_idx = _find_label_index(labels, ["positive", "pos"])
    negative_idx = _find_label_index(labels, ["negative", "neg"])

    positive = float(prob[positive_idx]) if positive_idx is not None else 0.0
    negative = float(prob[negative_idx]) if negative_idx is not None else 0.0

    return positive - negative


def _find_label_index(labels: dict, variants: list[str]) -> int | None:
    for idx, label in labels.items():
        if any(variant in label for variant in variants):
            return idx

    return None


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


def _count_news(value) -> int:
    if isinstance(value, list):
        return len(value)

    if isinstance(value, str):
        value = value.strip()

        if value == "[]":
            return 0

        if value.startswith("[") and value.endswith("]"):
            return len(ast.literal_eval(value))

        return int(len(value) > 0)

    return 0