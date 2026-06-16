import pandas as pd


def build_dataset(
    ohlcv: pd.DataFrame,
    news: pd.DataFrame,
    fundamentals: pd.DataFrame,
    ticker: str,
    ohlcv_time_col: str = "timestamp",
    news_time_col: str = "datetime",
    news_text_col: str = "text",
    fundamental_date_col: str = "available_date",
    news_output_col: str = "news",
    drop_nans: bool = False,
) -> pd.DataFrame:
    ohlcv = ohlcv.copy()
    news = news.copy()
    fundamentals = fundamentals.copy()

    if ohlcv_time_col not in ohlcv.columns:
        ohlcv = ohlcv.reset_index()

        if ohlcv_time_col not in ohlcv.columns:
            ohlcv = ohlcv.rename(columns={ohlcv.columns[0]: ohlcv_time_col})

    ohlcv[ohlcv_time_col] = pd.to_datetime(
        ohlcv[ohlcv_time_col],
        errors="coerce",
        utc=True,
    )

    if not news.empty:
        news[news_time_col] = pd.to_datetime(
            news[news_time_col],
            errors="coerce",
            utc=True,
        )

    fundamentals[fundamental_date_col] = pd.to_datetime(
        fundamentals[fundamental_date_col],
        errors="coerce",
        utc=True,
    )

    ohlcv = ohlcv.dropna(subset=[ohlcv_time_col])
    fundamentals = fundamentals.dropna(subset=[fundamental_date_col])

    if not news.empty:
        news = news.dropna(subset=[news_time_col])

    ohlcv["date"] = ohlcv[ohlcv_time_col].dt.date

    if news.empty:
        result = ohlcv.copy()
        result[news_output_col] = [[] for _ in range(len(result))]
    else:
        news["date"] = news[news_time_col].dt.date

        news_grouped = (
            news.groupby("date")[news_text_col]
            .apply(list)
            .reset_index()
            .rename(columns={news_text_col: news_output_col})
        )

        result = ohlcv.merge(
            news_grouped,
            on="date",
            how="left",
        )

        result[news_output_col] = result[news_output_col].apply(
            lambda x: x if isinstance(x, list) else []
        )

    ticker_fundamentals = fundamentals[
        fundamentals["ticker"] == ticker
    ].copy()

    if ticker_fundamentals.empty:
        result["ticker"] = ticker
        return result.reset_index(drop=True)

    ticker_fundamentals = ticker_fundamentals.drop(columns=["ticker"])

    result = result.sort_values(ohlcv_time_col).reset_index(drop=True)
    ticker_fundamentals = ticker_fundamentals.sort_values(
        fundamental_date_col
    ).reset_index(drop=True)

    result = pd.merge_asof(
        result,
        ticker_fundamentals,
        left_on=ohlcv_time_col,
        right_on=fundamental_date_col,
        direction="backward",
    )

    result["ticker"] = ticker

    if drop_nans:
        result = result[result[fundamental_date_col].notna()].copy()

    return result.reset_index(drop=True)