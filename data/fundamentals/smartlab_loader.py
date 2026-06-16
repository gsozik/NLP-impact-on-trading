from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


class FundamentalsDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return FundamentalsDataFrame

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(path, index=False)
        print(f"Saved {len(self)} rows to {path}")


@dataclass
class SmartLabFundamentalsLoader:
    tickers: list[str]
    metrics: list[str]
    start: str
    end: str
    period_type: str = "quarter"  # "quarter" или "year"
    report_type: str = "MSFO"
    save: bool = True
    filename: str = "storage/fundamentals.csv"

    base_url: str = "https://smart-lab.ru/q"

    def load(self) -> FundamentalsDataFrame:
        all_rows = []

        for ticker in self.tickers:
            print(f"\nTicker: {ticker}")

            ticker_df = None

            for metric in self.metrics:
                metric_df = self._load_metric(ticker, metric)

                if metric_df.empty:
                    print(f"  {metric}: empty")
                    continue

                print(f"  {metric}: {len(metric_df)} rows")

                if ticker_df is None:
                    ticker_df = metric_df
                else:
                    ticker_df = ticker_df.merge(
                        metric_df,
                        on=["ticker", "period", "period_date", "available_date"],
                        how="outer",
                    )

            if ticker_df is not None and not ticker_df.empty:
                all_rows.append(ticker_df)

        if not all_rows:
            return FundamentalsDataFrame()

        df = pd.concat(all_rows, ignore_index=True)
        df = df.sort_values(["ticker", "available_date"]).reset_index(drop=True)
        df = FundamentalsDataFrame(df)

        if self.save:
            df.save(self.filename)

        return df

    def _load_metric(self, ticker: str, metric: str) -> pd.DataFrame:
        url = f"{self.base_url}/{ticker}/{self.report_type}/{metric}/"

        html = self._get_html(url)
        if html is None:
            return pd.DataFrame()

        try:
            tables = pd.read_html(StringIO(html))
        except Exception:
            return pd.DataFrame()

        table_idx = 1 if self.period_type == "quarter" else 0

        if len(tables) <= table_idx:
            return pd.DataFrame()

        table = tables[table_idx]

        return self._parse_smartlab_table(
            table=table,
            ticker=ticker,
            metric=metric,
        )

    def _parse_smartlab_table(
        self,
        table: pd.DataFrame,
        ticker: str,
        metric: str,
    ) -> pd.DataFrame:
        rows = []
    
        metric_name = self._metric_name(metric)
    
        periods = table.iloc[2]
        values = table.iloc[3]
    
        for period_raw, value_raw in zip(periods, values):
            period = self._extract_period(period_raw)
    
            if period is None:
                continue
            
            period_date = pd.to_datetime(self._period_to_date(period))
            available_date = pd.to_datetime(self._period_to_available_date(period))
    
            start = pd.to_datetime(self.start) - pd.DateOffset(months=6)
            end = pd.to_datetime(self.end)
    
            if not (start <= available_date <= end):
                continue
            
            rows.append(
                {
                    "ticker": ticker,
                    "period": period,
                    "period_date": period_date,
                    "available_date": available_date,
                    metric_name: self._to_float(value_raw),
                }
            )
    
        return pd.DataFrame(rows)

    def _extract_period(self, value) -> str | None:
        value = str(value).strip().upper()

        if self.period_type == "quarter":
            match = re.search(r"(20\d{2})Q([1-4])", value)
            if match:
                return f"{match.group(1)}Q{match.group(2)}"

        if self.period_type == "year":
            match = re.search(r"^(20\d{2})$", value)
            if match:
                return match.group(1)

        return None

    def _period_to_date(self, period: str) -> str:
        if "Q" in period:
            year = int(period[:4])
            quarter = int(period[-1])

            return {
                1: f"{year}-03-31",
                2: f"{year}-06-30",
                3: f"{year}-09-30",
                4: f"{year}-12-31",
            }[quarter]

        return f"{period}-12-31"

    def _period_to_available_date(self, period: str) -> str:
        if "Q" in period:
            year = int(period[:4])
            quarter = int(period[-1])

            return {
                1: f"{year}-04-30",
                2: f"{year}-07-31",
                3: f"{year}-10-31",
                4: f"{year + 1}-01-31",
            }[quarter]

        return f"{int(period) + 1}-03-31"

    def _to_float(self, value) -> float | None:
        if pd.isna(value):
            return None

        value = str(value)
        value = value.replace("%", "")
        value = value.replace(",", ".")
        value = value.replace(" ", "")
        value = value.replace("\xa0", "")
        value = value.replace("−", "-")
        value = value.replace("—", "")

        value = re.sub(r"[^\d\.\-]", "", value)

        if value in ["", "-", ".", "-."]:
            return None

        try:
            return float(value)
        except ValueError:
            return None

    def _metric_name(self, metric: str) -> str:
        names = {
            "p_e": "pe",
            "p_b": "pb",
            "net_income":"net_income",
            "roe": "roe",
            "ev_ebitda": "ev_ebitda",
            "debt_ebitda": "debt_ebitda",
            "net_debt_ebitda": "net_debt_ebitda",
            "net_margin": "net_margin",
            "roa": "roa",
        }

        return names.get(metric, metric)

    def _get_html(self, url: str) -> str | None:
        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
                },
                timeout=20,
            )

            if response.status_code != 200:
                print(f"  bad status {response.status_code}: {url}")
                return None

            return response.text

        except requests.RequestException:
            print(f"  request error: {url}")
            return None