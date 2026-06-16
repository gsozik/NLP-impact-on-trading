from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import feedparser
import pandas as pd
import trafilatura
from tqdm import tqdm


@dataclass
class RssNewsLoader:
    rss_urls: list[str]
    start_date: str
    end_date: str
    keywords: Optional[list[str]] = None
    max_articles_per_feed: int = 100
    min_text_len: int = 100
    parse_full_text: bool = True

    def load(self) -> pd.DataFrame:
        rows = []

        for rss_url in self.rss_urls:
            feed = feedparser.parse(rss_url)

            for entry in tqdm(feed.entries[:self.max_articles_per_feed], desc=rss_url):
                url = entry.get("link", "")
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                published = entry.get("published") or entry.get("updated")

                if not url or not published:
                    continue

                dt = pd.to_datetime(published, errors="coerce", utc=True)

                if pd.isna(dt) or not self._in_date_range(dt):
                    continue

                base_text = f"{title} {summary}".strip()

                if self.keywords and not self._contains_keywords(base_text):
                    continue

                text = summary

                if self.parse_full_text:
                    parsed_text = self._extract_text(url)
                    if parsed_text:
                        text = parsed_text

                final_text = f"{title} {text}".strip()

                if len(final_text) < self.min_text_len:
                    continue

                if self.keywords and not self._contains_keywords(final_text):
                    continue

                rows.append(
                    {
                        "datetime": dt,
                        "title": title,
                        "summary": summary,
                        "text": text,
                        "source": self._source_from_url(url),
                        "rss_url": rss_url,
                        "url": url,
                    }
                )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return self._clean(df)

    def _extract_text(self, url: str) -> Optional[str]:
        try:
            html = trafilatura.fetch_url(url)
            if html is None:
                return None

            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )

            if text is None:
                return None

            text = text.strip()
            return text if len(text) >= self.min_text_len else None

        except Exception:
            return None

    def _contains_keywords(self, text: str) -> bool:
        text = str(text).lower()
        return any(keyword.lower() in text for keyword in self.keywords)

    def _in_date_range(self, dt: pd.Timestamp) -> bool:
        start = pd.to_datetime(self.start_date, utc=True)
        end = pd.to_datetime(self.end_date, utc=True)
        return start <= dt < end

    def _source_from_url(self, url: str) -> str:
        domain = urlparse(url).netloc.replace("www.", "")

        if "vedomosti.ru" in domain:
            return "Vedomosti"
        if "finam.ru" in domain:
            return "Finam"
        if "moex.com" in domain:
            return "MOEX"
        if "1prime.ru" in domain:
            return "Prime"
        if "investing.com" in domain:
            return "Investing"

        return domain or "unknown"

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df.dropna(subset=["datetime"])

        for col in ["title", "summary", "text", "source", "rss_url", "url"]:
            df[col] = df[col].fillna("").astype(str)

        df = df.drop_duplicates(subset=["url"])
        df = df.sort_values("datetime").reset_index(drop=True)

        return df

    def save(self, path: str) -> pd.DataFrame:
        df = self.load()

        if df.empty:
            print("No news parsed.")
            return df

        df.to_csv(path, index=False)
        print(f"Saved {len(df)} news to {path}")
        return df