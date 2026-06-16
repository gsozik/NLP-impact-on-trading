from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class NewsDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return NewsDataFrame

    @classmethod
    def read_csv(cls, path: str) -> "NewsDataFrame":
        df = pd.read_csv(path)

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

        return cls(df)

    def keywords(self, words: list[str]) -> "NewsDataFrame":
        if self.empty:
            return NewsDataFrame(self)

        pattern = "|".join(re.escape(word) for word in words)

        mask = (
            self["title"].fillna("").astype(str).str.contains(pattern, case=False, regex=True)
            | self["text"].fillna("").astype(str).str.contains(pattern, case=False, regex=True)
        )

        return NewsDataFrame(self.loc[mask].copy().reset_index(drop=True))

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.to_csv(path, index=False)
        print(f"Saved {len(self)} rows to {path}")


@dataclass
class LentaNewsLoader:
    start_date: str
    end_date: str
    max_pages_per_day: int = 20
    min_text_len: int = 80
    timeout: int = 20
    save: bool = True
    filename: str = "storage/lenta_news.csv"

    base_url: str = "https://lenta.ru"

    def load(self) -> NewsDataFrame:
        urls = self._collect_urls()
        rows = []

        print(f"Collected {len(urls)} urls")

        for url in tqdm(urls, desc="Parsing Lenta"):
            article = self._parse_article(url)

            if article is not None:
                rows.append(article)

        df = NewsDataFrame(rows)

        if not df.empty:
            df = NewsDataFrame(
                df.drop_duplicates(subset=["url"])
                  .sort_values("datetime")
                  .reset_index(drop=True)
            )

        if self.save:
            df.save(self.filename)

        return df

    def _collect_urls(self) -> list[str]:
        urls = []

        dates = pd.date_range(self.start_date, self.end_date, freq="D")

        for date in tqdm(dates, desc="Collecting Lenta pages"):
            day_url = f"{self.base_url}/{date.year}/{date.month:02d}/{date.day:02d}/"

            for page in range(1, self.max_pages_per_day + 1):
                page_url = day_url if page == 1 else f"{day_url}page/{page}/"
                soup = self._get_soup(page_url)

                if soup is None:
                    break

                page_urls = self._extract_article_urls(soup, page_url)

                if not page_urls:
                    break

                urls.extend(page_urls)

        return list(dict.fromkeys(urls))

    def _extract_article_urls(self, soup: BeautifulSoup, page_url: str) -> list[str]:
        urls = []

        for a in soup.find_all("a", href=True):
            url = urljoin(page_url, a["href"]).split("?")[0]

            if url.startswith(f"{self.base_url}/news/"):
                urls.append(url)

        return list(dict.fromkeys(urls))

    def _parse_article(self, url: str) -> Optional[dict]:
        soup = self._get_soup(url)

        if soup is None:
            return None

        title = self._get_title(soup)
        dt = self._get_datetime(url, soup)
        text = self._get_text(soup)

        if not title or dt is None:
            return None

        if len(text) < self.min_text_len:
            text = title

        return {
            "datetime": dt,
            "title": title,
            "text": text,
            "source": "Lenta.ru",
            "url": url,
        }

    def _get_title(self, soup: BeautifulSoup) -> str:
        h1 = soup.find("h1")

        if h1:
            return h1.get_text(" ", strip=True)

        meta = soup.find("meta", property="og:title")

        if meta:
            return meta.get("content", "").strip()

        return ""

    def _get_datetime(self, url: str, soup: BeautifulSoup) -> Optional[pd.Timestamp]:
        date_match = re.search(r"/news/(\d{4})/(\d{2})/(\d{2})/", url)

        if not date_match:
            return None

        year, month, day = date_match.groups()

        page_text = soup.get_text(" ", strip=True)
        time_match = re.search(r"(\d{1,2}:\d{2})", page_text)
        time_value = time_match.group(1) if time_match else "00:00"

        return pd.to_datetime(
            f"{year}-{month}-{day} {time_value}",
            errors="coerce",
            utc=True,
        )

    def _get_text(self, soup: BeautifulSoup) -> str:
        texts = []

        for p in soup.find_all("p"):
            text = p.get_text(" ", strip=True)

            if len(text) < 25:
                continue

            if self._is_bad_text(text):
                continue

            texts.append(text)

        return " ".join(texts).strip()

    def _is_bad_text(self, text: str) -> bool:
        bad_phrases = [
            "Что думаешь",
            "Нашли опечатку",
            "Комментарии",
            "Последние новости",
            "Подписывайтесь",
            "На сайте используются cookies",
        ]

        return any(phrase in text for phrase in bad_phrases)

    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        try:
            response = requests.get(
                url,
                headers=self._headers(),
                timeout=self.timeout,
            )

            if response.status_code != 200:
                return None

            return BeautifulSoup(response.text, "lxml")

        except requests.RequestException:
            return None

    def _headers(self) -> dict:
        return {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        }