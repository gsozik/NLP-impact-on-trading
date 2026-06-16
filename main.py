from data import CsvOHLCVLoader, MoexOHLCVLoader, BybitOHLCVLoader
from data.news import LentaNewsLoader
import pandas as pd


#df_btc = BybitOHLCVLoader(symbol = 'BTCUSDT',  timeframe = '4h', start = '01-01-2026', end = '06-01-2026').load() # mm-dd-yyyy
#print(df_btc)

news = LentaNewsLoader(
    start_date="2026-06-13",
    end_date="2026-06-15",
    max_pages_per_day=2,
    save=True,
    filename="storage/lenta_all_news.csv",
).load()

print(news.shape)
print(news.head())

sber_news = news.keywords(["SBER", "Сбербанк", "акции Сбербанка"])

print(sber_news.shape)
print(sber_news[["datetime", "title", "url"]].head())

sber_news.save("storage/lenta_sber_news.csv")
