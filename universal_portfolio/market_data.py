"""Real historical price data for Phase 5 (modern out-of-sample).

Adjusted daily close prices via yfinance (auto_adjust=True is this
installed version's default -- confirmed split/dividend-adjusted: no
discontinuity around NVDA's June 2024 10:1 split when checked directly).
Cached to `data/` (gitignored, like the sibling stock_price_forecast
repo's data/raw/) since this is fetched market data, not a fixed research
artifact -- rerun `fetch_and_cache` to refresh.

This is Phase 1-4's academic/synthetic data giving way to real, noisy,
single-realization history: there is exactly one historical path per
ticker, not thousands of Monte Carlo draws. Results here describe what
happened, not a statistically estimated expectation -- see README's
Phase 5 section for how that changes what can and can't be concluded.
"""
from __future__ import annotations

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data")
CACHE_PATH = os.path.join(DATA_DIR, "market_prices.csv")

ETF_TICKERS = ["SPY", "QQQ", "GLD", "TLT"]
STOCK_TICKERS = ["NVDA", "TSLA", "AMD", "MSFT", "GOOG", "ISRG", "WM"]
ALL_TICKERS = ETF_TICKERS + STOCK_TICKERS


def fetch_and_cache(start: str = "2016-01-01", end: str | None = None, tickers=ALL_TICKERS) -> pd.DataFrame:
    """Download adjusted close prices and cache to disk. Returns a
    DataFrame indexed by date, one column per ticker, dropped to rows
    where every requested ticker has a price (a common trading-day index
    across the whole set, so different pairs are compared over identical
    calendars).
    """
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    raw = raw[tickers].dropna(how="any")

    os.makedirs(DATA_DIR, exist_ok=True)
    raw.to_csv(CACHE_PATH)
    return raw


def load_prices(tickers=ALL_TICKERS) -> pd.DataFrame:
    """Load cached prices, fetching first if no cache exists."""
    if not os.path.exists(CACHE_PATH):
        return fetch_and_cache(tickers=tickers)
    return pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)[tickers]


def price_relatives(tickers=ALL_TICKERS) -> pd.DataFrame:
    """Day-over-day gross returns (x_t = P_t / P_{t-1}) for the cached
    tickers -- ready to feed into cover.py / strategies.py, same
    convention used throughout Phases 1-4.
    """
    prices = load_prices(tickers)
    return (prices / prices.shift(1)).dropna(how="any")
