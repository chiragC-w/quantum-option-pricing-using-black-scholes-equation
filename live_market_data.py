"""
live_market_data.py

Drop-in replacement for the hardcoded AAPL price list and hardcoded
T-bill yield in real_world_validation.py. Pulls fresh data from Yahoo
Finance (via yfinance) every time it's called, so each run of the
validation script reflects current market conditions instead of a
2026-06-23 -> 2026-08-31 snapshot.

Install:
    pip install yfinance --break-system-packages

Usage (inside real_world_validation.py):

    from live_market_data import get_aapl_closes, get_tbill_yield

    aapl_df = get_aapl_closes(n_days=50)     # replaces the hardcoded list
    closes = aapl_df["Close"].tolist()
    dates = aapl_df["Date"].tolist()

    r_free = get_tbill_yield()               # replaces the hardcoded 0.0386
"""

from __future__ import annotations
import datetime as dt
import os
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "market_data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def get_aapl_closes(n_days: int = 50, ticker: str = "AAPL") -> pd.DataFrame:
    """
    Pull the most recent n_days of daily closes for `ticker`.

    Returns a DataFrame with columns: Date, Close
    (newest date last, matching the ascending order the original
    hardcoded list used).

    Also writes a timestamped snapshot to market_data_cache/ so a run
    can be reproduced later even though the live source will have moved on.
    """
    # Pull extra calendar days to comfortably cover n_days of *trading* days
    # (weekends/holidays), then trim to the most recent n_days.
    lookback_days = int(n_days * 1.6) + 10
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)

    hist = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                        progress=False, auto_adjust=False)

    if hist.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker} ({start} to {end}). "
            "Check network access or ticker symbol."
        )

    hist = hist.tail(n_days).reset_index()
    df = hist[["Date", "Close"]].copy()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    snapshot_path = os.path.join(
        CACHE_DIR, f"{ticker}_{dt.date.today().isoformat()}.csv"
    )
    df.to_csv(snapshot_path, index=False)

    return df


def get_tbill_yield(maturity: str = "3mo") -> float:
    """
    Pull the current annualized T-bill yield as a decimal (e.g. 0.0386),
    replacing the hardcoded risk-free rate.

    maturity: "3mo" (^IRX, 13-week T-bill) or "13w" is the only option
    wired up here since that's what the script currently uses; extend
    the ticker map below if you want 1mo/6mo/1yr etc.
    """
    ticker_map = {"3mo": "^IRX"}  # 13-week Treasury Bill rate, quoted in %
    if maturity not in ticker_map:
        raise ValueError(f"Unsupported maturity '{maturity}'. Options: {list(ticker_map)}")

    symbol = ticker_map[maturity]
    hist = yf.Ticker(symbol).history(period="5d")

    if hist.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}.")

    latest_pct = hist["Close"].dropna().iloc[-1]  # e.g. 3.86 meaning 3.86%
    return float(latest_pct) / 100.0


if __name__ == "__main__":
    # Quick manual check: python live_market_data.py
    closes = get_aapl_closes(50)
    print(closes.head())
    print("...")
    print(closes.tail())
    print("Risk-free rate (3mo T-bill):", get_tbill_yield())
