"""Real-world validation: LIVE Apple (AAPL) market data vs. both pricing engines.

Unlike a hardcoded historical snapshot, this script fetches AAPL's real
price history via yfinance at run-time -- "the exact movement" as of
whenever you run it -- computes realized volatility from that live data,
and prices an option with both engines using it. You only supply the
option's strike (K) and time to maturity (T); spot price, volatility, and
the risk-free rate all come from the live market.

Requires ordinary internet access to Yahoo Finance (via yfinance). This
will NOT work in a network-sandboxed environment (e.g. this repo's CI or a
locked-down container) -- it needs a normal internet connection, the kind
your own machine has. If the fetch fails, the script prints a clear error
instead of silently falling back to fabricated data.

This script runs two checks, same as before, just on live rather than
hardcoded data:
  1. GBM ASSUMPTION CHECK -- are AAPL's real daily log-returns normal
     (skew, excess kurtosis, Jarque-Bera), as Black-Scholes and the IAE
     pricer both assume?
  2. WALK-FORWARD MODEL AGREEMENT -- for every day in the fetched window
     with a full trailing volatility window, price an at-the-money call
     with both engines using that day's real spot + real trailing
     realized volatility, and measure how closely they track each other.

Deliberately NOT tested: a single option price against a single path's
realized payoff -- that comparison is statistically misleading (an option
price is an expectation over many possible paths).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from option_engine import EuropeanOption, OptionType
from qae_pricer import QAEOptionPricer

TICKER = "AAPL"
FETCH_PERIOD = "6mo"     # history window to pull for the walk-forward test
VOL_WINDOW = 20          # trailing trading days used for realized volatility
TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.04  # used only if the live T-bill fetch fails


def _prompt_float(prompt_text: str, default: Optional[float] = None, min_value: Optional[float] = None) -> float:
    """Robust CLI helper that retries until a valid float is entered."""
    while True:
        raw = input(prompt_text).strip()
        if not raw:
            if default is not None:
                return default
            print("  [!] Input required. Please enter a numerical value.")
            continue
        try:
            val = float(raw)
            if min_value is not None and val < min_value:
                print(f"  [!] Value must be >= {min_value}. Try again.")
                continue
            return val
        except ValueError:
            print("  [!] Invalid input. Please enter a valid number.")


def fetch_price_history(ticker: str = TICKER, period: str = FETCH_PERIOD) -> pd.DataFrame:
    """Fetches real daily price history via yfinance. Raises with a clear message on failure."""
    import yfinance as yf
    try:
        hist = yf.Ticker(ticker).history(period=period)
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Yahoo Finance to fetch {ticker} data ({exc}). "
            "This needs a normal internet connection -- check your network and try again."
        ) from exc
    if hist.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {ticker}. Check the ticker symbol and try again.")
    return hist


def fetch_risk_free_rate(default: float = DEFAULT_RISK_FREE_RATE) -> float:
    """Fetches the live 13-week T-bill yield (^IRX) as a risk-free rate proxy; falls back on failure."""
    import yfinance as yf
    try:
        irx = yf.Ticker("^IRX").history(period="5d")
        if irx.empty:
            raise RuntimeError("empty response")
        return float(irx["Close"].iloc[-1]) / 100.0
    except Exception as exc:
        print(f"  [!] Could not fetch a live risk-free rate ({exc}); using fallback {default:.2%}.")
        return default


def run_validation(hist: pd.DataFrame, risk_free_rate: float, K: float, T_days: float) -> None:
    dates = hist.index.strftime("%Y-%m-%d").tolist()
    closes = hist["Close"].values
    log_returns = np.diff(np.log(closes))
    T = T_days / TRADING_DAYS_PER_YEAR

    print(f"\nFetched {len(closes)} real trading days for {TICKER} ({dates[0]} to {dates[-1]})")
    print(f"Risk-free rate used: {risk_free_rate:.4%}")

    # ------------------------------------------------------------------ #
    # 0. Price TODAY's live snapshot with both engines
    # ------------------------------------------------------------------ #
    S0_today = closes[-1]
    realized_vol_today = log_returns[-VOL_WINDOW:].std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

    print("\n" + "=" * 70)
    print(f"0. LIVE SNAPSHOT ({dates[-1]}): pricing with real S0 and real trailing vol")
    print("=" * 70)
    print(f"Live spot S0:            {S0_today:.2f}")
    print(f"Trailing {VOL_WINDOW}-day realized vol: {realized_vol_today:.2%}")
    print(f"Strike K:                {K}")
    print(f"Time to maturity:        {T_days:.0f} trading days ({T:.4f} yrs)")

    analytical_today = EuropeanOption(S0_today, K, risk_free_rate, realized_vol_today, T, option_type=OptionType.CALL).price()
    print(f"\nAnalytical (Black-Scholes) price: {analytical_today:.4f}")

    # Gracefully catch Qiskit breakpoint bounds errors on the live snapshot
    try:
        qae_today = QAEOptionPricer(S0_today, K, risk_free_rate, realized_vol_today, T, num_uncertainty_qubits=3).price(epsilon_target=0.01)
        print(f"Quantum (IAE, n=3) price:         {qae_today.price:.4f}  ({qae_today.oracle_queries} oracle queries)")
        print(f"Absolute disagreement:            {abs(qae_today.price - analytical_today):.4f}  "
              f"({abs(qae_today.price - analytical_today) / analytical_today * 100:.2f}%)")
    except ValueError as exc:
        if "Breakpoints" in str(exc):
            print("Quantum (IAE, n=3) price:         [ERROR] Strike K falls outside the quantum distribution bounds for this S0.")
        else:
            raise

    # ------------------------------------------------------------------ #
    # 1. GBM assumption check
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print(f"1. GBM ASSUMPTION CHECK: are {TICKER}'s real log-returns normal?")
    print("=" * 70)
    skew = stats.skew(log_returns)
    kurt = stats.kurtosis(log_returns)
    jb_stat, jb_pvalue = stats.jarque_bera(log_returns)
    print(f"Daily log-return stdev:  {log_returns.std(ddof=1):.5f}  "
          f"(annualized: {log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR):.2%})")
    print(f"Skewness:                {skew:+.3f}  (0.0 = symmetric, as GBM assumes)")
    print(f"Excess kurtosis:         {kurt:+.3f}  (0.0 = normal tails, as GBM assumes)")
    print(f"Jarque-Bera test:        stat={jb_stat:.3f}, p-value={jb_pvalue:.4f}")
    if jb_pvalue < 0.05:
        print("  -> Rejects normality at 5% significance over this window.")
    else:
        print("  -> Does not reject normality at 5% significance over this window.")

    # ------------------------------------------------------------------ #
    # 2. Walk-forward model agreement across the live fetched window
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("2. WALK-FORWARD: analytical vs. quantum pricer on real daily market snapshots")
    print("=" * 70)

    rows = []
    for i in range(VOL_WINDOW, len(closes)):
        trailing_returns = log_returns[i - VOL_WINDOW:i]
        realized_vol = trailing_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        S0 = closes[i]

        analytical_price = EuropeanOption(S0, K, risk_free_rate, realized_vol, T, option_type=OptionType.CALL).price()
        
        # Gracefully handle out-of-bounds strikes during the historical walk-forward
        try:
            qae_result = QAEOptionPricer(S0, K, risk_free_rate, realized_vol, T, num_uncertainty_qubits=3).price(epsilon_target=0.01)
            qae_price = qae_result.price
            queries = qae_result.oracle_queries
            abs_err = abs(qae_price - analytical_price)
            pct_err = abs_err / analytical_price * 100 if analytical_price > 0 else 0.0
        except ValueError as exc:
            if "Breakpoints" in str(exc):
                qae_price = np.nan
                queries = 0
                abs_err = np.nan
                pct_err = np.nan
            else:
                raise

        rows.append({
            "date": dates[i], "S0": S0, "realized_vol": realized_vol,
            "analytical_price": analytical_price, "qae_price": qae_price,
            "abs_error": abs_err,
            "pct_error": pct_err,
            "oracle_queries": queries,
        })

    df = pd.DataFrame(rows)
    df.to_csv("real_world_validation_results.csv", index=False)
    
    valid_df = df.dropna(subset=["qae_price"])
    print(f"Ran {len(df)} real market snapshots ({dates[VOL_WINDOW]} to {dates[-1]})")
    print(f"Valid QAE days (K within bounds):   {len(valid_df)} / {len(df)}")
    
    if not valid_df.empty:
        print(f"Mean absolute error (QAE vs. BS):   {valid_df['abs_error'].mean():.4f}")
        print(f"Mean percentage error:              {valid_df['pct_error'].mean():.2f}%")
        print(f"Correlation (QAE price, BS price):  {valid_df['analytical_price'].corr(valid_df['qae_price']):.4f}")
    
    print("Saved real_world_validation_results.csv")

    # ------------------------------------------------------------------ #
    # Plot: live price/vol series + walk-forward pricing agreement
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    ax0 = axes[0]
    ax0.plot(dates, closes, color="tab:blue", label=f"{TICKER} real close (live fetch)")
    ax0.axvline(dates[-1], color="tab:red", ls="--", alpha=0.5, label=f"Today ({dates[-1]})")
    ax0.set_ylabel(f"{TICKER} close ($)", color="tab:blue")
    ax0.tick_params(axis="y", labelcolor="tab:blue")
    ax0.set_xticks(ax0.get_xticks()[:: max(len(dates) // 10, 1)])
    ax0.tick_params(axis="x", rotation=45)
    ax0.set_title(f"Live {TICKER} closing price, {dates[0]} to {dates[-1]} (fetched via yfinance)")
    ax0.legend(loc="upper left")
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(df["date"], df["analytical_price"], marker="o", markersize=3, label="Analytical (Black-Scholes)")
    # Matplotlib will naturally skip NaN marker points where QAE failed
    ax1.plot(df["date"], df["qae_price"], marker="s", markersize=3, label="Quantum (IAE, n=3)")
    ax1.set_ylabel(f"{T_days:.0f}-day K={K} call price ($)")
    ax1.set_title("Walk-forward pricing on real daily market snapshots (real S0 + real trailing realized vol)")
    ax1.set_xticks(ax1.get_xticks()[:: max(len(df) // 10, 1)])
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("real_world_validation.png", dpi=150)
    print("Saved real_world_validation.png")


if __name__ == "__main__":
    print("=" * 70)
    print(f" LIVE {TICKER} MARKET DATA vs. ANALYTICAL + QUANTUM PRICING ENGINES")
    print("=" * 70)
    print(f"Fetching live {TICKER} price history ({FETCH_PERIOD})...")

    try:
        history = fetch_price_history()
    except RuntimeError as exc:
        print(f"\n[FATAL] {exc}")
        raise SystemExit(1)

    r = fetch_risk_free_rate()
    live_spot = history["Close"].iloc[-1]

    print(f"\nLive spot price right now: {live_spot:.2f}")
    print("Enter the option contract you want to price against this (blank = sensible default):\n")
    K = _prompt_float(f"1. Strike price (K) [default: at-the-money, {round(live_spot)}]: ",
                       default=round(live_spot), min_value=0.01)
    T_days = _prompt_float("2. Time to maturity in trading days (T) [default: 30]: ",
                            default=30.0, min_value=1.0)

    run_validation(history, r, K, T_days)
