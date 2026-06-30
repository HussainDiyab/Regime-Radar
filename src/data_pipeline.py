"""
RegimeRadar — data_pipeline.py  (Phase 0)

Fetch, cache, and align all raw data into a single daily DataFrame.

Sources
-------
- yfinance : SPY (OHLCV, adjusted), ^VIX, ^TNX (10y yield)
- FRED     : FEDFUNDS, CPIAUCSL, UNRATE (monthly), T10Y2Y (daily)

No-lookahead policy
-------------------
Market data is daily; FRED macro data is monthly and released with a lag
(e.g. March CPI is published in mid-April). If we naively forward-fill by the
*observation* date, the model sees March CPI on March 1 — that is lookahead
leakage. We instead align each monthly series by the date its value first became
public (FRED vintage / `realtime_start`), then forward-fill onto trading days.
Daily T10Y2Y is shifted one business day (published next day) as a conservative
guard. The result: every macro value is only "known" on/after its real release.

Caching
-------
Raw pulls are cached to data/raw/ as CSV so we never re-hit the APIs on rerun.
Use force_refresh=True (or `python data_pipeline.py --refresh`) to re-pull.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths & config
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2003-01-01"

MARKET_TICKERS = {
    "SPY": "SPY",       # S&P 500 ETF — OHLCV (adjusted)
    "VIX": "^VIX",      # CBOE Volatility Index
    "TNX": "^TNX",      # CBOE 10-Year Treasury Yield Index
}

# FRED series and their native frequency. Monthly series use vintage-aware
# (release-date) alignment; daily series use a 1-business-day publication lag.
FRED_SERIES = {
    "FEDFUNDS": "monthly",   # Effective Federal Funds Rate
    "CPIAUCSL": "monthly",   # CPI, all urban consumers
    "UNRATE": "monthly",     # Unemployment rate
    "T10Y2Y": "daily",       # 10Y-2Y Treasury spread
}

# Fallback publication lags (calendar days) if vintage history is unavailable.
FALLBACK_LAG_DAYS = {
    "FEDFUNDS": 5,
    "CPIAUCSL": 14,
    "UNRATE": 7,
    "T10Y2Y": 1,
}


# --------------------------------------------------------------------------- #
# Market data (yfinance)
# --------------------------------------------------------------------------- #
def fetch_market_data(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch SPY OHLCV + VIX/TNX closes, cache each, return aligned daily frame."""
    import yfinance as yf

    frames = {}
    for name, ticker in MARKET_TICKERS.items():
        cache = RAW_DIR / f"market_{name}.csv"
        if cache.exists() and not force_refresh:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
        else:
            print(f"  [yfinance] downloading {ticker} ...")
            df = yf.download(
                ticker,
                start=START_DATE,
                auto_adjust=True,
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                raise RuntimeError(f"yfinance returned no data for {ticker}")
            df.to_csv(cache)
        frames[name] = df

    # SPY: keep full OHLCV; VIX/TNX: keep Close only.
    spy = frames["SPY"][["Open", "High", "Low", "Close", "Volume"]].copy()
    spy.columns = [f"spy_{c.lower()}" for c in spy.columns]

    out = spy
    out["vix_close"] = frames["VIX"]["Close"]
    out["tnx_close"] = frames["TNX"]["Close"]
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.index.name = "date"
    # VIX/TNX occasionally miss a day SPY trades (different holiday calendars);
    # forward-fill those small gaps with the last known value (no lookahead).
    out[["vix_close", "tnx_close"]] = out[["vix_close", "tnx_close"]].ffill()
    return out


# --------------------------------------------------------------------------- #
# Macro data (FRED) — vintage-aware, no lookahead
# --------------------------------------------------------------------------- #
def _get_fred_client():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = os.getenv("FRED_API_KEY")
    if not key:
        return None
    from fredapi import Fred

    return Fred(api_key=key)


def _known_by_release_date(fred, series_id: str) -> pd.Series:
    """
    Return a series indexed by the date each value first became public.

    Uses FRED ALFRED vintages (`get_series_all_releases`): for each observation
    we take the earliest `realtime_start` (first release) and stamp the value to
    that knowledge date. This is the rigorous no-lookahead alignment.
    """
    allr = fred.get_series_all_releases(series_id)
    # Columns: 'realtime_start', 'date', 'value'
    allr = allr.dropna(subset=["value"]).copy()
    allr["realtime_start"] = pd.to_datetime(allr["realtime_start"])
    allr["date"] = pd.to_datetime(allr["date"])
    # First release per observation date.
    first = (
        allr.sort_values("realtime_start")
        .groupby("date", as_index=False)
        .first()
    )
    # Index by knowledge date; if two obs land on same release date keep the
    # latest observation's value.
    s = (
        first.sort_values("realtime_start")
        .groupby("realtime_start")["value"]
        .last()
    )
    s.index.name = "date"
    return s.astype(float)


def _known_by_lag(fred, series_id: str, lag_days: int) -> pd.Series:
    """Fallback: shift observation dates forward by a fixed publication lag."""
    s = fred.get_series(series_id, observation_start=START_DATE).dropna()
    s.index = pd.to_datetime(s.index) + pd.Timedelta(days=lag_days)
    s.index.name = "date"
    return s.astype(float)


def fetch_macro_data(force_refresh: bool = False) -> pd.DataFrame | None:
    """Fetch FRED series with release-date alignment; cache; return wide frame."""
    fred = _get_fred_client()
    if fred is None:
        print(
            "  [FRED] No FRED_API_KEY found in environment/.env — "
            "skipping macro pull. Copy .env.example to .env and add your key."
        )
        return None

    cols = {}
    for series_id, freq in FRED_SERIES.items():
        cache = RAW_DIR / f"fred_{series_id}.csv"
        if cache.exists() and not force_refresh:
            s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            print(f"  [FRED] downloading {series_id} ({freq}) ...")
            try:
                if freq == "monthly":
                    s = _known_by_release_date(fred, series_id)
                else:
                    s = _known_by_lag(fred, series_id, FALLBACK_LAG_DAYS[series_id])
            except Exception as exc:  # noqa: BLE001 — robust fallback
                print(f"    vintage fetch failed ({exc}); using fixed-lag fallback")
                s = _known_by_lag(fred, series_id, FALLBACK_LAG_DAYS[series_id])
            s = s[s.index >= pd.Timestamp(START_DATE)]
            s.name = series_id
            s.to_frame().to_csv(cache)
        cols[series_id.lower()] = s

    macro = pd.DataFrame(cols).sort_index()
    macro.index.name = "date"
    return macro


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def build_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """Align market + macro into one daily DataFrame on SPY trading days."""
    market = fetch_market_data(force_refresh=force_refresh)
    macro = fetch_macro_data(force_refresh=force_refresh)

    df = market.copy()

    if macro is not None:
        # Reindex macro onto the union of its own release dates and trading days,
        # forward-fill (a value persists until the next release), then restrict
        # to trading days. Forward-fill on a release-date index => no lookahead.
        combined_index = market.index.union(macro.index)
        macro_ff = macro.reindex(combined_index).ffill()
        macro_on_trading = macro_ff.reindex(market.index)
        df = df.join(macro_on_trading)

    df = df.sort_index()

    # Drop the leading warm-up rows before macro series have any release.
    if macro is not None:
        macro_cols = [c.lower() for c in FRED_SERIES]
        first_valid = df[macro_cols].dropna(how="all").index.min()
        df = df.loc[df.index >= first_valid]

    out_path = PROCESSED_DIR / "dataset.csv"
    df.to_csv(out_path)
    return df


# --------------------------------------------------------------------------- #
# Acceptance check
# --------------------------------------------------------------------------- #
def acceptance_check(df: pd.DataFrame) -> None:
    print("\n" + "=" * 64)
    print("PHASE 0 ACCEPTANCE CHECK")
    print("=" * 64)
    print(f"Shape            : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Date range       : {df.index.min().date()}  ->  {df.index.max().date()}")
    print(f"Columns          : {list(df.columns)}")

    na = df.isna().sum()
    na = na[na > 0]
    print("\nNaN counts (post-alignment):")
    if na.empty:
        print("  none - zero NaNs across all columns [OK]")
    else:
        for col, n in na.items():
            pct = 100 * n / len(df)
            print(f"  {col:<14} {n:>6}  ({pct:4.1f}%)")
        # Macro warm-up NaNs at the very start can be legitimate; flag anything
        # in the interior as unexpected.
        interior = df.iloc[63:]  # after a generous warm-up window
        interior_na = interior.isna().sum()
        interior_na = interior_na[interior_na > 0]
        if interior_na.empty:
            print("  (all NaNs are in the leading warm-up window - expected [OK])")
        else:
            print("  [!] UNEXPECTED interior NaNs:")
            print(interior_na.to_string())

    print("\nHead:")
    print(df.head(3).to_string())
    print("\nTail:")
    print(df.tail(3).to_string())
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="RegimeRadar data pipeline")
    parser.add_argument(
        "--refresh", action="store_true", help="force re-pull from APIs"
    )
    args = parser.parse_args()

    print("Building RegimeRadar dataset ...")
    df = build_dataset(force_refresh=args.refresh)
    acceptance_check(df)
    print(f"\nSaved -> {PROCESSED_DIR / 'dataset.csv'}")


if __name__ == "__main__":
    sys.exit(main())
