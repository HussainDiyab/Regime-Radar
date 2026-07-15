"""
RegimeRadar — features.py  (Phase 2)

Turn the Phase-0 aligned dataset into a model-ready feature matrix + target.

Design rules (non-negotiable)
-----------------------------
1. **No lookahead.** Every feature at row t is computed only from data known at
   the close of day t (returns/prices/vix/macro up to and including t). The
   *target* is realized volatility over the strictly-future window [t+1, t+21],
   so there is a clean gap between what the model sees and what it predicts.
2. **Risk, not direction.** The target is forward realized volatility. There is
   no next-day-return / up-down label anywhere.
3. Macro series are already release-date aligned in Phase 0, so their level at t
   is genuinely public at t and safe to use directly.

Output
------
data/processed/features.csv — features + `target_rv_21`, warm-up and the
trailing rows without a full forward window dropped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths & config
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

TRADING_DAYS = 252          # annualization factor
FWD_HORIZON = 21            # target: realized vol over the next ~1 month
VOL_WINDOWS = [5, 10, 21, 63]
RETURN_LAGS = [1, 2, 3, 5, 10]
RSI_WINDOW = 14


# --------------------------------------------------------------------------- #
# Indicator helpers (all backward-looking)
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Classic RSI on a rolling-mean of gains/losses (uses only past prices)."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Parkinson high-low range volatility estimator, annualized (%).
    Uses the day's own high/low, both known at the close — no lookahead.
    """
    hl = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    var = factor * hl.rolling(window).mean()
    return np.sqrt(var * TRADING_DAYS) * 100


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix + forward-vol target from the aligned dataset."""
    out = pd.DataFrame(index=df.index)

    # --- Returns (past-only) -------------------------------------------------
    ret = np.log(df["spy_close"]).diff()
    out["ret_1d"] = ret
    for lag in RETURN_LAGS:
        out[f"ret_lag_{lag}"] = ret.shift(lag)

    # --- Realized volatility over several trailing windows -------------------
    # rv_w at t uses returns in [t-w+1, t], i.e. known at t. Annualized %.
    for w in VOL_WINDOWS:
        out[f"rv_{w}"] = ret.rolling(w).std() * np.sqrt(TRADING_DAYS) * 100

    # Vol-of-vol and vol trend (is turbulence rising or falling?)
    out["vov_21"] = out["rv_21"].rolling(21).std()
    out["rv_ratio_5_63"] = out["rv_5"] / out["rv_63"]     # short vs long vol

    # Parkinson range vol (uses intraday high/low)
    out["parkinson_21"] = _parkinson_vol(df["spy_high"], df["spy_low"], 21)

    # --- Momentum / trend (context, not a direction label) -------------------
    out["mom_21"] = np.log(df["spy_close"]).diff(21)
    out["mom_63"] = np.log(df["spy_close"]).diff(63)
    out["rsi_14"] = _rsi(df["spy_close"], RSI_WINDOW)

    # --- Price z-score (mean-reversion context) ------------------------------
    ma21 = df["spy_close"].rolling(21).mean()
    sd21 = df["spy_close"].rolling(21).std()
    out["z_close_21"] = (df["spy_close"] - ma21) / sd21

    # --- VIX & rates ---------------------------------------------------------
    out["vix"] = df["vix_close"]
    out["vix_chg_5"] = df["vix_close"].diff(5)
    # Volatility risk premium: implied (VIX) minus trailing realized vol.
    out["vrp"] = df["vix_close"] - out["rv_21"]
    out["tnx"] = df["tnx_close"]
    out["tnx_chg_21"] = df["tnx_close"].diff(21)

    # --- Macro (already release-date aligned in Phase 0) ---------------------
    out["fedfunds"] = df["fedfunds"]
    out["unrate"] = df["unrate"]
    out["t10y2y"] = df["t10y2y"]                       # yield-curve slope
    # CPI level trends hard; year-over-year inflation is the useful stationary form.
    out["cpi_yoy"] = df["cpiaucsl"].pct_change(TRADING_DAYS) * 100

    # --- Target: forward realized vol over [t+1, t+FWD_HORIZON] --------------
    # rolling(H).std() at t+H covers returns [t+1, t+H]; shift(-H) brings it to t.
    fwd_rv = ret.rolling(FWD_HORIZON).std().shift(-FWD_HORIZON)
    out[f"target_rv_{FWD_HORIZON}"] = fwd_rv * np.sqrt(TRADING_DAYS) * 100

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature column names (everything except the target)."""
    return [c for c in df.columns if not c.startswith("target_")]


def build_and_save(force: bool = False) -> pd.DataFrame:
    src = PROCESSED_DIR / "dataset.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found — run `python -m src.data_pipeline` first (Phase 0)."
        )
    raw = pd.read_csv(src, index_col=0, parse_dates=True)
    raw.index.name = "date"

    feats = build_features(raw)

    # Drop warm-up rows (leading NaNs from rolling windows) and the trailing
    # rows that don't have a full forward window for the target.
    target_col = f"target_rv_{FWD_HORIZON}"
    clean = feats.dropna().copy()

    out_path = PROCESSED_DIR / "features.csv"
    clean.to_csv(out_path)
    return clean


# --------------------------------------------------------------------------- #
# Acceptance check
# --------------------------------------------------------------------------- #
def acceptance_check(df: pd.DataFrame) -> None:
    target_col = f"target_rv_{FWD_HORIZON}"
    feat_cols = feature_columns(df)

    print("\n" + "=" * 64)
    print("PHASE 2 ACCEPTANCE CHECK")
    print("=" * 64)
    print(f"Shape         : {df.shape[0]} rows x {df.shape[1]} cols "
          f"({len(feat_cols)} features + 1 target)")
    print(f"Date range    : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Target        : {target_col}  (forward realized vol, next {FWD_HORIZON}d)")

    na = df.isna().sum()
    na = na[na > 0]
    print(f"\nNaNs          : {'none [OK]' if na.empty else na.to_string()}")

    print("\nFeatures:")
    for c in feat_cols:
        print(f"  - {c}")

    # No-lookahead sanity: target must NOT be perfectly explained by current rv_21.
    # A high-but-imperfect correlation is expected (vol is persistent) and healthy.
    corr = df[["rv_21", target_col]].corr().iloc[0, 1]
    print(f"\ncorr(rv_21, {target_col}) = {corr:.3f}  "
          f"(persistent but < 1.0 => forecast has real work to do)")

    print("\nTarget summary (annualized %):")
    print(df[target_col].describe().round(2).to_string())
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="RegimeRadar feature engineering")
    parser.add_argument("--refresh", action="store_true",
                        help="rebuild even if features.csv exists")
    args = parser.parse_args()

    print("Building RegimeRadar features (Phase 2) ...")
    df = build_and_save(force=args.refresh)
    acceptance_check(df)
    print(f"\nSaved -> {PROCESSED_DIR / 'features.csv'}")


if __name__ == "__main__":
    sys.exit(main())
