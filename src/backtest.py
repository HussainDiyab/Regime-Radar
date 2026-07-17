"""
RegimeRadar — backtest.py  (Phase 7)

Does a better volatility forecast translate into better *risk-adjusted returns*?
We test a classic **volatility-targeting** strategy: size SPY exposure inversely
to forecast volatility so realized portfolio risk stays near a constant target
(lever up when calm, de-risk when turbulent), and compare it to buy-and-hold.

Three sleeves (the honest comparison)
-------------------------------------
1. Buy & hold SPY                         — the benchmark.
2. Vol-target on *persistence* (rv_21)     — naive forecast.
3. Vol-target on the *model* forecast      — walk-forward out-of-sample XGBoost.

Comparing (2) vs (3) isolates the question that matters: does the ML forecast
beat naive realized-vol targeting in a live-style strategy — not just on RMSE?

No-lookahead
------------
The forecast at day t (known at the close of t) sets the weight for t, which is
applied to the return earned from t to t+1 (`weight.shift(1) * ret`). Model
forecasts are out-of-sample (walk-forward test folds only). No future info sizes
a position. Financing/transaction costs are omitted (noted), so absolute Sharpes
are optimistic — but all sleeves are treated identically, so the *comparison* is
fair.

Outputs
-------
reports/metrics_backtest.csv
reports/figures/12_backtest_equity.png
reports/figures/13_backtest_drawdown.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

TARGET = "target_rv_21"
N_SPLITS = 5
TRADING_DAYS = 252
TARGET_VOL = 12.0          # annualized % risk target for the strategy
MAX_WEIGHT = 2.0           # leverage cap (position in [0, 2.0])

XGB_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
)


# --------------------------------------------------------------------------- #
# Out-of-sample model forecasts (walk-forward)
# --------------------------------------------------------------------------- #
def oos_model_forecast(feats: pd.DataFrame) -> pd.Series:
    """Concatenated out-of-sample XGBoost vol forecasts over the test folds."""
    from sklearn.model_selection import TimeSeriesSplit
    from xgboost import XGBRegressor

    feature_cols = [c for c in feats.columns if not c.startswith("target_")]
    X = feats[feature_cols].to_numpy(dtype=float)
    y = feats[TARGET].to_numpy(dtype=float)
    rv = feats["rv_21"].to_numpy(dtype=float)
    z = np.log(y) - np.log(rv)

    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    preds = pd.Series(index=feats.index, dtype=float)
    for tr, te in tss.split(X):
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(X[tr], z[tr])
        preds.iloc[te] = rv[te] * np.exp(m.predict(X[te]))
    return preds.dropna()


# --------------------------------------------------------------------------- #
# Strategy mechanics
# --------------------------------------------------------------------------- #
def vol_target_weights(forecast_vol: pd.Series) -> pd.Series:
    """Weight = target_vol / forecast_vol, clipped to [0, MAX_WEIGHT]."""
    return (TARGET_VOL / forecast_vol).clip(lower=0.0, upper=MAX_WEIGHT)


def strategy_returns(weights: pd.Series, spy_ret: pd.Series) -> pd.Series:
    """Apply yesterday's weight to today's return (no lookahead)."""
    return (weights.shift(1) * spy_ret).dropna()


# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #
def _max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def perf_stats(rets: pd.Series) -> dict:
    ann_ret = float(rets.mean() * TRADING_DAYS)
    ann_vol = float(rets.std() * np.sqrt(TRADING_DAYS))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    equity = (1 + rets).cumprod()
    mdd = _max_drawdown(equity)
    calmar = ann_ret / abs(mdd) if mdd < 0 else float("nan")
    return {
        "ann_return_%": 100 * ann_ret,
        "ann_vol_%": 100 * ann_vol,
        "sharpe": sharpe,
        "max_drawdown_%": 100 * mdd,
        "calmar": calmar,
    }


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def run():
    feats = pd.read_csv(PROCESSED_DIR / "features.csv", index_col=0, parse_dates=True)
    prices = pd.read_csv(PROCESSED_DIR / "dataset.csv", index_col=0, parse_dates=True)
    spy_ret = prices["spy_close"].pct_change()

    fc_model = oos_model_forecast(feats)          # OOS window defines the backtest
    idx = fc_model.index
    fc_persist = feats.loc[idx, "rv_21"]
    ret = spy_ret.reindex(idx)

    sleeves = {
        "buy_hold": strategy_returns(pd.Series(1.0, index=idx), ret),
        "voltarget_persistence": strategy_returns(vol_target_weights(fc_persist), ret),
        "voltarget_model": strategy_returns(vol_target_weights(fc_model), ret),
    }
    # Align all sleeves to a common date range for fair equity curves.
    common = sorted(set.intersection(*[set(s.index) for s in sleeves.values()]))
    sleeves = {k: v.reindex(common) for k, v in sleeves.items()}

    stats = pd.DataFrame({k: perf_stats(v) for k, v in sleeves.items()}).T
    equity = pd.DataFrame({k: (1 + v).cumprod() for k, v in sleeves.items()})
    return stats, equity, sleeves


def make_figures(equity: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = {"buy_hold": "Buy & hold", "voltarget_persistence": "Vol-target (persistence)",
              "voltarget_model": "Vol-target (model)"}
    colors = {"buy_hold": "#999999", "voltarget_persistence": "#ff7f0e",
              "voltarget_model": "#1f77b4"}

    # Equity curves
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for k in equity.columns:
        ax.plot(equity.index, equity[k], label=labels[k], color=colors[k], lw=1.3)
    ax.set_title(f"Growth of $1 — vol-targeting vs buy & hold (target {TARGET_VOL:.0f}% vol)")
    ax.set_ylabel("Equity (×)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_backtest_equity.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Drawdowns
    fig, ax = plt.subplots(figsize=(12, 4.0))
    for k in equity.columns:
        dd = 100 * (equity[k] / equity[k].cummax() - 1.0)
        ax.plot(equity.index, dd, label=labels[k], color=colors[k], lw=1.0)
    ax.set_title("Drawdown comparison")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_backtest_drawdown.png", dpi=120, bbox_inches="tight")
    plt.close()


def acceptance_check(stats: pd.DataFrame, equity: pd.DataFrame) -> None:
    print("\n" + "=" * 64)
    print("PHASE 7 ACCEPTANCE CHECK")
    print("=" * 64)
    print(f"Backtest window : {equity.index.min().date()} -> {equity.index.max().date()}"
          f"  ({len(equity)} days, out-of-sample)")
    print(f"Target vol {TARGET_VOL:.0f}%, leverage cap {MAX_WEIGHT:.1f}x, costs omitted\n")
    print(stats.round(2).to_string())

    bh = stats.loc["buy_hold"]
    vt = stats.loc["voltarget_model"]
    print(f"\nVol-target(model) vs buy&hold : "
          f"Sharpe {vt['sharpe']:.2f} vs {bh['sharpe']:.2f}, "
          f"maxDD {vt['max_drawdown_%']:.1f}% vs {bh['max_drawdown_%']:.1f}%")
    better_sharpe = vt["sharpe"] > bh["sharpe"]
    shallower_dd = vt["max_drawdown_%"] > bh["max_drawdown_%"]
    print(f"  Higher Sharpe   : {'YES' if better_sharpe else 'NO'}")
    print(f"  Shallower maxDD : {'YES' if shallower_dd else 'NO'}")

    vp = stats.loc["voltarget_persistence"]
    print(f"\nModel vs naive vol-target      : Sharpe {vt['sharpe']:.2f} vs {vp['sharpe']:.2f} "
          f"({'model better' if vt['sharpe'] > vp['sharpe'] else 'naive better/tied'})")
    print("=" * 64)


def main() -> None:
    argparse.ArgumentParser(description="RegimeRadar backtest (Phase 7)").parse_args()
    print("Backtesting volatility-targeting strategy (Phase 7) ...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stats, equity, _ = run()
    stats.to_csv(REPORTS_DIR / "metrics_backtest.csv")
    make_figures(equity)
    acceptance_check(stats, equity)
    print(f"\nSaved -> {REPORTS_DIR / 'metrics_backtest.csv'}")
    print(f"Figures -> {FIG_DIR / '12_backtest_equity.png'}, {FIG_DIR / '13_backtest_drawdown.png'}")


if __name__ == "__main__":
    sys.exit(main())
