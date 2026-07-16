"""
RegimeRadar — uncertainty.py  (Phase 6)

A point volatility forecast is not enough — risk management needs a *range*.
This phase wraps the forecaster in **split-conformal prediction intervals** and
checks that their coverage is actually calibrated (a 90% interval should contain
the truth ~90% of the time, out of sample).

Why conformal
-------------
Conformal prediction gives finite-sample, distribution-free coverage guarantees:
no Gaussian-error assumption (our residuals are fat-tailed), just the exchange-
ability we approximate with a time-respecting split. We calibrate in the anchored
z-space  z = log(target_rv_21) - log(rv_21),  where residuals are far more
homoscedastic, then map the band back to vol level:
    interval_vol = rv_21 * exp( zhat ± q )   = pred_vol * [exp(-q), exp(+q)]
This yields an *asymmetric, positivity-respecting, heteroscedastic* band in vol
space for free (wider bands when the vol level is higher).

Protocol (no leakage)
---------------------
Walk-forward `TimeSeriesSplit(5)`. Each train fold is split chronologically into
proper-train (fit the model) and a later calibration slice (measure residual
quantiles). Intervals on the still-later test fold use only calibration
residuals. Nothing peeks forward.

Outputs
-------
reports/metrics_uncertainty.csv        — per-fold coverage & width; calibration table
reports/figures/10_conformal_intervals.png
reports/figures/11_coverage_calibration.png
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
TARGET_COVERAGE = 0.90                 # main nominal level (alpha = 0.10)
CAL_FRACTION = 0.25                    # last 25% of each train fold = calibration
NOMINAL_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]

XGB_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
)


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "features.csv", index_col=0, parse_dates=True)
    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    rv = df["rv_21"].to_numpy(dtype=float)
    return df.index, X, y, rv


def conformal_quantile(residuals: np.ndarray, coverage: float) -> float:
    """Finite-sample conformal quantile of |residuals| for the target coverage."""
    n = len(residuals)
    level = np.ceil((n + 1) * coverage) / n     # conformal correction
    if level >= 1.0:
        return float(residuals.max())
    return float(np.quantile(residuals, level, method="higher"))


def run() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from sklearn.model_selection import TimeSeriesSplit
    from xgboost import XGBRegressor

    idx, X, y, rv = load_data()
    z = np.log(y) - np.log(rv)                   # anchored target
    tss = TimeSeriesSplit(n_splits=N_SPLITS)

    fold_rows = []
    calib_accum = {p: {"hit": 0, "n": 0} for p in NOMINAL_LEVELS}
    last_plot = None

    for fold, (tr, te) in enumerate(tss.split(X), start=1):
        n_cal = max(50, int(CAL_FRACTION * len(tr)))
        proper, cal = tr[:-n_cal], tr[-n_cal:]

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X[proper], z[proper])

        res_cal = np.abs(z[cal] - model.predict(X[cal]))     # calibration residuals
        zhat_te = model.predict(X[te])
        res_te = np.abs(z[te] - zhat_te)

        # Main 90% interval (in vol space)
        q = conformal_quantile(res_cal, TARGET_COVERAGE)
        pred_vol = rv[te] * np.exp(zhat_te)
        lower = rv[te] * np.exp(zhat_te - q)
        upper = rv[te] * np.exp(zhat_te + q)
        covered = (y[te] >= lower) & (y[te] <= upper)
        fold_rows.append({
            "fold": fold,
            "empirical_coverage": float(covered.mean()),
            "mean_width_vol": float(np.mean(upper - lower)),
        })

        # Calibration table: coverage at several nominal levels
        for p in NOMINAL_LEVELS:
            qp = conformal_quantile(res_cal, p)
            calib_accum[p]["hit"] += int(np.sum(res_te <= qp))
            calib_accum[p]["n"] += len(res_te)

        last_plot = {"dates": idx[te], "y": y[te], "pred": pred_vol,
                     "lower": lower, "upper": upper, "fold": fold}

    fold_df = pd.DataFrame(fold_rows)
    calib_df = pd.DataFrame({
        "nominal": NOMINAL_LEVELS,
        "empirical": [calib_accum[p]["hit"] / calib_accum[p]["n"] for p in NOMINAL_LEVELS],
    })
    return fold_df, calib_df, last_plot


def make_figures(last_plot: dict, calib_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Intervals on the last (most recent) out-of-sample fold
    d = last_plot
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.fill_between(d["dates"], d["lower"], d["upper"], color="#1f77b4", alpha=0.25,
                    label=f"{int(TARGET_COVERAGE*100)}% conformal interval")
    ax.plot(d["dates"], d["pred"], color="#1f77b4", lw=1.2, label="Forecast")
    ax.plot(d["dates"], d["y"], color="#d62728", lw=1.0, label="Realized (actual)")
    ax.set_title(f"Conformal prediction intervals — out-of-sample fold {d['fold']}")
    ax.set_ylabel("Annualized vol (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_conformal_intervals.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 2) Reliability diagram: nominal vs empirical coverage
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(calib_df["nominal"], calib_df["empirical"], "o-", color="#1f77b4",
            lw=1.5, label="Conformal")
    for _, r in calib_df.iterrows():
        ax.annotate(f"{r['empirical']:.2f}", (r["nominal"], r["empirical"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage (out-of-sample)")
    ax.set_title("Coverage calibration")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_coverage_calibration.png", dpi=120, bbox_inches="tight")
    plt.close()


def acceptance_check(fold_df: pd.DataFrame, calib_df: pd.DataFrame) -> None:
    print("\n" + "=" * 64)
    print("PHASE 6 ACCEPTANCE CHECK")
    print("=" * 64)
    print(f"Target coverage: {TARGET_COVERAGE:.0%}\n")
    print("Per-fold (out-of-sample):")
    print(fold_df.round(3).to_string(index=False))

    mean_cov = fold_df["empirical_coverage"].mean()
    mean_w = fold_df["mean_width_vol"].mean()
    print(f"\nMean empirical coverage : {mean_cov:.3f}  (target {TARGET_COVERAGE:.2f})")
    print(f"Mean interval width     : {mean_w:.2f} vol points")

    ok = abs(mean_cov - TARGET_COVERAGE) <= 0.05
    print(f"Coverage within +/-5pp of target : {'YES [OK]' if ok else 'NO [!]'}")

    print("\nCalibration table (nominal vs empirical):")
    tmp = calib_df.copy()
    tmp["abs_gap"] = (tmp["empirical"] - tmp["nominal"]).abs()
    print(tmp.round(3).to_string(index=False))
    print(f"\nMax calibration gap : {tmp['abs_gap'].max():.3f}")
    print("=" * 64)


def main() -> None:
    argparse.ArgumentParser(description="RegimeRadar uncertainty (Phase 6)").parse_args()
    print("Building conformal prediction intervals (Phase 6) ...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df, calib_df, last_plot = run()

    # Persist metrics (fold coverage + calibration table stacked).
    out = pd.concat([
        fold_df.assign(kind="fold"),
        calib_df.assign(kind="calibration"),
    ], ignore_index=True)
    out.to_csv(REPORTS_DIR / "metrics_uncertainty.csv", index=False)

    make_figures(last_plot, calib_df)
    acceptance_check(fold_df, calib_df)
    print(f"\nSaved -> {REPORTS_DIR / 'metrics_uncertainty.csv'}")
    print(f"Figures -> {FIG_DIR / '10_conformal_intervals.png'}, "
          f"{FIG_DIR / '11_coverage_calibration.png'}")


if __name__ == "__main__":
    sys.exit(main())
