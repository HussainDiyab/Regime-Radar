"""
RegimeRadar — ml_baseline.py  (Phase 4)

Classical-ML volatility forecaster: XGBoost predicting 21-day forward realized
volatility, validated with a strict time-respecting walk-forward.

What we test (all deliverables are honest comparisons)
------------------------------------------------------
1. **Global XGBoost vs a persistence baseline.** The hard-to-beat naive forecast
   for volatility is "next month's vol = this month's realized vol" (vol is very
   persistent). A model only earns its keep if it beats that.
2. **Global vs per-regime models.** Does training a separate XGBoost per HMM
   regime (Calm/Normal/Turbulent) beat one global model?
3. **SHAP** feature importances so the model is interpretable, not a black box.

Target parameterization — anchor to persistence
------------------------------------------------
Trees cannot extrapolate: a model trained on a calm period predicts capped,
too-low vol when tested on a later crisis, and loses badly to persistence. So
instead of the raw vol level we predict the *log mean-reversion adjustment*
    z = log(target_rv_21) - log(rv_21)
and reconstruct  pred = rv_21 * exp(z).  The model now learns deviations from the
strong persistence baseline; predicting z≈0 recovers persistence exactly, so the
model can only help. This is standard practice in volatility forecasting.

Anti-leakage protocol
----------------------
* Evaluation uses `TimeSeriesSplit` — every test fold is strictly *after* its
  train fold. No shuffling, no random split.
* Features are the Phase-2 no-lookahead matrix; the target is forward vol.
* CAVEAT on per-regime: the HMM regime labels are post-hoc (Phase 3), so the
  per-regime result is an optimistic *analysis*, not a deployable number. It is
  reported as such, clearly, next to the honest global number.

Outputs
-------
reports/metrics_ml.csv  — per-fold + overall metrics
reports/figures/08_shap_importance.png
models/xgb_global.json  — final global model refit on all data (for the dashboard)
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
MODELS_DIR = ROOT / "models"

TARGET = "target_rv_21"
N_SPLITS = 5
REGIME_ORDER = ["Calm", "Normal", "Turbulent"]

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data() -> tuple[pd.DataFrame, list[str]]:
    feats = pd.read_csv(PROCESSED_DIR / "features.csv", index_col=0, parse_dates=True)
    regimes = pd.read_csv(PROCESSED_DIR / "regimes.csv", index_col=0, parse_dates=True)
    df = feats.join(regimes[["regime", "regime_name"]], how="inner")
    feature_cols = [c for c in feats.columns if not c.startswith("target_")]
    return df, feature_cols


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# --------------------------------------------------------------------------- #
# Walk-forward evaluation
# --------------------------------------------------------------------------- #
def walk_forward(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Run TimeSeriesSplit for persistence baseline, global XGB, per-regime XGB."""
    from sklearn.model_selection import TimeSeriesSplit
    from xgboost import XGBRegressor

    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    persist = df["rv_21"].to_numpy(dtype=float)   # current realized vol
    # Anchored target: log mean-reversion adjustment to persistence.
    z = np.log(y) - np.log(persist)
    regime_name = df["regime_name"].to_numpy()

    def to_vol(rv, z_pred):
        return rv * np.exp(z_pred)                # reconstruct vol level

    rows = []
    for fold, (tr, te) in enumerate(tss.split(X), start=1):
        # 1) Persistence baseline (z == 0)
        m_base = _metrics(y[te], persist[te])

        # 2) Global XGBoost on the anchored target
        gm = XGBRegressor(**XGB_PARAMS)
        gm.fit(X[tr], z[tr])
        pred_g = to_vol(persist[te], gm.predict(X[te]))
        m_glob = _metrics(y[te], pred_g)

        # 3) Per-regime XGBoost (train one model per regime on the train fold;
        #    predict each test row with the model for its regime).
        pred_r = np.empty_like(y[te])
        for rname in REGIME_ORDER:
            tr_mask = regime_name[tr] == rname
            te_mask = regime_name[te] == rname
            if te_mask.sum() == 0:
                continue
            if tr_mask.sum() < 50:  # too few to train — fall back to global
                pred_r[te_mask] = to_vol(persist[te][te_mask], gm.predict(X[te][te_mask]))
                continue
            rm = XGBRegressor(**XGB_PARAMS)
            rm.fit(X[tr][tr_mask], z[tr][tr_mask])
            pred_r[te_mask] = to_vol(persist[te][te_mask], rm.predict(X[te][te_mask]))
        m_reg = _metrics(y[te], pred_r)

        for name, m in [("persistence", m_base), ("global_xgb", m_glob),
                        ("per_regime_xgb", m_reg)]:
            rows.append({"fold": fold, "model": name, **m})

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per model across folds, plus % RMSE improvement over baseline."""
    summ = results.groupby("model")[["RMSE", "MAE", "R2"]].mean()
    summ = summ.reindex(["persistence", "global_xgb", "per_regime_xgb"])
    base_rmse = summ.loc["persistence", "RMSE"]
    summ["RMSE_vs_persist_%"] = (100 * (base_rmse - summ["RMSE"]) / base_rmse).round(1)
    return summ.round(4)


# --------------------------------------------------------------------------- #
# Final model + SHAP
# --------------------------------------------------------------------------- #
def fit_final_and_shap(df: pd.DataFrame, feature_cols: list[str]) -> None:
    from xgboost import XGBRegressor

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X = df[feature_cols]
    # Same anchored target used in evaluation: z = log(fwd vol) - log(current vol).
    z = np.log(df[TARGET]) - np.log(df["rv_21"])
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X, z)
    model.save_model(MODELS_DIR / "xgb_global.json")

    try:
        import shap
        import matplotlib.pyplot as plt

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        shap.summary_plot(sv, X, plot_type="bar", show=False, max_display=15)
        plt.title("XGBoost SHAP feature importance (forward-vol forecast)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "08_shap_importance.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"SHAP figure -> {FIG_DIR / '08_shap_importance.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[SHAP skipped] {exc}")
        # Fall back to XGBoost gain importance so we still get interpretability.
        imp = (pd.Series(model.feature_importances_, index=feature_cols)
               .sort_values(ascending=False))
        print("\nTop feature importances (XGBoost gain):")
        print(imp.head(15).round(4).to_string())


# --------------------------------------------------------------------------- #
# Acceptance check
# --------------------------------------------------------------------------- #
def acceptance_check(summ: pd.DataFrame) -> None:
    print("\n" + "=" * 64)
    print("PHASE 4 ACCEPTANCE CHECK")
    print("=" * 64)
    print("Walk-forward (TimeSeriesSplit, 5 folds) — mean metrics:\n")
    print(summ.to_string())

    glob_beats = summ.loc["global_xgb", "RMSE"] < summ.loc["persistence", "RMSE"]
    print(f"\nGlobal XGB beats persistence baseline : "
          f"{'YES [OK]' if glob_beats else 'NO [!] — model adds no value'}")

    reg_vs_glob = summ.loc["per_regime_xgb", "RMSE"] - summ.loc["global_xgb", "RMSE"]
    verdict = "per-regime better" if reg_vs_glob < 0 else "global better/tied"
    print(f"Per-regime vs global (RMSE delta)     : {reg_vs_glob:+.4f}  ({verdict})")
    print("  NOTE: per-regime uses POST-HOC regime labels (lookahead) -> optimistic.")
    print("=" * 64)


def build_and_run() -> pd.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df, feature_cols = load_data()
    results = walk_forward(df, feature_cols)
    results.to_csv(REPORTS_DIR / "metrics_ml.csv", index=False)
    summ = summarize(results)
    fit_final_and_shap(df, feature_cols)
    return summ


def main() -> None:
    argparse.ArgumentParser(description="RegimeRadar XGBoost baseline").parse_args()
    print("Training XGBoost volatility forecaster (Phase 4) ...")
    summ = build_and_run()
    acceptance_check(summ)
    print(f"\nSaved -> {REPORTS_DIR / 'metrics_ml.csv'}")
    print(f"Model -> {MODELS_DIR / 'xgb_global.json'}")


if __name__ == "__main__":
    sys.exit(main())
