"""
RegimeRadar — dashboard/app.py  (Phase 8)

Interactive Streamlit dashboard that ties the whole pipeline together:
regimes, forecast + uncertainty, ML-vs-DL comparison, strategy backtest, and
model interpretability.

Run
---
    D:/regime-radar/.venv/Scripts/python.exe -m streamlit run dashboard/app.py
    (or:  streamlit run dashboard/app.py  from an env with streamlit installed)

Design
------
The heavy analysis already ran in Phases 0–7 and saved artifacts (CSV metrics +
PNG figures + the XGBoost model). The dashboard *reads* those artifacts and adds
one live element — the latest volatility forecast with a conformal band — using
the saved XGBoost model. It never re-runs the whole pipeline, so it loads fast
and always shows exactly what the analysis produced. No torch dependency here:
the deep-learning results are shown from saved metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
MODELS = ROOT / "models"

st.set_page_config(page_title="RegimeRadar", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data
def load_csv(path: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, **kw) if path.exists() else None


@st.cache_data
def load_features() -> pd.DataFrame | None:
    p = PROCESSED / "features.csv"
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else None


@st.cache_data
def load_regimes() -> pd.DataFrame | None:
    p = PROCESSED / "regimes.csv"
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else None


@st.cache_resource
def load_xgb():
    p = MODELS / "xgb_global.json"
    if not p.exists():
        return None
    from xgboost import XGBRegressor
    m = XGBRegressor()
    m.load_model(p)
    return m


def fig(name: str, caption: str | None = None):
    """Show a saved figure if present, else a friendly note."""
    p = FIGS / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"Figure `{name}` not found — run the matching phase to generate it.")


def _conformal_q(res: np.ndarray, coverage: float = 0.90) -> float:
    n = len(res)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(res, level, method="higher")) if level < 1 else float(res.max())


@st.cache_data
def latest_forecast() -> dict | None:
    """Latest 21-day vol forecast + conformal band from the saved XGBoost model."""
    feats = load_features()
    model = load_xgb()
    if feats is None or model is None:
        return None
    feature_cols = [c for c in feats.columns if not c.startswith("target_")]
    X = feats[feature_cols].to_numpy(dtype=float)
    rv = feats["rv_21"].to_numpy(dtype=float)
    zhat = model.predict(X)
    # In-sample residuals in z-space -> conformal band (illustrative; the
    # rigorously calibrated out-of-sample coverage is reported in the Uncertainty tab).
    y = feats["target_rv_21"].to_numpy(dtype=float)
    z = np.log(y) - np.log(rv)
    q = _conformal_q(np.abs(z - zhat), 0.90)
    pred = rv[-1] * np.exp(zhat[-1])
    return {
        "date": feats.index[-1],
        "current_rv": float(rv[-1]),
        "forecast": float(pred),
        "lower": float(rv[-1] * np.exp(zhat[-1] - q)),
        "upper": float(rv[-1] * np.exp(zhat[-1] + q)),
    }


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("RegimeRadar")
st.caption("Regime-aware volatility forecasting and risk dashboard — predicts risk, not direction.")

with st.sidebar:
    st.header("About")
    st.markdown(
        "- Forecasts **21-day forward realized volatility**\n"
        "- No price up/down prediction anywhere\n"
        "- Walk-forward validation (no lookahead)\n"
        "- Honest ML-vs-DL comparison"
    )
    st.divider()
    st.caption("Phases 0–7 produce the artifacts; this app reads them and adds a live forecast.")


# --------------------------------------------------------------------------- #
# Scorecard (computed live from saved metrics so it stays in sync)
# --------------------------------------------------------------------------- #
ml = load_csv(REPORTS / "metrics_ml.csv")
dl = load_csv(REPORTS / "metrics_dl.csv")
unc = load_csv(REPORTS / "metrics_uncertainty.csv")
bt = load_csv(REPORTS / "metrics_backtest.csv", index_col=0)

c1, c2, c3, c4 = st.columns(4)
if dl is not None:
    lstm_rmse = dl[dl["model"] == "lstm"]["RMSE"].mean()
    base_rmse = dl[dl["model"] == "persistence"]["RMSE"].mean()
    c1.metric("Best model RMSE (LSTM)", f"{lstm_rmse:.2f}",
              f"{100*(base_rmse-lstm_rmse)/base_rmse:+.0f}% vs persistence")
if ml is not None:
    r2 = ml[ml["model"] == "global_xgb"]["R2"].mean()
    c2.metric("XGBoost R²", f"{r2:.2f}", "vs 0.06 baseline")
if unc is not None:
    fold = unc[unc.get("kind", "") == "fold"] if "kind" in unc.columns else unc
    cov = fold["empirical_coverage"].mean() if "empirical_coverage" in fold else np.nan
    c3.metric("90% interval coverage", f"{cov:.0%}" if cov == cov else "—", "target 90%")
if bt is not None and "sharpe" in bt.columns:
    s_model = bt.loc["voltarget_model", "sharpe"]
    s_bh = bt.loc["buy_hold", "sharpe"]
    c4.metric("Strategy Sharpe", f"{s_model:.2f}", f"vs {s_bh:.2f} buy&hold")

st.divider()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_fc, tab_reg, tab_mdl, tab_unc, tab_bt, tab_eda = st.tabs(
    ["Forecast", "Regimes", "Models", "Uncertainty", "Strategy", "EDA"]
)

# --- Forecast -------------------------------------------------------------- #
with tab_fc:
    st.subheader("Latest volatility forecast")
    lf = latest_forecast()
    if lf is None:
        st.info("Run Phases 2 & 4 to enable the live forecast.")
    else:
        a, b, c = st.columns(3)
        a.metric("As of", lf["date"].date().isoformat())
        b.metric("Current realized vol (21d)", f"{lf['current_rv']:.1f}%")
        delta = lf["forecast"] - lf["current_rv"]
        c.metric("Forecast fwd vol (21d)", f"{lf['forecast']:.1f}%", f"{delta:+.1f} pts")
        st.markdown(
            f"**90% conformal interval:** "
            f"`{lf['lower']:.1f}%  –  {lf['upper']:.1f}%`  "
            f"(band widens with the vol level; calibration verified in the Uncertainty tab)."
        )
        regimes = load_regimes()
        if regimes is not None:
            last = regimes.iloc[-1]
            st.markdown(f"**Current regime:** {last['regime_name']} "
                        f"(P={last[f'p_{last.regime_name.lower()}']:.2f})")

# --- Regimes --------------------------------------------------------------- #
with tab_reg:
    st.subheader("Volatility regimes (3-state HMM)")
    fig("07_regimes.png", "Green=Calm, Orange=Normal, Red=Turbulent")
    regimes = load_regimes()
    if regimes is not None:
        dist = regimes["regime_name"].value_counts(normalize=True).reindex(
            ["Calm", "Normal", "Turbulent"]) * 100
        cols = st.columns(3)
        for col, (name, pct) in zip(cols, dist.items()):
            col.metric(f"{name} share", f"{pct:.0f}%")

# --- Models ---------------------------------------------------------------- #
with tab_mdl:
    st.subheader("Model comparison: ML vs DL vs baseline")
    fig("09_ml_vs_dl.png")
    if ml is not None and dl is not None:
        tbl = pd.concat([
            ml.groupby("model")[["RMSE", "MAE", "R2"]].mean(),
            dl.groupby("model")[["RMSE", "MAE", "R2"]].mean().loc[["lstm"]],
        ]).reindex(["persistence", "global_xgb", "per_regime_xgb", "lstm"])
        st.dataframe(tbl.style.format("{:.3f}"), width="stretch")
    st.markdown("**Interpretability — SHAP feature importance**")
    fig("08_shap_importance.png")
    st.caption("Volatility risk premium (VIX − realized vol) is the dominant driver.")

# --- Uncertainty ----------------------------------------------------------- #
with tab_unc:
    st.subheader("Conformal prediction intervals")
    c1, c2 = st.columns(2)
    with c1:
        fig("10_conformal_intervals.png")
    with c2:
        fig("11_coverage_calibration.png")
    if unc is not None and "empirical_coverage" in unc.columns:
        fold = unc[unc["kind"] == "fold"] if "kind" in unc.columns else unc
        st.metric("Mean out-of-sample coverage (90% target)",
                  f"{fold['empirical_coverage'].mean():.1%}")

# --- Strategy -------------------------------------------------------------- #
with tab_bt:
    st.subheader("Volatility-targeting backtest")
    c1, c2 = st.columns(2)
    with c1:
        fig("12_backtest_equity.png")
    with c2:
        fig("13_backtest_drawdown.png")
    if bt is not None:
        st.dataframe(bt.style.format("{:.2f}"), width="stretch")
        st.caption("Note: vol-targeting beats buy and hold, but the ML forecast did "
                   "not beat naive realized-vol targeting. Better RMSE did not mean a better strategy.")

# --- EDA ------------------------------------------------------------------- #
with tab_eda:
    st.subheader("Exploratory data analysis")
    fig("01_price_drawdown.png", "Price & drawdowns")
    c1, c2 = st.columns(2)
    with c1:
        fig("02_returns_clustering.png", "Volatility clustering")
        fig("04_acf_vol_memory.png", "Vol memory: ACF of squared returns")
    with c2:
        fig("03_fat_tails.png", "Fat tails vs Normal")
        fig("05_vix_vs_realized.png", "VIX vs realized vol")
