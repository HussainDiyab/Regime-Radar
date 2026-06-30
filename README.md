# RegimeRadar 🎯

**Regime-Aware Volatility Forecasting & Risk Dashboard**

RegimeRadar detects market volatility *regimes*, forecasts near-term **volatility
(risk, not price direction)** using both classical ML and deep learning, quantifies
forecast uncertainty with calibrated intervals, and validates real-world usefulness
through a backtested volatility-targeting strategy. Everything is surfaced in an
interactive Streamlit dashboard.

> This project predicts **risk, not direction.** There is no price up/down classifier
> anywhere in the pipeline — by design.

---

## Pipeline

```
raw data → EDA → regime detection → ML forecast → DL forecast
        → uncertainty-quantified forecast → backtested strategy → dashboard
```

| Phase | Module | What it does |
|------|--------|--------------|
| 0 | `src/data_pipeline.py` | Fetch + cache + align SPY/VIX/TNX (yfinance) and FRED macro, **no lookahead leakage** |
| 1 | `notebooks/01_eda.ipynb` | Vol clustering, fat tails, ACF of squared returns, VIX overlay, drawdowns |
| 2 | `src/features.py` | Lagged returns, rolling vol, RSI, z-score, lagged macro, forward-vol target |
| 3 | `src/regime_detection.py` | 3-state Gaussian HMM, post-hoc regime labels |
| 4 | `src/ml_baseline.py` | XGBoost (global vs per-regime), TimeSeriesSplit, SHAP |
| 5 | `src/dl_model.py` | LSTM, same walk-forward, honest ML-vs-DL comparison |
| 6 | `src/uncertainty.py` | Conformal / quantile intervals + coverage calibration |
| 7 | `src/backtest.py` | Vol-targeting vs buy-and-hold, Sharpe / max drawdown |
| 8 | `dashboard/app.py` | Streamlit app tying it all together |

## Data sources (all free, programmatic — no Kaggle)

- **yfinance**: `SPY` (OHLCV), `^VIX`, `^TNX` — daily, 2003–present
- **FRED** (`fredapi`): `FEDFUNDS`, `CPIAUCSL`, `UNRATE`, `T10Y2Y`

### No-lookahead macro alignment
Market data is daily; FRED macro is monthly and released with a lag (e.g. March CPI
is published mid-April). RegimeRadar aligns each monthly series by the date its value
**first became public** (FRED ALFRED vintage / `realtime_start`), then forward-fills
onto trading days. Verified: CPI steps on real mid-month release dates, not the 1st.

## Setup

```bash
pip install -r requirements.txt

# FRED API key (free, instant): https://fred.stlouisfed.org/docs/api/api_key.html
cp .env.example .env        # then paste your key into .env

# Phase 0: build the aligned dataset (caches raw pulls to data/raw/)
python -m src.data_pipeline           # add --refresh to re-pull from APIs
```

Output: `data/processed/dataset.csv` (5,909 daily rows, 2003-01-03 → present).

## Guardrails

- No price-direction / next-day-return prediction anywhere
- No random train/test splits — time-respecting walk-forward only
- Honest comparisons are deliverables: ML vs DL, regime vs no-regime, strategy vs benchmark
- API pulls are cached locally; never re-fetched on every run
