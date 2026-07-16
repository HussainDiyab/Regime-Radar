"""
RegimeRadar — dl_model.py  (Phase 5)

Deep-learning volatility forecaster: an LSTM that reads a rolling window of the
Phase-2 features and predicts 21-day forward realized volatility — put head-to-
head against the Phase-4 XGBoost baseline under an *identical* protocol.

Fair-fight rules (this is the whole point of the phase)
-------------------------------------------------------
* Same features, same target, same anchored parameterization
      z = log(target_rv_21) - log(rv_21),   pred = rv_21 * exp(z)
  so the LSTM also learns mean-reversion deviations from persistence, not raw
  levels (trees AND nets struggle to extrapolate vol levels).
* Same walk-forward `TimeSeriesSplit(5)` — test folds strictly after train.
* Feature scaling is fit on the TRAIN fold only (no leakage), then applied to
  test. Sequences use only past rows (window ending at t).

Reality check
-------------
On ~5k daily rows this is a small-data regime where gradient-boosted trees are
hard to beat. We report the comparison honestly rather than tuning the LSTM
until it "wins".

Outputs
-------
reports/metrics_dl.csv          — per-fold LSTM + persistence
reports/figures/09_ml_vs_dl.png — RMSE comparison bar chart
models/lstm.pt                  — final model refit on all data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

TARGET = "target_rv_21"
N_SPLITS = 5
LOOKBACK = 21            # trading-day window fed to the LSTM
HIDDEN = 32
LAYERS = 1
DROPOUT = 0.1
EPOCHS = 80
PATIENCE = 10
LR = 1e-3
BATCH = 64
SEED = 42

DEVICE = torch.device("cpu")


def _seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN, layers: int = LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=DROPOUT if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Linear(hidden // 2, 1))

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, H)
        return self.head(out[:, -1, :]).squeeze(-1)   # last timestep -> scalar


# --------------------------------------------------------------------------- #
# Data / sequences
# --------------------------------------------------------------------------- #
def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(PROCESSED_DIR / "features.csv", index_col=0, parse_dates=True)
    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    rv = df["rv_21"].to_numpy(dtype=np.float32)
    return X, y, rv, feature_cols


def make_sequences(Xs: np.ndarray, target_idx: np.ndarray, lookback: int):
    """Build (n, lookback, F) sequences ending at each valid target index."""
    seqs, keep = [], []
    for i in target_idx:
        if i >= lookback - 1:
            seqs.append(Xs[i - lookback + 1: i + 1])
            keep.append(i)
    return np.asarray(seqs, dtype=np.float32), np.asarray(keep)


# --------------------------------------------------------------------------- #
# Train one model (with chronological early-stopping split)
# --------------------------------------------------------------------------- #
def train_model(seqs: np.ndarray, z: np.ndarray) -> LSTMRegressor:
    _seed_everything()
    n = len(seqs)
    n_val = max(1, int(0.15 * n))          # last 15% (chronological) = validation
    tr_s, tr_z = seqs[:-n_val], z[:-n_val]
    va_s, va_z = seqs[-n_val:], z[-n_val:]

    model = LSTMRegressor(seqs.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    tr_s_t = torch.tensor(tr_s, device=DEVICE)
    tr_z_t = torch.tensor(tr_z, device=DEVICE)
    va_s_t = torch.tensor(va_s, device=DEVICE)
    va_z_t = torch.tensor(va_z, device=DEVICE)

    best_val, best_state, wait = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(tr_s_t))
        for b in range(0, len(perm), BATCH):
            idx = perm[b: b + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(tr_s_t[idx]), tr_z_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(va_s_t), va_z_t).item()
        if val_loss < best_val - 1e-6:
            best_val, best_state, wait = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _metrics(y_true, y_pred) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def walk_forward(X, y, rv) -> pd.DataFrame:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    z_all = np.log(y) - np.log(rv)          # anchored target
    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []
    for fold, (tr, te) in enumerate(tss.split(X), start=1):
        # Scale on train fold only, apply everywhere (test uses train stats).
        scaler = StandardScaler().fit(X[tr])
        Xs = scaler.transform(X).astype(np.float32)

        tr_seqs, tr_keep = make_sequences(Xs, tr, LOOKBACK)
        te_seqs, te_keep = make_sequences(Xs, te, LOOKBACK)

        model = train_model(tr_seqs, z_all[tr_keep])
        model.eval()
        with torch.no_grad():
            z_pred = model(torch.tensor(te_seqs, device=DEVICE)).cpu().numpy()
        pred_vol = rv[te_keep] * np.exp(z_pred)

        m_lstm = _metrics(y[te_keep], pred_vol)
        m_base = _metrics(y[te_keep], rv[te_keep])   # persistence on same rows
        rows.append({"fold": fold, "model": "persistence", **m_base})
        rows.append({"fold": fold, "model": "lstm", **m_lstm})
        print(f"  fold {fold}: LSTM RMSE {m_lstm['RMSE']:.3f} | "
              f"persistence {m_base['RMSE']:.3f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Final model + comparison
# --------------------------------------------------------------------------- #
def fit_final(X, y, rv, n_features: int) -> None:
    from sklearn.preprocessing import StandardScaler

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    z = np.log(y) - np.log(rv)
    seqs, keep = make_sequences(Xs, np.arange(len(X)), LOOKBACK)
    model = train_model(seqs, z[keep])
    torch.save({"state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                "lookback": LOOKBACK, "n_features": n_features}, MODELS_DIR / "lstm.pt")


def comparison_figure(dl_summary: pd.DataFrame) -> pd.DataFrame:
    """Merge with the Phase-4 XGB result and draw the RMSE bar chart."""
    import matplotlib.pyplot as plt

    table = {"persistence": dl_summary.loc["persistence", "RMSE"],
             "lstm": dl_summary.loc["lstm", "RMSE"]}
    ml_path = REPORTS_DIR / "metrics_ml.csv"
    if ml_path.exists():
        ml = pd.read_csv(ml_path)
        table["global_xgb"] = ml[ml["model"] == "global_xgb"]["RMSE"].mean()

    order = [k for k in ["persistence", "global_xgb", "lstm"] if k in table]
    vals = [table[k] for k in order]
    labels = {"persistence": "Persistence", "global_xgb": "XGBoost", "lstm": "LSTM"}
    colors = {"persistence": "#999999", "global_xgb": "#1f77b4", "lstm": "#d62728"}

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar([labels[k] for k in order], vals, color=[colors[k] for k in order])
    ax.set_ylabel("Walk-forward RMSE (vol %, lower = better)")
    ax.set_title("Volatility forecast: ML vs DL vs baseline")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_ml_vs_dl.png", dpi=120, bbox_inches="tight")
    plt.close()
    return pd.Series(table).to_frame("RMSE")


def acceptance_check(dl_summary: pd.DataFrame, table: pd.DataFrame) -> None:
    print("\n" + "=" * 64)
    print("PHASE 5 ACCEPTANCE CHECK")
    print("=" * 64)
    print("Walk-forward RMSE (mean over 5 folds):\n")
    print(table.round(4).to_string())

    lstm_rmse = dl_summary.loc["lstm", "RMSE"]
    base_rmse = dl_summary.loc["persistence", "RMSE"]
    print(f"\nLSTM beats persistence : "
          f"{'YES' if lstm_rmse < base_rmse else 'NO'} "
          f"({100*(base_rmse-lstm_rmse)/base_rmse:+.1f}% RMSE)")
    if "global_xgb" in table.index:
        xgb = table.loc["global_xgb", "RMSE"]
        winner = "XGBoost" if xgb < lstm_rmse else "LSTM"
        print(f"Head-to-head winner    : {winner} "
              f"(XGB {xgb:.3f} vs LSTM {lstm_rmse:.3f})")
    print("=" * 64)


def main() -> None:
    argparse.ArgumentParser(description="RegimeRadar LSTM (Phase 5)").parse_args()
    print("Training LSTM volatility forecaster (Phase 5) ...")
    X, y, rv, feature_cols = load_data()
    results = walk_forward(X, y, rv)
    results.to_csv(REPORTS_DIR / "metrics_dl.csv", index=False)
    dl_summary = results.groupby("model")[["RMSE", "MAE", "R2"]].mean()
    fit_final(X, y, rv, len(feature_cols))
    table = comparison_figure(dl_summary)
    acceptance_check(dl_summary, table)
    print(f"\nSaved -> {REPORTS_DIR / 'metrics_dl.csv'}")
    print(f"Model -> {MODELS_DIR / 'lstm.pt'}")
    print(f"Figure -> {FIG_DIR / '09_ml_vs_dl.png'}")


if __name__ == "__main__":
    sys.exit(main())
