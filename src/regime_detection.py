"""
RegimeRadar — regime_detection.py  (Phase 3)

Fit a 3-state Gaussian HMM to discover latent volatility *regimes*
(calm / normal / turbulent) and attach a regime label to every trading day.

What this is
------------
A hidden Markov model treats the market as switching between a few unobserved
states, each with its own return/volatility distribution and its own persistence
(states are sticky — turbulence begets turbulence). We fit 3 states on daily
returns + trailing realized vol; the states separate naturally by volatility.

Important — these are POST-HOC labels
-------------------------------------
The Viterbi decode uses the *whole* return sequence, so the label at day t peeks
at the future. That is fine (and standard) for the project's purposes here:
  * describing history / visualizing regimes,
  * the "regime vs no-regime" analysis and per-regime models in Phase 4.
It is NOT safe to feed a full-sample regime label into a live forecast as-is —
that would be lookahead. When regimes become a live model input we will need an
online (forward-filtered, walk-forward) assignment. This module therefore also
saves smoothed state probabilities and flags the distinction loudly.

Output
------
data/processed/regimes.csv — regime id/name + per-state posterior probabilities,
aligned to the feature rows. Figure: reports/figures/07_regimes.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"

N_STATES = 3
REGIME_NAMES = {0: "Calm", 1: "Normal", 2: "Turbulent"}
REGIME_COLORS = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}
RANDOM_STATE = 42

# Observations the HMM sees. Daily return carries the sign/shock; realized vol
# carries the turbulence level — together they give clean volatility regimes.
OBS_COLS = ["ret_1d", "rv_21"]


def _load_features() -> pd.DataFrame:
    src = PROCESSED_DIR / "features.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found — run `python -m src.features` first (Phase 2)."
        )
    df = pd.read_csv(src, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def fit_hmm(df: pd.DataFrame):
    """Fit a 3-state Gaussian HMM; return (model, standardized X, mean, std)."""
    from hmmlearn.hmm import GaussianHMM

    X = df[OBS_COLS].to_numpy(dtype=float)
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    Xs = (X - mu) / sigma  # standardize so both obs contribute on equal footing

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=500,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )
    model.fit(Xs)
    return model, Xs, mu, sigma


def _order_states_by_vol(df: pd.DataFrame, raw_states: np.ndarray) -> dict[int, int]:
    """
    Map raw HMM state ids -> ranked ids so 0=lowest vol (Calm), 2=highest
    (Turbulent). HMM state numbering is arbitrary, so we sort by each state's
    mean realized vol.
    """
    tmp = pd.DataFrame({"state": raw_states, "rv": df["rv_21"].to_numpy()})
    order = tmp.groupby("state")["rv"].mean().sort_values().index.tolist()
    return {raw: rank for rank, raw in enumerate(order)}


def detect_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Fit HMM, decode regimes, relabel by volatility, return a tidy frame."""
    model, Xs, _, _ = fit_hmm(df)

    raw_states = model.predict(Xs)                 # Viterbi (post-hoc)
    proba = model.predict_proba(Xs)                # smoothed posteriors
    remap = _order_states_by_vol(df, raw_states)

    ranked = np.array([remap[s] for s in raw_states])
    # Reorder posterior columns to match the ranked labels.
    inv = {rank: raw for raw, rank in remap.items()}
    proba_ranked = np.column_stack([proba[:, inv[r]] for r in range(N_STATES)])

    out = pd.DataFrame(index=df.index)
    out["regime"] = ranked
    out["regime_name"] = [REGIME_NAMES[r] for r in ranked]
    for r in range(N_STATES):
        out[f"p_{REGIME_NAMES[r].lower()}"] = proba_ranked[:, r]
    # Carry realized + forward vol for the acceptance check / plotting.
    out["rv_21"] = df["rv_21"].to_numpy()
    out["target_rv_21"] = df["target_rv_21"].to_numpy()
    return out, model


def _persistence(regimes: pd.Series) -> pd.Series:
    """Average run length (in trading days) of each regime — how sticky it is."""
    runs = (regimes != regimes.shift()).cumsum()
    lengths = regimes.groupby(runs).transform("size")
    first = regimes[regimes != regimes.shift()]
    run_len = lengths[regimes != regimes.shift()]
    return pd.Series(run_len.to_numpy(), index=first.to_numpy()).groupby(level=0).mean()


def plot_regimes(out: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(out.index, out["rv_21"], lw=0.7, color="#333333", label="Realized vol (21d)")
    # Shade the background by regime.
    changes = out.index[out["regime"] != out["regime"].shift()].tolist()
    changes = [out.index[0]] + changes + [out.index[-1]]
    for i in range(len(changes) - 1):
        seg = out.loc[changes[i]:changes[i + 1]]
        if len(seg):
            ax.axvspan(changes[i], changes[i + 1],
                       color=REGIME_COLORS[int(seg["regime"].iloc[0])], alpha=0.15)
    for r in range(N_STATES):
        ax.plot([], [], color=REGIME_COLORS[r], lw=6, alpha=0.4, label=REGIME_NAMES[r])
    ax.set_title("HMM volatility regimes over realized volatility")
    ax.set_ylabel("Annualized vol (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", ncol=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_regimes.png", dpi=120, bbox_inches="tight")
    plt.close()


def acceptance_check(out: pd.DataFrame, model) -> None:
    print("\n" + "=" * 64)
    print("PHASE 3 ACCEPTANCE CHECK")
    print("=" * 64)
    n = len(out)
    print(f"Rows          : {n}   ({out.index.min().date()} -> {out.index.max().date()})")

    print("\nRegime distribution & mean realized vol (must increase Calm->Turbulent):")
    grp = out.groupby("regime_name", sort=False).agg(
        days=("regime", "size"),
        mean_rv=("rv_21", "mean"),
        mean_fwd_vol=("target_rv_21", "mean"),
    )
    grp = grp.reindex([REGIME_NAMES[r] for r in range(N_STATES)])
    grp["share_%"] = (100 * grp["days"] / n).round(1)
    print(grp.round(2).to_string())

    monotonic = grp["mean_rv"].is_monotonic_increasing
    print(f"\nMonotonic vol ordering Calm<Normal<Turbulent : "
          f"{'YES [OK]' if monotonic else 'NO [!]'}")

    # Persistence: regimes should be sticky (avg run length >> 1 day).
    pers = _persistence(out["regime_name"])
    print("\nAvg run length (trading days) — regimes should be sticky:")
    for name in [REGIME_NAMES[r] for r in range(N_STATES)]:
        if name in pers.index:
            print(f"  {name:<10} {pers[name]:.1f}")

    # Crisis sanity: known turbulent windows should be mostly Turbulent.
    def turb_share(a, b):
        w = out.loc[a:b]
        return 100 * (w["regime_name"] == "Turbulent").mean() if len(w) else float("nan")

    print("\nCrisis check (% of window flagged Turbulent):")
    print(f"  2008 GFC   (2008-09-01..2009-03-31): {turb_share('2008-09-01','2009-03-31'):.0f}%")
    print(f"  COVID      (2020-02-20..2020-04-30): {turb_share('2020-02-20','2020-04-30'):.0f}%")
    print(f"  Calm 2017  (2017-01-01..2017-12-31): {turb_share('2017-01-01','2017-12-31'):.0f}%  (expect ~0%)")
    print("=" * 64)


def build_and_save() -> pd.DataFrame:
    df = _load_features()
    out, model = detect_regimes(df)
    out.to_csv(PROCESSED_DIR / "regimes.csv")
    plot_regimes(out)
    return out, model


def main() -> None:
    argparse.ArgumentParser(description="RegimeRadar regime detection").parse_args()
    print("Detecting volatility regimes (Phase 3) ...")
    out, model = build_and_save()
    acceptance_check(out, model)
    print(f"\nSaved -> {PROCESSED_DIR / 'regimes.csv'}")
    print(f"Figure -> {FIG_DIR / '07_regimes.png'}")


if __name__ == "__main__":
    sys.exit(main())
