"""
Phase 5a (step 2) - Window size sweep
=====================================
Place at:  module3/experiments/window_size/run_window_experiment.py
Run like your other scripts (Code Runner, or python -m experiments.window_size.run_window_experiment)

WHAT IT DOES
Trains one model per window size, changing NOTHING else, and records validation
PR-AUC. This is the empirical half of your window justification (the burst
analysis was the other half).

METHOD NOTES (these matter for your write-up)
  - Selection is on the OUT-OF-TIME VALIDATION split only. fraudTest.csv is never
    touched here. Choosing a hyperparameter on test data is leakage.
  - Everything except `window` is held fixed: seed, epochs, architecture, split.
  - SEEDS: with only ~165 validation frauds, a single run is noisy. Running 3 seeds
    and reporting mean +/- std is far more defensible than one number per window.
    Set SEEDS = [42] for a quick look, [42, 7, 123] for the reportable result.
  - Exact PR-AUC is computed with sklearn.average_precision_score on the restored
    best weights (Keras' AUC(curve='PR') is a threshold approximation - fine for
    early stopping, but report the exact value).
"""
import sys, os, copy, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import load_config, path
from src.features.phase1_features import build_dataset
from src.models.phase2_model import train_model

WINDOWS = [3, 5, 10, 20]      # 12 added: burst analysis pointed near 10-12
SEEDS = [42, 7, 123]                          # -> [42, 7, 123] for the reportable multi-seed run


def run_one(window, seed, base_cfg):
    """Build data + train one model at this window size. Returns metrics dict."""
    cfg = copy.deepcopy(base_cfg)
    cfg["features"]["window"] = window          # the ONLY thing that changes

    t0 = time.time()
    data = build_dataset(path(cfg["data"]["train_csv"]),
                         nrows=cfg["data"]["nrows"],
                         window=window,
                         val_fraction=cfg["data"]["val_fraction"])

    m, hist, (Xsv, Xnv, yv) = train_model(data, cfg,
                                          bidirectional=cfg["model"]["bidirectional"],
                                          seed=seed)

    # exact PR-AUC on the validation split, using the weights the callback restored
    p = m.predict([Xsv.astype("float32"), Xnv.astype("float32")], verbose=0).ravel()
    pr_auc = average_precision_score(yv, p)
    roc = roc_auc_score(yv, p)

    return {
        "window": window,
        "seed": seed,
        "val_pr_auc": round(float(pr_auc), 4),
        "val_roc_auc": round(float(roc), 4),
        "keras_val_pr_auc_best": round(float(max(hist.history["val_pr_auc"])), 4),
        "epochs_run": len(hist.history["loss"]),
        "val_frauds": int(yv.sum()),
        "minutes": round((time.time() - t0) / 60, 2),
    }


def main():
    base_cfg = load_config()
    print(f"windows={WINDOWS}  seeds={SEEDS}  nrows={base_cfg['data']['nrows']}")
    print(f"total runs = {len(WINDOWS) * len(SEEDS)}\n")

    rows = []
    for w in WINDOWS:
        for s in SEEDS:
            print(f"--- window={w} seed={s} ---")
            r = run_one(w, s, base_cfg)
            rows.append(r)
            print(f"    val PR-AUC={r['val_pr_auc']}  ROC={r['val_roc_auc']}  "
                  f"epochs={r['epochs_run']}  ({r['minutes']} min)\n")

    df = pd.DataFrame(rows)
    out_raw = path("results/metrics/phase5a_window_sweep_raw.csv")
    df.to_csv(out_raw, index=False)

    # summary across seeds
    summ = (df.groupby("window")["val_pr_auc"]
              .agg(mean="mean", std="std", n="count")
              .round(4).reset_index())
    out_sum = path("results/metrics/phase5a_window_sweep_summary.csv")
    summ.to_csv(out_sum, index=False)

    print("=" * 60)
    print("WINDOW SWEEP RESULT  (validation PR-AUC)")
    print("=" * 60)
    print(summ.to_string(index=False))

    best = summ.loc[summ["mean"].idxmax()]
    top = summ[summ["mean"] >= best["mean"] - 0.01]        # within 0.01 = effectively tied
    print(f"\n  best mean PR-AUC : window={int(best['window'])} ({best['mean']:.4f})")
    if len(top) > 1:
        print(f"  within 0.01 of best: windows {sorted(top['window'].astype(int).tolist())}")
        print(f"  -> prefer the SMALLEST of these ({int(top['window'].min())}): "
              f"simpler model, cheaper inference, same performance.")
    if int(best["window"]) == max(WINDOWS):
        print("  !! BOUNDARY WIN: best window is the largest tested.")
        print("     Extend the sweep (e.g. add 25, 30) before claiming an optimum.")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    if summ["std"].notna().any():
        plt.errorbar(summ["window"], summ["mean"], yerr=summ["std"].fillna(0),
                     marker="o", capsize=4)
    else:
        plt.plot(summ["window"], summ["mean"], marker="o")
    plt.xlabel("window size (transactions)")
    plt.ylabel("validation PR-AUC")
    plt.title("Phase 5a - window size sweep")
    plt.grid(alpha=0.3); plt.tight_layout()
    fig = path("results/figures/phase5a_window_sweep.png")
    plt.savefig(fig, dpi=130)

    print(f"\nsaved -> {out_sum}, {out_raw}, {fig}")


if __name__ == "__main__":
    main()