"""
Module 3 - Phase 2 Evaluation (interim presentation)
====================================================
Trains the full Bi-LSTM + network model, evaluates it on the temporal test
split, and writes presentation-ready figures + a metrics summary.

Outputs:
  eval_metrics.txt        - text summary you can read off in the talk
  fig_pr_curve.png        - precision-recall curve (the headline figure)
  fig_roc_curve.png       - ROC curve
  fig_confusion.png       - confusion matrix at the chosen threshold
  fig_threshold.png       - precision/recall/F1 vs threshold

Run:
    python3 evaluate_model.py             # default 300k rows (fast, representative)
    python3 evaluate_model.py 0           # full dataset
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)

from phase1_features import build_dataset, FEATURE_CONFIG
from phase2_model import build_model, binary_focal_loss
from tensorflow import keras

NROWS = 300000
if len(sys.argv) > 1:
    NROWS = None if sys.argv[1] == "0" else int(sys.argv[1])

TRAIN = "/mnt/user-data/uploads/fraudTrain.csv"
TEST = "/mnt/user-data/uploads/fraudTest.csv"

# ---- build features ----
train = build_dataset(TRAIN, nrows=NROWS)
test = build_dataset(TEST, fit_from=train, nrows=NROWS)
cfg = FEATURE_CONFIG
Xs, Xn, y = train["X_seq"], train["X_net"], train["y"]
cut = int(len(y) * 0.85)

# ---- train full bidirectional model ----
model = build_model(window=cfg["window"], n_seq_features=cfg["n_seq_features"],
                    n_categories=train["n_categories"],
                    n_net_features=len(cfg["net_features"]), bidirectional=True)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=binary_focal_loss(gamma=2.0, alpha=0.25))
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                   restore_best_weights=True)
model.fit([Xs[:cut], Xn[:cut]], y[:cut],
          validation_data=([Xs[cut:], Xn[cut:]], y[cut:]),
          epochs=12, batch_size=2048, callbacks=[es], verbose=2)

# ---- predict ----
va = model.predict([Xs[cut:], Xn[cut:]], batch_size=8192, verbose=0).ravel()
yv = y[cut:]
te = model.predict([test["X_seq"], test["X_net"]], batch_size=8192, verbose=0).ravel()
yt = test["y"]

# ---- choose threshold on validation, NOT test ----
p, r, thr = precision_recall_curve(yv, va)
f1v = 2 * p * r / (p + r + 1e-9)
best = thr[max(0, np.argmax(f1v) - 1)]

pred = (te >= best).astype(int)
roc = roc_auc_score(yt, te)
prauc = average_precision_score(yt, te)
P = precision_score(yt, pred, zero_division=0)
R = recall_score(yt, pred, zero_division=0)
F = f1_score(yt, pred, zero_division=0)
cm = confusion_matrix(yt, pred)

# ---- text summary ----
lines = [
    "MODULE 3 - EVALUATION SUMMARY",
    "=" * 40,
    f"Train rows used : {len(y):,}   (val slice {len(yv):,})",
    f"Test rows       : {len(yt):,}",
    f"Test fraud rate : {yt.mean():.4f}  ({int(yt.sum())} fraud / {len(yt):,})",
    "",
    "PRIMARY METRICS (imbalanced -> PR-AUC is the headline)",
    f"  PR-AUC (average precision) : {prauc:.4f}",
    f"  ROC-AUC                    : {roc:.4f}",
    "",
    f"AT OPERATING THRESHOLD = {best:.3f}  (chosen on validation set)",
    f"  Precision : {P:.3f}",
    f"  Recall    : {R:.3f}",
    f"  F1        : {F:.3f}",
    "",
    "CONFUSION MATRIX  [[TN, FP], [FN, TP]]",
    f"  {cm.tolist()}",
    f"  True negatives : {cm[0,0]:,}",
    f"  False positives: {cm[0,1]:,}   (legit flagged as fraud)",
    f"  False negatives: {cm[1,0]:,}   (fraud missed)",
    f"  True positives : {cm[1,1]:,}   (fraud caught)",
    "",
    "NOTE: accuracy is omitted on purpose - at <1% fraud a model that",
    "predicts 'never fraud' scores >99% accuracy, so accuracy is misleading.",
]
summary = "\n".join(lines)
open("/mnt/user-data/outputs/eval_metrics.txt", "w").write(summary)
print(summary)

# ---- figures ----
def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)

# PR curve
pr_p, pr_r, _ = precision_recall_curve(yt, te)
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.plot(pr_r, pr_p, color="#534AB7", lw=2)
ax.axhline(yt.mean(), color="#999", ls="--", lw=1,
           label=f"baseline (random) = {yt.mean():.3f}")
ax.scatter([R], [P], color="#D85A30", zorder=5,
           label=f"operating point\nP={P:.2f} R={R:.2f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title(f"Precision-Recall curve  (PR-AUC = {prauc:.3f})")
ax.legend(fontsize=9); style(ax); fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_pr_curve.png", dpi=150); plt.close(fig)

# ROC curve
fpr, tpr, _ = roc_curve(yt, te)
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.plot(fpr, tpr, color="#1D9E75", lw=2)
ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=1, label="random")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title(f"ROC curve  (ROC-AUC = {roc:.3f})")
ax.legend(fontsize=9); style(ax); fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_roc_curve.png", dpi=150); plt.close(fig)

# Confusion matrix
fig, ax = plt.subplots(figsize=(4.6, 4.2))
im = ax.imshow(cm, cmap="Purples")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["pred legit", "pred fraud"])
ax.set_yticklabels(["true legit", "true fraud"])
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, f"{v:,}", ha="center", va="center",
            color="white" if v > cm.max() / 2 else "#333", fontsize=12)
ax.set_title(f"Confusion matrix @ thr={best:.2f}")
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_confusion.png", dpi=150); plt.close(fig)

# Threshold sweep
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.plot(thr, p[:-1], label="precision", color="#185FA5", lw=1.8)
ax.plot(thr, r[:-1], label="recall", color="#D85A30", lw=1.8)
ax.plot(thr, f1v[:-1], label="F1", color="#534AB7", lw=1.8)
ax.axvline(best, color="#999", ls="--", lw=1, label=f"chosen thr={best:.2f}")
ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 vs threshold (validation)")
ax.legend(fontsize=9); style(ax); fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_threshold.png", dpi=150); plt.close(fig)

model.save("/mnt/user-data/outputs/module3_model.keras")
print("\nSaved: eval_metrics.txt, fig_pr_curve.png, fig_roc_curve.png, "
      "fig_confusion.png, fig_threshold.png, module3_model.keras")
