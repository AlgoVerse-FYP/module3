"""
Verify the two foundation fixes before retraining.
=================================================
Place at:  module3/experiments/ablation/verify_fixes.py   (next to leakage_audit.py)
Run it the SAME way you run leakage_audit.py (Code Runner, or python -m ...).

CHECK 1 -> Fix 1: out-of-time split + is_val mask + train-only normalization
CHECK 2 -> Fix 2: masking actually ignores padded steps

It trains nothing. If an edit wasn't applied, it says so in plain language.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from src.config import load_config, path
from src.features.phase1_features import build_dataset, SEQ_NUMERIC, NET_FEATURES

cfg = load_config()
nrows = cfg["data"]["nrows"]
vf = cfg["data"]["val_fraction"]

# ---------------------------------------------------------------
print("=" * 60)
print("CHECK 1 - out-of-time split (Fix 1)")
print("=" * 60)
try:
    d = build_dataset(path(cfg["data"]["train_csv"]), nrows=nrows,
                      val_fraction=vf, return_df=True)
except TypeError as e:
    print("FAIL: build_dataset() does not accept val_fraction yet.")
    print("      -> Fix 1 is not applied. Re-check the phase1_features.py edit.")
    print("      error:", e)
    sys.exit(1)

is_val = d.get("is_val")
if is_val is None:
    print("FAIL: build_dataset() returned no 'is_val' mask.")
    print("      -> Fix 1 is not applied. Re-check the phase1_features.py edit.")
    sys.exit(1)

df = d["df"]
overlap = bool(df.ts[is_val].min() <= df.ts[~is_val].max())
print("train ts :", df.ts[~is_val].min(), "->", df.ts[~is_val].max())
print("val   ts :", df.ts[ is_val].min(), "->", df.ts[ is_val].max())
print("val fraud:", int(df.is_fraud[is_val].sum()))
print("overlap  :", overlap)
print("RESULT   :", "PASS - split is temporal" if not overlap
      else "FAIL - train/val still overlap in time")

# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("CHECK 2 - masking ignores padded steps (Fix 2)")
print("=" * 60)
from src.models.phase2_model import build_model
w = cfg["features"]["window"]
d2 = build_dataset(path(cfg["data"]["train_csv"]), nrows=nrows, val_fraction=vf)
m = build_model(w, 1 + len(SEQ_NUMERIC), d2["n_categories"], len(NET_FEATURES))

i = 0                                              # first row = a customer's first txn -> leading pads
xs = d2["X_seq"][i:i+1].astype("float32")
xn = d2["X_net"][i:i+1].astype("float32")
p1 = float(m.predict([xs, xn], verbose=0)[0, 0])

xs2 = xs.copy()
xs2[:, :w-1, 1:] = 999.0                           # trash pad numeric cols; keep cat idx 0 so they stay pads
p2 = float(m.predict([xs2, xn], verbose=0)[0, 0])

print("cat indices in window:", xs[0, :, 0], " (leading 0s = pad steps)")
print("pred            :", round(p1, 6))
print("pred trashed pad:", round(p2, 6))
works = abs(p1 - p2) < 1e-6
print("RESULT   :", "PASS - pads are ignored" if works
      else "FAIL - pads still change the prediction (Fix 2 not applied)")

print("\n" + "=" * 60)
print("Both PASS  ->  ready to retrain and read the honest baseline.")
print("=" * 60)