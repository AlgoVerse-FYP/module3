import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, auc

from src.config import load_config, path
from src.features.phase1_features import build_dataset, SEQ_NUMERIC, NET_FEATURES


def single_feature_auc(df):
    """(A) For each engineered feature, AUC of that ONE feature vs is_fraud.
    A single feature scoring ~0.95+ is a red flag: either leakage or a
    feature that trivially encodes the label."""
    y = df["is_fraud"].astype(int).to_numpy()
    feats = [c for c in (SEQ_NUMERIC + NET_FEATURES) if c in df.columns]
    rows = []
    for c in feats:
        x = df[c].to_numpy("float64")
        if np.all(x == x[0]):            # constant column -> skip
            continue
        auc = roc_auc_score(y, x)
        auc = max(auc, 1 - auc)          # direction-agnostic (leak leaks either way)
        rows.append({"feature": c, "solo_auc": round(auc, 4)})
    out = pd.DataFrame(rows).sort_values("solo_auc", ascending=False).reset_index(drop=True)
    out["flag"] = np.where(out["solo_auc"] >= 0.95, "SUSPICIOUS",
                    np.where(out["solo_auc"] >= 0.85, "strong", ""))
    return out


def diagnose_current_split(df, val_fraction):
    """(B) Reproduce phase2_model.make_splits (positional cut on cc_num,ts order)
    and report the TIME range of each side. Overlap => not a temporal split."""
    cut = int(len(df) * (1 - val_fraction))
    tr, va = df.iloc[:cut], df.iloc[cut:]
    overlap = va["ts"].min() <= tr["ts"].max()
    return {
        "train_ts": (tr["ts"].min(), tr["ts"].max()),
        "val_ts":   (va["ts"].min(), va["ts"].max()),
        "train_customers": tr["cc_num"].nunique(),
        "val_customers":   va["cc_num"].nunique(),
        "shared_customers": len(set(tr["cc_num"]) & set(va["cc_num"])),
        "time_overlap": bool(overlap),
        "train_fraud": int(tr["is_fraud"].sum()),
        "val_fraud":   int(va["is_fraud"].sum()),
    }


def propose_temporal_split(df, val_fraction):
    """(C) A real out-of-time split: cut by a TIME threshold so val is strictly
    later than train. Windowing is unaffected (it is per-customer and already built);
    you apply this as a row mask on the SAME cc_num,ts-ordered arrays."""
    cutoff = df["ts"].quantile(1 - val_fraction)
    tr, va = df[df["ts"] < cutoff], df[df["ts"] >= cutoff]
    return cutoff, {
        "cutoff": cutoff,
        "train_ts": (tr["ts"].min(), tr["ts"].max()),
        "val_ts":   (va["ts"].min(), va["ts"].max()),
        "time_overlap": bool(va["ts"].min() <= tr["ts"].max()),
        "train_fraud": int(tr["is_fraud"].sum()),
        "val_fraud":   int(va["is_fraud"].sum()),
        "val_rows": int(len(va)),
    }


def main():
    cfg = load_config()
    nrows = cfg["data"]["nrows"]
    vf = cfg["data"]["val_fraction"]

    data = build_dataset(path(cfg["data"]["train_csv"]), nrows=nrows, return_df=True)
    df = data["df"]                      # same order as the X tensors (cc_num, ts)
    print(f"rows={len(df)}  fraud={int(df['is_fraud'].sum())} "
          f"({100*df['is_fraud'].mean():.3f}%)\n")

    print("=" * 60)
    print("(A) SINGLE-FEATURE AUC  (>=0.95 = investigate before trusting model)")
    print("=" * 60)
    print(single_feature_auc(df).to_string(index=False), "\n")

    print("=" * 60)
    print("(B) THE SPLIT YOU ACTUALLY TRAIN ON  (phase2_model.make_splits)")
    print("=" * 60)
    cur = diagnose_current_split(df, vf)
    for k, v in cur.items():
        print(f"  {k:18s}: {v}")
    print(f"  --> temporal? {'NO - train/val overlap in time' if cur['time_overlap'] else 'yes'}\n")

    print("=" * 60)
    print("(C) PROPOSED OUT-OF-TIME SPLIT  (val strictly later than train)")
    print("=" * 60)
    _, prop = propose_temporal_split(df, vf)
    for k, v in prop.items():
        print(f"  {k:18s}: {v}")
    if prop["val_fraud"] < 20:
        print("  WARNING: <20 fraud cases in val -> PR-AUC will be noisy; "
              "raise val_fraction or use full data.")


if __name__ == "__main__":
    main()