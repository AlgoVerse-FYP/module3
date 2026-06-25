"""Phase 1: raw data cleaning (timestamp parse, invalid-row removal, temporal split). Currently inside phase1_features.py; split out here if it grows."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from src.config import load_config, path

def clean_data(df):
    df = df.copy()
    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])          # text -> datetime
    critical = ["cc_num", "amt", "ts", "merchant", "category", "is_fraud"]
    df = df.dropna(subset=critical)                                 # drop broken rows
    df = df[df["amt"] >= 0]                                         # no negative amounts
    df = df.sort_values(["cc_num", "ts"]).reset_index(drop=True)
    df["amt_log"] = np.log1p(df["amt"])                            # compress skewed amount
    cats = sorted(df["category"].unique())
    df["cat_idx"] = df["category"].map({c: i + 1 for i, c in enumerate(cats)})  # category -> integer
    return df

def _run_report():
    cfg = load_config()
    nrows = cfg["data"]["nrows"]
    raw = pd.read_csv(path(cfg["data"]["train_csv"]), nrows=nrows)
    before_rows = len(raw)
    before_amt = (raw.amt.min(), raw.amt.max())

    df = clean_data(raw)

    # every customer is time-ordered
    ordered = df.groupby("cc_num")["ts"].apply(lambda s: s.is_monotonic_increasing).all()

    # temporal split has no time overlap (no leakage)
    df_t = df.sort_values("ts").reset_index(drop=True)
    cut = int(len(df_t) * (1 - cfg["data"]["val_fraction"]))
    tr, va = df_t.iloc[:cut], df_t.iloc[cut:]
    no_leak = va.ts.min() >= tr.ts.max()

    # cache cleaned data so later phases don't re-clean
    df.to_parquet(path("data/processed/train_clean.parquet"))

    out = path("results/metrics/phase1_preprocessing_report.txt")
    with open(out, "w") as f:
        f.write("PHASE 1 - PREPROCESSING\n")
        f.write(f"rows before {before_rows}  after {len(df)}\n")
        f.write(f"amount raw range {before_amt[0]:.2f}-{before_amt[1]:.2f}  "
                f"log range {df.amt_log.min():.2f}-{df.amt_log.max():.2f}\n")
        f.write(f"all customers time-ordered: {ordered}\n")
        f.write(f"temporal split leak-free: {no_leak}\n")
    print(open(out).read())
    print("saved", out)

if __name__ == "__main__":
    _run_report()

