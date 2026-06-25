import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from src.config import load_config, path
from src.preprocessing.clean_data import clean_data

DIGITAL = {"grocery_net", "misc_net", "shopping_net"}

SEQ_NUMERIC = ["amt_log", "is_digital", "hour_sin", "hour_cos", "dow_sin",
               "dow_cos", "dt_log", "step_dist", "speed_log", "channel_switch"]
NET_FEATURES = ["recv_count_prior", "recv_unique_cust_prior", "cust_merch_novelty",
                "recv_vol_prior_log", "merch_amt_z"]

# Feature engineering
def engineer(df):
    df = df.copy()
    df["is_digital"] = df["category"].isin(DIGITAL).astype("float32")   # channel flag

    h, d = df.ts.dt.hour, df.ts.dt.dayofweek
    df["hour_sin"] = np.sin(2*np.pi*h/24).astype("float32")             # cyclical time
    df["hour_cos"] = np.cos(2*np.pi*h/24).astype("float32")
    df["dow_sin"] = np.sin(2*np.pi*d/7).astype("float32")
    df["dow_cos"] = np.cos(2*np.pi*d/7).astype("float32")

    g = df.groupby("cc_num", sort=False)                               # per customer
    dt = g["ts"].diff().dt.total_seconds().div(3600).fillna(0).clip(lower=0)
    df["dt_hours"] = dt
    df["dt_log"] = np.log1p(dt).astype("float32")                      # time since last

    plat, plong = g["merch_lat"].shift(1), g["merch_long"].shift(1)
    df["step_dist"] = np.sqrt((df.merch_lat-plat)**2 +
                              (df.merch_long-plong)**2).fillna(0).astype("float32")  # location move

    speed = (df.step_dist / df.dt_hours.replace(0, np.nan)).fillna(0)
    df["speed_log"] = np.log1p(speed.clip(lower=0).replace([np.inf,-np.inf],0)).astype("float32")  # implied speed

    prev_dig = g["is_digital"].shift(1)
    df["channel_switch"] = (df.is_digital != prev_dig).fillna(False).astype("float32")  # channel change
    return df

def merchant_features(df):
    df = df.sort_values(["merchant", "ts"]).reset_index()   # sort by merchant+time
    gm = df.groupby("merchant", sort=False)

    df["recv_count_prior"] = gm.cumcount().astype("float32")        # txns before this one
    csum = gm["amt"].cumsum() - df["amt"]                           # prior volume (exclude self)
    df["recv_vol_prior_log"] = np.log1p(csum.clip(lower=0)).astype("float32")

    cmean = csum / df["recv_count_prior"].replace(0, np.nan)        # prior mean amount
    df["merch_amt_z"] = ((df.amt-cmean)/(cmean.abs()+1)).fillna(0).clip(-10,10).astype("float32")

    df["_first"] = ~df.duplicated(["merchant", "cc_num"])           # first time this pair?
    df["recv_unique_cust_prior"] = (gm["_first"].cumsum() - df["_first"].astype(int)).astype("float32")
    df["cust_merch_novelty"] = df["_first"].astype("float32")

    df = df.set_index("index").sort_index()                        # restore original order
    return df[NET_FEATURES]

def build_dataset(csv_path, fit_from=None, nrows=None, window=None, return_df=False):
    cfg = load_config()
    if window is None:
        window = cfg["features"]["window"]

    raw = pd.read_csv(csv_path, nrows=nrows)
    df = clean_data(raw)            # Phase 1
    df = engineer(df)              # Part 1
    df = df.join(merchant_features(df))   # Part 2

    if fit_from is None:           # training: learn the encoders
        cats = sorted(df["category"].unique())
        cat_index = {c: i+1 for i, c in enumerate(cats)}
        norm = None
    else:                          # test: reuse training encoders (no leakage)
        cat_index, norm = fit_from["cat_index"], fit_from["norm"]
    df["cat_idx"] = df["category"].map(cat_index).fillna(0).astype("float32")

    to_norm = ["amt_log","dt_log","step_dist","speed_log","recv_count_prior",
               "recv_unique_cust_prior","recv_vol_prior_log","merch_amt_z"]
    if norm is None:
        norm = {c: (float(df[c].mean()), float(df[c].std()+1e-6)) for c in to_norm}
    dfn = df.copy()
    for c in to_norm:
        m, s = norm[c]; dfn[c] = ((dfn[c]-m)/s).astype("float32")

    step_cols = ["cat_idx"] + SEQ_NUMERIC
    step_mat = dfn[step_cols].to_numpy("float32")
    net_mat = dfn[NET_FEATURES].to_numpy("float32")
    y = dfn["is_fraud"].to_numpy("int8")
    cc = dfn["cc_num"].to_numpy()
    n, f = step_mat.shape

    X = np.zeros((n, window, f), dtype="float16")
    is_new = np.empty(n, bool); is_new[0]=True; is_new[1:] = cc[1:]!=cc[:-1]
    first_idx = np.maximum.accumulate(np.where(is_new, np.arange(n), 0))
    pos = np.arange(n) - first_idx
    for p_ in range(window):
        back = window-1-p_
        src = np.arange(n) - back
        ok = (src >= 0) & (pos >= back)
        rows = np.where(ok)[0]
        X[rows, p_, :] = step_mat[src[rows]].astype("float16")

    result = {"X_seq": X, "X_net": net_mat.astype("float16"), "y": y,
              "cat_index": cat_index, "norm": norm, "n_categories": len(cat_index)+1}
    if return_df:
        result["df"] = df
    return result

def _run_report():
    cfg = load_config()
    nrows = cfg["data"]["nrows"]
    data = build_dataset(path(cfg["data"]["train_csv"]), nrows=nrows, return_df=True)
    df = data["df"]

    # separation table: each feature's fraud-vs-legit difference
    sep = []
    for c in ["amt_log","speed_log","step_dist","dt_log","channel_switch",
              "is_digital","cust_merch_novelty","merch_amt_z"]:
        g = df.groupby("is_fraud")[c].mean()
        sep.append({"feature": c, "legit_mean": round(g[0],4),
                    "fraud_mean": round(g[1],4), "abs_diff": round(abs(g[1]-g[0]),4)})
    sep_df = pd.DataFrame(sep).sort_values("abs_diff", ascending=False)
    sep_df.to_csv(path("results/metrics/phase2_separation_table.csv"), index=False)

    # sample of engineered rows: raw columns next to computed features
    show = ["cc_num","amt","category","is_fraud"] + SEQ_NUMERIC + NET_FEATURES
    df[show].head(20).to_csv(path("results/metrics/phase2_engineered_sample.csv"), index=False)

    print("SEQUENCE tensor:", data["X_seq"].shape, "| NETWORK:", data["X_net"].shape)
    print(sep_df.to_string(index=False))
    print("saved -> phase2_separation_table.csv + phase2_engineered_sample.csv")

if __name__ == "__main__":
    _run_report()