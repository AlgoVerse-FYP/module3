"""Phase 0: baseline learnability check (gradient-boosted baseline AUC) to validate any dataset before use."""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from src.config import load_config, path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# ----- read settings + load data -----
cfg = load_config()
nrows = cfg["data"]["nrows"]
df = pd.read_csv(path(cfg["data"]["train_csv"]), nrows=nrows)
test = pd.read_csv(path(cfg["data"]["test_csv"]), nrows=nrows)

# ----- signal check -----
df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"] = df["ts"].dt.hour
night = [22, 23, 0, 1, 2, 3]

amt_legit = df[df.is_fraud == 0].amt.mean()
amt_fraud = df[df.is_fraud == 1].amt.mean()
night_rate = df[df.hour.isin(night)].is_fraud.mean()
day_rate = df[~df.hour.isin(night)].is_fraud.mean()
print("amount legit", round(amt_legit,2), "fraud", round(amt_fraud,2))
print("night rate", round(night_rate,4), "day rate", round(day_rate,4))

# ----- simple feature-set baseline (gradient-boosted) -----
def features(d):
    d = d.copy()
    d["ts"] = pd.to_datetime(d["trans_date_trans_time"])
    d["hour"] = d["ts"].dt.hour
    d["age"] = 2019 - pd.to_datetime(d["dob"]).dt.year
    X = d[["amt", "hour", "city_pop", "age"]].copy()
    X["category"] = d["category"].astype("category").cat.codes
    return X

m = HistGradientBoostingClassifier(max_iter=200, random_state=42)
m.fit(features(df), df.is_fraud)
p = m.predict_proba(features(test))[:, 1]
roc = roc_auc_score(test.is_fraud, p)
print("ROC-AUC", round(roc, 4))

out = path("results/metrics/phase0_dataset_validation.txt")
with open(out, "w") as f:
    f.write("PHASE 0 - DATASET VALIDATION\n")
    f.write(f"fraud rate: {df.is_fraud.mean():.4f}\n")
    f.write(f"amount legit {amt_legit:.2f} fraud {amt_fraud:.2f}\n")
    f.write(f"night rate {night_rate:.4f} day rate {day_rate:.4f}\n")
    f.write(f"baseline ROC-AUC: {roc:.4f}\n")
    f.write("VERDICT: learnable\n" if roc > 0.7 else "VERDICT: NOT learnable\n")
print("saved", out)