"""Naive/seasonal baselines and cold-start slices on the Huawei R5 trace.

Task (same as main pipeline): will function f have >=1 cold-start event in
5-minute bucket t?  Rules use only information from buckets < t.
Thresholds/rules are fixed a priori, so no tuning on test days.
Run: python experiments/baseline_eval.py
"""
import glob, os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B, PER_DAY, DAYS = 300, 288, 31
TOT = DAYS * PER_DAY

df = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f"{ROOT}/dataset/raw/region5_cold_starts/R5/day_*.csv")))
pool = df.podID.str.rsplit("-", n=1).str[0]
df["fid"] = df.funcName.astype(str) + "--" + df.userID.astype(str) + "--" + pool
df["b"] = (df.time // B).astype(int)

fids = np.sort(df.fid.unique()); idx = {f: i for i, f in enumerate(fids)}
M = np.zeros((len(fids), TOT), dtype=np.int8)
g = df.groupby(["fid", "b"]).size().reset_index()
M[g.fid.map(idx).values, g.b.values] = 1

train_ev = df[df.day < 24].groupby("fid").size().reindex(fids).fillna(0)
seen = (train_ev >= 10).values            # eligible, as in the main pipeline
lo, hi = 27 * PER_DAY, TOT                # test days 27-30

def lag(k): return M[:, lo - k:hi - k]
y = M[:, lo:hi]
rules = {
    "last_bucket": lag(1),
    "same_bucket_1d": lag(PER_DAY),
    "same_bucket_7d": lag(7 * PER_DAY) if lo >= 7 * PER_DAY else None,
    "1d OR 7d": np.maximum(lag(PER_DAY), lag(7 * PER_DAY)),
    "1d +-1 bucket": np.maximum.reduce([lag(PER_DAY - 1), lag(PER_DAY), lag(PER_DAY + 1)]),
    "any in last 1h": np.maximum.reduce([lag(k) for k in range(1, 13)]),
    "any in last 1h OR 1d": np.maximum(np.maximum.reduce([lag(k) for k in range(1, 13)]), lag(PER_DAY)),
}
# 'recent-empty' mask: nothing in previous hour -> a "reactivation" cold start
recent = np.maximum.reduce([lag(k) for k in range(1, 13)]).astype(bool)

def report(title, mask):
    print(f"\n== {title}  (n={mask.sum():,}, positives={int(y[mask].sum()):,}, prev={y[mask].mean():.4f})")
    print(f"{'rule':24s}{'prec':>7s}{'rec':>7s}{'F1':>7s}{'MCC':>7s}")
    for name, p in rules.items():
        if p is None: continue
        yt, pt = y[mask], p[mask]
        if pt.std() == 0 or yt.std() == 0:
            print(f"{name:24s}  (degenerate)"); continue
        print(f"{name:24s}{precision_score(yt,pt,zero_division=0):7.3f}{recall_score(yt,pt):7.3f}"
              f"{f1_score(yt,pt):7.3f}{matthews_corrcoef(yt,pt):7.3f}")

report("ALL eligible functions (>=10 train events)", np.broadcast_to(seen[:, None], y.shape))
report("Only buckets with NO event in previous hour (reactivation / true cold-ish)",
       seen[:, None] & ~recent)
unseen = ~seen
report("Functions with <10 train events (low-history / new)", np.broadcast_to(unseen[:, None], y.shape))
# no-history 7d lag exists only because lo=27d>=7d; periodicity share:
n_ev = y[seen].sum()
print(f"\nTest events (function-buckets) from eligible funcs: {int(n_ev):,}; from low-history funcs: {int(y[unseen].sum())}")
print(f"Positives in 'no event in prior hour' slice that same-bucket-yesterday also catches: "
      f"{(y & lag(PER_DAY))[seen[:, None] & ~recent].sum() / max(1, y[seen[:, None] & ~recent].sum()):.3f}")
