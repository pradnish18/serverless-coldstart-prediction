"""Probability calibration of the main-seed models (no new models).
Brier score and expected calibration error (ECE, 15 equal-width bins) on test, before and after isotonic
recalibration fitted on validation only. Also a 10-bin reliability table. Output: v2/results/calibration.md"""
import os, sys, numpy as np
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
S = np.load(f"{RES}/frames.npz"); P = {}
for f in ("preds_gbdt.npz", "preds_abigru_s0.npz", "preds_gru_s0.npz", "preds_bigru_s0.npz", "preds_mlp_static_s0.npz"):
    z = np.load(f"{RES}/{f}")
    for k in z.files:
        m, fr = k.split("__"); P.setdefault(m.replace("_s0", ""), {})[fr] = z[k]
mv, mt = S["val_main_role"] == 0, S["test_main_role"] == 0; yv, yt = S["val_main_y"][mv].astype(float), S["test_main_y"][mt].astype(float)

def ece(y, p, bins=15):
    e = np.clip((p * bins).astype(int), 0, bins - 1); tot = 0.0
    for b in range(bins):
        k = e == b
        if k.any(): tot += k.mean() * abs(y[k].mean() - p[k].mean())
    return tot
brier = lambda y, p: float(np.mean((p - y) ** 2))
out = ["# Probability calibration (MAIN all slice, test days 27-30)\n",
       "Isotonic recalibration is fitted on validation predictions only and applied to test. "
       "Lower is better for both Brier score and ECE (expected calibration error). Base rate: %.4f.\n" % yt.mean(),
       "| model | Brier (raw) | ECE (raw) | Brier (isotonic) | ECE (isotonic) | mean predicted p | observed rate |", "|---|---|---|---|---|---|---|"]
rel = {}
for m in ["xgb_full", "xgb_no_static", "abigru", "bigru", "gru", "mlp_static"]:
    pv, pt = P[m]["val_main"][mv], P[m]["test_main"][mt]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv, yv); pc = iso.predict(pt)
    out.append(f"| {m} | {brier(yt, pt):.5f} | {ece(yt, pt):.4f} | {brier(yt, pc):.5f} | {ece(yt, pc):.4f} | {pt.mean():.4f} | {yt.mean():.4f} |")
    rel[m] = pt
out += ["\n## Reliability table (raw probabilities, 10 bins): observed cold-start rate per predicted-probability bin\n",
        "| bin | " + " | ".join(rel) + " |", "|---|" + "---|" * len(rel)]
for b in range(10):
    lo, hi = b / 10, (b + 1) / 10; cells = []
    for m, p in rel.items():
        k = (p >= lo) & ((p < hi) if b < 9 else (p <= hi))
        cells.append(f"{yt[k].mean():.3f} (n={k.sum():,})" if k.any() else "–")
    out.append(f"| {lo:.1f}–{hi:.1f} | " + " | ".join(cells) + " |")
open(f"{RES}/calibration.md", "w").write("\n".join(out)); print("\n".join(out))
