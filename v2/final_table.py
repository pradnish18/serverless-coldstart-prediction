import glob, numpy as np, os
from evalx import *
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
S = np.load(f"{RES}/frames.npz"); P = {}
for fp in glob.glob(f"{RES}/preds_*.npz"):
    z = np.load(fp)
    for k in z.files: m, fr = k.split("__"); P.setdefault(m, {})[fr] = z[k]
role_t, role_v = S["test_main_role"], S["val_main_role"]
mt, mv = role_t == 0, role_v == 0
yt, ft = S["test_main_y"][mt], S["test_main_f"][mt]; yv = S["val_main_y"][mv]
print(f"TEST (days 27-30), eligible functions: n={len(yt):,}, positives={int(yt.sum()):,} ({yt.mean():.1%}); always-negative accuracy={1-yt.mean():.4f}\n")
print(f"{'model':22s}{'thr':>7s}{'Acc':>8s}{'Prec':>8s}{'Rec':>8s}{'F1':>8s}{'MCC':>8s}{'PR-AUC':>8s}{'ROC-AUC':>8s}   val-F1")
rows = []
for m, dd in P.items():
    if m in ("xgb_cold_newfunc", "score_pool_hour_rate"): continue
    thr = best_thr(yv, dd["val_main"][mv])
    r = metrics(yt, dd["test_main"][mt], thr, ft, nboot=2)
    rv = metrics(yv, dd["val_main"][mv], thr, S["val_main_f"][mv], nboot=2)
    rows.append((m, thr, r, rv["f1"]))
for m, thr, r, vf in sorted(rows, key=lambda x: -x[2]["f1"]):
    print(f"{m:22s}{thr:7.3f}{r['acc']:8.4f}{r['precision']:8.4f}{r['recall']:8.4f}{r['f1']:8.4f}{r['mcc']:8.4f}{r['pr_auc']:8.4f}{r['roc_auc']:8.4f}   {vf:.4f}")
