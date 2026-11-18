"""Multi-seed summary: mean, sd, min-max of test metrics per model family (threshold tuned on validation per seed)."""
import glob, os, re, numpy as np
from evalx import best_thr, metrics
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
S = np.load(f"{RES}/frames.npz"); P = {}
for fp in glob.glob(f"{RES}/preds_*.npz"):
    if "smoke" in fp: continue
    z = np.load(fp)
    for k in z.files:
        m, fr = k.split("__"); P.setdefault(m, {})[fr] = z[k]
fam = {}
for m in P:
    if m.startswith(("rule_", "score_")) or m in ("xgb_no_pool", "xgb_cold_newfunc"): continue
    fam.setdefault(re.sub(r"_s\d+$", "", m), []).append(m)
slices = {"MAIN all (eligible functions)": lambda fr: S[f"{fr}_role"] == 0,
          "MAIN reactivation (no event in prior hour)": lambda fr: (S[f"{fr}_role"] == 0) & (S[f"{fr}_roll12"] == 0)}
out = ["# Multi-seed results (test days 27-30)\n", "Each seed's threshold is tuned on its own validation predictions. sd = standard deviation across seeds.\n"]
for title, mk in slices.items():
    mv, mt = mk("val_main"), mk("test_main"); yv, yt = S["val_main_y"][mv], S["test_main_y"][mt]; ft = S["test_main_f"][mt]
    out += [f"\n## {title}\n", "| model | seeds | F1 mean ± sd | F1 min–max | MCC mean ± sd | PR-AUC mean ± sd | Accuracy mean |", "|---|---|---|---|---|---|---|"]
    rows = []
    for b, ms in fam.items():
        r = [metrics(yt, P[m]["test_main"][mt], best_thr(yv, P[m]["val_main"][mv]), ft, nboot=2) for m in sorted(ms)]
        a = lambda k: np.array([x[k] for x in r])
        rows.append((a("f1").mean(), f"| {b} | {len(r)} | {a('f1').mean():.4f} ± {a('f1').std():.4f} | {a('f1').min():.4f}–{a('f1').max():.4f} | {a('mcc').mean():.4f} ± {a('mcc').std():.4f} | {a('pr_auc').mean():.4f} ± {a('pr_auc').std():.4f} | {a('acc').mean():.4f} |"))
    out += [x[1] for x in sorted(rows, key=lambda x: -x[0])]
    rr = [(n, metrics(yt, P[n]["test_main"][mt], best_thr(yv, P[n]["val_main"][mv]), ft, nboot=2)) for n in ("rule_majority_7d", "rule_same_bucket_1d")]
    out += [f"| {n} (rule, deterministic) | – | {r['f1']:.4f} | – | {r['mcc']:.4f} | {r['pr_auc']:.4f} | {r['acc']:.4f} |" for n, r in rr]
txt = "\n".join(out); open(f"{RES}/multiseed_summary.md", "w").write(txt); print(txt)
