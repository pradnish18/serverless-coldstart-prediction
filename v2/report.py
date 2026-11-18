import glob, re, json, numpy as np, os
from evalx import *
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
S = np.load(f"{RES}/frames.npz")
P = {}
for fpath in glob.glob(f"{RES}/preds_*.npz"):
    z = np.load(fpath)
    for k in z.files:
        m, fr = k.split("__"); P.setdefault(m, {})[fr] = z[k]

def msk(fr, kind):
    role, r12 = S[f"{fr}_role"], S[f"{fr}_roll12"]
    return {"all": role == 0 if fr.endswith("main") else np.ones_like(role, bool),
            "react": (role == 0) & (r12 == 0) if fr.endswith("main") else (r12 == 0),
            "lowhist": role == 2}[kind]

SLICES = [("MAIN all (eligible fns)", "main", "all"), ("MAIN reactivation (no event prior 1h)", "main", "react"),
          ("LOW-HISTORY fns (<10 train events)", "main", "lowhist"),
          ("HOLDOUT unseen fns, own-history features", "hold", "all"), ("HOLDOUT reactivation", "hold", "react")]
COLD_ONLY = {"xgb_cold_newfunc", "score_pool_hour_rate"}
rows = {}
for m, dd in sorted(P.items()):
    base = re.sub(r"_s\d+$", "", m)
    for title, fam, kind in SLICES:
        if base in COLD_ONLY and fam == "main": continue
        if base in COLD_ONLY and "own-history" not in title and kind != "all": continue
        if base in COLD_ONLY and "own-history" not in title: continue
        v, t = f"val_{fam}", f"test_{fam}"
        mv, mt = msk(v, kind), msk(t, kind)
        yv, pv = S[f"{v}_y"][mv], dd[v][mv]; yt, pt = S[f"{t}_y"][mt], dd[t][mt]
        if yt.sum() == 0: continue
        thr = best_thr(yv, pv) if yv.sum() >= 5 else best_thr(S[f"{v}_y"], dd[v])
        r = metrics(yt, pt, thr, S[f"{t}_f"][mt], nboot=100 if not re.search(r"_s[1-9]$", m) else 0)
        r["cost@2%"] = cost_capture(pt, S[f"{t}_cost"][mt])[0.02]
        rows.setdefault((title, base), []).append(r)

out = []
for title, fam, kind in SLICES:
    out.append(f"\n### {title}\n")
    out.append("| model | n_seeds | F1 [95% CI] | MCC | precision | recall | PR-AUC | ROC-AUC | cost captured @2% cells |\n|---|---|---|---|---|---|---|---|---|")
    items = [(b, rs) for (t_, b), rs in rows.items() if t_ == title]
    items.sort(key=lambda x: -np.mean([r["f1"] for r in x[1]]))
    for b, rs in items:
        a = lambda k: np.mean([r[k] for r in rs])
        sdv = np.std([r["f1"] for r in rs]) if len(rs) > 1 else 0
        ci = rs[0]["f1_ci"]
        out.append(f"| {b} | {len(rs)} | {a('f1'):.3f}{'±%.3f' % sdv if len(rs) > 1 else ''} [{ci[0]:.3f},{ci[1]:.3f}] | {a('mcc'):.3f} | {a('precision'):.3f} | {a('recall'):.3f} | {a('pr_auc'):.3f} | {a('roc_auc'):.3f} | {a('cost@2%'):.3f} |")
    n = rs[0]; out.append(f"\n(n={n['n']:,}, positives={n['pos']:,})")
txt = "\n".join(out)
open(f"{RES}/benchmark.md", "w").write(txt); print(txt)
