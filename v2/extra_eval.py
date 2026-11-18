"""Extra evaluation (no new models):
  1. confusion matrices per slice,
  2. training-set metrics vs validation/test (overfitting check),
  3. paired statistical tests between models (function-level paired bootstrap + McNemar).
Uses the main seed of each model (XGBoost seed 42, neural seed 0) and the saved models in v2/models/.
Output: v2/results/extra_eval.md
"""
import os, sys, json, numpy as np, torch
from scipy.stats import chi2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frames import get_ctx, build
from evalx import best_thr, metrics
import nets, xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results"); MOD = os.path.join(HERE, "models")
S = np.load(f"{RES}/frames.npz")
P = {}
for f in ("preds_gbdt.npz", "preds_abigru_s0.npz", "preds_gru_s0.npz", "preds_bigru_s0.npz", "preds_mlp_static_s0.npz"):
    z = np.load(f"{RES}/{f}")
    for k in z.files:
        m, fr = k.split("__"); P.setdefault(m.replace("_s0", ""), {})[fr] = z[k]
MODELS = ["xgb_full", "xgb_no_static", "abigru", "bigru", "gru", "mlp_static", "rule_majority_7d", "rule_same_bucket_1d"]
out = ["# Extra evaluation (main seed of each model)\n"]

def slice_mask(fr, kind):
    role, r12 = S[f"{fr}_role"], S[f"{fr}_roll12"]
    return {"all": role == 0, "react": (role == 0) & (r12 == 0), "lowhist": role == 2,
            "hold": np.ones_like(role, bool), "hold_react": r12 == 0}[kind]

SL = [("MAIN all", "main", "all"), ("MAIN reactivation", "main", "react"), ("LOW-HISTORY", "main", "lowhist"),
      ("HOLDOUT (unseen functions, own history)", "hold", "hold")]

# ---------------- 1. confusion matrices per slice ----------------
out.append("## 1. Confusion matrices per slice (test days 27-30; threshold tuned on the same slice of validation)\n")
thr_main = {}
for title, fam, kind in SL:
    v, t = f"val_{fam}", f"test_{fam}"; mv, mt = slice_mask(v, kind), slice_mask(t, kind)
    yv, yt = S[f"{v}_y"][mv], S[f"{t}_y"][mt]
    out += [f"\n### {title} (n={mt.sum():,}, positives={int(yt.sum()):,})\n", "| model | threshold | TP | FP | FN | TN | precision | recall | F1 |", "|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        pv, pt = P[m][v][mv], P[m][t][mt]
        thr = best_thr(yv, pv) if yv.sum() >= 5 else best_thr(S[f"{v}_y"], P[m][v])
        if title == "MAIN all": thr_main[m] = thr
        pr = pt >= thr; tp = int((pr & (yt > 0)).sum()); fp = int((pr & (yt == 0)).sum()); fn = int((~pr & (yt > 0)).sum()); tn = len(yt) - tp - fp - fn
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        out.append(f"| {m} | {thr:.3f} | {tp:,} | {fp:,} | {fn:,} | {tn:,} | {prec:.3f} | {rec:.3f} | {f1:.3f} |")

# ---------------- 2. training-set metrics ----------------
out.append("\n## 2. Training vs validation vs test (MAIN all slice, eligible functions; threshold from validation)\n")
out += ["| model | split | accuracy | precision | recall | F1 | MCC | PR-AUC |", "|---|---|---|---|---|---|---|---|"]
d, ctx = get_ctx()
full, names = build(ctx, d, "full", names=("train",))
cold, cnames = build(ctx, d, "cold", names=("train",))
tr = full["train"]; mtr = tr["role"] == 0; ytr = tr["y"][mtr]; ftr = tr["f"][mtr]
T = nets.make_tensors(ctx, torch.device("cpu"))
def train_scores(m):
    if m.startswith("xgb_"):
        feats = json.load(open(f"{MOD}/{m}_features.json"))["features"]
        mdl = xgb.XGBClassifier(); mdl.load_model(f"{MOD}/{m}_s42.json")
        return mdl.predict_proba(tr["X"][mtr][:, [names.index(x) for x in feats]])[:, 1]
    if m.startswith("rule_"):
        X = tr["X"][mtr]; g = lambda n: X[:, names.index(n)]
        return (g("slot_hits_7d") >= 4).astype(float) if m == "rule_majority_7d" else g("d1_+0")
    net, ck = nets.load_checkpoint(f"{MOD}/{m}_s0.pt", "cpu")
    Sx = torch.tensor(nets.standardize(tr["X"][mtr], ck["mu"], ck["sd"]))
    return nets.predict(net, T, Sx, torch.tensor(tr["f"][mtr]), torch.tensor(tr["t"][mtr]))
for m in MODELS:
    rows = [("train (days 7-23)", ytr, train_scores(m), ftr)]
    for sp in ("val_main", "test_main"):
        mk = S[f"{sp}_role"] == 0; rows.append((sp.replace("_main", "").replace("val", "validation (24-26)").replace("test", "test (27-30)"), S[f"{sp}_y"][mk], P[m][sp][mk], S[f"{sp}_f"][mk]))
    for nm, y, p, f in rows:
        r = metrics(y, p, thr_main[m], f, nboot=0)
        out.append(f"| {m} | {nm} | {r['acc']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['mcc']:.4f} | {r['pr_auc']:.4f} |")
    print("train metrics done:", m, flush=True)

# ---------------- 3. paired statistical tests ----------------
out.append("\n## 3. Paired statistical tests on the test set (MAIN all slice)\n")
out.append("Paired bootstrap: 1,000 resamples of whole functions; each model keeps its validation threshold. "
           "p = two-sided bootstrap p-value for 'no difference'. McNemar compares row-level disagreements; rows of one "
           "function are correlated, so its p-value is optimistic and the bootstrap is the primary test.\n")
out += ["| model A | model B | F1(A) − F1(B) | 95% CI | bootstrap p | McNemar χ² | McNemar p |", "|---|---|---|---|---|---|---|"]
mt = S["test_main_role"] == 0; yt = S["test_main_y"][mt]; ft = S["test_main_f"][mt]
u, inv = np.unique(ft, return_inverse=True)
def per_fn(pred):
    a = np.zeros((len(u), 3))
    for j, msk in enumerate((pred & (yt > 0), pred & (yt == 0), ~pred & (yt > 0))):
        a[:, j] = np.bincount(inv, weights=msk.astype(float), minlength=len(u))
    return a
f1 = lambda c: 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1)
PRED = {m: P[m]["test_main"][mt] >= thr_main[m] for m in MODELS}
CNT = {m: per_fn(PRED[m]) for m in MODELS}
rng = np.random.RandomState(0); B = [rng.randint(0, len(u), len(u)) for _ in range(1000)]
PAIRS = [("xgb_full", "rule_majority_7d"), ("abigru", "rule_majority_7d"), ("mlp_static", "rule_majority_7d"),
         ("abigru", "xgb_full"), ("xgb_no_static", "xgb_full"), ("abigru", "mlp_static"), ("abigru", "bigru"),
         ("abigru", "gru"), ("bigru", "mlp_static"), ("gru", "mlp_static")]
for a, b in PAIRS:
    obs = f1(CNT[a].sum(0)) - f1(CNT[b].sum(0))
    diffs = np.array([f1(CNT[a][i].sum(0)) - f1(CNT[b][i].sum(0)) for i in B])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    ca, cb = PRED[a] == (yt > 0), PRED[b] == (yt > 0)
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    chi = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1); pm = float(chi2.sf(chi, 1))
    out.append(f"| {a} | {b} | {obs:+.4f} | [{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}] | {p:.3f} | {chi:.1f} | {pm:.2g} |")

open(f"{RES}/extra_eval.md", "w").write("\n".join(out)); print("\n".join(out))
