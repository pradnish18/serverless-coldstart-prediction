"""Robustness checks (no new models):
  A. Rolling-origin evaluation: the same XGBoost (all 57 features, seed 42, same hyperparameters) and the two best
     rules evaluated on three consecutive 4-day test windows, each with an expanding training window.
  B. XGBoost hyperparameter search on the ORIGINAL split, selected on validation only, then scored once on test.
Neural models are not re-run here (about 15 min per model per fold).
Output: v2/results/robustness.md
"""
import os, sys, itertools, time, numpy as np, xgboost as xgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load, PD
from features import Ctx, make
from evalx import best_thr, metrics
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
d = load(); ctx = Ctx(d); role = d["role"]
main_fns = np.where(role != 1)[0]                     # training functions (roles 0 and 2), as in the main experiment
P0 = dict(max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0)

def frame(lo_day, hi_day):
    X, names, f, t = make(ctx, main_fns, lo_day * PD, hi_day * PD, "full")
    return X, names, f, t, ctx.ev[f, t].astype(np.int8)

def fit(Xtr, ytr, Xv, yv, params, seed=42):
    m = xgb.XGBClassifier(n_estimators=1500, early_stopping_rounds=40, tree_method="hist", n_jobs=10, eval_metric="aucpr", random_state=seed, **params)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False); return m

def score(y, p, yv, pv, f):
    return metrics(y, p, best_thr(yv, pv), f, nboot=0)

out = ["# Robustness checks\n", "## A. Rolling-origin evaluation (eligible functions, role 0)\n",
       "Each fold trains on days 7..(val start − 1), tunes the threshold on the next 3 days and tests on the following 4 days. "
       "Fold 3 is the original split. Static function statistics always come from days 0–6, so no fold leaks.\n",
       "| fold | train days | val days | test days | test positives | model | F1 | MCC | PR-AUC | best trees |", "|---|---|---|---|---|---|---|---|---|---|"]
FOLDS = [(1, 7, 16, 19, 23), (2, 7, 20, 23, 27), (3, 7, 24, 27, 31)]   # fold, train_lo, val_lo, test_lo, test_hi(excl)
fold_f1 = {}
for k, a, v, s, e in FOLDS:
    t0 = time.time()
    Xtr, names, _, _, ytr = frame(a, v); Xv, _, fv, tv, yv = frame(v, s); Xt, _, ft, tt, yt = frame(s, e)
    rv, rt = role[fv] == 0, role[ft] == 0
    m = fit(Xtr, ytr, Xv, yv, P0)
    pv, pt = m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
    g = lambda X, n: X[:, names.index(n)]
    cands = {"XGBoost full": (pv, pt, m.best_iteration),
             "rule: 7-day majority": ((g(Xv, "slot_hits_7d") >= 4).astype(float), (g(Xt, "slot_hits_7d") >= 4).astype(float), "–"),
             "rule: same slot yesterday": (g(Xv, "d1_+0"), g(Xt, "d1_+0"), "–")}
    for nm, (a1, b1, it) in cands.items():
        r = score(yt[rt], b1[rt], yv[rv], a1[rv], ft[rt]); fold_f1.setdefault(nm, []).append(r["f1"])
        out.append(f"| {k} | {a}–{v-1} | {v}–{s-1} | {s}–{e-1} | {int(yt[rt].sum()):,} | {nm} | {r['f1']:.4f} | {r['mcc']:.4f} | {r['pr_auc']:.4f} | {it} |")
    print(f"fold {k} done in {time.time()-t0:.0f}s", flush=True)
out += ["\n**Mean over folds:** " + "; ".join(f"{nm} F1 {np.mean(v):.4f} (range {min(v):.4f}–{max(v):.4f})" for nm, v in fold_f1.items()) + ".",
        "**Gap XGBoost − 7-day rule per fold:** " + ", ".join(f"{x - y:+.4f}" for x, y in zip(fold_f1["XGBoost full"], fold_f1["rule: 7-day majority"])) + ".\n"]

# ---------------- B. hyperparameter search ----------------
Xtr, names, _, _, ytr = frame(7, 24); Xv, _, fv, tv, yv = frame(24, 27); Xt, _, ft, tt, yt = frame(27, 31)
rv, rt = role[fv] == 0, role[ft] == 0
GRID = list(itertools.product([4, 6, 8, 10], [0.03, 0.1], [5, 20]))
out += ["## B. XGBoost hyperparameter search (original split; selection on validation PR-AUC only)\n",
        "Fixed: subsample 0.8, colsample_bytree 0.8, reg_lambda 1.0, up to 1,500 trees with early stopping (40). "
        "Grid: max_depth × learning_rate × min_child_weight = %d configurations. The default used everywhere else is depth 8, learning rate 0.05, min_child_weight 5.\n" % len(GRID),
        "| max_depth | learning_rate | min_child_weight | trees | val PR-AUC | val F1 |", "|---|---|---|---|---|---|"]
results = []
for md, lr, mcw in GRID:
    p = dict(P0, max_depth=md, learning_rate=lr, min_child_weight=mcw)
    m = fit(Xtr, ytr, Xv, yv, p); pv = m.predict_proba(Xv)[:, 1]
    rvv = score(yv[rv], pv[rv], yv[rv], pv[rv], fv[rv])
    results.append((rvv["pr_auc"], rvv["f1"], md, lr, mcw, m.best_iteration, m))
    out.append(f"| {md} | {lr} | {mcw} | {m.best_iteration} | {rvv['pr_auc']:.4f} | {rvv['f1']:.4f} |"); print("grid", md, lr, mcw, flush=True)
best = max(results, key=lambda r: r[0]); _, _, md, lr, mcw, it, m = best
pv, pt = m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
rb = score(yt[rt], pt[rt], yv[rv], pv[rv], ft[rt])
m0 = fit(Xtr, ytr, Xv, yv, P0); p0v, p0t = m0.predict_proba(Xv)[:, 1], m0.predict_proba(Xt)[:, 1]
r0 = score(yt[rt], p0t[rt], yv[rv], p0v[rv], ft[rt])
out += [f"\n**Selected on validation:** max_depth {md}, learning_rate {lr}, min_child_weight {mcw} ({it} trees).",
        f"**Test (scored once):** tuned F1 {rb['f1']:.4f}, MCC {rb['mcc']:.4f}, PR-AUC {rb['pr_auc']:.4f} — default F1 {r0['f1']:.4f}, MCC {r0['mcc']:.4f}, PR-AUC {r0['pr_auc']:.4f}. "
        f"Difference {rb['f1'] - r0['f1']:+.4f} F1 (seed-to-seed sd of XGBoost is about 0.0005)."]
open(f"{RES}/robustness.md", "w").write("\n".join(out)); print("\n".join(out))
