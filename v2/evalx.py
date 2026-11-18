"""Evaluation: threshold from validation, slice metrics, function-cluster bootstrap CI, cost capture."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def best_thr(y, p, grid=None):
    if np.unique(p).size <= 3:
        return 0.5
    qs = np.unique(np.quantile(p, np.linspace(0.5, 0.9995, 400)))
    best, bt = -1, 0.5
    for q in qs:
        pr = p >= q; tp = (pr & (y > 0)).sum(); fp = (pr & (y == 0)).sum(); fn = ((~pr) & (y > 0)).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best: best, bt = f1, q
    return bt


def prf(tp, fp, fn, tn):
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return pr, rc, f1, mcc


def metrics(y, p, thr, fn_ids, nboot=200, seed=0):
    pred = p >= thr
    tp = (pred & (y > 0)).sum(); fp = (pred & (y == 0)).sum(); fn = ((~pred) & (y > 0)).sum(); tn = len(y) - tp - fp - fn
    pr, rc, f1, mcc = prf(tp, fp, fn, tn)
    out = dict(n=int(len(y)), pos=int(y.sum()), precision=pr, recall=rc, f1=f1, mcc=mcc,
               acc=(tp + tn) / len(y))
    out["pr_auc"] = average_precision_score(y, p) if 0 < y.sum() < len(y) else float("nan")
    out["roc_auc"] = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")
    # cluster bootstrap over functions
    u, inv = np.unique(fn_ids, return_inverse=True)
    S = np.zeros((len(u), 4))
    for j, a in enumerate((pred & (y > 0), pred & (y == 0), (~pred) & (y > 0), (~pred) & (y == 0))):
        S[:, j] = np.bincount(inv, weights=a.astype(float), minlength=len(u))
    rng = np.random.RandomState(seed); f1s, mccs = [], []
    for _ in range(nboot):
        s = S[rng.randint(0, len(u), len(u))].sum(0)
        _, _, a, b = prf(*s); f1s.append(a); mccs.append(b)
    if nboot > 0:
        out["f1_ci"] = (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5)))
        out["mcc_ci"] = (float(np.percentile(mccs, 2.5)), float(np.percentile(mccs, 97.5)))
    else:                                   # bootstrap skipped
        out["f1_ci"] = out["mcc_ci"] = (float("nan"), float("nan"))
    return out


def cost_capture(p, cost, budgets=(0.01, 0.02, 0.05, 0.1)):
    """Fraction of total cold-start cost captured when pre-warming the top-k fraction of (func,bucket) cells."""
    o = np.argsort(-p, kind="stable"); c = np.cumsum(cost[o]); tot = max(cost.sum(), 1e-9)
    return {b: float(c[max(int(len(p) * b) - 1, 0)] / tot) for b in budgets}
