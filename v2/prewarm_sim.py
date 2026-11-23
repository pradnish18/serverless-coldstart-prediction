"""Pre-warming simulator (evaluation tool; no new model).

Replays test days 27-30 for the eligible functions (role 0). Before each 5-minute bucket t, a policy chooses which
functions get ONE pre-warmed container for bucket t. Scoring, per bucket:
  * if the function has >= 1 cold start in t and was pre-warmed: its FIRST cold start is avoided and we save that
    event's latency, estimated as the bucket's mean cold-start cost (total cost / count);
  * further cold starts in the same bucket (concurrency) are NOT avoided - one container serves one cold start;
  * a pre-warm in a bucket with no cold start is WASTED: one container idles for 5 minutes.

Assumptions (stated so the numbers are not over-read):
  * a container started at the end of bucket t-1 is ready for bucket t (mean cold start is ~0.9 s);
  * the trace contains only cold starts, so counterfactual effects (a pre-warmed container also serving later
    requests, or routing to another cluster) cannot be simulated; avoided counts are therefore conservative;
  * no change to the platform's own keep-alive is modelled.

Policies:
  no_prewarm, keepalive_15min / keepalive_60min (warm if any cold start in the last 3 / 12 buckets),
  rule_same_slot_yesterday, rule_majority_7d, model@threshold (validation-tuned F1 threshold),
  and per-bucket budgets: warm the top-N functions by score each bucket (N = containers per 5 minutes).
Outputs: v2/results/prewarm_sim.md, v2/figures/fig7_prewarm_tradeoff.png
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load
from hh_policy import hh_warm, RANGE
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
S = np.load(f"{RES}/frames.npz"); d = load()
mt = S["test_main_role"] == 0
f, t, y = S["test_main_f"][mt], S["test_main_t"][mt], S["test_main_y"][mt].astype(bool)
cnt = d["cnt"][f, t]; cost = d["cost"][f, t]
first_cost = np.where(cnt > 0, cost / np.maximum(cnt, 1), 0.0)          # latency of the first cold start in the bucket
TOT_CS, TOT_SEC = cnt.sum(), cost.sum()
n_buckets = len(np.unique(t)); n_fn = len(np.unique(f))

def load_scores(fname, key):
    z = np.load(f"{RES}/{fname}"); return z[key][mt]
SC = {"xgb_full": load_scores("preds_gbdt.npz", "xgb_full__test_main"),
      "abigru": load_scores("preds_abigru_s0.npz", "abigru_s0__test_main"),
      "mlp_static": load_scores("preds_mlp_static_s0.npz", "mlp_static_s0__test_main")}
G = np.load(f"{RES}/preds_gbdt.npz")
SC["rule_7d_votes"] = G["rule_majority_7d__test_main"][mt].astype(float)          # used only for its 0/1 decision
thr = json.load(open(os.path.join(HERE, "models", "thresholds.json")))
roll = lambda w: S["test_main_roll12"][mt] > 0 if w == 12 else None

# rolling counts for keep-alive baselines straight from the matrices
ev = (d["cnt"] > 0).astype(np.int32); cs = np.concatenate([np.zeros((ev.shape[0], 1), np.int32), np.cumsum(ev, 1)], 1)
def recent(w): return (cs[f, t] - cs[f, np.maximum(t - w, 0)]) > 0
slot_votes = sum(ev[f, t - k * 288] for k in range(1, 8))                       # 7-day vote count (a ranking score for the rule)
yday = ev[f, t - 288] > 0

def evaluate(warm):
    hit = warm & y
    return dict(prewarms=int(warm.sum()), per_bucket=warm.sum() / n_buckets, avoided=int(hit.sum()),
                avoided_pct=100 * hit.sum() / TOT_CS, sec_saved=float(first_cost[hit].sum()),
                sec_pct=100 * first_cost[hit].sum() / TOT_SEC, wasted=int((warm & ~y).sum()),
                waste_pct=100 * (warm & ~y).sum() / max(warm.sum(), 1), idle_container_hours=(warm & ~y).sum() * 5 / 60)

def topn(score, n):
    """Per-bucket budget: warm the n highest-scoring functions in every bucket (ties broken by function index)."""
    order = np.lexsort((f, -score, t)); rank = np.empty_like(order)
    tt = t[order]; starts = np.r_[0, np.flatnonzero(np.diff(tt)) + 1]
    pos = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    rank[order] = pos
    return rank < n

pol = {"no_prewarm": np.zeros_like(y), "keepalive_15min": recent(3), "keepalive_60min": recent(12),
       "rule_same_slot_yesterday": yday, "rule_majority_7d": slot_votes >= 4,
       "xgb_full @ threshold %.3f" % thr["xgb_full_s42"]: SC["xgb_full"] >= thr["xgb_full_s42"],
       "abigru @ threshold %.3f" % thr["abigru_s0"]: SC["abigru"] >= thr["abigru_s0"],
       "mlp_static @ threshold %.3f" % thr["mlp_static_s0"]: SC["mlp_static"] >= thr["mlp_static_s0"]}
# hybrid-histogram baseline (Shahrad et al., ATC 2020), adapted; see hh_policy.py
hh, hh_branch = hh_warm(ev, f, t, fallback_ka=RANGE)
hh60, _ = hh_warm(ev, f, t, fallback_ka=12)
pol["hybrid_histogram (ATC'20, fallback 4 h)"] = hh
pol["hybrid_histogram (fallback 60 min)"] = hh60
fn_branch = {b: len(np.unique(f[hh_branch == b])) for b in (0, 1, 2)}
res = {k: evaluate(v) for k, v in pol.items()}

BUDGETS = [5, 10, 20, 30, 40, 50, 60, 80, 100, 125]
curves = {"xgb_full": SC["xgb_full"], "abigru": SC["abigru"], "rule votes (7-day)": slot_votes + 0.01 * yday}
bud = {name: [evaluate(topn(sc, n)) for n in BUDGETS] for name, sc in curves.items()}

out = ["# Pre-warming simulation (test days 27-30, %d eligible functions, %d five-minute buckets)\n" % (n_fn, n_buckets),
       "Total cold starts: **%s** in %s function-buckets; total cold-start latency: **%.0f s** (%.1f h).\n" % (f"{int(TOT_CS):,}", f"{int(y.sum()):,}", TOT_SEC, TOT_SEC / 3600),
       "One pre-warm = one container kept ready for one 5-minute bucket; it avoids at most the first cold start in that bucket. "
       "Assumptions and limits are listed in `v2/prewarm_sim.py`. An upper bound for one container per function-bucket: "
       "avoiding every function-bucket's first cold start = %.1f%% of cold starts and %.1f%% of latency.\n" % (100 * y.sum() / TOT_CS, 100 * first_cost[y].sum() / TOT_SEC),
       "## Policies\n", "| policy | pre-warms per bucket | cold starts avoided | % of cold starts | latency saved (s) | % of latency | wasted pre-warms | % wasted | idle container-hours |", "|---|---|---|---|---|---|---|---|---|"]
for k, r in res.items():
    out.append(f"| {k} | {r['per_bucket']:.1f} | {r['avoided']:,} | {r['avoided_pct']:.1f}% | {r['sec_saved']:,.0f} | {r['sec_pct']:.1f}% | {r['wasted']:,} | {r['waste_pct']:.1f}% | {r['idle_container_hours']:,.0f} |")
out += ["\n## Fixed budget: N pre-warmed containers per 5-minute bucket (top-N functions by score)\n",
        "| N per bucket | " + " | ".join(f"{c}: % cold starts avoided / % wasted" for c in curves) + " |", "|---|" + "---|" * len(curves)]
for i, n in enumerate(BUDGETS):
    out.append(f"| {n} | " + " | ".join(f"{bud[c][i]['avoided_pct']:.1f}% / {bud[c][i]['waste_pct']:.1f}%" for c in curves) + " |")
# classification view of the binary policies (same rows as the benchmark's MAIN slice)
def cls(pred, m):
    p, yy = pred[m], y[m]; tp = (p & yy).sum(); fp = (p & ~yy).sum(); fn_ = (~p & yy).sum(); tn = (~p & ~yy).sum()
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn_, 1); f1 = 2 * pr * rc / max(pr + rc, 1e-9)
    mcc = (tp * tn - fp * fn_) / max(np.sqrt(float(tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)), 1e-9)
    return pr, rc, f1, mcc
react = S["test_main_roll12"][mt] == 0
out += ["\n## Policies as classifiers (test days 27-30)\n",
        "| policy | MAIN all: precision / recall / F1 / MCC | MAIN reactivation: precision / recall / F1 / MCC |", "|---|---|---|"]
for k, v in pol.items():
    if k == "no_prewarm": continue
    a, r_ = cls(v, np.ones_like(y)), cls(v, react)
    out.append(f"| {k} | {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f} / {a[3]:.3f} | {r_[0]:.3f} / {r_[1]:.3f} / {r_[2]:.3f} / {r_[3]:.3f} |")
out += ["\nHybrid-histogram branch use (functions; a function can use several over time): fixed keep-alive %d, "
        "histogram %d, out-of-range forecaster %d.\n" % (fn_branch[0], fn_branch[1], fn_branch[2])]
out += ["\nAverage number of functions with a cold start per bucket: %.1f.\n" % (y.sum() / n_buckets)]
open(f"{RES}/prewarm_sim.md", "w").write("\n".join(out)); print("\n".join(out))

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for c, st in zip(curves, ("-o", "-s", "--^")):
        ax[0].plot(BUDGETS, [b["avoided_pct"] for b in bud[c]], st, label=c, ms=4)
        ax[1].plot([b["waste_pct"] for b in bud[c]], [b["avoided_pct"] for b in bud[c]], st, label=c, ms=4)
    # threshold policies (adaptive number of containers per bucket) as single points; keep-alive in neutral grey
    pts = [("keepalive_15min", "keep-alive 15 min", "x", "0.35"), ("keepalive_60min", "keep-alive 60 min", "P", "0.35"),
           ("rule_majority_7d", "rule 7-day majority @ 0/1", "D", "C2"),
           ("hybrid_histogram (ATC'20, fallback 4 h)", "hybrid histogram (ATC'20)", "v", "C3")]
    pts += [(k, k.split(" @")[0] + " @ val threshold", mk, c) for k, mk, c in
            ((next(x for x in res if x.startswith("xgb_full")), "*", "C0"), (next(x for x in res if x.startswith("abigru")), "*", "C1"))]
    for k, lab, mk, c in pts:
        ax[1].scatter(res[k]["waste_pct"], res[k]["avoided_pct"], marker=mk, s=110 if mk == "*" else 60, color=c,
                      edgecolor="black" if mk == "*" else None, linewidths=0.6, label=lab, zorder=5)
    ax[0].set(xlabel="pre-warmed containers per 5-min bucket", ylabel="% of cold starts avoided", title="Budget vs cold starts avoided")
    ax[1].set(xlabel="% of pre-warms wasted (container idled)", ylabel="% of cold starts avoided", title="Benefit vs waste")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=7.5, title="lines: fixed N per bucket\nmarkers: threshold policies", title_fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "fig7_prewarm_tradeoff.png"), dpi=150); print("figure saved")
except ImportError:
    print("matplotlib not installed: figure skipped")
