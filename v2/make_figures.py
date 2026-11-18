"""Paper figures for the v2 benchmark -> v2/figures/*.png (static, light theme, 200 dpi).

Needs v2/results/frames.npz and preds_*.npz (produced by the training scripts).
  python3 v2/make_figures.py
"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
from evalx import best_thr, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# ---- palette (validated categorical slots 1-4, light mode) and text/surface tokens ----
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
RULE = "#8a8985"                       # neutral gray for the non-learned baseline
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "font.family": "DejaVu Sans", "lines.linewidth": 1.6, "axes.axisbelow": True,
})
NICE = {"xgb_full": "XGBoost (all features)", "xgb_no_static": "XGBoost (no static)", "abigru": "ABiGRU",
        "bigru": "BiGRU", "gru": "GRU", "mlp_static": "MLP (no sequence)", "rule_majority_7d": "Rule: 7-day majority"}

S = np.load(f"{RES}/frames.npz"); P = {}
for fp in glob.glob(f"{RES}/preds_*.npz"):
    if "smoke" in fp: continue
    z = np.load(fp)
    for k in z.files:
        m, fr = k.split("__"); P.setdefault(m, {})[fr] = z[k]

def mask(fr, kind):
    m = S[f"{fr}_role"] == 0
    return m & (S[f"{fr}_roll12"] == 0) if kind == "react" else m

def seed_f1(names, kind):
    mv, mt = mask("val_main", kind), mask("test_main", kind)
    yv, yt, ft = S["val_main_y"][mv], S["test_main_y"][mt], S["test_main_f"][mt]
    return np.array([metrics(yt, P[n]["test_main"][mt], best_thr(yv, P[n]["val_main"][mv]), ft, nboot=0)["f1"] for n in names])

def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight"); plt.close(fig); print("saved", name)

# ---------- Figure 1: multi-seed F1 (dot + range), main and reactivation slices ----------
fams = {}
for m in P:
    b = re.sub(r"_s\d+$", "", m)
    if b in NICE and not b.startswith("rule"): fams.setdefault(b, []).append(m)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
for ax, (kind, title) in zip(axes, [("main", "All test buckets"), ("react", "Reactivation buckets (no event in prior hour)")]):
    rows = [(NICE[b], seed_f1(sorted(ms), kind)) for b, ms in fams.items()]
    rows.sort(key=lambda r: r[1].mean())
    rule = seed_f1(["rule_majority_7d"], kind)[0]
    y = np.arange(len(rows))
    for i, (lab, v) in enumerate(rows):
        ax.plot([v.min(), v.max()], [i, i], color=C[0], lw=2.2, solid_capstyle="round", alpha=0.45)
        ax.scatter(v, [i] * len(v), s=14, color=C[0], alpha=0.55, zorder=3, linewidths=0)
        ax.scatter([v.mean()], [i], s=46, color=C[0], zorder=4, edgecolors=SURF, linewidths=1.5)
        ax.annotate(f"{v.mean():.4f}", (v.max(), i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.axvline(rule, color=RULE, lw=1.4, ls=(0, (4, 3)))
    ax.annotate(f"7-day rule {rule:.3f}", (rule, len(rows) - 0.45), xytext=(-4, 0), textcoords="offset points",
                ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows]); ax.set_title(title, loc="left"); ax.set_xlabel("F1 on test days 27–30")
    lo = min(rule, min(r[1].min() for r in rows)); hi = max(r[1].max() for r in rows)
    ax.set_xlim(lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.38); ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.grid(axis="y", visible=False)
fig.text(0.0, -0.06, "Large dot = mean of 3 seeds; small dots = individual seeds; bar = min–max. Threshold tuned on validation per seed.\n"
         "Axes are zoomed (not starting at 0) to show differences between models.", fontsize=7.5, color=INK2)
fig.tight_layout(); save(fig, "fig1_multiseed_f1.png")

# ---------- Figure 2: precision-recall curves (main, reactivation) ----------
curves = [("xgb_full", "XGBoost (all features)", C[0]), ("abigru_s0", "ABiGRU", C[1]), ("mlp_static_s0", "MLP (no sequence)", C[2])]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
for ax, (kind, title, lim) in zip(axes, [("main", "All test buckets", 0.6), ("react", "Reactivation buckets", 0.0)]):
    mt = mask("test_main", kind); yt = S["test_main_y"][mt]
    for key, lab, col in curves:
        p = P[key]["test_main"][mt]; pr, rc, _ = precision_recall_curve(yt, p)
        ax.plot(rc, pr, color=col, label=f"{lab}  (AP {average_precision_score(yt, p):.3f})")
    r = P["rule_majority_7d"]["test_main"][mt] >= 0.5
    rp, rr = (r & (yt > 0)).sum() / max(r.sum(), 1), (r & (yt > 0)).sum() / max((yt > 0).sum(), 1)
    ax.scatter([rr], [rp], s=60, marker="D", color=RULE, edgecolors=SURF, linewidths=1.5, zorder=5,
               label=f"Rule: 7-day majority  (P {rp:.3f}, R {rr:.3f})")
    ax.set_xlim(lim, 1.005); ax.set_ylim(lim, 1.005); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title + (" (axes from 0.6)" if lim else ""), loc="left"); ax.legend(loc="lower left")
fig.tight_layout(); save(fig, "fig2_precision_recall.png")

# ---------- Figure 3: ROC (main slice, zoomed false-positive range) ----------
fig, ax = plt.subplots(figsize=(4.8, 3.9))
mt = mask("test_main", "main"); yt = S["test_main_y"][mt]
for key, lab, col in curves:
    p = P[key]["test_main"][mt]; fpr, tpr, _ = roc_curve(yt, p)
    ax.plot(fpr, tpr, color=col, label=f"{lab}  (AUC {roc_auc_score(yt, p):.4f})")
r = P["rule_majority_7d"]["test_main"][mt] >= 0.5
ax.scatter([(r & (yt == 0)).sum() / (yt == 0).sum()], [(r & (yt > 0)).sum() / (yt > 0).sum()], s=60, marker="D", color=RULE,
           edgecolors=SURF, linewidths=1.5, zorder=5, label="Rule: 7-day majority")
ax.set_xlim(0, 0.05); ax.set_ylim(0.6, 1.005); ax.set_xlabel("False-positive rate (zoomed to 0–0.05)"); ax.set_ylabel("True-positive rate")
ax.set_title("ROC, all test buckets", loc="left"); ax.legend(loc="lower right")
fig.tight_layout(); save(fig, "fig3_roc.png")

# ---------- Figure 4: confusion matrices (seed-0 / seed-42 models, main slice) ----------
from matplotlib.colors import LinearSegmentedColormap
blues = LinearSegmentedColormap.from_list("b", ["#eef4fc", "#2a78d6", "#123b6e"])
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
mv = mask("val_main", "main"); yv = S["val_main_y"][mv]
for ax, (key, lab) in zip(axes, [("xgb_full", "XGBoost (all features)"), ("abigru_s0", "ABiGRU")]):
    thr = best_thr(yv, P[key]["val_main"][mv]); pr = P[key]["test_main"][mt] >= thr
    cm = np.array([[(~pr & (yt == 0)).sum(), (pr & (yt == 0)).sum()], [(~pr & (yt > 0)).sum(), (pr & (yt > 0)).sum()]])
    share = cm / cm.sum(1, keepdims=True)
    ax.imshow(share, cmap=blues, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.1%} of row", ha="center", va="center", fontsize=9,
                    color="white" if share[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1], ["Predicted: no", "Predicted: yes"]); ax.set_yticks([0, 1], ["Actual: no", "Actual: yes"])
    ax.grid(False); ax.set_title(f"{lab}  (threshold {thr:.2f})", loc="left")
    for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); save(fig, "fig4_confusion_matrices.png")

# ---------- Figure 5: XGBoost feature importance (top 15) ----------
imp = json.load(open(f"{RES}/info_xgb_full.json"))["importance"]
top = sorted(imp.items(), key=lambda x: -x[1])[:15][::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.barh([t[0] for t in top], [t[1] for t in top], color=C[0], height=0.62)
for i, (_, v) in enumerate(top):
    ax.annotate(f"{v:.3f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
ax.set_xlim(0, max(t[1] for t in top) * 1.15); ax.set_xlabel("Importance share (XGBoost gain)"); ax.grid(axis="y", visible=False)
ax.set_title("Top 15 features, XGBoost (all 57 features)", loc="left")
fig.text(0.0, -0.02, "slot_hits_7d = days of the last 7 with an event in this 5-minute slot; d1_+0 = event in this slot yesterday.",
         fontsize=7.5, color=INK2)
fig.tight_layout(); save(fig, "fig5_feature_importance.png")

# ---------- Figure 6: neural training curves (validation PR-AUC per epoch, seed 0) ----------
fig, ax = plt.subplots(figsize=(6.0, 3.7))
for (m, lab), col in zip([("abigru", "ABiGRU"), ("bigru", "BiGRU"), ("gru", "GRU"), ("mlp_static", "MLP (no sequence)")], C):
    h = json.load(open(f"{RES}/info_{m}_s0.json"))["history"]
    ax.plot([e["epoch"] for e in h], [e["val_pr_auc"] for e in h], color=col, marker="o", ms=3.5, label=lab)
ax.set_xlabel("Epoch (700,000 sampled training rows each)"); ax.set_ylabel("Validation PR-AUC")
ax.set_title("Neural models: validation PR-AUC during training (seed 0)", loc="left"); ax.legend(loc="lower right")
fig.tight_layout(); save(fig, "fig6_training_curves.png")
