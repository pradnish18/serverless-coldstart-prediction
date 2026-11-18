"""Hybrid sequence models: (24-bucket window incl. seasonally aligned channels) encoder + tabular context.
usage: python run_deep.py <model> <seed>      model in mlp_static|gru|bigru|abigru
Saves predictions (results/preds_*.npz), training info (results/info_*.json) AND the trained model (models/<model>_s<seed>.pt).
Set COLDSTART_SMOKE=1 for a tiny test run that writes to a temp folder."""
import sys, os, time, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.metrics import average_precision_score
from frames import *
import nets
HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("COLDSTART_SMOKE") == "1"
RES = os.environ.get("COLDSTART_OUT", os.path.join(HERE, "results")); MODELS = os.environ.get("COLDSTART_MODELS", os.path.join(HERE, "models"))
os.makedirs(RES, exist_ok=True); os.makedirs(MODELS, exist_ok=True)
name, seed = sys.argv[1], int(sys.argv[2])
EPOCHS, PATIENCE, BS, SAMPLE = (1, 3, 2048, 30_000) if SMOKE else (14, 3, 2048, 700_000)
torch.manual_seed(seed); np.random.seed(seed)
dev = torch.device(os.environ.get("COLDSTART_DEVICE") or                       # optional override: cpu / cuda / mps
                   ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
print(f"device: {dev}", flush=True)

d, ctx = get_ctx()
full, names = build(ctx, d, "full")
mu = np.nanmean(full["train"]["X"], 0); sd = np.nanstd(full["train"]["X"], 0) + 1e-6
T = nets.make_tensors(ctx, dev)
net = nets.Net(name, len(names)).to(dev)
opt = torch.optim.RMSprop(net.parameters(), lr=1e-3, alpha=0.9, eps=1e-8)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
D = {k: (torch.tensor(nets.standardize(v["X"], mu, sd), device=dev), torch.tensor(v["f"], device=dev), torch.tensor(v["t"], device=dev), v["y"])
     for k, v in full.items()}
pred = lambda key: nets.predict(net, T, D[key][0], D[key][1], D[key][2])

S, f, t, y = D["train"]; ytr = torch.tensor(y, dtype=torch.float32, device=dev)
best, bad, best_state, hist = -1, 0, None, []
t0 = time.time()
for ep in range(EPOCHS):
    net.train(); perm = torch.randperm(len(f), device=dev)[:SAMPLE]; tl = 0
    for i in range(0, len(perm), BS):
        j = perm[i:i + BS]
        loss = Fn.binary_cross_entropy_with_logits(net(S[j], nets.seq(T, f[j], t[j])), ytr[j])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tl += loss.item() * len(j)
    ap = average_precision_score(D["val_main"][3], pred("val_main"))
    sched.step(ap); hist.append({"epoch": ep + 1, "train_loss": tl / len(perm), "val_pr_auc": float(ap)})
    print(f"{name} s{seed} ep{ep+1} loss {tl/len(perm):.4f} val PR-AUC {ap:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if ap > best: best, bad = ap, 0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
    else:
        bad += 1
        if bad >= PATIENCE: break
net.load_state_dict(best_state)
res = {k: pred(k) for k in ("val_main", "test_main", "val_hold", "test_hold")}
tag = "smoke_" if SMOKE else ""
np.savez_compressed(f"{RES}/preds_{tag}{name}_s{seed}.npz", **{f"{name}_s{seed}__{k}": v for k, v in res.items()})
json.dump({"history": hist, "best_val_pr_auc": best, "params": sum(p.numel() for p in net.parameters()), "epochs": len(hist)},
          open(f"{RES}/info_{tag}{name}_s{seed}.json", "w"), indent=1)
nets.save_checkpoint(f"{MODELS}/{tag}{name}_s{seed}.pt", net, name, len(names), mu, sd, names, seed, hist)
print("DONE", name, seed, "model saved ->", f"{MODELS}/{tag}{name}_s{seed}.pt", flush=True)
