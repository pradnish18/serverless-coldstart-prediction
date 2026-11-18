"""Neural model definitions + checkpoint save/load for the v2 pipeline (independent of the v1 `src/` package).

Checkpoint format (torch.save dict):
    state_dict, kind, n_static, window, mu, sd, feature_names, seed, history, torch_version
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
from data import PD, TOT

W = 24                      # sequence window (buckets)
KINDS = ("mlp_static", "gru", "bigru", "abigru")


class Attention(nn.Module):
    """Additive attention: score_t = v^T tanh(W h_t); alpha = softmax(score); context = sum_t alpha_t h_t."""
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, h):
        scores = self.v(torch.tanh(self.W(h))).squeeze(-1)
        w = Fn.softmax(scores, dim=1)
        return torch.bmm(w.unsqueeze(1), h).squeeze(1), w


class Net(nn.Module):
    def __init__(self, kind, ns, hid=64):
        super().__init__()
        assert kind in KINDS, kind
        self.kind = kind
        self.static = nn.Sequential(nn.Linear(ns, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
        enc = 0
        if kind != "mlp_static":
            bi = kind in ("bigru", "abigru")
            self.rnn = nn.GRU(7, hid, 2, batch_first=True, bidirectional=bi, dropout=0.3)
            enc = hid * (2 if bi else 1)
            if kind == "abigru":
                self.att = Attention(enc, 32); enc = enc * 2
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 + enc, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, s, x):
        z = [self.static(s)]
        if self.kind != "mlp_static":
            o, h = self.rnn(x)
            if self.kind == "gru":
                z.append(h[-1])
            elif self.kind == "bigru":
                z.append(torch.cat([h[-2], h[-1]], 1))
            else:
                c, _ = self.att(o); z.append(torch.cat([c, o.max(1).values], 1))
        return self.head(torch.cat(z, 1)).squeeze(-1)       # logit


def make_tensors(ctx, dev):
    """Per-bucket channel tensors (F x T) used to build the 24 x 7 sequence on the fly."""
    ev = torch.tensor(ctx.ev, device=dev)
    lc = torch.tensor(np.log1p(ctx.cost), device=dev)
    pr = torch.tensor(ctx.P[ctx.pidx] / np.maximum(ctx.psize[ctx.pidx], 1)[:, None], device=dev, dtype=torch.float32)
    tt = torch.arange(TOT, device=dev)
    return dict(ev=ev, lc=lc, pr=pr, hs=torch.sin(2 * np.pi * (tt % PD) / PD), hc=torch.cos(2 * np.pi * (tt % PD) / PD),
                offs=torch.arange(-W, 0, device=dev))


def seq(T, f, t):
    """Channels per step (7): event, log cost, sin, cos, event 1 day earlier, event 7 days earlier, pool activity."""
    idx = t[:, None] + T["offs"][None, :]
    idx1 = (idx - PD).clamp(min=0); idx7 = (idx - 7 * PD).clamp(min=0); fi = f[:, None]
    ch = [T["ev"][fi, idx], T["lc"][fi, idx], T["hs"][idx], T["hc"][idx], T["ev"][fi, idx1], T["ev"][fi, idx7], T["pr"][fi, idx]]
    return torch.stack(ch, -1)


def standardize(X, mu, sd):
    return np.nan_to_num(np.clip((X - mu) / sd, -6, 6)).astype(np.float32)   # NaN (2 fns with unknown pool) -> 0


def predict(net, T, S, f, t, bs=8192):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(f), bs):
            out.append(torch.sigmoid(net(S[i:i + bs], seq(T, f[i:i + bs], t[i:i + bs]))).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def save_checkpoint(path, net, kind, n_static, mu, sd, feature_names, seed, history):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "kind": kind, "n_static": int(n_static),
                "window": W, "mu": np.asarray(mu, np.float32), "sd": np.asarray(sd, np.float32),
                "feature_names": list(feature_names), "seed": int(seed), "history": history,
                "torch_version": torch.__version__}, path)


def load_checkpoint(path, device="cpu"):
    # weights_only=False: our checkpoints also store numpy scaler arrays. Only load checkpoints you trust.
    ck = torch.load(path, map_location=device, weights_only=False)
    net = Net(ck["kind"], ck["n_static"]).to(device)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, ck
