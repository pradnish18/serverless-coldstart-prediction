"""Leakage-safe vectorized features. Every feature for target bucket t uses buckets < t only
(or train-period-only static stats). Returns flat (function, bucket) rows."""
import numpy as np
from data import PD, TOT

CAP = 7 * PD
NAMES_SEQ = None


def _cs(a):  # cumulative sum with leading zero along time
    return np.concatenate([np.zeros((a.shape[0], 1), np.float32), np.cumsum(a, 1, dtype=np.float32)], 1)


class Ctx:
    def __init__(self, d):
        self.d = d
        self.cnt, self.cost, self.role = d["cnt"], d["cost"], d["role"]
        self.ev = (self.cnt > 0).astype(np.float32)
        self.cs = _cs(self.ev)
        self.meta = d["meta"]; self.mc = list(d["meta_cols"])
        F = self.ev.shape[0]
        idx = np.arange(TOT)
        last = np.where(self.ev > 0, idx[None, :], -10**6)
        self.last = np.maximum.accumulate(last, 1)
        # pool dynamics from non-holdout functions only
        pool = self.meta[:, self.mc.index("pool_id")]
        self.pool = np.nan_to_num(pool, nan=-1).astype(int)
        up = np.unique(self.pool)
        self.pidx = np.searchsorted(up, self.pool)
        P = np.zeros((len(up), TOT), np.float32)
        nh = self.role != 1
        np.add.at(P, self.pidx[nh], self.ev[nh])
        self.P = P; self.Pcs = _cs(P)
        self.psize = np.bincount(self.pidx[nh], minlength=len(up)).astype(np.float32)

    def rsum(self, f, t, w):          # events in [t-w, t-1]
        return self.cs[f, t] - self.cs[f, np.maximum(t - w, 0)]


def make(ctx, fsel, t_lo, t_hi, kind="full"):
    """kind: 'full' (own history + static + pool) or 'cold' (no own history: pool + hardware + time)."""
    fsel = np.asarray(fsel)
    ts = np.arange(t_lo, t_hi)
    f = np.repeat(fsel, len(ts)); t = np.tile(ts, len(fsel))
    ev, m, mc = ctx.ev, ctx.meta, ctx.mc
    cols, names = [], []

    def add(n, v): names.append(n); cols.append(np.asarray(v, np.float32))

    hod = (t % PD) / 12.0
    add("hour_sin", np.sin(2 * np.pi * hod / 24)); add("hour_cos", np.cos(2 * np.pi * hod / 24))
    add("slot", (t % PD))
    hour = ((t % PD) // 12).astype(int)
    for c in ("pool_cpu", "pool_mem", "cluster"):
        add(c, m[f, mc.index(c)])
    add("pool_hour_rate", ctx.d["hour_rate"][f, hour])
    # live pool load (other functions of same pool, not self), last bucket / 1h / same slot yesterday
    pi = ctx.pidx[f]; selfev = np.where(ctx.role[f] != 1, 1.0, 0.0)
    def pool_sum(lagk, w=1):
        s = ctx.Pcs[pi, np.maximum(t - lagk + w, 0)] - ctx.Pcs[pi, np.maximum(t - lagk, 0)]
        return s
    own1 = ev[f, np.maximum(t - 1, 0)] * selfev
    add("pool_last1", (pool_sum(1) - own1) / np.maximum(ctx.psize[pi], 1))
    own12 = ctx.rsum(f, t, 12) * selfev
    add("pool_last12", (pool_sum(12, 12) - own12) / np.maximum(ctx.psize[pi] * 12, 1))
    add("pool_yday", pool_sum(PD, 1) / np.maximum(ctx.psize[pi], 1))   # t-288 bucket incl. self hist ok (past)
    if kind == "cold":
        return np.stack(cols, 1), names, f, t

    for k in range(1, 13):
        add(f"lag{k}", ev[f, np.maximum(t - k, 0)])
    for d_ in (1, 2, 3, 7):
        for off in (-1, 0, 1):
            add(f"d{d_}_{off:+d}", ev[f, np.maximum(t - d_ * PD + off, 0)])
    hits = sum(ev[f, np.maximum(t - d_ * PD, 0)] for d_ in range(1, 8))
    add("slot_hits_7d", hits)
    add("slot_hits_7d_pm1", sum(ev[f, np.maximum(t - d_ * PD + o, 0)] for d_ in range(1, 8) for o in (-1, 1)))
    for w in (3, 12, 72, 288, 2016):
        add(f"roll{w}", ctx.rsum(f, t, w))
    since = (t - 1) - ctx.last[f, t - 1]
    add("since_last", np.where(ctx.last[f, t - 1] < 0, CAP, np.minimum(since, CAP)))
    add("cnt_lag1", np.log1p(ctx.cnt[f, np.maximum(t - 1, 0)])); add("cnt_yday", np.log1p(ctx.cnt[f, np.maximum(t - PD, 0)]))
    add("cost_lag1", np.log1p(ctx.cost[f, np.maximum(t - 1, 0)])); add("cost_yday", np.log1p(ctx.cost[f, np.maximum(t - PD, 0)]))
    for c in ("tr_events", "tr_mean_cost", "tr_dep_frac", "tr_code", "tr_sched", "tr_alloc", "tr_active_days", "tr_slot_conc", "tr_event_buckets"):
        v = m[f, mc.index(c)]; add(c, np.log1p(v) if c in ("tr_events", "tr_event_buckets") else v)
    add("fn_hour_rate", ctx.d["fn_hour_rate"][f, hour])
    add("age_buckets", np.clip(t - m[f, mc.index("first_b")], -1, CAP))
    return np.stack(cols, 1), names, f, t
