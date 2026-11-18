"""Vectorized data cache + function metadata for Huawei R5 cold-start trace (v2 pipeline)."""
import glob, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "v2", "cache", "matrices.npz")
B, PD, DAYS = 300, 288, 31
TOT = DAYS * PD
TRAIN_END, VAL_END = 24 * PD, 27 * PD
STAT_END = 7 * PD                           # static stats use days 0-6 only (before any training target)      # days 0-23 / 24-26 / 27-30
MIN_TRAIN_EV = 10
HOLDOUT_FRAC, SEED = 0.2, 42


def build():
    df = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f"{ROOT}/dataset/raw/region5_cold_starts/R5/day_*.csv")))
    pool = df.podID.str.rsplit("-", n=1).str[0]
    df["pool"] = pool
    df["fid"] = df.funcName.astype(str) + "--" + df.userID.astype(str) + "--" + pool
    df["b"] = (df.time // B).astype(int)
    fids = np.sort(df.fid.unique()); idx = {f: i for i, f in enumerate(fids)}
    df["fi"] = df.fid.map(idx)
    F = len(fids)
    cnt = np.zeros((F, TOT), np.float32); cost = np.zeros((F, TOT), np.float32)
    g = df.groupby(["fi", "b"]).agg(n=("totalCost_cold_start", "size"), c=("totalCost_cold_start", "sum")).reset_index()
    cnt[g.fi.values, g.b.values] = g.n.values; cost[g.fi.values, g.b.values] = g.c.values

    # ---- static metadata, computed ONLY from the training period (b < TRAIN_END) ----
    tr = df[df.b < STAT_END]
    trfull = df[df.b < TRAIN_END]   # used ONLY to define slice membership (role), never as a feature
    meta = pd.DataFrame(index=range(F))
    pp = df.groupby("fi").pool.first().str.extract(r"pool(\d+)-(\d+)-(\d+)").astype(float)
    meta["pool_id"], meta["pool_cpu"], meta["pool_mem"] = pp[0].values, pp[1].values, pp[2].values
    meta["cluster"] = df.groupby("fi").clusterName.agg(lambda s: s.mode().iloc[0]).values
    meta["user"] = df.groupby("fi").userID.first().values
    st = tr.groupby("fi").agg(tr_events=("b", "size"), tr_mean_cost=("totalCost_cold_start", "mean"),
                              tr_dep_frac=("deployDependencyCost", lambda s: (s > 0).mean()),
                              tr_code=("deployCodeCost", "mean"), tr_sched=("schedulingCost", "mean"),
                              tr_alloc=("podAllocationCost", "mean"), first_b=("b", "min"))
    meta = meta.join(st).fillna({"tr_events": 0, "tr_mean_cost": 0, "tr_dep_frac": 0, "tr_code": 0, "tr_sched": 0, "tr_alloc": 0})
    meta["first_b"] = meta["first_b"].fillna(TOT).astype(float)
    # per-function periodicity from training days: active-bucket-of-day concentration
    ev = (cnt[:, :STAT_END] > 0).reshape(F, STAT_END // PD, PD)
    meta["tr_active_days"] = ev.any(2).sum(1).astype(float)
    per_slot = ev.sum(1)                                   # (F, 288) days active per slot
    meta["tr_slot_conc"] = np.where(per_slot.sum(1) > 0, per_slot.max(1) / np.maximum(per_slot.sum(1), 1), 0)
    meta["tr_event_buckets"] = ev.reshape(F, -1).sum(1).astype(float)
    rng = np.random.RandomState(SEED)
    trev = trfull.groupby('fi').size().reindex(range(F)).fillna(0).values
    elig = np.where(trev >= MIN_TRAIN_EV)[0]
    holdout = rng.choice(elig, int(len(elig) * HOLDOUT_FRAC), replace=False)
    role = np.zeros(F, np.int8)             # 0 normal-eligible, 1 holdout(new), 2 low-history(<10 train events)
    role[trev < MIN_TRAIN_EV] = 2
    role[holdout] = 1
    # pool-level hourly profile from training period (shared across functions of the pool)
    pool_ids = meta["pool_id"].fillna(-1).values
    hour_rate = np.zeros((F, 24), np.float32)
    for p in np.unique(pool_ids):
        m = (pool_ids == p) & (role != 1)   # holdout (new) functions never contribute
        if m.sum() == 0:
            continue
        e = (cnt[m, :STAT_END] > 0).sum(0).reshape(-1, PD).reshape(-1, 24, 12).sum((0, 2)) / (m.sum() * (STAT_END // PD) * 12)
        hour_rate[pool_ids == p] = e
    # function-level hourly rate from training period
    fh = (cnt[:, :STAT_END] > 0).reshape(F, -1, 24, 12).mean((1, 3)).astype(np.float32)

    np.savez_compressed(CACHE, cnt=cnt, cost=cost, fids=fids, role=role, hour_rate=hour_rate, fn_hour_rate=fh,
                        meta=meta.values.astype(np.float32), meta_cols=np.array(meta.columns))
    return CACHE


def load():
    if not os.path.exists(CACHE):
        build()
    d = np.load(CACHE, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    d = load()
    print({k: v.shape for k, v in d.items()})
    r = d["role"]; print("normal", (r == 0).sum(), "holdout", (r == 1).sum(), "low-history", (r == 2).sum())
