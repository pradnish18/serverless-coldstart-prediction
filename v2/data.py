"""Vectorized data cache + function metadata for Huawei R5 cold-start trace (v2 pipeline)."""
import glob, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGION = os.environ.get("CS_REGION", "5")      # Huawei region 1-5; region 5 keeps the original cache name
CACHE = os.path.join(ROOT, "v2", "cache", "matrices.npz" if REGION == "5" else f"matrices_R{REGION}.npz")
AGG_DIR = os.path.join(ROOT, "v2", "cache", f"agg_R{REGION}")   # per-day aggregates written by region_prep.py
B, PD, DAYS = 300, 288, 31
TOT = DAYS * PD
TRAIN_END, VAL_END = 24 * PD, 27 * PD
STAT_END = 7 * PD                           # static stats use days 0-6 only (before any training target)      # days 0-23 / 24-26 / 27-30
MIN_TRAIN_EV = 10
HOLDOUT_FRAC, SEED = 0.2, 42


AGG_KEYS = ["fid", "b", "clusterName"]


def aggregate(df):
    """Per (function, bucket, cluster) sums. Everything the cache needs is derivable from these sums, so large regions
    can be aggregated one day at a time (region_prep.py) without loading the whole trace into memory."""
    pool = df.podID.str.rsplit("-", n=1).str[0]
    df = df.assign(fid=df.funcName.astype(str) + "--" + df.userID.astype(str) + "--" + pool,
                   b=(df.time // B).astype(int), depn=(df.deployDependencyCost > 0).astype(np.int64))
    return df.groupby(AGG_KEYS, sort=False).agg(n=("totalCost_cold_start", "size"), c=("totalCost_cold_start", "sum"),
                                                 dep=("depn", "sum"), code=("deployCodeCost", "sum"),
                                                 sched=("schedulingCost", "sum"), alloc=("podAllocationCost", "sum")).reset_index()


def raw_files():
    pats = [f"{ROOT}/dataset/raw/region{REGION}_cold_starts/R{REGION}/day_*.csv", f"{ROOT}/dataset/raw/**/R{REGION}/day_*.csv"]
    for p in pats:
        fs = sorted(glob.glob(p, recursive=True))
        if fs:
            return fs
    return []


def build():
    if os.path.isdir(AGG_DIR) and glob.glob(f"{AGG_DIR}/day_*.pkl"):
        A = pd.concat(pd.read_pickle(f) for f in sorted(glob.glob(f"{AGG_DIR}/day_*.pkl")))
    else:
        A = pd.concat(aggregate(pd.read_csv(f)) for f in raw_files())
    A = A.groupby(AGG_KEYS, sort=False).sum().reset_index()   # a key can only repeat across files at day borders
    A["pool"] = A.fid.str.split("--").str[2]
    A["user"] = A.fid.str.split("--").str[1]
    fids = np.sort(A.fid.unique()); idx = {f: i for i, f in enumerate(fids)}
    A["fi"] = A.fid.map(idx)
    F = len(fids)
    cnt = np.zeros((F, TOT), np.float32); cost = np.zeros((F, TOT), np.float32)
    g = A.groupby(["fi", "b"]).agg(n=("n", "sum"), c=("c", "sum")).reset_index()
    cnt[g.fi.values, g.b.values] = g.n.values; cost[g.fi.values, g.b.values] = g.c.values

    # ---- static metadata, computed ONLY from days 0-6 (STAT_END); role uses event counts of days 0-23 ----
    tr = A[A.b < STAT_END]
    trfull = A[A.b < TRAIN_END]   # used ONLY to define slice membership (role), never as a feature
    meta = pd.DataFrame(index=range(F))
    first = A.groupby("fi").first()
    pp = first.pool.str.extract(r"pool(\d+)-(\d+)-(\d+)").astype(float)
    meta["pool_id"], meta["pool_cpu"], meta["pool_mem"] = pp[0].values, pp[1].values, pp[2].values
    cl = A.groupby(["fi", "clusterName"]).n.sum().reset_index().sort_values(["fi", "n", "clusterName"], ascending=[True, False, True])
    meta["cluster"] = cl.groupby("fi").clusterName.first().values          # mode, ties -> smallest (as Series.mode)
    meta["user"] = pd.to_numeric(first.user, errors="coerce").values
    s_ = tr.groupby("fi").agg(n=("n", "sum"), c=("c", "sum"), dep=("dep", "sum"), code=("code", "sum"), sched=("sched", "sum"),
                               alloc=("alloc", "sum"), first_b=("b", "min"))
    st = pd.DataFrame({"tr_events": s_.n, "tr_mean_cost": s_.c / s_.n, "tr_dep_frac": s_.dep / s_.n, "tr_code": s_.code / s_.n,
                       "tr_sched": s_.sched / s_.n, "tr_alloc": s_.alloc / s_.n, "first_b": s_.first_b})
    meta = meta.join(st).fillna({"tr_events": 0, "tr_mean_cost": 0, "tr_dep_frac": 0, "tr_code": 0, "tr_sched": 0, "tr_alloc": 0})
    meta["first_b"] = meta["first_b"].fillna(TOT).astype(float)
    # per-function periodicity from training days: active-bucket-of-day concentration
    ev = (cnt[:, :STAT_END] > 0).reshape(F, STAT_END // PD, PD)
    meta["tr_active_days"] = ev.any(2).sum(1).astype(float)
    per_slot = ev.sum(1)                                   # (F, 288) days active per slot
    meta["tr_slot_conc"] = np.where(per_slot.sum(1) > 0, per_slot.max(1) / np.maximum(per_slot.sum(1), 1), 0)
    meta["tr_event_buckets"] = ev.reshape(F, -1).sum(1).astype(float)
    rng = np.random.RandomState(SEED)
    trev = trfull.groupby('fi').n.sum().reindex(range(F)).fillna(0).values
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

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)   # cache/ is not in git; create it on a fresh clone
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
