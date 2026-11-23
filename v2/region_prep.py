"""Aggregate one Huawei region's cold-start CSVs (read straight from the zip, no extraction) into per-day
(function, bucket, cluster) sums under v2/cache/agg_R<n>/, then build the region cache v2/cache/matrices_R<n>.npz.
usage: python region_prep.py <region> [zip path]      (re-running skips days already aggregated)"""
import os, re, sys, glob, zipfile, time
import pandas as pd
r = sys.argv[1]; os.environ["CS_REGION"] = r
import data
ROOT = data.ROOT
zp = sys.argv[2] if len(sys.argv) > 2 else next(iter(sorted(glob.glob(f"{ROOT}/dataset/raw/R{r}*.zip") + glob.glob(f"{ROOT}/dataset/raw/region{r}*.zip"))), None)
os.makedirs(data.AGG_DIR, exist_ok=True)
z = zipfile.ZipFile(zp)
days = sorted(n for n in z.namelist() if re.search(r"day_\d+\.csv$", n) and "__MACOSX" not in n)
print(zp, len(days), "day files", flush=True)
t0 = time.time()
for n in days:
    out = os.path.join(data.AGG_DIR, os.path.basename(n).replace(".csv", ".pkl"))
    if os.path.exists(out):
        continue
    with z.open(n) as fh:
        df = pd.read_csv(fh, usecols=["time", "clusterName", "funcName", "userID", "totalCost_cold_start", "podAllocationCost",
                                       "deployCodeCost", "deployDependencyCost", "schedulingCost", "podID"])
    data.aggregate(df).to_pickle(out)
    print(n, len(df), f"{time.time()-t0:.0f}s", flush=True)
    if time.time() - t0 > float(os.environ.get("PREP_BUDGET", "1e9")):
        print("budget reached, re-run to continue"); sys.exit(0)
data.build(); print("cache ->", data.CACHE, f"{time.time()-t0:.0f}s")
d = data.load(); rl = d["role"]
print("functions", len(rl), "normal", (rl == 0).sum(), "holdout", (rl == 1).sum(), "low-history", (rl == 2).sum(),
      "events", int(d["cnt"].sum()))
