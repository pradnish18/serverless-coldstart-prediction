"""Load a saved v2 model and score one function over one day (historical replay).

  python3 v2/predict.py --list
  python3 v2/predict.py --model xgb_full_s42 --function "210--261--pool24-600-512" --day 28
  python3 v2/predict.py --model abigru_s0 --function 0 --day 28 --threshold 0.44     # function may be an index too

Models live in v2/models/ : xgb_<variant>_s<seed>.json (+ <variant>_features.json) and <kind>_s<seed>.pt (neural).
Day must be >= 7 (7 days of history are needed). Requires v2/cache/matrices.npz (python3 v2/data.py).
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frames import get_ctx
from features import make
from data import PD
HERE = os.path.dirname(os.path.abspath(__file__)); MODELS = os.path.join(HERE, "models")


def thresholds():
    p = os.path.join(MODELS, "thresholds.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def make_scorer(model_name, ctx):
    """Return score(function_indices, t_lo, t_hi) -> (probabilities, f, t) for the named saved model."""
    if model_name.startswith("xgb_"):
        import xgboost as xgb
        variant = model_name.rsplit("_s", 1)[0]
        feats = json.load(open(os.path.join(MODELS, f"{variant}_features.json")))["features"]
        m = xgb.XGBClassifier(); m.load_model(os.path.join(MODELS, model_name + ".json"))
        kind = "cold" if "cold" in variant else "full"
        def score(fs, lo, hi):
            X, names, f, t = make(ctx, fs, lo, hi, kind)
            return m.predict_proba(X[:, [names.index(x) for x in feats]])[:, 1], f, t
        return score
    import torch, nets
    net, ck = nets.load_checkpoint(os.path.join(MODELS, model_name + ".pt"), "cpu")
    T = nets.make_tensors(ctx, torch.device("cpu"))
    def score(fs, lo, hi):
        X, names, f, t = make(ctx, fs, lo, hi, "full")
        assert names == ck["feature_names"], "feature order mismatch"
        S = torch.tensor(nets.standardize(X, ck["mu"], ck["sd"]))
        return nets.predict(net, T, S, torch.tensor(f), torch.tensor(t)), f, t
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true"); ap.add_argument("--model"); ap.add_argument("--function"); ap.add_argument("--day", type=int)
    ap.add_argument("--threshold", type=float); a = ap.parse_args()
    if a.list or not a.model:
        for p in sorted(glob.glob(os.path.join(MODELS, "*.json")) + glob.glob(os.path.join(MODELS, "*.pt"))):
            b = os.path.basename(p)
            if "features" not in b and b != "thresholds.json": print(b.rsplit(".", 1)[0])
        return
    d, ctx = get_ctx(); fids = list(d["fids"])
    fi = int(a.function) if a.function.isdigit() else fids.index(a.function)
    if a.day is None or a.day < 7: sys.exit("--day must be >= 7")
    thr = a.threshold if a.threshold is not None else thresholds().get(a.model, 0.5)
    p, f, t = make_scorer(a.model, ctx)([fi], a.day * PD, (a.day + 1) * PD)
    print(f"model={a.model} function={fids[fi]} day={a.day} threshold={thr:.3f}")
    print("bucket  clock  probability  predicted  actual")
    for pp, tt in zip(p, t):
        pred = pp >= thr; act = ctx.ev[fi, tt] > 0
        if pred or act:
            print(f"{tt:6d}  {int(tt%PD*5//60):02d}:{int(tt%PD*5%60):02d}  {pp:11.3f}  {int(pred):9d}  {int(act):6d}")
    pr = p >= thr; ac = ctx.ev[fi, t] > 0
    print(f"predicted-yes buckets: {int(pr.sum())}  actual cold-start buckets: {int(ac.sum())}  hits: {int((pr & ac).sum())}")


if __name__ == "__main__":
    main()
