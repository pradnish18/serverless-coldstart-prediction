import sys, time, numpy as np, xgboost as xgb, json, os
from frames import *
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"); os.makedirs(MODELS, exist_ok=True)
d, ctx = get_ctx()
full, names = build(ctx, d, "full")
cold, cnames = build(ctx, d, "cold")
ni = {n: i for i, n in enumerate(names)}

# side info for the report
side = {}
for k, v in full.items():
    side[k + "_y"] = v["y"]; side[k + "_f"] = v["f"]; side[k + "_t"] = v["t"]; side[k + "_cost"] = v["cost"]
    side[k + "_role"] = v["role"]; side[k + "_roll12"] = v["X"][:, ni["roll12"]]
    side[k + "_poolhr"] = v["X"][:, ni["pool_hour_rate"]]
np.savez_compressed(f"{RES}/frames.npz", **side)

preds = {}
def rule(name, fn):
    preds[name] = {k: fn(v["X"]).astype(np.float32) for k, v in full.items() if k != "train"}
g = lambda X, n: X[:, ni[n]]
rule("rule_last_bucket", lambda X: g(X, "lag1"))
rule("rule_same_bucket_1d", lambda X: g(X, "d1_+0"))
rule("rule_same_bucket_7d", lambda X: g(X, "d7_+0"))
rule("rule_1d_or_7d", lambda X: np.maximum(g(X, "d1_+0"), g(X, "d7_+0")))
rule("rule_majority_7d", lambda X: (g(X, "slot_hits_7d") >= 4))
rule("rule_any_last_hour", lambda X: g(X, "roll12") > 0)
preds["score_pool_hour_rate"] = {k: v["X"][:, ni["pool_hour_rate"]] for k, v in cold.items() if k != "train"}

def fit(name, data, cols_idx, params, rounds=1500):
    t0 = time.time()
    Xtr = data["train"]["X"][:, cols_idx]; ytr = data["train"]["y"]
    Xv = data["val_main"]["X"][:, cols_idx]; yv = data["val_main"]["y"]
    m = xgb.XGBClassifier(n_estimators=rounds, early_stopping_rounds=40, tree_method="hist", n_jobs=10,
                          eval_metric="aucpr", random_state=42, **params)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    m.save_model(f"{MODELS}/{name}_s42.json")   # portable native XGBoost format
    json.dump({"features": [(names if data is full else cnames)[i] for i in cols_idx]}, open(f"{MODELS}/{name}_features.json", "w"), indent=1)
    preds[name] = {k: m.predict_proba(v["X"][:, cols_idx])[:, 1].astype(np.float32) for k, v in data.items() if k != "train"}
    imp = dict(zip([ (names if data is full else cnames)[i] for i in cols_idx], m.feature_importances_.round(4).tolist()))
    print(f"{name}: best_iter={m.best_iteration} time={time.time()-t0:.0f}s", flush=True)
    json.dump({"best_iter": int(m.best_iteration), "params": params, "importance": imp, "n_features": len(cols_idx)},
              open(f"{RES}/info_{name}.json", "w"), indent=1)

P = dict(max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0)
allc = list(range(len(names)))
fit("xgb_full", full, allc, P)
nometa = [i for i, n in enumerate(names) if not (n.startswith("tr_") or n in ("fn_hour_rate", "age_buckets"))]
fit("xgb_no_static", full, nometa, P)
nopool = [i for i, n in enumerate(names) if not n.startswith("pool_")]
fit("xgb_no_pool", full, nopool, P)
fit("xgb_cold_newfunc", cold, list(range(len(cnames))), P)
np.savez_compressed(f"{RES}/preds_gbdt.npz", **{f"{m}__{k}": v for m, dd in preds.items() for k, v in dd.items()})
print("DONE", flush=True)
