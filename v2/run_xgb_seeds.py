import os, numpy as np, xgboost as xgb
from frames import *
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"); os.makedirs(MODELS, exist_ok=True)
d, ctx = get_ctx(); full, names = build(ctx, d, "full")
nometa = [i for i, n in enumerate(names) if not (n.startswith("tr_") or n in ("fn_hour_rate", "age_buckets"))]
P = dict(max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0)
out = {}
for seed in (1, 2):
    for nm, cols in (("xgb_full", list(range(len(names)))), ("xgb_no_static", nometa)):
        m = xgb.XGBClassifier(n_estimators=1500, early_stopping_rounds=40, tree_method="hist", n_jobs=10, eval_metric="aucpr", random_state=seed, **P)
        m.fit(full["train"]["X"][:, cols], full["train"]["y"], eval_set=[(full["val_main"]["X"][:, cols], full["val_main"]["y"])], verbose=False)
        m.save_model(f"{MODELS}/{nm}_s{seed}.json")
        for k in ("val_main", "test_main", "val_hold", "test_hold"):
            out[f"{nm}_s{seed}__{k}"] = m.predict_proba(full[k]["X"][:, cols])[:, 1].astype(np.float32)
        print(nm, seed, m.best_iteration, flush=True)
np.savez_compressed(f"{RES}/preds_xgbseeds.npz", **out); print("DONE xgb seeds")
