"""Write v2/models/thresholds.json: the validation-tuned (max-F1) threshold of each model on the main validation frame."""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalx import best_thr
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
S = np.load(f"{RES}/frames.npz"); mv = S["val_main_role"] == 0; yv = S["val_main_y"][mv]
out = {}
for fp in glob.glob(f"{RES}/preds_*.npz"):
    z = np.load(fp)
    for k in z.files:
        m, fr = k.split("__")
        if m.startswith(("rule_", "score_")) or "smoke" in fp:
            continue
        key = m + "_s42" if m.startswith("xgb_") and not re.search(r"_s\d+$", m) else m   # match model file names
        if m == "xgb_cold_newfunc" and fr == "val_hold":        # new-function model: tune on held-out functions
            out[key] = round(float(best_thr(S["val_hold_y"], z[k])), 4)
        elif m != "xgb_cold_newfunc" and fr == "val_main":
            out[key] = round(float(best_thr(yv, z[k][mv])), 4)
os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
json.dump(dict(sorted(out.items())), open(os.path.join(HERE, "models", "thresholds.json"), "w"), indent=1)
print(json.dumps(dict(sorted(out.items())), indent=1))
