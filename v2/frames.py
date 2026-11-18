import numpy as np
from data import load, PD, TOT, TRAIN_END, VAL_END
from features import Ctx, make

T0 = 7 * PD   # first target bucket (needs 7 days of lag history)

def get_ctx():
    d = load(); return d, Ctx(d)

def frame_defs(d):
    role = d["role"]
    main = np.where(role != 1)[0]; hold = np.where(role == 1)[0]
    return {"train": (main, T0, TRAIN_END), "val_main": (main, TRAIN_END, VAL_END), "test_main": (main, VAL_END, TOT),
            "val_hold": (hold, TRAIN_END, VAL_END), "test_hold": (hold, VAL_END, TOT)}

def build(ctx, d, kind="full", names=("train", "val_main", "test_main", "val_hold", "test_hold")):
    out = {}
    fd = frame_defs(d)
    for n in names:
        fs, lo, hi = fd[n]
        X, cols, f, t = make(ctx, fs, lo, hi, kind)
        out[n] = dict(X=X, f=f, t=t, y=ctx.ev[f, t].astype(np.int8), cost=ctx.cost[f, t],
                      role=ctx.role[f])
    return out, cols
