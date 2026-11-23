"""Hybrid-histogram (HH) keep-alive / pre-warm policy, adapted from Shahrad et al.,
"Serverless in the Wild", USENIX ATC 2020 (the standard pre-warming baseline from Azure Functions).

Original policy (per application, 1-minute bins, 4-hour range):
  * keep a histogram of idle times (ITs) between invocations;
  * if many ITs fall outside the range, predict the next IT with ARIMA
    (pre-warm = 85% of the prediction, keep-alive window = +/-15% around it);
  * else if the histogram is representative (coefficient of variation of bin counts > 2),
    pre-warm window = 5th percentile of ITs (minus 10% margin) and keep-alive window ends at the
    99th percentile (plus 10% margin); the container is unloaded after each execution and re-loaded
    at last_event + pre-warm;
  * otherwise fall back to a standard fixed keep-alive equal to the histogram range (4 h).

Adaptation to this trace (stated honestly in the paper):
  * the trace has cold-start events only, so the "idle time" is the gap between consecutive buckets that
    contain a cold start (not between invocations);
  * bins are 5-minute buckets; range = 48 buckets (4 h);
  * the ARIMA branch is replaced by a simple forecaster (median of the last 5 out-of-range gaps),
    because statsmodels is not part of the pipeline; it is used by few functions (reported);
  * strictly causal: the policy parameters used during the idle period that starts at event E[i] are
    computed only from gaps that ended at or before E[i].

Warm decision for bucket t: L = last event bucket < t, g = t - L.  warm  <=>  pw <= g <= ka_end.
"""
import numpy as np

RANGE = 48            # 4 h in 5-minute buckets
HEAD_Q, TAIL_Q = 5, 99
MARGIN = 0.10
CV_MIN = 2.0
MIN_ITS = 10          # fewer observed gaps -> not representative
OOB_FRAC = 0.5        # share of out-of-range gaps that switches to the forecaster branch


def _params(hist, n_oob, recent_oob, fallback_ka):
    """hist: counts of in-range gaps for gap = 1..RANGE (index 0 -> gap 1). Returns (pw, ka_end, branch)."""
    n_in = hist.sum(); n = n_in + n_oob
    if n >= MIN_ITS and n_oob / n > OOB_FRAC and len(recent_oob):
        pred = float(np.median(recent_oob[-5:]))
        return int(np.floor(0.85 * pred)), int(np.ceil(1.15 * pred)), 2
    if n_in >= MIN_ITS:
        cv = hist.std() / max(hist.mean(), 1e-9)
        if cv > CV_MIN:
            c = np.cumsum(hist) / n_in
            head = int(np.searchsorted(c, HEAD_Q / 100)) + 1
            tail = int(np.searchsorted(c, TAIL_Q / 100)) + 1
            return int(np.floor(head * (1 - MARGIN))), int(np.ceil(tail * (1 + MARGIN))), 1
    return 0, fallback_ka, 0


def hh_warm(ev, f, t, fallback_ka=RANGE):
    """ev: (F, T) 0/1 event matrix. f, t: query rows. Returns bool warm decisions and branch used per row."""
    warm = np.zeros(len(f), bool); branch = np.full(len(f), -1, np.int8)
    order = np.lexsort((t, f)); fs, ts = f[order], t[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(fs)) + 1, len(fs)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        fi = fs[a]; tq = ts[a:b]
        E = np.flatnonzero(ev[fi] > 0)
        if len(E) == 0:
            continue
        gaps = np.diff(E)                          # gaps[i] ends at event E[i+1]
        # parameters valid during idle period after event E[i] use gaps[:i]
        k = np.searchsorted(E, tq, side="left") - 1   # index of last event strictly before t
        has = k >= 0
        uk = np.unique(k[has])
        P = {}
        hist = np.zeros(RANGE, np.int64); n_oob = 0; oob = []; done = 0
        for i in uk:                               # incremental, causal
            for gg in gaps[done:i]:
                if gg <= RANGE: hist[gg - 1] += 1
                else: n_oob += 1; oob.append(gg)
            done = i
            P[i] = _params(hist, n_oob, np.asarray(oob), fallback_ka)
        res_w = np.zeros(b - a, bool); res_b = np.full(b - a, -1, np.int8)
        for j in np.flatnonzero(has):
            pw, ka, br = P[k[j]]; g = tq[j] - E[k[j]]
            res_w[j] = pw <= g <= ka; res_b[j] = br
        warm[order[a:b]] = res_w; branch[order[a:b]] = res_b
    return warm, branch
