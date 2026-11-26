"""Quality-aware reconstruction of short gaps in minute stock series.

Only gaps with at most MAX_RECONSTRUCT_MINUTES missing observations are
reconstructed. Longer gaps split the series into separate contiguous segments.
Synthetic values are explicitly marked and must never be used as realized
backtest labels or represented as provider observations.
"""
from __future__ import annotations

import numpy as np

MAX_RECONSTRUCT_MINUTES = 5


def reconstruct_short_gaps(times, prices, *, max_gap=MAX_RECONSTRUCT_MINUTES):
    """Return a minute-contiguous series for short gaps plus provenance flags.

    Parameters
    ----------
    times:
        Strictly increasing UTC minute indices (Unix timestamp // 60).
    prices:
        Positive finite observed prices corresponding to times.
    max_gap:
        Maximum number of missing minutes eligible for reconstruction.

    Returns
    -------
    (times, prices, is_synthetic, methods)
        Short gaps are linearly reconstructed between genuine endpoints.
        Gaps larger than max_gap remain unfilled. methods maps the first
        synthetic minute index to linear_average (one/two missing minutes)
        or local_slope_bridge (three to five by default).
    """
    t = np.asarray(times, dtype=np.int64)
    p = np.asarray(prices, dtype=np.float64)
    if (t.ndim != 1 or p.ndim != 1 or len(t) != len(p) or not len(t)
            or np.any(np.diff(t) <= 0) or np.any(~np.isfinite(p))
            or np.any(p <= 0)):
        raise ValueError("Expected matching, strictly increasing timestamps and positive finite prices")
    if not isinstance(max_gap, int) or not 0 <= max_gap <= 60:
        raise ValueError("max_gap must be an integer between 0 and 60")

    out_t = [int(t[0])]
    out_p = [float(p[0])]
    synthetic = [False]
    methods = {}

    for i in range(1, len(t)):
        left_t, right_t = int(t[i - 1]), int(t[i])
        missing = right_t - left_t - 1
        if 0 < missing <= max_gap:
            slope = (float(p[i]) - float(p[i - 1])) / (missing + 1)
            method = "linear_average" if missing <= 2 else "local_slope_bridge"
            methods[left_t + 1] = method
            for step in range(1, missing + 1):
                out_t.append(left_t + step)
                out_p.append(float(p[i - 1]) + slope * step)
                synthetic.append(True)
        out_t.append(right_t)
        out_p.append(float(p[i]))
        synthetic.append(False)

    return (np.asarray(out_t, dtype=np.int64),
            np.asarray(out_p, dtype=np.float64),
            np.asarray(synthetic, dtype=bool),
            methods)


def contiguous_segments(times, prices, is_synthetic):
    """Split at unresolved gaps; return independent arrays for each segment."""
    t = np.asarray(times, dtype=np.int64)
    p = np.asarray(prices, dtype=np.float64)
    s = np.asarray(is_synthetic, dtype=bool)
    if len(t) != len(p) or len(t) != len(s):
        raise ValueError("times, prices and provenance mask must have equal lengths")
    if len(t) == 0:
        return []
    if np.any(np.diff(t) <= 0) or np.any(~np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("Invalid cleaned series")
    cuts = np.flatnonzero(np.diff(t) > 1) + 1
    bounds = np.r_[0, cuts, len(t)]
    return [(t[a:b], p[a:b], s[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
