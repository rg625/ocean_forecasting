"""Confidence intervals that respect how the DA problems were actually drawn.

The campaign draws each problem as a (trajectory, analysis time) pair.  Drawing 48
problems does not give 48 independent cases: `gt_interp` holds 6 trajectories and
`gt_longer` holds 4, so many problems share a trajectory, and two problems from the same
trajectory whose 25-frame observation windows overlap are strongly dependent.  Treating
them as independent produces a t-interval that is roughly sqrt(n / n_trajectories) times
too narrow -- a factor of ~2.8 at n = 48 over 6 trajectories.

So every interval here is reported twice:

  ci95_naive     the usual t-interval over all problems.  Quoted only for comparison; it
                 is the number an unclustered analysis would give, and it is too tight.
  ci95_cluster   the t-interval over PER-TRAJECTORY means, with df = n_trajectories - 1.
                 This is the defensible one: it treats the trajectory as the unit of
                 replication, which is what it is.  With 4-6 trajectories the critical
                 value is large (t_.975 = 3.18 at df 3, 2.57 at df 5), and that is the
                 honest cost of a small trajectory pool.

`ci95_cluster` is what the deck reports.  Widening the pool needs more trajectories, not
more draws from the ones that exist.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import stats


def summarise(
    values: Sequence[float], clusters: Optional[Sequence] = None, alpha: float = 0.05
) -> dict:
    """Mean with naive and cluster-robust confidence intervals.

    ``clusters`` labels which trajectory each value came from.  Non-finite values are
    dropped and counted, so a diverged solve reduces n rather than poisoning the mean.
    """
    v = np.asarray(values, dtype=float).ravel()
    ok = np.isfinite(v)
    n_bad = int((~ok).sum())
    v = v[ok]
    out: dict = {"n": int(v.size), "n_dropped_nonfinite": n_bad}
    if v.size == 0:
        return {
            **out,
            "mean": None,
            "sem": None,
            "ci95_naive": None,
            "ci95_cluster": None,
            "n_clusters": 0,
        }
    mean = float(v.mean())
    out["mean"] = mean
    if v.size > 1:
        sem = float(v.std(ddof=1) / np.sqrt(v.size))
        t = float(stats.t.ppf(1 - alpha / 2, v.size - 1))
        out["sem"] = sem
        out["ci95_naive"] = [mean - t * sem, mean + t * sem]
        out["ci95_naive_halfwidth"] = t * sem
    else:
        out["sem"] = None
        out["ci95_naive"] = None

    if clusters is None:
        out["n_clusters"] = None
        out["ci95_cluster"] = None
        return out

    c = np.asarray(clusters).ravel()[ok]
    uniq = np.unique(c)
    out["n_clusters"] = int(uniq.size)
    out["cluster_sizes"] = [int((c == u).sum()) for u in uniq]
    if uniq.size < 2:
        out["ci95_cluster"] = None
        out["cluster_note"] = "only one trajectory; no between-trajectory variance"
        return out
    # the trajectory is the unit of replication: average within, then interval across
    cm = np.array([v[c == u].mean() for u in uniq], dtype=float)
    cmean = float(cm.mean())
    csem = float(cm.std(ddof=1) / np.sqrt(cm.size))
    tc = float(stats.t.ppf(1 - alpha / 2, cm.size - 1))
    out["cluster_mean"] = cmean
    out["cluster_sem"] = csem
    out["ci95_cluster"] = [cmean - tc * csem, cmean + tc * csem]
    out["ci95_cluster_halfwidth"] = tc * csem
    out["t_crit_cluster"] = tc
    if out.get("ci95_naive_halfwidth"):
        out["cluster_inflation"] = (
            out["ci95_cluster_halfwidth"] / out["ci95_naive_halfwidth"]
        )
    return out


def fmt(s: dict, digits: int = 4) -> str:
    """`mean [lo, hi]` using the CLUSTER interval, which is the one the deck reports."""
    if s.get("mean") is None:
        return "n/a"
    ci = s.get("ci95_cluster") or s.get("ci95_naive")
    if ci is None:
        return f"{s['mean']:.{digits}f}"
    return f"{s['mean']:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"
