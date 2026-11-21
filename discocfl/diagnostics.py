"""Identifiability and calibration diagnostics for DisCo similarity objects.

All functions take a `SimilarityResult` (see discocfl.disco) or plain arrays and
use only numpy. Conventions shared by the functions:

* W[c, i, j] is the evidence weight of class c for the pair (i, j); rho[c, i, j]
  its disattenuated similarity. Entries with W = 0 carry no evidence and are
  ignored (np.nan_to_num is applied to rho only where W > 0).
* The bottleneck linkage (num + lam) / (den + lam) shrinks towards 1 ("no
  evidence of a difference") in proportion to the missing evidence, as in
  discocfl.disco._linkage.
"""
from __future__ import annotations

import numpy as np


def _masked_rho(sim):
    """rho with NaN replaced by 0 where it is not defined (W = 0 there anyway)."""
    return np.where(sim.W > 0, np.nan_to_num(sim.rho), 0.0)


def _pooled_bottleneck(W, rho, lam):
    """min over classes with positive pooled weight of (sum W rho + lam)/(sum W + lam).
    W, rho: (C, ...) already pooled over the pair set; returns NaN when no class has weight."""
    num = (W * rho).sum(axis=(1, 2)) if W.ndim == 3 else (W * rho).sum(axis=1)
    den = W.sum(axis=(1, 2)) if W.ndim == 3 else W.sum(axis=1)
    ok = den > 0
    if not ok.any():
        return float("nan")
    return float(((num[ok] + lam) / (den[ok] + lam)).min())


def group_linkage(sim, true_labels, lam=0.5):
    """Pooled bottleneck linkage between ground-truth groups.

    For groups a <= b, the pair set is {(i, j): i in a, j in b, i != j}; the linkage
    is min over classes c with positive pooled weight of
        (sum_pairs W[c,i,j] rho[c,i,j] + lam) / (sum_pairs W[c,i,j] + lam).
    Returns a dict with
        between   : {"a-b": linkage} for a < b,
        within    : {a: linkage of the pairs inside group a},
        delta_hat : 1 - max_{a<b} between  (estimated separation of the closest pair
                    of groups; NaN if fewer than two groups or no defined linkage),
        within_min: min_a within[a] (NaN if undefined).
    """
    labels = np.asarray(true_labels)
    groups = sorted(np.unique(labels).tolist())
    rho = _masked_rho(sim)
    W = sim.W
    idx = {g: np.where(labels == g)[0] for g in groups}
    between, within = {}, {}
    for x, a in enumerate(groups):
        for b in groups[x:]:
            ia, ib = idx[a], idx[b]
            Wab = W[:, ia][:, :, ib]
            rab = rho[:, ia][:, :, ib]
            if a == b:
                # exclude i == j (the diagonal of the submatrix)
                off = ~np.eye(len(ia), dtype=bool)
                Wab = Wab * off
                rab = rab * off
            # pool over the pair set: keep the class axis, sum over i and j
            value = _pooled_bottleneck(Wab, rab, lam)
            if a == b:
                within[int(a)] = value
            else:
                between[f"{int(a)}-{int(b)}"] = value
    finite_between = [v for v in between.values() if np.isfinite(v)]
    finite_within = [v for v in within.values() if np.isfinite(v)]
    delta_hat = float(1.0 - max(finite_between)) if finite_between else float("nan")
    within_min = float(min(finite_within)) if finite_within else float("nan")
    return {"between": between, "within": within, "delta_hat": delta_hat, "within_min": within_min}


def pairwise_bottleneck(sim, lam=0.5):
    """Matrix S_b[i, j] = min over classes with W[c, i, j] > 0 of
    (W rho + lam) / (W + lam) for the single pair (i, j).
    NaN when the pair shares no evidence; the diagonal is NaN."""
    rho = _masked_rho(sim)
    W = sim.W
    vals = np.where(W > 0, (W * rho + lam) / (W + lam), np.inf)
    S = vals.min(axis=0)
    S = np.where(np.isfinite(S), S, np.nan)
    np.fill_diagonal(S, np.nan)
    return S


def otsu_threshold(values, lo=0.0, hi=1.0, bins=100):
    """Otsu's threshold on the finite values clipped to [lo, hi].

    Maximises the between-class variance of the histogram over `bins` equal bins.
    Ties (a plateau of equally good thresholds, e.g. an empty gap between two
    modes) are resolved to the midpoint of the plateau. Returns (tau_hat, eta):
    eta = between-class variance / total variance of the values at tau_hat, with
    the two classes {v < tau} and {v >= tau}; eta = 0 when there are fewer than
    two distinct values. tau_hat is NaN for an empty input.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = np.clip(v[np.isfinite(v)], lo, hi)
    if v.size == 0:
        return float("nan"), 0.0
    if np.unique(v).size < 2:
        return float(v[0]), 0.0
    edges = np.linspace(lo, hi, bins + 1)
    hist, _ = np.histogram(v, bins=edges)
    p = hist / hist.sum()
    centres = 0.5 * (edges[:-1] + edges[1:])
    # candidate split after bin t-1 (t = 1 .. bins-1): classes bins[:t] and bins[t:]
    cum_p = np.cumsum(p)
    cum_m = np.cumsum(p * centres)
    total_m = cum_m[-1]
    best_score, best_ts = -np.inf, []
    for t in range(1, bins):
        w0, w1 = cum_p[t - 1], 1.0 - cum_p[t - 1]
        if w0 <= 0 or w1 <= 0:
            continue
        m0 = cum_m[t - 1] / w0
        m1 = (total_m - cum_m[t - 1]) / w1
        score = w0 * w1 * (m0 - m1) ** 2
        if score > best_score + 1e-15:
            best_score, best_ts = score, [t]
        elif abs(score - best_score) <= 1e-15:
            best_ts.append(t)
    if not best_ts:
        return float(lo), 0.0
    # midpoint of the plateau of optimal split positions (edges[t] is the split)
    tau = 0.5 * (edges[min(best_ts)] + edges[max(best_ts)])
    below = v < tau
    var_total = v.var()
    if var_total <= 0 or below.all() or (~below).all():
        return float(tau), 0.0
    w0, w1 = below.mean(), 1.0 - below.mean()
    m0, m1 = v[below].mean(), v[~below].mean()
    eta = float(w0 * w1 * (m0 - m1) ** 2 / var_total)
    return float(tau), eta


def coverage(sim, disc_classes):
    """Fraction of conflicting pairs for which the evidence is not empty on the
    discriminative classes: sum_{c in disc ∩ classes} W[c, i, j] > 0.

    disc_classes: {(i, j): iterable of discriminative class labels} for the pairs
    that must be distinguished. Returns NaN when there are no pairs."""
    ci = {int(c): k for k, c in enumerate(sim.classes)}
    if not disc_classes:
        return float("nan")
    hit = 0
    for (i, j), classes in disc_classes.items():
        ks = [ci[int(c)] for c in classes if int(c) in ci]
        if ks and float(sim.W[ks, i, j].sum()) > 0:
            hit += 1
    return float(hit / len(disc_classes))
