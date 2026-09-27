"""Synthetic validation of the theoretical results (results/theory/*.json).

1. estimation error of the disattenuated similarity vs. half size h (consistency,
   removal of the attenuation bias of the raw cosine);
2. probability of the recovery guarantees (safety, maximality, exact recovery) vs. h
   for several separations Delta: the h needed for 90% recovery is compared with
   the Delta^-2 (upper bound) and Delta^-1 (lower bound) rates;
3. group-size independence: recovery of a minority group of 1..8 clients.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from discocfl.disco import compute_similarity, disco_cluster  # noqa: E402
from discocfl.synthetic import concept_vectors, synthetic_signatures  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "theory")
P, R_EFF, SIGMA, C = 512, 16, 2.0, 10


def basis(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((P, R_EFF)))
    return Q


def estimation(trials=40):
    rows = []
    for h in [2, 4, 8, 16, 32, 64, 128, 256]:
        for truth in [1.0, 0.5]:
            err_c, err_r = [], []
            for t in range(trials):
                rng = np.random.default_rng(1000 * h + t)
                mu = concept_vectors(2, 1, P, 1 - truth, {1: [0]}, rng)
                sigs = synthetic_signatures([0, 1], [[0], [0]], [h, 4 * h], mu, SIGMA, rng, basis(rng))
                true = float(mu[0, 0] @ mu[1, 0])
                for dis, acc in ((True, err_c), (False, err_r)):
                    s = compute_similarity(sigs, disattenuate=dis).S[0, 1]
                    acc.append(s - true)
            rows.append(dict(h=h, truth=truth,
                             bias_disatt=float(np.mean(err_c)), rmse_disatt=float(np.sqrt(np.mean(np.square(err_c)))),
                             bias_raw=float(np.mean(err_r)), rmse_raw=float(np.sqrt(np.mean(np.square(err_r))))))
    return rows


def _population(rng, K=4, per=8, sizes=None, full_support=False):
    """full_support: every client holds all discriminative classes (then the
    theorem gives exact recovery); otherwise 6 random classes (+ own ones)."""
    differ = {1: [0, 1], 2: [4, 5], 3: [7]}
    groups = np.concatenate([[k] * (sizes[k] if sizes else per) for k in range(K)])
    allc = {c for v in differ.values() for c in v}
    cs = []
    for g in groups:
        s = set(rng.choice(C, 6, replace=False).tolist())
        if full_support:
            s |= allc
        elif g > 0:
            s |= set(differ[g])
        cs.append(sorted(s))
    return differ, groups, cs


def _conflicts(groups, cs, differ):
    n = len(groups)
    M = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(n):
            if groups[i] != groups[j]:
                d = set(differ.get(groups[i], [])) | set(differ.get(groups[j], []))
                M[i, j] = bool(d & set(cs[i]) & set(cs[j]))
    return M


def _guarantees(groups, cs, differ, lab):
    """(safe, maximal, exact): no conflicting pair merged; every pair of output
    clusters contains a conflicting pair; partition equals the ground truth."""
    M = _conflicts(groups, cs, differ)
    same = np.equal.outer(lab, lab)
    safe = not (same & M).any()
    ks = np.unique(lab)
    maximal = all(M[np.ix_(lab == a, lab == b)].any() for i, a in enumerate(ks) for b in ks[i + 1:])
    from sklearn.metrics import adjusted_rand_score
    return safe, safe and maximal, adjusted_rand_score(groups, lab) == 1.0


def recovery(trials=30):
    rows = []
    for full in (False, True):
        for delta in [0.2, 0.35, 0.5, 0.7]:
            for h in [1, 2, 4, 8, 16, 32, 64, 128]:
                acc = np.zeros(3)
                for t in range(trials):
                    rng = np.random.default_rng(7919 * h + t + int(delta * 1000))
                    differ, groups, cs = _population(rng, full_support=full)
                    mu = concept_vectors(4, C, P, delta, differ, rng)
                    sigs = synthetic_signatures(groups, cs, h, mu, SIGMA, rng, basis(rng))
                    acc += _guarantees(groups, cs, differ, disco_cluster(sigs, tau=1 - delta / 2).labels)
                rows.append(dict(full_support=full, delta=delta, h=h, p_safe=acc[0] / trials,
                                 p_safe_maximal=acc[1] / trials, p_exact=acc[2] / trials))
    return rows


def minority(trials=30, delta=0.5):
    rows = []
    for m in [1, 2, 4, 8]:
        for h in [2, 4, 8, 16, 32, 64]:
            acc = np.zeros(3)
            for t in range(trials):
                rng = np.random.default_rng(31 * h + t + 1000 * m)
                differ, groups, cs = _population(rng, sizes=[8, 8, 8, m], full_support=True)
                mu = concept_vectors(4, C, P, delta, differ, rng)
                sigs = synthetic_signatures(groups, cs, h, mu, SIGMA, rng, basis(rng))
                acc += _guarantees(groups, cs, differ, disco_cluster(sigs, tau=1 - delta / 2).labels)
            rows.append(dict(minority_size=m, h=h, p_safe=acc[0] / trials,
                             p_safe_maximal=acc[1] / trials, p_exact=acc[2] / trials))
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("estimation", estimation), ("recovery", recovery), ("minority", minority)]:
        rows = fn()
        json.dump(rows, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
        print(name, len(rows))
        for r in rows[:: max(1, len(rows) // 8)]:
            print("  ", r)
