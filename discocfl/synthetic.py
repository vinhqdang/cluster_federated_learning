"""Synthetic class-conditional signatures with known ground truth.

Used by the unit tests and the theory-validation experiment: every client of
group k holds, for each class c it owns, noisy half-means
    a = mu_k^c + xi_a,  b = mu_k^c + xi_b,  xi ~ N(0, sigma^2 / (p h) I_p)
so the per-sample noise has total variance sigma^2 and the signal ||mu||^2 = 1.
Groups differ from group 0 on a chosen set of classes, where the population
cosine is 1 - Delta.
"""
from __future__ import annotations

import numpy as np
import torch


def concept_vectors(K, C, p, delta, differ, rng):
    """mu[k, c] unit vectors; groups k>0 differ from group 0 on classes differ[k]
    with cos(mu_0^c, mu_k^c) = 1 - delta, and different groups are orthogonal-ish
    perturbations of each other."""
    base = rng.standard_normal((C, p))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    mu = np.repeat(base[None], K, axis=0)
    for k in range(1, K):
        for c in differ[k]:
            z = rng.standard_normal(p)
            z -= z @ base[c] * base[c]
            z /= np.linalg.norm(z)
            cosv = 1 - delta
            mu[k, c] = cosv * base[c] + np.sqrt(1 - cosv ** 2) * z
    return mu


def synthetic_signatures(groups, class_sets, h, mu, sigma=1.0, rng=None, basis=None):
    """groups: group of each client; class_sets: classes held by each client;
    h: per-class half size (int or per-client list). basis: optional (p, r)
    orthonormal matrix; the per-sample noise then lives in an r-dimensional
    subspace (effective rank r) with the same total variance sigma^2."""
    rng = rng or np.random.default_rng(0)
    p = mu.shape[-1]
    sigs = []

    def noise(hi):
        if basis is None:
            return sigma / np.sqrt(p * hi) * rng.standard_normal(p)
        r = basis.shape[1]
        return basis @ (sigma / np.sqrt(r * hi) * rng.standard_normal(r))

    for i, (g, cs) in enumerate(zip(groups, class_sets)):
        hi = h[i] if np.ndim(h) else h
        s = {}
        for c in cs:
            a = mu[g, c] + noise(hi)
            b = mu[g, c] + noise(hi)
            s[int(c)] = (torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32), 2 * hi)
        sigs.append(s)
    return sigs
