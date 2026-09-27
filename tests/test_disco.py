import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from discocfl.disco import (CountSketch, assign_newcomer, compute_similarity,  # noqa: E402
                            disco_cluster, gaussian_dp_epsilon)
from discocfl.synthetic import concept_vectors, synthetic_signatures  # noqa: E402


def _setup(h=20, delta=0.6, K=4, per=6, C=10, p=512, seed=0, sigma=1.0):
    rng = np.random.default_rng(seed)
    differ = {1: [0, 1], 2: [4, 5], 3: list(range(C))}
    mu = concept_vectors(K, C, p, delta, differ, rng)
    groups = np.repeat(np.arange(K), per)
    class_sets = [sorted(rng.choice(C, 7, replace=False)) for _ in groups]
    # every client of groups 1/2 holds its discriminative classes
    for i, g in enumerate(groups):
        if g in (1, 2):
            class_sets[i] = sorted(set(class_sets[i]) | set(differ[g]))
    sigs = synthetic_signatures(groups, class_sets, h, mu, sigma, rng)
    return groups, sigs


def _conflicts(groups, sigs, C=10):
    differ = {0: set(), 1: {0, 1}, 2: {4, 5}, 3: set(range(C))}
    n = len(groups)
    M = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(n):
            gi, gj = groups[i], groups[j]
            if gi == gj:
                continue
            d = differ[gi] | differ[gj]
            M[i, j] = bool(d & set(sigs[i]) & set(sigs[j]))
    return M


def test_count_sketch_preserves_inner_products():
    torch.manual_seed(0)
    d, p = 5000, 2048
    x, y = torch.randn(d), torch.randn(d)
    y = 0.7 * x + 0.3 * y
    errs = []
    for s in range(20):
        S = CountSketch(d, p, seed=s)
        errs.append(float(S(x) @ S(y) - x @ y) / float(x.norm() * y.norm()))
    assert abs(np.mean(errs)) < 0.02
    assert np.std(errs) < 0.05


def test_disattenuation_is_calibrated_across_sample_sizes():
    """Same concept, very different sample sizes: corrected similarity ~ 1,
    the raw cosine is attenuated."""
    rng = np.random.default_rng(1)
    mu = concept_vectors(1, 1, 1024, 0.0, {}, rng)
    sigs = synthetic_signatures([0, 0], [[0], [0]], [3, 300], mu, sigma=2.0, rng=rng)
    corr = compute_similarity(sigs, disattenuate=True).S[0, 1]
    raw = compute_similarity(sigs, disattenuate=False).S[0, 1]
    assert corr > 0.9
    assert raw < 0.85


def test_label_skew_does_not_split_and_concept_shift_does():
    # safety (no conflicting pair merged) and K recovery, the guarantees of the
    # recovery theorem; clients lacking the discriminative classes are compatible
    # with several groups and may legitimately join any of them
    for seed in range(5):
        groups, sigs = _setup(seed=seed)
        lab = disco_cluster(sigs, tau=0.8).labels
        same = np.equal.outer(lab, lab)
        assert not (same & _conflicts(groups, sigs)).any()
        assert len(np.unique(lab)) == 4
    # single concept with arbitrary class supports -> one cluster
    rng = np.random.default_rng(3)
    mu = concept_vectors(1, 10, 512, 0.0, {}, rng)
    cs = [sorted(rng.choice(10, rng.integers(2, 10), replace=False)) for _ in range(20)]
    sigs = synthetic_signatures([0] * 20, cs, list(rng.integers(3, 100, 20)), mu, rng=rng)
    assert disco_cluster(sigs, tau=0.8).labels.max() == 0


def test_explanations_identify_divergent_classes():
    groups, sigs = _setup(h=50)
    res = disco_cluster(sigs, tau=0.8)
    lab = res.labels
    maj = {k: np.bincount(groups[lab == k]).argmax() for k in np.unique(lab)}
    inv = {v: k for k, v in maj.items()}
    pair = res.explanations["between"]
    key = f"{min(inv[0], inv[1])}-{max(inv[0], inv[1])}"
    assert pair[key]["divergent_classes"] == [0, 1]
    key3 = f"{min(inv[0], inv[3])}-{max(inv[0], inv[3])}"
    assert pair[key3]["diagnosis"].startswith("global")


def test_certificates_and_audit_log():
    groups, sigs = _setup(h=50)
    res = disco_cluster(sigs, tau=0.8)
    assert len(res.certificates) == len(groups)
    assert all(c["verdict"] in ("confident", "uncertain", "low-evidence") for c in res.certificates)
    assert sum(c["verdict"] == "confident" for c in res.certificates) >= len(groups) // 2
    assert any("merge" in e for e in res.audit_log)


def test_newcomer_assignment_and_novelty():
    groups, sigs = _setup(h=50)
    res = disco_cluster(sigs[:-1], tau=0.8)
    k, s = assign_newcomer(sigs[:-1], res.labels, sigs[-1], 0.8)
    assert k == res.labels[np.where(groups[:-1] == groups[-1])[0][0]]
    rng = np.random.default_rng(9)
    mu = concept_vectors(2, 10, 512, 0.9, {1: list(range(10))}, rng)
    novel = synthetic_signatures([1], [list(range(10))], 50, mu, rng=rng)[0]
    k, _ = assign_newcomer(sigs[:-1], res.labels, novel, 0.8)
    assert k == -1


def test_gaussian_dp_epsilon_matches_definition():
    from scipy.stats import norm
    for sigma in (0.7, 1.0, 2.0, 4.0):
        eps = gaussian_dp_epsilon(sigma, 1e-5)
        d = norm.cdf(0.5 / sigma - eps * sigma) - math.exp(eps) * norm.cdf(-0.5 / sigma - eps * sigma)
        assert d <= 1.01e-5
        assert gaussian_dp_epsilon(sigma * 2) < eps


def test_data_partition_applies_group_transforms():
    pytest.importorskip("torch")
    from discocfl.data import make_federation
    try:
        fed = make_federation("fmnist", "swap", clients_per_group=2, mean_size=60, seed=0)
    except FileNotFoundError:
        pytest.skip("dataset not available")
    assert len(fed.clients) == 8
    assert fed.groups[1].perm[0] == 1 and fed.groups[1].perm[1] == 0
    assert all(c.n >= 30 for c in fed.clients)
