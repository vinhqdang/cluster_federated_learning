"""Regression tests added in the revision: the shared-vector error of local
training, and the edge-case rules of the signatures (class cut-off m, the
reliability floor r_min, low total evidence)."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from discocfl.disco import compute_similarity, disco_cluster  # noqa: E402
from discocfl.fl import Trainer  # noqa: E402


def _toy_federation(n_clients=3, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    clients = []
    for i in range(n_clients):
        x = torch.randn(n, 1, 28, 28, generator=g)
        y = torch.randint(0, 10, (n,), generator=g)
        clients.append(SimpleNamespace(cid=i, group=0, x=x, y=y, x_test=x, y_test=y, n=n))
    return SimpleNamespace(clients=clients, groups=[None], num_classes=10, in_shape=(1, 28, 28),
                           dataset="toy", true_labels=np.zeros(n_clients, int))


def test_local_training_does_not_modify_the_input_vector():
    """A model vector passed to several clients in one round must not change
    (the error corrected in the revision trained them sequentially)."""
    fed = _toy_federation()
    tr = Trainer(fed, seed=0)
    w = tr.get().clone()
    w_copy = w.clone()
    out1 = tr.local_train(w, fed.clients[0].x, fed.clients[0].y)
    assert torch.equal(w, w_copy)
    out2 = tr.local_train(w, fed.clients[1].x, fed.clients[1].y)
    assert torch.equal(w, w_copy)
    assert not torch.equal(out1, out2)


def _sig(rng, classes, p=128, signal=1.0, noise=0.1, n=20):
    mu = {c: rng.standard_normal(p) for c in classes}
    return {c: (torch.tensor(signal * mu[c] + noise * rng.standard_normal(p), dtype=torch.float32),
                torch.tensor(signal * mu[c] + noise * rng.standard_normal(p), dtype=torch.float32), n)
            for c in classes}


def test_r_min_floor_sets_weak_reliability_to_zero():
    rng = np.random.default_rng(0)
    p = 4096
    # nearly independent halves: split-half cosine of order 1/sqrt(p) > 0 but tiny
    a = torch.tensor(rng.standard_normal(p), dtype=torch.float32)
    b = a * 0.005 + torch.tensor(rng.standard_normal(p), dtype=torch.float32)
    good = _sig(rng, [0, 1], p=p)
    sigs = [{0: (a, b, 20), 1: good[1]}, good]
    R = compute_similarity(sigs, r_min=0.2).R
    assert R[0, 0] == 0.0          # below the floor: no evidence
    assert R[1, 0] > 0.2


def test_low_total_evidence_is_flagged():
    rng = np.random.default_rng(1)
    sigs = [_sig(rng, [0, 1, 2]) for _ in range(4)]
    p = 128
    noise_only = {0: (torch.tensor(rng.standard_normal(p), dtype=torch.float32),
                      torch.tensor(rng.standard_normal(p), dtype=torch.float32), 8)}
    sigs.append(noise_only)
    res = disco_cluster(sigs)
    assert 4 in res.flags["low_evidence"]


def test_class_below_cut_off_is_not_reported():
    from discocfl.disco import CountSketch, SignatureConfig, client_signature
    fed = _toy_federation(n_clients=1, n=40)
    c = fed.clients[0]
    c.y = torch.cat([torch.zeros(37, dtype=torch.long), torch.ones(3, dtype=torch.long)])
    tr = Trainer(fed, seed=0)
    cfg = SignatureConfig(seed=0, min_per_class=4)
    sig = client_signature(tr, tr.get(), c, CountSketch(tr.dim, 64, seed=0), cfg, np.random.default_rng(0))
    assert 0 in sig and 1 not in sig
