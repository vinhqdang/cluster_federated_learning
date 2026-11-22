"""Tests of the label-free stable warm-up (variant DisCo-AS)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from discocfl.algorithms import _stable_warmup  # noqa: E402
from discocfl.fl import Trainer  # noqa: E402

from test_regression_r1 import _toy_federation  # noqa: E402

CHECKPOINTS = (10, 15, 20, 25, 30)


def _no_sigs(w):
    return None


def test_stable_warmup_returns_a_checkpoint_from_the_set():
    fed = _toy_federation(n_clients=3, n=32)
    tr = Trainer(fed, seed=0)
    w, t_used, record = _stable_warmup(tr, fed, tr.get(), _no_sigs,
                                       lambda s: np.array([0, 0, 1]),
                                       checkpoints=(2, 3, 4), loss_frac=1.0)
    assert t_used in (2, 3, 4)
    assert record[-1]["t"] == t_used
    assert [r["t"] for r in record] == [2, 3, 4][:len(record)]


def test_stable_warmup_default_checkpoints_stop_only_after_two_learned():
    fed = _toy_federation(n_clients=3, n=32)
    tr = Trainer(fed, seed=1)
    rng = np.random.default_rng(0)

    def cluster_fn(sigs):
        return rng.integers(0, 2, 3)  # unstable partitions: ARI rarely >= 0.9

    w, t_used, record = _stable_warmup(tr, fed, tr.get(), _no_sigs, cluster_fn,
                                       loss_frac=1.0)
    assert t_used in CHECKPOINTS
    assert record[-1]["t"] == t_used
    assert [r["t"] for r in record] == list(CHECKPOINTS[:len(record)])
    # no stop before two consecutive learned checkpoints with agreeing partitions
    for r in record[:-1]:
        assert not (r["learned"] and r["ari_prev"] is not None and r["ari_prev"] >= 0.9)
    if t_used != CHECKPOINTS[-1]:
        assert record[-1]["learned"] and record[-1]["ari_prev"] >= 0.9


def test_stable_warmup_stops_at_second_checkpoint_when_always_learned_and_stable():
    fed = _toy_federation(n_clients=3, n=32)
    tr = Trainer(fed, seed=2)
    w, t_used, record = _stable_warmup(tr, fed, tr.get(), _no_sigs,
                                       lambda s: np.zeros(3, int),
                                       checkpoints=(2, 4), loss_frac=10)
    assert t_used == 4
    assert [r["t"] for r in record] == [2, 4]
    assert record[1]["ari_prev"] == 1.0
    assert all(r["learned"] for r in record)
