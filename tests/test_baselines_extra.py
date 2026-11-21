import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from discocfl.baselines_extra import (_features, _classifier, _local_train_prox,  # noqa: E402
                                      _stocfl_cluster, run_feddag, run_fedccfa,
                                      run_stocfl)
from discocfl.data import Client, Federation, Group, swap_perm  # noqa: E402
from discocfl.fl import Trainer  # noqa: E402


def _federation(seed=0, C=10, n=40, n_test=20, n_group_test=50):
    """8 clients in 2 concept groups; group 1 swaps the labels 0 and 1."""
    g = torch.Generator().manual_seed(seed)
    groups = [Group(), Group(perm=swap_perm(C, [(0, 1)]))]
    clients = []
    for cid in range(8):
        grp = cid // 4
        x = torch.randn(n, 1, 28, 28, generator=g)
        y = torch.randint(0, C, (n,), generator=g)
        xt = torch.randn(n_test, 1, 28, 28, generator=g)
        yt = torch.randint(0, C, (n_test,), generator=g)
        x, y = groups[grp].apply(x, y)
        xt, yt = groups[grp].apply(xt, yt)
        clients.append(Client(cid, grp, x, y, xt, yt))
    group_test = []
    for grp in groups:
        x = torch.randn(n_group_test, 1, 28, 28, generator=g)
        y = torch.randint(0, C, (n_group_test,), generator=g)
        group_test.append(grp.apply(x, y))
    return Federation(clients, groups, group_test, C, (1, 28, 28), "synthetic")


@pytest.fixture
def setup():
    fed = _federation()
    tr = Trainer(fed, lr=0.05, batch_size=16, local_epochs=1, seed=0)
    return fed, tr


CFG = dict(rounds=3, warmup=1, seed=0, K=2)


def _check_output(out, fed, tr):
    N = len(fed.clients)
    assert len(out["assign"]) == N
    assert set(out["assign"].tolist()) <= set(out["models"].keys())
    for k, vec in out["models"].items():
        assert isinstance(vec, torch.Tensor)
        assert vec.numel() == tr.dim
        assert torch.isfinite(vec).all()
    assert isinstance(out["cluster_floats"], int) and out["cluster_floats"] > 0


def test_stocfl_runs(setup):
    fed, tr = setup
    out = run_stocfl(fed, tr, CFG)
    _check_output(out, fed, tr)
    assert out["cluster_floats"] == len(fed.clients) * tr.dim
    assert out["extra_local_epochs_per_round"] == 2


def test_fedccfa_runs(setup):
    fed, tr = setup
    out = run_fedccfa(fed, tr, dict(CFG, fedccfa_align_start=1))
    _check_output(out, fed, tr)
    assert len(out["partition_assign"]) == 8
    assert set(out["partition_assign"].tolist()) == set(range(out["partition_assign"].max() + 1))
    assert np.array_equal(out["assign"], np.arange(8))


def test_feddag_runs(setup):
    fed, tr = setup
    out = run_feddag(fed, tr, CFG)
    _check_output(out, fed, tr)
    assert len(np.unique(out["assign"])) == CFG["K"]
    assert set(out["models"]) == set(np.unique(out["assign"]).tolist())


def test_feature_split_reproduces_forward(setup):
    _, tr = setup
    m = tr.model
    x = torch.randn(5, 1, 28, 28)
    with torch.no_grad():
        assert torch.allclose(_classifier(m)(_features(m, x)), m(x), atol=1e-6)


def test_prox_with_zero_lambda_equals_local_train(setup):
    fed, tr = setup
    c = fed.clients[0]
    w = tr.get()
    anchor = torch.randn_like(w)
    tr.rng = np.random.default_rng(5)
    a = tr.local_train(w.clone(), c.x, c.y)
    tr.rng = np.random.default_rng(5)
    b = _local_train_prox(tr, w.clone(), c.x, c.y, anchor, lam=0.0)
    assert torch.allclose(a, b, atol=1e-6)
    tr.rng = np.random.default_rng(5)
    c_ = _local_train_prox(tr, w.clone(), c.x, c.y, anchor, lam=1.0)
    assert not torch.allclose(a, c_, atol=1e-6)


def test_stocfl_merges_only_aligned_gradients():
    v = torch.tensor([1.0, 0.0, 0.0])
    G = [v, v.clone(), -v]
    labels = _stocfl_cluster(G, tau=0.5)
    assert labels[0] == labels[1] != labels[2]
    assert _stocfl_cluster([v, -v], tau=0.5).tolist() == [0, 1]
