"""Pilot runs used to (re-)select hyperparameters after correcting the training
primitive: seed 0 of the four pilot configurations only (rot, swap, mixed_qs,
label). Clustering quality is measured against the ground truth, exactly as in the
original pilot; no other seed or dataset is used for these choices.

DisCo-CFL: warm-up length T0 in {10, 20, 30} and threshold tau in a grid.
StoCFL: merging threshold tau (its clustering happens at the anchor model).
Also records the median split-half reliability of the signatures at every T0,
the quantity an adaptive warm-up rule would monitor.
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "..", "results", "r1", "pilot")
PILOT = ["rot", "swap", "mixed_qs", "label"]
T0S = [10, 20, 30]
TAUS = [0.7, 0.75, 0.8, 0.85, 0.9]
STO_TAUS = [0.1, 0.3, 0.5, 0.7, 0.9]


def study(scen):
    torch.set_num_threads(1)
    from sklearn.metrics import adjusted_rand_score
    import run as R
    from discocfl.algorithms import _fedavg_rounds
    from discocfl.data import make_federation
    from discocfl.diagnostics import group_linkage
    from discocfl.disco import CountSketch, SignatureConfig, client_signature, disco_cluster
    from discocfl.fl import Trainer

    fcfg = dict(R.BASE, **R.SCENARIOS[scen])
    fcfg.pop("_train", None)
    fed = make_federation(seed=0, **fcfg)
    tr = Trainer(fed, lr=R.TRAIN["lr"], batch_size=R.TRAIN["batch_size"],
                 local_epochs=R.TRAIN["local_epochs"], seed=0)
    conf = R.conflict_matrix(fed)
    g = fed.true_labels

    def score(labels):
        A = np.asarray(labels)
        same = np.equal.outer(A, A) & ~np.eye(len(A), dtype=bool)
        ari = float(adjusted_rand_score(g, A)) if len(set(g)) > 1 or len(set(A)) > 1 else 1.0
        return {"ari": ari, "k": int(len(set(A))),
                "conflict": float((same & conf).sum() / max(conf.sum(), 1))}

    out = {"scenario": scen, "disco": [], "stocfl": []}
    # StoCFL: clustering of normalised gradients at the anchor (initial) model
    w0 = tr.get()
    G = torch.stack([tr.mean_grad(w0, c.x, c.y) for c in fed.clients])
    G = G / G.norm(dim=1, keepdim=True).clamp_min(1e-12)
    for tau in STO_TAUS:
        reps = {i: G[i].clone() for i in range(len(fed.clients))}
        mem = {i: [i] for i in reps}
        while len(reps) > 1:
            keys = list(reps)
            M = torch.stack([reps[k] / reps[k].norm() for k in keys])
            S = (M @ M.T).numpy()
            np.fill_diagonal(S, -np.inf)
            a, b = np.unravel_index(np.argmax(S), S.shape)
            if S[a, b] <= tau:
                break
            ka, kb = keys[a], keys[b]
            reps[ka] = reps[ka] + reps.pop(kb)
            mem[ka] += mem.pop(kb)
        lab = np.zeros(len(fed.clients), int)
        for k, (_, m) in enumerate(mem.items()):
            lab[m] = k
        out["stocfl"].append(dict(tau=tau, **score(lab)))
    # DisCo-CFL: signatures after T0 warm-up rounds
    w = w0
    done = 0
    sketch = CountSketch(tr.dim, 1024, seed=0)
    for T0 in T0S:
        w = _fedavg_rounds(tr, fed, w, T0 - done)
        done = T0
        sigs = [client_signature(tr, w, c, sketch, SignatureConfig(seed=0), np.random.default_rng(7))
                for c in fed.clients]
        Rv = []
        for s in sigs:
            for c, (a, b, n) in s.items():
                r = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
                Rv.append(2 * r / (1 + r) if r > 0 else 0.0)
        acc = float(np.mean([tr.evaluate(w, c.x_test, c.y_test)[0] for c in fed.clients]))
        for tau in TAUS:
            res = disco_cluster(sigs, tau=tau, lam=0.5)
            gl = group_linkage(res.sim, g, lam=0.5)
            out["disco"].append(dict(T0=T0, tau=tau, median_R=float(np.median(Rv)),
                                     frac_R_pos=float(np.mean(np.array(Rv) > 0)), ref_acc=acc,
                                     delta_hat=gl["delta_hat"] if np.isfinite(gl["delta_hat"]) else None,
                                     **score(res.labels)))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f"pilot_{scen}.json"), "w"), indent=1)
    return scen


if __name__ == "__main__":
    with Pool(4) as p:
        for s in p.imap_unordered(study, PILOT):
            print("done", s, flush=True)
