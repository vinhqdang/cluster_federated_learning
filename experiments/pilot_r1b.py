"""Pilot (seed 0 of rot, swap, mixed_qs, label) for the baseline thresholds:
FedCCFA DBSCAN radius eps (full runs) and StoCFL merging threshold tau with the
gradient taken at the warm-up model (clustering only)."""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
os.environ.setdefault("DISCO_RESULTS", os.path.join(HERE, "..", "results", "r1", "pilot", "raw"))
import run as R  # noqa: E402

PILOT = ["rot", "swap", "mixed_qs", "label"]
EPS = [0.02, 0.05, 0.1, 0.2, 0.4]
STO_TAUS = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]


def sto_warm(scen):
    import torch
    torch.set_num_threads(1)
    from sklearn.metrics import adjusted_rand_score
    from discocfl.algorithms import _fedavg_rounds
    from discocfl.baselines_extra import _stocfl_cluster
    from discocfl.data import make_federation
    from discocfl.fl import Trainer
    fcfg = dict(R.BASE, **R.SCENARIOS[scen]); fcfg.pop("_train", None)
    fed = make_federation(seed=0, **fcfg)
    tr = Trainer(fed, lr=R.TRAIN["lr"], batch_size=R.TRAIN["batch_size"],
                 local_epochs=R.TRAIN["local_epochs"], seed=0)
    w = _fedavg_rounds(tr, fed, tr.get(), R.TRAIN["warmup"])
    G = []
    for c in fed.clients:
        g = tr.mean_grad(w, c.x, c.y)
        G.append(g / g.norm().clamp_min(1e-12))
    out = []
    for tau in STO_TAUS:
        lab = _stocfl_cluster(G, tau)
        out.append(dict(scenario=scen, tau=tau, k=int(lab.max()) + 1,
                        ari=float(adjusted_rand_score(fed.true_labels, lab))))
    return out


def job(j):
    if j[0] == "sto":
        return ("sto", sto_warm(j[1]))
    scen, eps = j[1], j[2]
    return ("ccfa", R.run_one((scen, "FedCCFA", 0, {"fedccfa_eps": eps, "_tag": f"-eps{eps}"})))


if __name__ == "__main__":
    jobs = [("sto", s) for s in PILOT] + [("ccfa", s, e) for e in EPS for s in PILOT]
    sto = []
    with Pool(int(os.environ.get("W", "4"))) as p:
        for kind, r in p.imap_unordered(job, jobs):
            if kind == "sto":
                sto += r
                print(r, flush=True)
    out = os.path.join(HERE, "..", "results", "r1", "pilot", "stocfl_warm.json")
    json.dump(sto, open(out, "w"), indent=1)
