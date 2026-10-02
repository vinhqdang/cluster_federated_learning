"""Realistic benchmarks (GPU): PACS, Office-Home, CIFAR-100, FEMNIST.

usage: python experiments/run_real.py --suite real --device cuda
Results: results/raw_real/{scenario}__{method}__s{seed}.json (resumable).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from run import VARIANTS, metrics  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "raw_real")

# size: image side; arch: l4 = pretrained ResNet-18 head, smallcnn = small CNN
SCENARIOS = {
    "pacs": dict(kind="domain", name="pacs", clients_per_domain=10, mean_size=200, alpha=0.3),
    "pacs_qs": dict(kind="domain", name="pacs", clients_per_domain=10, mean_size=200, alpha=0.3,
                    quantity_sigma=1.0),
    "officehome": dict(kind="domain", name="office-home", clients_per_domain=10, mean_size=300,
                       alpha=0.3),
    "c100_rot": dict(kind="c100", groups="rotation", clients_per_group=10, mean_size=300, alpha=0.5),
    "c100_swap": dict(kind="c100", groups="bigswap4", clients_per_group=10, mean_size=300, alpha=0.5),
    "femnist": dict(kind="femnist", n_writers=100),
}
TRAIN = dict(rounds=50, warmup=10, lr=0.01, batch_size=64, local_epochs=1)
FEMNIST_TRAIN = dict(rounds=50, warmup=10, lr=0.05, batch_size=32, local_epochs=1)

METHODS = ["FedAvg", "Local", "FedAvg-FT", "IFCA", "FeSEM", "MTCFL", "FL+HC", "PACFL", "DisCo", "Oracle"]
SUITES = {
    "real": dict(scenarios=["pacs", "officehome", "c100_swap", "c100_rot", "pacs_qs"],
                 methods=METHODS, seeds=[0, 1, 2]),
    "femnist": dict(scenarios=["femnist"], methods=["FedAvg", "Local", "FedAvg-FT", "IFCA", "FeSEM",
                                                      "MTCFL", "FL+HC", "DisCo"], seeds=[0, 1, 2]),
    "variants": dict(scenarios=["officehome", "c100_rot", "pacs"], methods=["DisCo-full", "DisCo-T30"],
                     seeds=[0, 1, 2]),
    "ablation_real": dict(scenarios=["pacs", "c100_swap"],
                          methods=["DisCo-noDisatt", "DisCo-noClass", "DisCo-mean", "DisCo-linkorder",
                                   "DisCo-noClip"], seeds=[0, 1, 2]),
}
FEMNIST_K = [1, 3, 5]      # IFCA / FeSEM / FL+HC have no true K on natural writers


def build(scen, seed, device, pretrained, size):
    from discocfl import real_data as R
    c = dict(SCENARIOS[scen])
    kind = c.pop("kind")
    if kind == "domain":
        name = c.pop("name")
        return R.make_domain_federation(name, seed=seed, size=size, device=device,
                                        pretrained=pretrained, **c), "l4", TRAIN
    if kind == "c100":
        return R.make_cifar100_federation(device=device, pretrained=pretrained, seed=seed, **c), "l4", TRAIN
    return R.make_femnist_federation(seed=seed, device=device, **c), "smallcnn", FEMNIST_TRAIN


def run_one(scen, method, seed, device, pretrained=True, size=112, out_dir=RESULTS, extra=None):
    from discocfl.algorithms import ALGORITHMS
    from discocfl.fl import Trainer
    path = os.path.join(out_dir, f"{scen}__{method}__s{seed}.json")
    if os.path.exists(path):
        return None
    fed, arch, train = build(scen, seed, device, pretrained, size)
    K_true = len(fed.groups)
    tr = Trainer(fed, lr=train["lr"], batch_size=train["batch_size"],
                 local_epochs=train["local_epochs"], seed=seed, arch=arch, device=device,
                 pretrained=pretrained)
    tr.participation = 1.0
    cfg = dict(train, seed=seed, K=K_true)
    if scen == "femnist":
        cfg["K"] = 4          # natural writers have no true K; baselines get the same K as elsewhere
    base = method
    if method in VARIANTS:
        cfg.update(VARIANTS[method])
        base = "DisCo"
    cfg.update(extra or {})
    t0 = time.time()
    out = ALGORITHMS[base](fed, tr, cfg)
    elapsed = time.time() - t0
    res = metrics(fed, tr, out)
    res.update({"scenario": scen, "method": method, "seed": seed, "time_s": elapsed,
                "model_dim": tr.dim, "n_clients": len(fed.clients), "k_true": K_true,
                "cluster_upload_floats": int(out.get("cluster_floats", 0)),
                "assign": np.asarray(out["assign"]).tolist(), "true": fed.true_labels.tolist(),
                "extra": {k: v for k, v in out.items() if k in
                          ("n_confident", "flags", "time_signature_s", "time_server_s")},
                "config": cfg})
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[done] {scen} {method} s{seed} acc={res['acc_mean']:.3f} concept={res['acc_concept_mean']:.3f} "
          f"ari={res['ari']:.2f} K={res['k_found']} ({elapsed:.0f}s)", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="real")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default=None)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    a = ap.parse_args()
    s = SUITES[a.suite]
    jobs = [(sc, m, sd) for sc, sd, m in itertools.product(s["scenarios"], s["seeds"], s["methods"])]
    if a.only:
        jobs = [j for j in jobs if j[0] in a.only.split(",")]
    if a.methods:
        jobs = [j for j in jobs if j[1] in a.methods.split(",")]
    if a.seeds:
        jobs = [j for j in jobs if str(j[2]) in a.seeds.split(",")]
    i, n = map(int, a.shard.split("/"))
    jobs = jobs[i::n]
    print(f"{len(jobs)} jobs on {a.device}", flush=True)
    for scen, method, seed in jobs:
        try:
            run_one(scen, method, seed, a.device, not a.no_pretrained, a.size, a.out)
        except Exception as e:  # keep going; failures are listed at the end
            import traceback
            traceback.print_exc()
            print(f"[FAILED] {scen} {method} s{seed}: {e}", flush=True)
