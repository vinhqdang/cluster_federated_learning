"""Run the benchmark grid: scenarios x methods x seeds (results/raw/*.json).

usage: python experiments/run.py --suite main --workers 4
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "raw")

BASE = dict(dataset="fmnist", clients_per_group=10, mean_size=400, alpha=0.3,
            quantity_sigma=0.0)

SCENARIOS = {
    # concept shift on features + target skew
    "rot": dict(groups="rotation"),
    # concept shift on targets (label swaps of 2 class pairs) + target skew
    "swap": dict(groups="swap"),
    # rotation + target skew + quantity skew
    "rot_qs": dict(groups="rotation", quantity_sigma=1.0),
    # 2 rotations x 2 labelings + target skew + quantity skew
    "mixed_qs": dict(groups="mixed", quantity_sigma=1.0),
    # a single concept: only target skew and quantity skew (true K = 1)
    "label": dict(groups="single", clients_per_group=40, alpha=0.1, quantity_sigma=1.0),
    # imbalanced groups: 25/9/4/2 clients (minority concepts)
    "minority": dict(groups="rotation", group_counts=[25, 9, 4, 2]),
    # second dataset
    "mnist_rot": dict(dataset="mnist", groups="rotation"),
    "mnist_swap": dict(dataset="mnist", groups="swap"),
    "cifar_rot": dict(dataset="cifar10", groups="rotation", mean_size=500),
}

TRAIN = dict(rounds=50, warmup=10, lr=0.05, batch_size=32, local_epochs=1)

METHODS = ["FedAvg", "Local", "IFCA", "MTCFL", "FL+HC", "PACFL", "DisCo", "Oracle"]

SUITES = {
    "main": dict(scenarios=["rot", "swap", "rot_qs", "mixed_qs", "label", "minority"],
                 methods=METHODS, seeds=[0, 1, 2]),
    "mnist": dict(scenarios=["mnist_rot", "mnist_swap"], methods=METHODS, seeds=[0, 1, 2]),
    "cifar": dict(scenarios=["cifar_rot"], methods=METHODS, seeds=[0, 1, 2]),
    # ablations of the DisCo components
    "ablation": dict(scenarios=["rot", "swap", "rot_qs", "mixed_qs", "label"],
                     methods=["DisCo-noDisatt", "DisCo-noClass", "DisCo-mean", "DisCo-linkorder",
                              "DisCo-noClip"],
                     seeds=[0, 1, 2]),
}

VARIANTS = {
    "DisCo-noDisatt": dict(disattenuate=False),
    "DisCo-noClass": dict(class_conditional=False),
    "DisCo-mean": dict(linkage="mean"),
    "DisCo-linkorder": dict(order="link"),
    "DisCo-noClip": dict(clip=0.0),
}


def conflict_matrix(fed, min_count=4):
    """Two clients conflict when they hold a common label (>= min_count samples
    each) whose input->label mapping differs between their concept groups."""
    n = len(fed.clients)
    C = fed.num_classes
    counts = np.stack([np.bincount(c.y.numpy(), minlength=C) for c in fed.clients])
    inv = []
    for c in fed.clients:
        g = fed.groups[c.group]
        perm = np.arange(C) if g.perm is None else np.array(g.perm)
        inv.append((g.rotation % 4, np.argsort(perm)))
    M = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            if inv[i][0] != inv[j][0]:
                bad = True
            else:
                shared = (counts[i] >= min_count) & (counts[j] >= min_count)
                bad = bool((shared & (inv[i][1] != inv[j][1])).any())
            M[i, j] = M[j, i] = bad
    return M


def metrics(fed, tr, out):
    from sklearn.metrics import adjusted_rand_score
    from discocfl.fl import evaluate_assignment

    local, bal = evaluate_assignment(tr, fed, out["models"], out["assign"])
    g = fed.true_labels
    A = np.asarray(out["assign"])
    group_acc = [float(local[g == k].mean()) for k in np.unique(g)]
    conf = conflict_matrix(fed)
    same = np.equal.outer(A, A) & ~np.eye(len(A), dtype=bool)
    compatible = ~conf & ~np.eye(len(A), dtype=bool)
    return {
        "acc_mean": float(local.mean()),
        "acc_p10": float(np.percentile(local, 10)),
        "acc_min": float(local.min()),
        "acc_std": float(local.std()),
        "jain": float(local.sum() ** 2 / (len(local) * (local ** 2).sum())),
        "worst_group_acc": float(min(group_acc)),
        "group_acc": group_acc,
        "acc_concept_mean": float(bal.mean()),
        "ari": float(adjusted_rand_score(g, A)) if len(np.unique(g)) > 1 or len(np.unique(A)) > 1 else 1.0,
        "k_found": int(len(np.unique(A))),
        "conflict_merge_rate": float((same & conf).sum() / max(conf.sum(), 1)),
        "compatible_split_rate": float((~same & compatible & np.equal.outer(g, g)).sum()
                                       / max((compatible & np.equal.outer(g, g)).sum(), 1)),
        "per_client_acc": local.tolist(),
    }


def run_one(job):
    scen, method, seed, extra = job
    import torch
    torch.set_num_threads(1)
    from discocfl.algorithms import ALGORITHMS
    from discocfl.data import make_federation
    from discocfl.fl import Trainer

    tag = extra.pop("_tag", "")
    path = os.path.join(RESULTS, f"{scen}__{method}{tag}__s{seed}.json")
    if os.path.exists(path):
        return path
    fcfg = dict(BASE, **SCENARIOS[scen])
    fed = make_federation(seed=seed, **fcfg)
    K_true = len(fed.groups)
    tr = Trainer(fed, lr=TRAIN["lr"], batch_size=TRAIN["batch_size"],
                 local_epochs=TRAIN["local_epochs"], seed=seed)
    cfg = dict(TRAIN, seed=seed, K=K_true)
    base_method = method
    if method in VARIANTS:
        cfg.update(VARIANTS[method])
        base_method = "DisCo"
    cfg.update(extra)
    t0 = time.time()
    out = ALGORITHMS[base_method](fed, tr, cfg)
    elapsed = time.time() - t0
    res = metrics(fed, tr, out)
    res.update({"scenario": scen, "method": method + tag, "seed": seed, "time_s": elapsed,
                "cluster_upload_floats": int(out.get("cluster_floats", 0)),
                "model_dim": tr.dim, "n_clients": len(fed.clients), "k_true": K_true,
                "assign": np.asarray(out["assign"]).tolist(), "true": fed.true_labels.tolist(),
                "extra": {k: v for k, v in out.items() if k in
                          ("n_confident", "flags", "time_signature_s", "time_server_s",
                           "extra_download_floats")},
                "config": {k: v for k, v in cfg.items()}})
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[done] {scen} {method}{tag} s{seed} acc={res['acc_mean']:.3f} ari={res['ari']:.2f} "
          f"K={res['k_found']} ({elapsed:.0f}s)", flush=True)
    return path


def jobs_for(suite):
    s = SUITES[suite]
    return [(sc, m, sd, {}) for sc, m, sd in itertools.product(s["scenarios"], s["methods"], s["seeds"])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="main")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="comma-separated scenario filter")
    ap.add_argument("--methods", default=None)
    a = ap.parse_args()
    jobs = jobs_for(a.suite)
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j[0] in keep]
    if a.methods:
        keep = set(a.methods.split(","))
        jobs = [j for j in jobs if j[1] in keep]
    # longest jobs first
    order = {"IFCA": 0, "MTCFL": 1, "Local": 2}
    jobs.sort(key=lambda j: order.get(j[1], 5))
    with Pool(a.workers) as p:
        for _ in p.imap_unordered(run_one, jobs):
            pass
