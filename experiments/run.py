"""Run the benchmark grid: scenarios x methods x seeds (results/raw/*.json).

usage: python experiments/run.py --suite main --workers 4
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = os.environ.get("DISCO_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results", "r1", "raw"))

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
    "cifar_swap": dict(dataset="cifar10", groups="swap", mean_size=500),
    "svhn_rot": dict(dataset="svhn", groups="rotation", mean_size=500),
    "svhn_swap": dict(dataset="svhn", groups="swap", mean_size=500),
    # heterogeneity strength: 4 groups rotated by 0, a, 2a, 3a degrees
    "angle15": dict(groups="angle15"),
    "angle30": dict(groups="angle30"),
    "angle45": dict(groups="angle45"),
    # number of concept groups (label permutations)
    "perm2": dict(groups="perm2", clients_per_group=20),
    "perm8": dict(groups="perm8", clients_per_group=5),
    # partial participation: 30% of the clients per round
    "rot_part30": dict(groups="rotation", _train=dict(participation=0.3)),
    "swap_part30": dict(groups="swap", _train=dict(participation=0.3)),
    # more clients
    "rot_n200": dict(groups="rotation", clients_per_group=50, mean_size=200),
}

TRAIN = dict(rounds=50, warmup=10, lr=0.05, batch_size=32, local_epochs=1)

METHODS = ["FedAvg", "Local", "IFCA", "MTCFL", "FL+HC", "PACFL", "DisCo", "Oracle"]
NEW_BASELINES = ["FeSEM", "FedAvg-FT"]
NEW_R1 = ["StoCFL", "StoCFL-W", "FedCCFA", "FedDAG-C"]
ALL = METHODS + NEW_BASELINES
SWEEP = ["FedAvg", "IFCA", "FL+HC", "PACFL", "FeSEM", "DisCo", "Oracle"]

SUITES = {
    "main": dict(scenarios=["rot", "swap", "rot_qs", "mixed_qs", "label", "minority"],
                 methods=METHODS, seeds=[0, 1, 2]),
    "mnist": dict(scenarios=["mnist_rot", "mnist_swap"], methods=METHODS, seeds=[0, 1, 2]),
    "cifar": dict(scenarios=["cifar_rot"], methods=METHODS, seeds=[0, 1, 2]),
    # second round of experiments
    "color": dict(scenarios=["svhn_rot", "svhn_swap", "cifar_swap"], methods=ALL + ["DisCo-T30"],
                  seeds=[0, 1, 2]),
    "newbase": dict(scenarios=["rot", "swap", "rot_qs", "mixed_qs", "label", "minority",
                               "mnist_rot", "mnist_swap", "cifar_rot"],
                    methods=NEW_BASELINES, seeds=[0, 1, 2]),
    "strength": dict(scenarios=["angle15", "angle30", "angle45"], methods=SWEEP, seeds=[0, 1, 2]),
    "groups": dict(scenarios=["perm2", "perm8"], methods=SWEEP, seeds=[0, 1, 2]),
    "participation": dict(scenarios=["rot_part30", "swap_part30"],
                          methods=["FedAvg", "IFCA", "MTCFL", "FL+HC", "PACFL", "FeSEM", "DisCo", "Oracle"],
                          seeds=[0, 1, 2]),
    "scale": dict(scenarios=["rot_n200"], methods=["FedAvg", "FL+HC", "PACFL", "FeSEM", "DisCo", "Oracle"],
                  seeds=[0, 1, 2]),
    # ablations of the DisCo components
    "ablation": dict(scenarios=["rot", "swap", "rot_qs", "mixed_qs", "label"],
                     methods=["DisCo-noDisatt", "DisCo-noClass", "DisCo-mean", "DisCo-linkorder",
                              "DisCo-noClip"],
                     seeds=[0, 1, 2]),
    # round 1 of revisions: new baselines, DisCo without attachment, and the seeds 3-4
    "r1_main": dict(scenarios=["rot", "swap", "rot_qs", "mixed_qs", "label", "minority"],
                    methods=METHODS + NEW_BASELINES + NEW_R1 + ["DisCo-noAttach"], seeds=[0, 1, 2, 3, 4]),
    "r1_other": dict(scenarios=["mnist_rot", "mnist_swap", "svhn_rot", "svhn_swap", "cifar_rot", "cifar_swap"],
                     methods=NEW_R1 + ["DisCo-noAttach"], seeds=[0, 1, 2]),
    "r1_other_seeds": dict(scenarios=["mnist_rot", "mnist_swap", "svhn_rot", "svhn_swap", "cifar_rot",
                                      "cifar_swap"],
                           methods=METHODS + NEW_BASELINES + NEW_R1 + ["DisCo-noAttach", "DisCo-T30"],
                           seeds=[3, 4]),
    # full re-run of the revision with the corrected training primitive
    "r1_other_all": dict(scenarios=["mnist_rot", "mnist_swap", "svhn_rot", "svhn_swap", "cifar_rot",
                                    "cifar_swap"],
                         methods=METHODS + NEW_BASELINES + NEW_R1 + ["DisCo-noAttach", "DisCo-T30"],
                         seeds=[0, 1, 2, 3, 4]),
    "r1_sweeps": dict(scenarios=["angle15", "angle30", "angle45", "perm2", "perm8", "rot_part30",
                                 "swap_part30", "rot_n200"],
                      methods=SWEEP + ["MTCFL"] + NEW_R1, seeds=[0, 1, 2]),
    "r1_ablation": dict(scenarios=["rot", "swap", "rot_qs", "mixed_qs", "label"],
                        methods=["DisCo-noDisatt", "DisCo-noClass", "DisCo-mean", "DisCo-linkorder",
                                 "DisCo-noClip"], seeds=[0, 1, 2]),
    "r1_aw": dict(scenarios=["rot", "swap", "rot_qs", "mixed_qs", "label", "minority",
                             "mnist_rot", "mnist_swap", "svhn_rot", "svhn_swap", "cifar_rot",
                             "cifar_swap"], methods=["DisCo-AW"], seeds=[0, 1, 2, 3, 4]),
    "r1_aw_sweeps": dict(scenarios=["angle15", "angle30", "angle45", "perm2", "perm8", "rot_part30",
                                    "swap_part30", "rot_n200"], methods=["DisCo-AW"], seeds=[0, 1, 2]),
}

VARIANTS = {
    "DisCo-noDisatt": dict(disattenuate=False),
    "DisCo-noClass": dict(class_conditional=False),
    "DisCo-mean": dict(linkage="mean"),
    "DisCo-linkorder": dict(order="link"),
    "DisCo-noClip": dict(clip=0.0),
    "DisCo-T30": dict(warmup=30),
    "DisCo-full": dict(sig="full"),
    "DisCo-noAttach": dict(attach=False),
    # adaptive warm-up: FedAvg until the training loss plateaus (5% over 5 rounds), 10..30 rounds
    "DisCo-AW": dict(adaptive_warmup=True),
    # StoCFL with its gradient representation computed at the warm-up model
    "StoCFL-W": dict(_base="StoCFL", stocfl_warmup="warmup"),
}


def conflict_classes(fed, min_count=4):
    """Conflict matrix and, for every conflicting pair i < j, the discriminative
    classes. Two clients conflict when they hold a common label (>= min_count
    samples each) whose input->label mapping differs between their concept groups.
    If the feature transformations differ (key() differs, e.g. rotations or
    domains) every shared class is discriminative; otherwise only the shared
    classes whose label permutation differs are.
    Returns (M, disc) with disc = {(i, j): np.ndarray of class indices}."""
    n = len(fed.clients)
    C = fed.num_classes
    counts = np.stack([np.bincount(c.y.cpu().numpy(), minlength=C) for c in fed.clients])
    inv = []
    for c in fed.clients:
        g = fed.groups[c.group]
        perm = np.arange(C) if g.perm is None else np.array(g.perm)
        inv.append((g.key(), np.argsort(perm)))
    M = np.zeros((n, n), bool)
    disc = {}
    for i in range(n):
        for j in range(i + 1, n):
            shared = (counts[i] >= min_count) & (counts[j] >= min_count)
            if inv[i][0] != inv[j][0]:
                bad = True
                cls = shared
            else:
                cls = shared & (inv[i][1] != inv[j][1])
                bad = bool(cls.any())
            M[i, j] = M[j, i] = bad
            if bad:
                disc[(i, j)] = np.where(cls)[0]
    return M, disc


def conflict_matrix(fed, min_count=4):
    """Two clients conflict when they hold a common label (>= min_count samples
    each) whose input->label mapping differs between their concept groups."""
    return conflict_classes(fed, min_count)[0]


def _fl(x):
    """JSON-safe float: NaN / inf become None."""
    x = float(x)
    return x if math.isfinite(x) else None


def _diagnostics(fed, sim, conf_disc):
    """Identifiability / threshold diagnostics of the stored similarity object."""
    from sklearn.metrics import adjusted_rand_score
    from discocfl.diagnostics import coverage, group_linkage, otsu_threshold, pairwise_bottleneck
    from discocfl.disco import threshold_cluster

    g = fed.true_labels
    n = len(g)
    gl = group_linkage(sim, g, lam=0.5)
    cov = coverage(sim, conf_disc)
    iu = np.triu_indices(n, 1)
    vals = pairwise_bottleneck(sim, 0.5)[iu]
    tau, eta = otsu_threshold(vals[np.isfinite(vals)])
    lab_o, _ = threshold_cluster(sim, tau, lam=0.5)
    if len(np.unique(g)) == 1 and len(np.unique(lab_o)) == 1:
        ari_o = 1.0
    else:
        ari_o = float(adjusted_rand_score(g, lab_o))
    return {"delta_hat": _fl(gl["delta_hat"]), "within_min": _fl(gl["within_min"]),
            "coverage": _fl(cov), "otsu_tau": _fl(tau), "otsu_eta": _fl(eta),
            "ari_otsu": _fl(ari_o), "k_otsu": int(len(np.unique(lab_o)))}


def metrics(fed, tr, out):
    from sklearn.metrics import adjusted_rand_score
    from discocfl.fl import evaluate_assignment

    local, bal = evaluate_assignment(tr, fed, out["models"], out["assign"])
    local_pc, bal_pc = evaluate_assignment(tr, fed, out["models"], out["assign"], prior_correction=True)
    g = fed.true_labels
    # clustering metrics use the partition before attachment when a method reports one;
    # accuracy always uses the final assignment (out["assign"]) and the cluster models
    A = np.asarray(out.get("partition_assign", out["assign"]))
    group_acc = [float(local[g == k].mean()) for k in np.unique(g)]
    conf, disc = conflict_classes(fed)
    same = np.equal.outer(A, A) & ~np.eye(len(A), dtype=bool)
    compatible = ~conf & ~np.eye(len(A), dtype=bool)
    res = {
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
        # with Bayes prior (logit) correction, applied identically to every method
        "pc_acc_mean": float(local_pc.mean()),
        "pc_acc_p10": float(np.percentile(local_pc, 10)),
        "pc_worst_group_acc": float(min(local_pc[g == k].mean() for k in np.unique(g))),
        "pc_jain": float(local_pc.sum() ** 2 / (len(local_pc) * (local_pc ** 2).sum())),
        "pc_acc_concept_mean": float(bal_pc.mean()),
        "pc_per_client_acc": local_pc.tolist(),
    }
    # attachment accounting. A conflicting pair (i < j) is an error of the merge stage
    # when both clients stay in their clustering-stage cluster and share the final
    # cluster; it is an error of the attach stage when one of them was attached.
    flags = out.get("flags") or {}
    if "attached_small" in flags or "attached_low" in flags:
        att_ids = [int(i) for i in list(flags.get("attached_small", [])) + list(flags.get("attached_low", []))]
        att = np.zeros(len(g), bool)
        att[att_ids] = True
        A_final = np.asarray(out["assign"])
        iu = np.triu_indices(len(g), 1)
        conf_u = conf[iu]
        same_final = np.equal.outer(A_final, A_final)[iu]
        touched = att[iu[0]] | att[iu[1]]
        n_conf = max(int(conf_u.sum()), 1)
        res["n_attached"] = int(att.sum())
        res["conflict_rate_merge_stage"] = float((conf_u & same_final & ~touched).sum() / n_conf)
        res["conflict_rate_attach_stage"] = float((conf_u & same_final & touched).sum() / n_conf)
    # identifiability and threshold diagnostics; never allowed to break a run
    if "sim" in out:
        try:
            res["diag"] = _diagnostics(fed, out["sim"], disc)
        except Exception as e:  # noqa: BLE001
            res["diag"] = {"error": f"{type(e).__name__}: {e}"}
    return res


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
    train_extra = fcfg.pop("_train", {})
    fed = make_federation(seed=seed, **fcfg)
    K_true = len(fed.groups)
    tr = Trainer(fed, lr=TRAIN["lr"], batch_size=TRAIN["batch_size"],
                 local_epochs=TRAIN["local_epochs"], seed=seed)
    cfg = dict(TRAIN, seed=seed, K=K_true)
    cfg.update(train_extra)
    tr.participation = cfg.get("participation", 1.0)
    base_method = method
    if method in VARIANTS:
        cfg.update(VARIANTS[method])
        base_method = cfg.pop("_base", "DisCo")
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
                           "extra_download_floats", "warmup_used", "warmup_losses")},
                "config": {k: v for k, v in cfg.items()}})
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[done] {scen} {method}{tag} s{seed} acc={res['acc_mean']:.3f} ari={res['ari']:.2f} "
          f"K={res['k_found']} ({elapsed:.0f}s)", flush=True)
    return path


def jobs_for(suite):
    # seed-major: all methods (and scenarios) of seed s come before seed s+1, so an
    # interrupted run leaves complete seeds behind
    s = SUITES[suite]
    return [(sc, m, sd, {}) for sd, sc, m in itertools.product(s["seeds"], s["scenarios"], s["methods"])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="main")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="comma-separated scenario filter")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated seed filter")
    ap.add_argument("--out", default=None, help="results directory")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    ap.add_argument("--reverse", action="store_true", help="run the selected jobs in reverse order")
    a = ap.parse_args()
    if a.out:
        RESULTS = a.out
        os.environ["DISCO_RESULTS"] = a.out
    jobs = jobs_for(a.suite)
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j[0] in keep]
    if a.methods:
        keep = set(a.methods.split(","))
        jobs = [j for j in jobs if j[1] in keep]
    # seed-major, longest methods first within a seed (stable sort keeps seed order)
    order = {"IFCA": 0, "FeSEM": 1, "MTCFL": 1, "Local": 2}
    jobs.sort(key=lambda j: (j[2], order.get(j[1], 5)))
    if a.seeds:
        keep = set(int(v) for v in a.seeds.split(","))
        jobs = [j for j in jobs if j[2] in keep]
    i, n = map(int, a.shard.split("/"))
    jobs = jobs[i::n]
    if a.reverse:
        jobs = jobs[::-1]
    print(f"{len(jobs)} jobs, {a.workers} workers -> {RESULTS}", flush=True)
    with Pool(a.workers) as p:
        for _ in p.imap_unordered(run_one, jobs):
            pass
