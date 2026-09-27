"""Clustering-level studies of DisCo-CFL (no cluster training needed).

Studies: tau sensitivity, differential privacy, sample size, minority groups,
explanation fidelity, certificate calibration, newcomers, scalability.
Results go to results/clustering/<study>.json.

usage: python experiments/clustering_study.py --study all --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "..", "results", "clustering")
CACHE = os.path.join(HERE, "..", "results", "cache")


def _setup():
    import torch
    torch.set_num_threads(1)


def prepare(fcfg, seed, warmup=10):
    """Federation, trainer and warm-up reference model (cached on disk)."""
    import torch
    from discocfl.algorithms import _fedavg_rounds
    from discocfl.data import make_federation
    from discocfl.fl import Trainer
    fed = make_federation(seed=seed, **fcfg)
    tr = Trainer(fed, seed=seed)
    key = "_".join(f"{k}={v}" for k, v in sorted(fcfg.items())).replace("/", "")
    path = os.path.join(CACHE, f"wref_{hashlib.md5(key.encode()).hexdigest()[:12]}_{seed}_{warmup}.pt")
    if os.path.exists(path):
        w = torch.load(path)
    else:
        w = _fedavg_rounds(tr, fed, tr.get(), warmup)
        os.makedirs(CACHE, exist_ok=True)
        torch.save(w, path)
    return fed, tr, w


def signatures(fed, tr, w, seed, **kw):
    from discocfl.disco import CountSketch, SignatureConfig, client_signature
    scfg = SignatureConfig(seed=seed, **kw)
    sk = CountSketch(tr.dim, scfg.sketch_dim, seed=seed)
    rng = np.random.default_rng(seed + 7)
    return [client_signature(tr, w, c, sk, scfg, rng) for c in fed.clients]


def cluster_scores(fed, labels):
    from sklearn.metrics import adjusted_rand_score
    from run import conflict_matrix
    g = fed.true_labels
    conf = conflict_matrix(fed)
    n = len(g)
    same = np.equal.outer(labels, labels) & ~np.eye(n, dtype=bool)
    comp = ~conf & ~np.eye(n, dtype=bool) & np.equal.outer(g, g)
    return {"ari": float(adjusted_rand_score(g, labels)), "k": int(len(np.unique(labels))),
            "conflict_merge_rate": float((same & conf).sum() / max(conf.sum(), 1)),
            "compatible_split_rate": float((~same & comp).sum() / max(comp.sum(), 1))}


def base(**kw):
    from run import BASE, SCENARIOS
    d = dict(BASE)
    d.update(kw)
    return d


# ---------------------------------------------------------------------------
def job_tau(args):
    scen, seed = args
    _setup()
    from run import SCENARIOS
    from discocfl.disco import disco_cluster
    fed, tr, w = prepare(base(**SCENARIOS[scen]), seed)
    sigs = signatures(fed, tr, w, seed)
    rows = []
    for dis in (True, False):
        for tau in np.round(np.arange(0.5, 0.96, 0.05), 2):
            r = disco_cluster(sigs, tau=float(tau), disattenuate=dis)
            rows.append(dict(scenario=scen, seed=seed, tau=float(tau), disattenuate=dis, lam=0.5,
                             **cluster_scores(fed, r.labels)))
    for lam in (0.02, 0.1, 0.2, 1.0):
        r = disco_cluster(sigs, tau=0.8, lam=lam)
        rows.append(dict(scenario=scen, seed=seed, tau=0.8, disattenuate=True, lam=lam,
                         **cluster_scores(fed, r.labels)))
    return rows


def job_dp(args):
    scen, size, seed, sigma = args
    _setup()
    from run import SCENARIOS
    from discocfl.disco import disco_cluster, gaussian_dp_epsilon
    fed, tr, w = prepare(base(**dict(SCENARIOS[scen], mean_size=size)), seed)
    sigs = signatures(fed, tr, w, seed, dp_sigma=sigma, dp_clip=0.5,
                      max_per_class=10 ** 6)
    rows = []
    for dis in (True, False):
        r = disco_cluster(sigs, disattenuate=dis)
        rows.append(dict(scenario=scen, mean_size=size, seed=seed, sigma=sigma, disattenuate=dis,
                         epsilon=gaussian_dp_epsilon(sigma), **cluster_scores(fed, r.labels)))
    return rows


def job_size(args):
    groups, size, seed = args
    _setup()
    from discocfl.disco import disco_cluster
    fed, tr, w = prepare(base(groups=groups, mean_size=size), seed)
    sigs = signatures(fed, tr, w, seed)
    rows = []
    for dis in (True, False):
        r = disco_cluster(sigs, disattenuate=dis)
        S = r.sim.S
        g = fed.true_labels
        m = np.equal.outer(g, g) & ~np.eye(len(g), dtype=bool)
        rows.append(dict(groups=groups, mean_size=size, seed=seed, disattenuate=dis,
                         same_group_sim=float(np.nanmean(S[m])),
                         diff_group_sim=float(np.nanmean(S[~np.equal.outer(g, g)])),
                         **cluster_scores(fed, r.labels)))
    return rows


def job_minority(args):
    m, seed = args
    _setup()
    import torch
    from sklearn.cluster import AgglomerativeClustering
    from discocfl.algorithms import run_pacfl
    from discocfl.disco import disco_cluster
    counts = [20, 10, 10, m]
    fed, tr, w = prepare(base(groups="rotation", group_counts=counts), seed)
    g = fed.true_labels
    sigs = signatures(fed, tr, w, seed)
    out = {}
    out["DisCo"] = disco_cluster(sigs).labels
    ups = torch.stack([tr.local_train(w, c.x, c.y) - w for c in fed.clients]).numpy()
    out["FL+HC"] = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(ups)
    # PACFL clustering only (no training): reuse its distance computation
    out["PACFL"] = run_pacfl(fed, tr, dict(K=4, rounds=0))["assign"]
    rows = []
    mi = np.where(g == 3)[0]
    for name, lab in out.items():
        # minority recall: fraction of minority clients whose cluster is majority-minority
        maj = {k: np.bincount(g[lab == k], minlength=4).argmax() for k in np.unique(lab)}
        recall = float(np.mean([maj[lab[i]] == 3 for i in mi]))
        purity = float(np.mean([g[i] == 3 for i in np.where(np.isin(lab, [k for k, v in maj.items() if v == 3]))[0]])) \
            if any(v == 3 for v in maj.values()) else 0.0
        rows.append(dict(method=name, minority_size=m, seed=seed, minority_recall=recall,
                         minority_purity=purity, **cluster_scores(fed, np.asarray(lab))))
    return rows


def job_explain(args):
    scen, seed = args
    _setup()
    from run import SCENARIOS, conflict_matrix
    from discocfl.disco import disco_cluster
    fed, tr, w = prepare(base(**SCENARIOS[scen]), seed)
    sigs = signatures(fed, tr, w, seed)
    r = disco_cluster(sigs)
    g, lab = fed.true_labels, r.labels
    C = fed.num_classes
    maj = {k: int(np.bincount(g[lab == k]).argmax()) for k in np.unique(lab)}
    rows = []
    for key, v in r.explanations["between"].items():
        a, b = map(int, key.split("-"))
        ga, gb = maj[a], maj[b]
        if ga == gb:
            continue
        A, B = fed.groups[ga], fed.groups[gb]
        pa = np.arange(C) if A.perm is None else np.array(A.perm)
        pb = np.arange(C) if B.perm is None else np.array(B.perm)
        if A.rotation % 4 != B.rotation % 4:
            truth, kind = set(range(C)), "global"
        else:
            truth, kind = set(np.where(np.argsort(pa) != np.argsort(pb))[0].tolist()), "class-specific"
        shared = set(v["per_class"].keys())
        truth = truth & shared
        pred = set(v["divergent_classes"])
        tp = len(pred & truth)
        rows.append(dict(scenario=scen, seed=seed, pair=key, groups=[ga, gb], kind=kind,
                         truth=sorted(truth), pred=sorted(pred),
                         precision=tp / max(len(pred), 1), recall=tp / max(len(truth), 1),
                         exact=pred == truth,
                         diagnosis_ok=(kind == "global") == v["diagnosis"].startswith("global")))
    # certificates
    conf = conflict_matrix(fed)
    cert_rows = []
    for c in r.certificates:
        i = c["client"]
        mates = np.where(lab == lab[i])[0]
        mates = mates[mates != i]
        correct = bool(not conf[i, mates].any()) if len(mates) else True
        cert_rows.append(dict(scenario=scen, seed=seed, client=i, verdict=c["verdict"],
                              correct=correct, margin=c["margin"]))
    return [dict(type="explain", **x) for x in rows] + [dict(type="cert", **x) for x in cert_rows]


def job_newcomer(args):
    scen, seed = args
    _setup()
    from run import SCENARIOS
    from discocfl.data import Group, swap_perm
    from discocfl.disco import assign_newcomer, disco_cluster
    cfg = base(**SCENARIOS[scen])
    fed, tr, w = prepare(cfg, seed)
    sigs = signatures(fed, tr, w, seed)
    g = fed.true_labels
    rng = np.random.default_rng(seed)
    held = np.concatenate([rng.choice(np.where(g == k)[0], 2, replace=False) for k in np.unique(g)])
    keep = np.setdiff1d(np.arange(len(g)), held)
    r = disco_cluster([sigs[i] for i in keep])
    lab = r.labels
    maj = {k: int(np.bincount(g[keep][lab == k]).argmax()) for k in np.unique(lab)}
    rows = []
    for i in held:
        k, s = assign_newcomer([sigs[j] for j in keep], lab, sigs[i], 0.8)
        rows.append(dict(scenario=scen, seed=seed, kind="known", correct=(k >= 0 and maj[k] == g[i]),
                         similarity=float(s)))
    # novel concept newcomers: a labeling never seen during clustering
    from discocfl.data import make_federation
    novel = make_federation(seed=seed + 99, **dict(cfg, groups=[Group(rotation=0, perm=swap_perm(10, [(0, 9), (3, 6)]))],
                                                    clients_per_group=4))
    nsig = signatures(novel, tr, w, seed + 99)
    for s_ in nsig:
        k, s = assign_newcomer([sigs[j] for j in keep], lab, s_, 0.8)
        rows.append(dict(scenario=scen, seed=seed, kind="novel", correct=(k == -1), similarity=float(s)))
    return rows


def job_scale(args):
    n, seed = args
    _setup()
    from discocfl.disco import disco_cluster
    fed, tr, w = prepare(base(groups="rotation", clients_per_group=n // 4, mean_size=200), seed,
                         warmup=5)
    t0 = time.time()
    sigs = signatures(fed, tr, w, seed)
    t_sig = (time.time() - t0) / n
    t0 = time.time()
    r = disco_cluster(sigs)
    t_srv = time.time() - t0
    up = float(np.mean([2 * len(s) * 1024 for s in sigs]))
    return [dict(n_clients=n, seed=seed, sig_time_per_client_s=t_sig, server_time_s=t_srv,
                 upload_floats_per_client=up, model_dim=tr.dim, **cluster_scores(fed, r.labels))]


STUDIES = {
    "tau": (job_tau, lambda: [(s, sd) for s in ["rot", "swap", "rot_qs", "mixed_qs", "label", "minority"]
                              for sd in range(3)]),
    "dp": (job_dp, lambda: [(s, n, sd, sg) for s in ["rot", "swap"] for n in [400, 1500]
                            for sd in range(3) for sg in [0.0, 0.5, 1.0, 2.0, 4.0]]),
    "size": (job_size, lambda: [(g, n, sd) for g in ["rotation", "swap"] for n in [50, 100, 200, 400, 800]
                                for sd in range(3)]),
    "minority": (job_minority, lambda: [(m, sd) for m in [1, 2, 3, 5] for sd in range(3)]),
    "explain": (job_explain, lambda: [(s, sd) for s in ["swap", "mixed_qs", "rot"] for sd in range(3)]),
    "newcomer": (job_newcomer, lambda: [(s, sd) for s in ["rot", "swap"] for sd in range(3)]),
    "scale": (job_scale, lambda: [(n, 0) for n in [40, 80, 160, 320]]),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="all")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    names = list(STUDIES) if a.study == "all" else a.study.split(",")
    os.makedirs(OUT, exist_ok=True)
    for name in names:
        path = os.path.join(OUT, f"{name}.json")
        if os.path.exists(path):
            print("skip", name)
            continue
        fn, jobs = STUDIES[name]
        t0 = time.time()
        with Pool(a.workers) as p:
            rows = [r for rs in p.map(fn, jobs()) for r in rs]
        with open(path, "w") as f:
            json.dump(rows, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
        print(f"[study] {name}: {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)
