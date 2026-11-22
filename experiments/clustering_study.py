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

OUT = os.path.join(HERE, "..", "results", "r1", "clustering")
CACHE = os.path.join(HERE, "..", "results", "r1", "cache")


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
        r = disco_cluster(sigs, tau=0.9, lam=lam)
        rows.append(dict(scenario=scen, seed=seed, tau=0.9, disattenuate=True, lam=lam,
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
        if A.key() != B.key():
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


def job_edge(args):
    """Edge-case constants: minimum class size m reported in a signature, the
    reliability floor r_min, and an extreme label-skew federation (alpha = 0.1,
    many missing and tiny classes) and a small-client federation (mean size 100)."""
    scen, seed = args
    _setup()
    from run import SCENARIOS
    from discocfl.disco import compute_similarity, disco_cluster
    variants = {"base": {}, "alpha0.1": {"alpha": 0.1}, "size100": {"mean_size": 100}}
    rows = []
    for vname, over in variants.items():
        fed, tr, w = prepare(base(**dict(SCENARIOS[scen], **over)), seed)
        for m in (2, 4, 8, 16):
            sigs = signatures(fed, tr, w, seed, min_per_class=m)
            held = [len(s) for s in sigs]
            for r_min in (0.0, 0.02, 0.1, 0.2):
                import discocfl.disco as D
                orig = D.compute_similarity
                D.compute_similarity = (lambda sg, disattenuate=True, r_min=r_min, _o=orig:
                                        _o(sg, disattenuate=disattenuate, r_min=r_min))
                try:
                    r = disco_cluster(sigs)
                finally:
                    D.compute_similarity = orig
                rows.append(dict(scenario=scen, variant=vname, seed=seed, m=m, r_min=r_min,
                                 mean_classes_reported=float(np.mean(held)),
                                 n_low_evidence=len(r.flags["low_evidence"]),
                                 n_anomalous=len(r.flags["anomalous"]),
                                 **cluster_scores(fed, r.labels)))
    return rows


def job_support(args):
    """Class-support disclosure. With DP noise, every client also sends, for each
    class it does not report, two independent noise vectors with the noise scale of
    a class with the client's median half size ("padding"). We measure clustering
    quality and how well the server can infer the class support from R > 0."""
    scen, size, seed, sigma = args
    _setup()
    import torch
    from run import SCENARIOS
    from discocfl.disco import compute_similarity, disco_cluster, gaussian_dp_epsilon
    fed, tr, w = prepare(base(**dict(SCENARIOS[scen], mean_size=size)), seed)
    sigs = signatures(fed, tr, w, seed, dp_sigma=sigma, dp_clip=0.5, max_per_class=10 ** 6)
    C = fed.num_classes
    g = torch.Generator().manual_seed(seed + 11)
    padded = []
    for s in sigs:
        hs = [n // 2 for (_, _, n) in s.values()] or [2]
        h = int(np.median(hs))
        std = sigma * 2 * 0.5 / max(h, 1)
        p = next(iter(s.values()))[0].numel() if s else 1024
        q = dict(s)
        for c in range(C):
            if c not in q:
                q[c] = (torch.randn(p, generator=g) * std, torch.randn(p, generator=g) * std, 2 * h)
        padded.append(q)
    rows = []
    for name, sg in (("plain", sigs), ("padded", padded)):
        r = disco_cluster(sg)
        sim = compute_similarity(sg)
        pred = sim.R > 0                                    # (N, classes present in sim)
        cls = list(sim.classes)
        truth = np.array([[c in s0 for c in cls] for s0 in sigs])
        tp = float((pred & truth).sum()); fp = float((pred & ~truth).sum())
        fn = float((~pred & truth).sum()); tn = float((~pred & ~truth).sum())
        rows.append(dict(scenario=scen, mean_size=size, seed=seed, sigma=sigma, signature=name,
                         epsilon=gaussian_dp_epsilon(sigma),
                         support_acc=(tp + tn) / max(tp + tn + fp + fn, 1),
                         support_fpr=fp / max(fp + tn, 1), support_tpr=tp / max(tp + fn, 1),
                         **cluster_scores(fed, r.labels)))
    return rows


def _replay(fcfg, seed):
    """Replays the sampling of discocfl.data.make_federation with the same rng calls in
    the same order. The raw sample indices of a client do not depend on its group, so
    the draws can be re-transformed with any group. Returns (draws, data, groups) where
    draws[i] = (group index, train indices, test indices)."""
    import inspect
    from discocfl.data import _draw, _sample_sizes, load_dataset, make_federation, make_groups
    kw = {k: p.default for k, p in inspect.signature(make_federation).parameters.items()}
    kw.pop("seed", None)
    kw.update(fcfg)
    rng = np.random.default_rng(seed)
    data = load_dataset(kw["dataset"])
    xtr, ytr, xte, yte = data
    C = int(ytr.max()) + 1
    gs = make_groups(kw["groups"], C) if isinstance(kw["groups"], str) else kw["groups"]
    counts = kw["group_counts"] or [kw["clients_per_group"]] * len(gs)
    ytr_np, yte_np = ytr.numpy(), yte.numpy()
    tr_pools = [np.where(ytr_np == c)[0] for c in range(C)]
    te_pools = [np.where(yte_np == c)[0] for c in range(C)]
    sizes = _sample_sizes(rng, sum(counts), kw["mean_size"], kw["quantity_sigma"], kw["min_size"])
    draws = []
    for g, cnt in enumerate(counts):
        for _ in range(cnt):
            cid = len(draws)
            if kw["alpha"] is None:
                prop = np.full(C, 1.0 / C)
            else:
                prop = rng.dirichlet(np.full(C, kw["alpha"]))
            itr = _draw(rng, tr_pools, prop, int(sizes[cid]))
            ite = _draw(rng, te_pools, prop, kw["test_size"])
            draws.append((g, itr, ite))
    return draws, data, gs


def _build_clients(draws, data, gs, assign):
    """Clients from the raw draws, transformed with the group assign[i] of each client."""
    from discocfl.data import Client
    xtr, ytr, xte, yte = data
    out = []
    for cid, (_, itr, ite) in enumerate(draws):
        grp = gs[assign[cid]]
        x, y = grp.apply(xtr[itr], ytr[itr])
        xt, yt = grp.apply(xte[ite], yte[ite])
        out.append(Client(cid, int(assign[cid]), x, y, xt, yt))
    return out


def _same_client(c, d):
    import torch
    return c.group == d.group and torch.equal(c.x, d.x) and torch.equal(c.y, d.y) \
        and torch.equal(c.x_test, d.x_test) and torch.equal(c.y_test, d.y_test)


def _same_sig(s, t):
    import torch
    return s.keys() == t.keys() and all(
        torch.equal(s[c][0], t[c][0]) and torch.equal(s[c][1], t[c][1]) and s[c][2] == t[c][2]
        for c in s)


def _keyed_signatures(clients, tr, w, seed, **kw):
    """Signatures with one rng stream per client (stream key (seed + 7, index)), so that a
    client whose data are unchanged gets an identical signature in every call."""
    from discocfl.disco import CountSketch, SignatureConfig, client_signature
    scfg = SignatureConfig(seed=seed, **kw)
    sk = CountSketch(tr.dim, scfg.sketch_dim, seed=seed)
    return [client_signature(tr, w, c, sk, scfg, np.random.default_rng([seed + 7, i]))
            for i, c in enumerate(clients)]


def job_drift(args):
    scen, seed, q = args
    _setup()
    from sklearn.metrics import adjusted_rand_score
    from run import SCENARIOS
    from discocfl.disco import client_to_cluster_jk, compute_similarity, disco_cluster
    TAU = 0.9
    cfg = base(**SCENARIOS[scen])
    fed, tr, w = prepare(cfg, seed)
    N, G = len(fed.clients), len(fed.groups)
    g0 = fed.true_labels
    draws, data, gs = _replay(cfg, seed)
    orig = [c.group for c in fed.clients]
    rebuilt = _build_clients(draws, data, gs, orig)
    assert all(_same_client(c, d) for c, d in zip(fed.clients, rebuilt)), "replay mismatch"

    # drift: round(q N) clients move to a different group, chosen uniformly
    rng = np.random.default_rng(seed + 100)
    n_drift = int(round(q * N))
    chosen = rng.choice(N, n_drift, replace=False)
    new = list(orig)
    for i in chosen:
        others = [k for k in range(G) if k != orig[i]]
        new[i] = int(rng.choice(others))
    drifted = np.zeros(N, dtype=bool)
    drifted[chosen] = True
    clients1 = _build_clients(draws, data, gs, new)
    for i in np.where(~drifted)[0]:
        assert _same_client(fed.clients[i], clients1[i]), "unchanged client altered"

    # clustering before drift; signatures of all clients at the same reference model
    sigs0 = _keyed_signatures(fed.clients, tr, w, seed)
    sigs1 = _keyed_signatures(clients1, tr, w, seed)
    for i in np.where(~drifted)[0]:
        assert _same_sig(sigs0[i], sigs1[i]), "unchanged signature altered"
    r0 = disco_cluster(sigs0, tau=TAU)
    lab0 = r0.labels
    ari0 = float(adjusted_rand_score(g0, lab0))
    sim1 = compute_similarity(sigs1)

    # majority true group of each pre-drift cluster, over its unchanged members
    maj = {}
    for k in np.unique(lab0):
        mem = np.where((lab0 == k) & ~drifted)[0]
        if len(mem) == 0:
            mem = np.where(lab0 == k)[0]
        maj[int(k)] = int(np.bincount(g0[mem]).argmax())

    # newcomer test applied to every client
    flagged = np.zeros(N, dtype=bool)
    reassigned = -np.ones(N, dtype=int)
    n_undefined = 0
    for i in range(N):
        own = int(lab0[i])
        s_own = client_to_cluster_jk(sim1, i, np.where(lab0 == own)[0])["s"]
        if np.isnan(s_own):
            n_undefined += 1
        flagged[i] = bool(s_own < TAU)          # NaN (no evidence) is not flagged
        if flagged[i]:
            best, arg = -np.inf, -1
            for k in np.unique(lab0):
                s = client_to_cluster_jk(sim1, i, np.where(lab0 == k)[0])["s"]
                if not np.isnan(s) and s > best:
                    best, arg = s, int(k)
            if arg >= 0 and best >= TAU:
                reassigned[i] = arg

    # supplementary (not the pre-registered measure): the same flag, but the cluster
    # members are restricted to unchanged clients, to separate the effect of drifted
    # members (whose new signatures point to another concept) from the test itself
    flagged_alt = np.zeros(N, dtype=bool)
    for i in range(N):
        mem = np.where((lab0 == lab0[i]) & ~drifted)[0]
        flagged_alt[i] = bool(client_to_cluster_jk(sim1, i, mem)["s"] < TAU)

    reassign_ok = [reassigned[i] >= 0 and maj[int(reassigned[i])] == new[i] for i in chosen]
    return [dict(scenario=scen, seed=seed, q=q, n_drift=int(n_drift), n_clients=N,
                 detection_rate=float(flagged[drifted].mean()),
                 false_alarm_rate=float(flagged[~drifted].mean()),
                 reassign_correct=float(np.mean(reassign_ok)),
                 reassign_rate=float(np.mean(reassigned[drifted] >= 0)),
                 detection_rate_unchanged_members=float(flagged_alt[drifted].mean()),
                 false_alarm_rate_unchanged_members=float(flagged_alt[~drifted].mean()),
                 ari_pre=ari0, k_pre=int(len(np.unique(lab0))), n_undefined=int(n_undefined))]


def job_byz(args):
    scen, seed, m, attack = args
    _setup()
    import torch
    from run import SCENARIOS
    from discocfl.data import Federation
    from discocfl.disco import disco_cluster
    fed, tr, w = prepare(base(**SCENARIOS[scen]), seed)
    N = len(fed.clients)
    g = fed.true_labels
    sigs = signatures(fed, tr, w, seed)
    rng = np.random.default_rng(seed + 200)
    bad = np.sort(rng.choice(N, m, replace=False))
    honest = np.setdiff1d(np.arange(N), bad)
    gen = torch.Generator().manual_seed(int(rng.integers(2 ** 31 - 1)))
    forged = list(sigs)
    for i in bad:
        h = sigs[i]
        if attack == "random":
            # two independent Gaussian vectors per class, rescaled to the norms of the
            # client's honest half-vectors (a for the first half, b for the second)
            def fake(ref):
                v = torch.randn(ref.numel(), generator=gen)
                return v * (ref.norm() / v.norm().clamp_min(1e-12))
            forged[i] = {c: (fake(a), fake(b), n) for c, (a, b, n) in h.items()}
        elif attack == "mimic":
            cands = [j for j in honest if g[j] != g[i]]
            j = cands[int(rng.integers(len(cands)))]
            forged[i] = {c: (a.clone(), b.clone(), n) for c, (a, b, n) in sigs[j].items()}
        elif attack == "flip":
            forged[i] = {c: (-a, -b, n) for c, (a, b, n) in h.items()}
        else:
            raise ValueError(attack)

    r = disco_cluster(forged)
    lab = r.labels
    sub = Federation(clients=[fed.clients[i] for i in honest], groups=fed.groups,
                     group_test=fed.group_test, num_classes=fed.num_classes,
                     in_shape=fed.in_shape, dataset=fed.dataset)
    hs = cluster_scores(sub, lab[honest])
    sizes = np.bincount(lab[lab >= 0])
    flagged = set(r.flags["low_evidence"]) | set(r.flags["anomalous"])
    fates = []
    for i in bad:
        if sizes[lab[i]] == 1:
            f = "isolated"
        elif int(i) in flagged:
            f = "flagged"
        elif np.any(lab[honest] == lab[i]):
            f = "merged"
        else:
            f = "other"                          # cluster made of malicious clients only
        fates.append(f)
    cnt = {k: fates.count(k) for k in ("isolated", "flagged", "merged", "other")}
    return [dict(scenario=scen, seed=seed, m=m, attack=attack, honest_ari=hs["ari"],
                 honest_conflict_merge_rate=hs["conflict_merge_rate"], honest_k=hs["k"],
                 n_isolated=cnt["isolated"], n_flagged=cnt["flagged"], n_merged=cnt["merged"],
                 n_other=cnt["other"], fates=fates, malicious=[int(i) for i in bad])]


def _subset_warmup(tr, fed, keep, rounds):
    """FedAvg for `rounds` rounds in which only the clients in `keep` take part."""
    import dataclasses
    from discocfl.fl import cluster_fedavg_round
    sub = dataclasses.replace(fed, clients=[c for c in fed.clients if c.cid in set(keep)])
    assign = np.zeros(len(fed.clients), int)
    models = {0: tr.get()}
    for _ in range(rounds):
        models = cluster_fedavg_round(tr, sub, models, assign)
    return models[0]


def job_refmodel(args):
    """Dependence on the reference model (PREREGISTRATION_R1_ROUND3.md, section 2):
    reference model from all clients, from a random half, or without group 0; all
    clients are clustered at that model."""
    from sklearn.metrics import adjusted_rand_score
    from run import SCENARIOS, conflict_matrix
    from discocfl.disco import disco_cluster
    scen, seed = args
    _setup()
    fed, tr, w_full = prepare(base(**SCENARIOS[scen]), seed)
    g = fed.true_labels
    n = len(g)
    rng = np.random.default_rng(seed + 991)
    half = np.sort(rng.choice(n, n // 2, replace=False))
    refs = {"full": (w_full, np.arange(n)), "half": (None, half)}
    if len(np.unique(g)) > 1:
        refs["drop_group"] = (None, np.where(g != 0)[0])
    conf = conflict_matrix(fed)
    rows = []
    for name, (w, keep) in refs.items():
        if w is None:
            from discocfl.fl import Trainer
            w = _subset_warmup(Trainer(fed, seed=seed), fed, keep, 10)
        sigs = signatures(fed, tr, w, seed)
        lab = disco_cluster(sigs).labels
        inside = np.isin(np.arange(n), keep)
        row = dict(scenario=scen, seed=seed, ref=name, n_train=int(len(keep)),
                   **cluster_scores(fed, lab))
        for part, mask in (("in", inside), ("out", ~inside)):
            row[f"ari_{part}"] = (float(adjusted_rand_score(g[mask], lab[mask]))
                                  if mask.sum() > 1 else None)
        pair0 = conf & (np.equal.outer(g == 0, np.ones(n, bool)) | np.equal.outer(np.ones(n, bool), g == 0))
        same = np.equal.outer(lab, lab) & ~np.eye(n, dtype=bool)
        row["conflict_merge_group0"] = float((same & pair0).sum() / max(pair0.sum(), 1))
        rows.append(row)
    return rows


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
    "edge": (job_edge, lambda: [(s, sd) for s in ["rot", "swap", "mixed_qs", "label"] for sd in range(3)]),
    "support": (job_support, lambda: [(s, n, sd, sg) for s in ["rot", "swap"] for n in [400, 1500]
                                      for sd in range(3) for sg in [0.5, 1.0, 2.0]]),
    "scale": (job_scale, lambda: [(n, 0) for n in [40, 80, 160, 320]]),
    "drift": (job_drift, lambda: [(s, sd, q) for s in ["rot", "swap"] for sd in range(3)
                                  for q in [0.1, 0.2]]),
    "byz": (job_byz, lambda: [(s, sd, m, a) for s in ["rot", "swap"] for sd in range(3)
                              for m in [1, 2, 4] for a in ["random", "mimic", "flip"]]),
    "refmodel": (job_refmodel, lambda: [(s, sd) for s in ["rot", "swap", "mixed_qs", "label",
                                                          "mnist_rot", "mnist_swap"] for sd in range(3)]),
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
