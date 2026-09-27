"""DisCo-CFL and the baselines, all run under the same training budget.

Every algorithm returns a dict with the final per-client assignment, the cluster
models and bookkeeping (uploaded floats used for clustering, number of rounds).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from .disco import CountSketch, SignatureConfig, client_signature, disco_cluster
from .fl import Trainer, cluster_fedavg_round, weighted_average


def _fedavg_rounds(tr, fed, w, rounds):
    assign = np.zeros(len(fed.clients), int)
    models = {0: w}
    for _ in range(rounds):
        models = cluster_fedavg_round(tr, fed, models, assign)
    return models[0]


def _train_clusters(tr, fed, init, assign, rounds):
    models = {int(k): init.clone() if isinstance(init, torch.Tensor) else init[k]
              for k in np.unique(assign)}
    for _ in range(rounds):
        models = cluster_fedavg_round(tr, fed, models, assign)
    return models


def run_fedavg(fed, tr, cfg):
    w = _fedavg_rounds(tr, fed, tr.get(), cfg["rounds"])
    return {"assign": np.zeros(len(fed.clients), int), "models": {0: w}, "cluster_floats": 0}


def run_local(fed, tr, cfg):
    """Each client trains alone (same number of local epochs as the FL budget)."""
    w0 = tr.get()
    models = {c.cid: tr.local_train(w0, c.x, c.y, epochs=cfg["rounds"]) for c in fed.clients}
    return {"assign": np.arange(len(fed.clients)), "models": models, "cluster_floats": 0}


def run_oracle(fed, tr, cfg):
    w = _fedavg_rounds(tr, fed, tr.get(), cfg["warmup"])
    assign = fed.true_labels
    models = _train_clusters(tr, fed, w, assign, cfg["rounds"] - cfg["warmup"])
    return {"assign": assign, "models": models, "cluster_floats": 0}


def run_disco(fed, tr, cfg, return_details=False):
    """DisCo-CFL: warm-up FedAvg -> one-shot signatures -> calibrated clustering
    -> per-cluster FedAvg from the reference model."""
    w_ref = _fedavg_rounds(tr, fed, tr.get(), cfg["warmup"])
    scfg = SignatureConfig(sketch_dim=cfg.get("sketch_dim", 1024),
                           class_conditional=cfg.get("class_conditional", True),
                           clip=cfg.get("clip", 0.5),
                           dp_sigma=cfg.get("dp_sigma", 0.0), dp_clip=cfg.get("dp_clip", 0.5),
                           seed=cfg["seed"])
    sketch = CountSketch(tr.dim, scfg.sketch_dim, seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"] + 7)
    t0 = time.time()
    sigs = [client_signature(tr, w_ref, c, sketch, scfg, rng) for c in fed.clients]
    t_sig = time.time() - t0
    t0 = time.time()
    res = disco_cluster(sigs, tau=cfg.get("tau", 0.8), disattenuate=cfg.get("disattenuate", True),
                        lam=cfg.get("lam", 0.5), mode=cfg.get("linkage", "min"),
                        order=cfg.get("order", "evidence"))
    t_srv = time.time() - t0
    assign = res.labels
    models = _train_clusters(tr, fed, w_ref, assign, cfg["rounds"] - cfg["warmup"])
    floats = sum(2 * len(s) * scfg.sketch_dim for s in sigs)
    out = {"assign": assign, "models": models, "cluster_floats": floats,
           "time_signature_s": t_sig, "time_server_s": t_srv,
           "n_confident": sum(c["verdict"] == "confident" for c in res.certificates),
           "flags": res.flags}
    if return_details:
        out.update({"result": res, "signatures": sigs, "w_ref": w_ref, "sketch": sketch})
    return out


def run_ifca(fed, tr, cfg):
    """IFCA (Ghosh et al. 2020): K models, clients pick the lowest-loss model."""
    K = cfg["K"]
    models = {k: tr.init_vector(cfg["seed"] * 100 + k) for k in range(K)}
    assign = np.zeros(len(fed.clients), int)
    for _ in range(cfg["rounds"]):
        for c in fed.clients:
            m = min(200, c.n)
            losses = [tr.evaluate(models[k], c.x[:m], c.y[:m])[1] for k in range(K)]
            assign[c.cid] = int(np.argmin(losses))
        models = cluster_fedavg_round(tr, fed, models, assign)
    # IFCA downloads K models per client per round; no dedicated upload
    return {"assign": assign.copy(), "models": models, "cluster_floats": 0,
            "extra_download_floats": (K - 1) * tr.dim * len(fed.clients) * cfg["rounds"]}


def run_flhc(fed, tr, cfg):
    """FL+HC (Briggs et al. 2020): warm-up FedAvg, then one-shot agglomerative
    clustering (Ward, Euclidean) of the clients' local updates; K given."""
    w = _fedavg_rounds(tr, fed, tr.get(), cfg["warmup"])
    ups = torch.stack([tr.local_train(w, c.x, c.y) - w for c in fed.clients]).numpy()
    K = cfg["K"]
    assign = np.zeros(len(fed.clients), int) if K == 1 else \
        AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(ups)
    models = _train_clusters(tr, fed, w, assign, cfg["rounds"] - cfg["warmup"])
    return {"assign": assign, "models": models, "cluster_floats": ups.size}


def _bipartition(sim):
    lab = AgglomerativeClustering(n_clusters=2, metric="precomputed",
                                  linkage="complete").fit_predict(-sim)
    return lab


def run_mtcfl(fed, tr, cfg):
    """CFL / MTCFL (Sattler et al. 2020): recursive cosine bipartitioning once a
    cluster is near a stationary point of its joint objective but individual
    updates are still large. Norm thresholds are expressed relative to the mean
    client-update norm so they do not depend on the model scale. The defaults
    were selected on the same pilot run as DisCo-CFL (best ARI and accuracy
    among eps1 in {0.4,0.8,0.9,0.95}, eps2 in {0,0.6,1.2,1.4}, gamma in {0,0.2,0.5,1})."""
    eps1, eps2 = cfg.get("eps1", 0.9), cfg.get("eps2", 0.0)
    assign = np.zeros(len(fed.clients), int)
    models = {0: tr.get()}
    next_id = 1
    for r in range(cfg["rounds"]):
        new_models = dict(models)
        for k in list(models):
            members = [c for c in fed.clients if assign[c.cid] == k]
            if not members:
                continue
            ws = [tr.local_train(models[k], c.x, c.y) for c in members]
            dW = torch.stack([w - models[k] for w in ws])
            ns = [c.n for c in members]
            new_models[k] = weighted_average(ws, ns)
            mean_norm = torch.norm(weighted_average(list(dW), ns)).item()
            avg_norm = dW.norm(dim=1).mean().item()
            max_norm = dW.norm(dim=1).max().item()
            if (r >= cfg["warmup"] and len(members) > 2 and mean_norm < eps1 * avg_norm
                    and max_norm > eps2 * avg_norm):
                U = torch.nn.functional.normalize(dW, dim=1)
                sim = (U @ U.T).numpy()
                lab = _bipartition(sim)
                cross = sim[np.ix_(lab == 0, lab == 1)].max()
                if cross < cfg.get("gamma", 1.0):
                    for c, l in zip(members, lab):
                        if l == 1:
                            assign[c.cid] = next_id
                    new_models[next_id] = new_models[k].clone()
                    next_id += 1
        models = new_models
    return {"assign": assign.copy(), "models": models, "cluster_floats": 0}


def run_pacfl(fed, tr, cfg):
    """PACFL (Vahidian et al. 2023): clients send the top-p left singular vectors of
    their raw data matrix; the server clusters principal-angle distances with
    hierarchical clustering (K given)."""
    p = cfg.get("pacfl_p", 3)
    Us = []
    for c in fed.clients:
        A = c.x[: min(200, c.n)].flatten(1).T.numpy()
        U, _, _ = np.linalg.svd(A, full_matrices=False)
        Us.append(U[:, :p])
    n = len(Us)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = np.linalg.svd(Us[i].T @ Us[j], compute_uv=False).clip(-1, 1)
            D[i, j] = D[j, i] = np.degrees(np.arccos(s)).sum()
    K = cfg["K"]
    assign = np.zeros(n, int) if K == 1 else AgglomerativeClustering(
        n_clusters=K, metric="precomputed", linkage="complete").fit_predict(D)
    w = tr.get()
    models = _train_clusters(tr, fed, w, assign, cfg["rounds"])
    return {"assign": assign, "models": models, "cluster_floats": n * p * Us[0].shape[0]}


ALGORITHMS = {
    "FedAvg": run_fedavg,
    "Local": run_local,
    "IFCA": run_ifca,
    "MTCFL": run_mtcfl,
    "FL+HC": run_flhc,
    "PACFL": run_pacfl,
    "DisCo": run_disco,
    "Oracle": run_oracle,
}
