"""Three additional clustered-FL baselines: StoCFL, FedCCFA and FedDAG (clustering stage).

All three use the same training budget as the other methods (plain SGD, no
momentum, learning rate ``tr.lr``, batch size ``tr.bs``, one local epoch = one
pass over the local data) and respect partial participation through
``participating``. Every algorithm returns the same dictionary as those in
``algorithms.py``.

Note on aliasing: ``Trainer.local_train`` (through ``vector_to_parameters``) writes
its SGD updates into the vector it receives. Every call in this module therefore
passes a clone, and ``_cluster_fedavg`` is a copy of ``cluster_fedavg_round`` that
does the same.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from .fl import Trainer, participating, weighted_average


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _features(model, x):
    """Penultimate features: the input of the final linear layer."""
    if hasattr(model, "fc2"):                       # SmallCNN
        h = F.max_pool2d(F.relu(model.conv1(x)), 2)
        h = F.max_pool2d(F.relu(model.conv2(h)), 2)
        return F.relu(model.fc1(h.flatten(1)))
    return model.features(x)                        # L4Head


def _classifier(model):
    return model.fc2 if hasattr(model, "fc2") else model.fc


def _head_shape(model):
    """(C, F, numel) of the final linear layer (its weight and bias are the last two tensors)."""
    params = list(model.parameters())
    C, Fd = params[-2].shape
    return C, Fd, params[-2].numel() + params[-1].numel()


def _rows(v, C, Fd):
    """Flat classifier vector (weight then bias) -> (C, F+1) matrix with rows [W_c, b_c]."""
    return torch.cat([v[:C * Fd].reshape(C, Fd), v[C * Fd:].reshape(C, 1)], 1)


def _flat(rows, Fd):
    """Inverse of ``_rows``."""
    return torch.cat([rows[:, :Fd].reshape(-1), rows[:, Fd].reshape(-1)])


@torch.no_grad()
def _feat_all(model, x, chunk=512):
    model.eval()
    return torch.cat([_features(model, x[s:s + chunk]) for s in range(0, len(x), chunk)])


def _sgd(tr: Trainer, vec, x, y, part, epochs=1, proto=None):
    """Plain SGD with the same loop as ``Trainer.local_train``, updating only the
    classifier (``part="clf"``) or only the feature extractor (``part="ext"``).

    proto = (anchors (C, F), temperature, weight) adds
    weight * CE(cos(f, anchors) / temperature, y) to the loss (FedCCFA alignment).
    """
    tr.set(vec.clone())
    model = tr.model
    model.train()
    params = list(model.parameters())
    head = set(range(len(params) - 2, len(params)))
    trainable = head if part == "clf" else set(range(len(params) - 2))
    for i, p in enumerate(params):
        p.requires_grad_(i in trainable)
    opt = torch.optim.SGD([p for i, p in enumerate(params) if i in trainable],
                          lr=tr.lr, momentum=0.0)
    n = len(y)
    for _ in range(epochs):
        order = torch.from_numpy(tr.rng.permutation(n)).to(x.device)
        for s in range(0, n, tr.bs):
            b = order[s:s + tr.bs]
            opt.zero_grad()
            f = _features(model, x[b])
            loss = F.cross_entropy(_classifier(model)(f), y[b])
            if proto is not None:
                anchors, temp, weight = proto
                sim = F.cosine_similarity(f[:, None, :], anchors[None, :, :], dim=2) / temp
                loss = loss + weight * F.cross_entropy(sim, y[b])
            loss.backward()
            opt.step()
    for p in params:
        p.requires_grad_(True)
    return tr.get()


def _local_train_prox(tr: Trainer, vec, x, y, anchor_vec, lam, epochs=None):
    """``Trainer.local_train`` with the proximal gradient lam * (theta - anchor) added
    to the gradient of every parameter before each SGD step."""
    tr.set(vec.clone())
    model = tr.model
    model.train()
    params = list(model.parameters())
    sizes = [p.numel() for p in params]
    anchors = [a.view_as(p) for a, p in zip(torch.split(anchor_vec.clone(), sizes), params)]
    opt = torch.optim.SGD(params, lr=tr.lr, momentum=0.0)
    n = len(y)
    for _ in range(epochs or tr.epochs):
        order = torch.from_numpy(tr.rng.permutation(n)).to(x.device)
        for s in range(0, n, tr.bs):
            b = order[s:s + tr.bs]
            opt.zero_grad()
            F.cross_entropy(model(x[b]), y[b]).backward()
            with torch.no_grad():
                for p, a in zip(params, anchors):
                    p.grad.add_(lam * (p - a))
            opt.step()
    return tr.get()


def _init_clusters(init, assign):
    return {int(k): init.clone() for k in np.unique(assign)}


def _cluster_fedavg(tr, fed, models, assign, rounds):
    """``cluster_fedavg_round`` for ``rounds`` rounds, with defensive copies."""
    for _ in range(rounds):
        updates = {}
        for c in participating(tr, fed):
            k = int(assign[c.cid])
            updates.setdefault(k, []).append((tr.local_train(models[k].clone(), c.x, c.y), c.n))
        models = dict(models)
        for k, lst in updates.items():
            models[k] = weighted_average([v for v, _ in lst], [n for _, n in lst])
    return models


# ---------------------------------------------------------------------------
# StoCFL
# ---------------------------------------------------------------------------

def _stocfl_cluster(G, tau):
    """Stochastic federated client clustering with every client sampled.

    Starts from singletons. The representation of a cluster is the sum of its
    members' normalised gradients. Repeatedly merge the pair of clusters with the
    highest cosine similarity while that similarity exceeds tau."""
    members = [[i] for i in range(len(G))]
    reps = [g.clone() for g in G]
    while len(reps) > 1:
        R = F.normalize(torch.stack(reps), dim=1)
        S = R @ R.T
        S.fill_diagonal_(-float("inf"))
        a, b = divmod(int(S.argmax()), S.shape[1])
        if S[a, b].item() <= tau:
            break
        a, b = min(a, b), max(a, b)
        reps[a] = reps[a] + reps[b]
        members[a] += members[b]
        del reps[b], members[b]
    labels = np.zeros(len(G), int)
    for k, mem in enumerate(members):
        labels[mem] = k
    return labels


def _fedavg(tr, fed, w, rounds):
    for _ in range(rounds):
        ups = [(tr.local_train(w.clone(), c.x, c.y), c.n) for c in participating(tr, fed)]
        if ups:
            w = weighted_average([u for u, _ in ups], [n for _, n in ups])
    return w


def run_stocfl(fed, tr, cfg):
    """StoCFL (Zeng et al., Neural Networks 2025), following the paper's
    Algorithm 1 under full participation in the first round.

    Clustering: each client uploads the normalised full-batch gradient of its
    loss at the anchor w0 (the paper's anchor is a random initialisation; here
    it is the common initial model, which every client already holds). Clients
    are merged greedily by cosine similarity of cluster representations while the
    maximum similarity exceeds ``stocfl_tau`` (default 0.5). The number of
    clusters is therefore determined by tau; ``cfg["K"]`` is not used.

    Training: bi-level. A global model omega and one model theta_k per cluster.
    A participating client in cluster k runs one local epoch on theta_k with the
    proximal term lam * (theta - omega) (lam = ``stocfl_lambda``, default 0.05)
    and one local epoch on omega. The server averages omega and each theta_k with
    data-size weights over the participants.

    Deviations: the proximal term is added to the gradient of the local SGD
    step (the same optimiser as the other methods, no momentum); the anchor for
    the gradient representation is w0 instead of a random initialisation; the
    number of clusters is not given to the algorithm, it follows from tau.
    """
    tau = cfg.get("stocfl_tau", 0.5)
    lam = cfg.get("stocfl_lambda", 0.05)
    w0 = tr.get()
    # stocfl_warmup > 0 computes the gradient representation at a FedAvg model
    # after that many rounds (the same reference model DisCo-CFL uses) instead of
    # the initial anchor; those rounds count towards the budget.
    T_w = cfg.get("stocfl_warmup", 0)
    T_w = int(cfg["warmup"] if T_w == "warmup" else T_w)
    w_rep = _fedavg(tr, fed, w0, T_w) if T_w else w0
    G = []
    for c in fed.clients:
        g = tr.mean_grad(w_rep, c.x, c.y)
        G.append(g / g.norm().clamp_min(1e-12))
    labels = _stocfl_cluster(G, tau)
    K = int(labels.max()) + 1
    omega = w_rep.clone()
    theta = {k: w_rep.clone() for k in range(K)}
    for _ in range(cfg["rounds"] - T_w):
        omegas, thetas = [], {}
        for c in participating(tr, fed):
            k = int(labels[c.cid])
            ti = _local_train_prox(tr, theta[k], c.x, c.y, omega, lam, epochs=1)
            oi = tr.local_train(omega.clone(), c.x, c.y, epochs=1)
            omegas.append((oi, c.n))
            thetas.setdefault(k, []).append((ti, c.n))
        if omegas:
            omega = weighted_average([v for v, _ in omegas], [n for _, n in omegas])
        for k, lst in thetas.items():
            theta[k] = weighted_average([v for v, _ in lst], [n for _, n in lst])
    models = {k: theta[k] for k in range(K)}
    return {"assign": labels, "models": models, "cluster_floats": len(fed.clients) * tr.dim,
            "extra_local_epochs_per_round": 2}


# ---------------------------------------------------------------------------
# FedCCFA
# ---------------------------------------------------------------------------

def run_fedccfa(fed, tr, cfg):
    """FedCCFA (Chen et al., arXiv:2410.18478; reference code
    github.com/Chen-Junbao/FedCCFA, file entities/FedCCFA.py), ported to the common budget.

    The model is split into a shared feature extractor (all parameters except the
    final linear layer) and a personal classifier per client. Each round every
    participant (i) trains a balanced classifier on a class-balanced subset of its
    data (extractor from the server, classifier from the initialisation, used
    only for clustering), (ii) trains its personal classifier for one epoch and
    then its extractor for one epoch with CE + w_i * L_proto, where w_i is the
    label entropy divided by ``fedccfa_gamma`` and L_proto aligns features with
    the class anchors of its cluster (cosine logits, temperature
    ``fedccfa_temperature``, active from round ``fedccfa_align_start``), and (iii)
    computes its local class anchors. The server averages the extractors, and
    for each class clusters the balanced classifiers with DBSCAN on the MADD
    distance (mean absolute difference of cosine similarities to the other
    participants). Members of a non-trivial cluster share the mean of their local
    class-c classifier rows and class-c anchors.

    Deviations: all parameters use the same plain SGD with ``tr.lr`` and no
    momentum (the paper uses 0.1 for the classifier, 0.01 with momentum for the
    extractor). Missing anchors (classes not held by any member of the cluster
    so far) are zero vectors, which give a cosine of 0. The partition is taken
    from the last round in which a client participated (the last round when the
    participation is full). ``fedccfa_balanced_epochs`` (default 5) is the
    number of epochs on the balanced subset.
    """
    C, Fd, hd = _head_shape(tr.model)
    N = len(fed.clients)
    rounds = cfg["rounds"]
    gamma = cfg.get("fedccfa_gamma", 20.0)
    temp = cfg.get("fedccfa_temperature", 0.1)
    T_s = cfg.get("fedccfa_align_start", max(1, rounds // 10))
    E_bal = cfg.get("fedccfa_balanced_epochs", 5)
    eps = cfg.get("fedccfa_eps", 0.1)

    w0 = tr.get()
    ext = w0[:-hd].clone()                        # global feature extractor
    phi0 = w0[-hd:].clone()                       # initial classifier
    P = [_rows(phi0, C, Fd).clone() for _ in range(N)]        # personal classifiers
    A = [torch.zeros(C, Fd, device=w0.device) for _ in range(N)]  # class anchors
    last = [None] * N
    floats = 0

    for r in range(rounds):
        part = participating(tr, fed)
        ids = [c.cid for c in part]
        res = {}
        for c in part:
            i = c.cid
            y_np = c.y.cpu().numpy()
            counts = np.bincount(y_np, minlength=C).astype(float)
            p = counts / counts.sum()
            nz = p[p > 0]
            H = float(-(nz * np.log(nz)).sum())
            pw = H / gamma
            held = counts > 0

            # (2) balanced classifier, used only for clustering
            k_bal = int(min(5, counts[held].min()))
            sub = np.concatenate([tr.rng.choice(np.where(y_np == cl)[0], k_bal, replace=False)
                                  for cl in np.where(held)[0]])
            sub = torch.from_numpy(sub).to(c.x.device)
            bal = _sgd(tr, torch.cat([ext, phi0]), c.x[sub], c.y[sub], "clf", epochs=E_bal)
            bal_rows = _rows(bal[-hd:], C, Fd)

            # (3) local training: personal classifier, then extractor with alignment
            vec = torch.cat([ext, _flat(P[i], Fd)])
            vec = _sgd(tr, vec, c.x, c.y, "clf", epochs=1)
            proto = (A[i], temp, pw) if r >= T_s else None
            vec = _sgd(tr, vec, c.x, c.y, "ext", epochs=1, proto=proto)
            loc_rows = _rows(vec[-hd:], C, Fd)

            # (4) local class anchors with the trained extractor
            tr.set(vec.clone())
            f = _feat_all(tr.model, c.x)
            loc_anc = torch.zeros(C, Fd, device=f.device)
            for cl in np.where(held)[0]:
                loc_anc[cl] = f[c.y == cl].mean(0)
            res[i] = dict(ext=vec[:-hd], rows=loc_rows, anc=loc_anc, held=held,
                          bal=bal_rows, n=c.n)

        # server: extractor average, then per-class DBSCAN on the balanced classifiers
        ext = weighted_average([res[i]["ext"] for i in ids], [res[i]["n"] for i in ids])
        m = len(ids)
        labels = np.zeros((C, m), int)
        for cl in range(C):
            V = torch.stack([res[i]["bal"][cl] for i in ids])
            if m < 3:
                D = np.zeros((m, m))
            else:
                Vn = F.normalize(V, dim=1)
                S = (Vn @ Vn.T).clamp(-1, 1).double().cpu().numpy()
                D = np.zeros((m, m))
                for a in range(m):
                    # MADD: |S[a,q] - S[b,q]| summed over q, minus the terms q = a, q = b
                    tot = np.abs(S[a][None, :] - S).sum(1) - 2 * np.abs(1 - S[a])
                    D[a] = np.clip(tot, 0, None) / (m - 2)
                D = (D + D.T) / 2
                np.fill_diagonal(D, 0.0)
            labels[cl] = DBSCAN(eps=eps, min_samples=1, metric="precomputed").fit_predict(D)
            for L in np.unique(labels[cl]):
                mem = np.where(labels[cl] == L)[0]
                new_row = torch.stack([res[ids[q]]["rows"][cl] for q in mem]).mean(0)
                holders = [q for q in mem if res[ids[q]]["held"][cl]]
                new_anc = (torch.stack([res[ids[q]]["anc"][cl] for q in holders]).mean(0)
                           if holders else None)
                for q in mem:
                    i = ids[q]
                    P[i][cl] = new_row
                    if new_anc is not None:
                        A[i][cl] = new_anc
        for q, i in enumerate(ids):
            last[i] = tuple(int(v) for v in labels[:, q])
        floats += m * (C * (Fd + 1) + C * Fd)

    keys = {}
    partition = np.zeros(N, int)
    for i in range(N):
        key = last[i] if last[i] is not None else ("unseen", i)
        partition[i] = keys.setdefault(key, len(keys))
    models = {c.cid: torch.cat([ext, _flat(P[c.cid], Fd)]) for c in fed.clients}
    return {"assign": np.arange(N), "models": models, "partition_assign": partition,
            "cluster_floats": int(floats)}


# ---------------------------------------------------------------------------
# FedDAG (clustering stage only, "FedDAG-C")
# ---------------------------------------------------------------------------

def _minmax_offdiag(M):
    N = M.shape[0]
    off = ~np.eye(N, dtype=bool)
    lo, hi = M[off].min(), M[off].max()
    out = np.zeros(M.shape, float)
    if hi > lo:
        out[off] = (M[off] - lo) / (hi - lo)
    return out


def run_feddag(fed, tr, cfg):
    """Clustering stage of FedDAG (Pramanik et al., arXiv:2602.23504), "FedDAG-C".

    Only the clustering signal is reproduced; the training stage of FedDAG (dual
    encoder and cross-cluster graph) is not. The true K is given (``cfg["K"]``)
    instead of FedDAG's own K selection, whose constants are not specified.

    1. Warm-up: every client trains locally from w0 for ``feddag_tg`` (default 2)
       epochs; Delta_i = w_i - w0.
    2. Gradient similarity: angles (degrees) between Delta_i and Delta_j restricted
       to a shared random index set of size ceil(0.02 * dim) (seeded with cfg["seed"]).
    3. Class-wise data subspaces: for every class held by a client, the top
       p_c = max(1, ceil(0.01 n_c)) left singular vectors of its class-c data
       matrix (raw inputs ``x_raw`` if available, otherwise ``x``). The angle
       V'_ijc is the minimal principal angle between the two subspaces. It is 90
       when the class is on exactly one client and 0 when on neither. The
       frequency weight W'_ijc (log count ratio, min-max scaled to [1-0.1, 1+0.1];
       1+0.1 when a count is zero) multiplies it, and V_ij is the mean over the C
       classes.
    4. Fusion: G and V are min-max scaled to [0, 1] over off-diagonal entries and
       combined as A = 0.5 G + 0.5 V. The paper learns per-client fusion weights with
       an entropy-minimising MLP; here the weight is fixed.
    5. Clustering: average-linkage agglomerative clustering on A with K clusters.
    6. Training: the data-size-weighted average of the warm-up models initialises
       one model per cluster, followed by cluster-wise FedAvg for the remaining
       rounds.

    cluster_floats counts, per client, the sparsified gradient coordinates plus
    p_c * input_dim for each class it holds.
    """
    C = fed.num_classes
    N = len(fed.clients)
    K = cfg["K"]
    t_g = cfg.get("feddag_tg", 2)
    delta = 0.1
    eps_w = 1.0 + 1e-6
    w0 = tr.get()
    weights = [c.n for c in fed.clients]

    # 1. warm-up
    local = [tr.local_train(w0.clone(), c.x, c.y, epochs=t_g) for c in fed.clients]
    Delta = torch.stack([w - w0 for w in local])

    # 2. sparsified gradient-angle matrix
    m_idx = int(math.ceil(0.02 * tr.dim))
    idx = torch.from_numpy(np.random.default_rng(cfg["seed"]).choice(
        tr.dim, m_idx, replace=False)).to(Delta.device)
    U = F.normalize(Delta[:, idx], dim=1)
    G = torch.rad2deg(torch.arccos((U @ U.T).clamp(-1, 1))).double().cpu().numpy()

    # 3. class-wise subspaces
    counts = np.zeros((N, C))
    bases = [[None] * C for _ in range(N)]
    floats_cls = 0
    for c in fed.clients:
        xr = c.x_raw if getattr(c, "x_raw", None) is not None else c.x
        X = xr.flatten(1).double().cpu().numpy()
        d = X.shape[1]
        y_np = c.y.cpu().numpy()
        for cl in range(C):
            sel = np.where(y_np == cl)[0]
            counts[c.cid, cl] = len(sel)
            if not len(sel):
                continue
            Ucl, _, _ = np.linalg.svd(X[sel].T, full_matrices=False)
            p_c = max(1, int(math.ceil(0.01 * len(sel))))
            bases[c.cid][cl] = Ucl[:, :min(p_c, Ucl.shape[1])]
            floats_cls += p_c * d

    Vp = np.zeros((C, N, N))
    Wp = np.full((C, N, N), 1.0 + delta)
    raw = []
    for cl in range(C):
        for i in range(N):
            for j in range(i + 1, N):
                bi, bj = bases[i][cl], bases[j][cl]
                if bi is not None and bj is not None:
                    s = min(1.0, float(np.abs(bi.T @ bj).max()))
                    Vp[cl, i, j] = Vp[cl, j, i] = math.degrees(math.acos(s))
                    a = math.log(counts[i, cl] + eps_w)
                    b = math.log(counts[j, cl] + eps_w)
                    raw.append((cl, i, j, max(a, b) / min(a, b)))
                elif bi is None and bj is None:
                    Vp[cl, i, j] = Vp[cl, j, i] = 0.0
                else:
                    Vp[cl, i, j] = Vp[cl, j, i] = 90.0
    if raw:
        ws = np.array([r[3] for r in raw])
        lo, hi = ws.min(), ws.max()
        for (cl, i, j, w) in raw:
            wn = (1 - delta) + 2 * delta * (w - lo) / (hi - lo) if hi > lo else 1.0
            Wp[cl, i, j] = Wp[cl, j, i] = wn
    V = (Vp * Wp).mean(0)

    # 4. fusion
    A = 0.5 * _minmax_offdiag(G) + 0.5 * _minmax_offdiag(V)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0.0)

    # 5. clustering
    assign = np.zeros(N, int) if K == 1 else AgglomerativeClustering(
        n_clusters=K, metric="precomputed", linkage="average").fit_predict(A)

    # 6. training
    init = weighted_average(local, weights)
    models = _cluster_fedavg(tr, fed, _init_clusters(init, assign), assign,
                             cfg["rounds"] - t_g)
    floats = N * m_idx + floats_cls
    return {"assign": assign, "models": models, "cluster_floats": int(floats)}
