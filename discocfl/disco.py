"""DisCo-CFL: Disattenuated Class-cOnditional Clustered Federated Learning.

Server-side CFL in which every client uploads, once, a compact *class-conditional
gradient signature* computed at a common reference model:

    for every class c the client holds, the samples of class c are split into two
    random halves A and B, and the client sends the count-sketched mean gradients
    a_c = S g(A), b_c = S g(B)            (S: shared random count-sketch, R^d -> R^p)

From the two halves the server estimates the reliability (signal-to-noise) of
each class signature and applies Spearman's correction for attenuation,

    rho_ij^c = cos(m_i^c, m_j^c) / sqrt(R_i^c R_j^c),   m = (a + b)/2,
    R^c = 2 r^c / (1 + r^c),                              r^c = cos(a_c, b_c),

which is a consistent estimator of the cosine between the *population*
class-conditional expected gradients of the two clients. Consequences:

* label-distribution skew does not change the class-conditional gradients, so it
  does not split clients (only concept shift does);
* quantity skew and differential-privacy noise only lower the reliability R,
  which the correction removes, so the similarity scale is calibrated
  (1 = same concept) independently of the number of samples;
* a single threshold tau on this calibrated scale determines the number of
  clusters automatically, independently of cluster size (minority groups are not
  absorbed by large ones).

The module also produces the transparency / accountability outputs: per-class
explanations of why clusters differ, a heterogeneity-type diagnosis, per-client
assignment certificates with jackknife confidence intervals, an audit log of all
merge decisions, and low-evidence / anomaly flags.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


# ----------------------------------------------------------------------------
# Client side: signatures
# ----------------------------------------------------------------------------
class CountSketch:
    """Linear count-sketch R^d -> R^p; E<Sx, Sy> = <x, y>."""

    def __init__(self, d, p, seed=0, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        self.p = p
        self.bucket = torch.randint(0, p, (d,), generator=g).to(device)
        self.sign = (torch.randint(0, 2, (d,), generator=g).float() * 2 - 1).to(device)

    def __call__(self, v):  # v: (..., d)
        out = torch.zeros(*v.shape[:-1], self.p, device=v.device)
        return out.index_add_(-1, self.bucket, v * self.sign)


@dataclass
class SignatureConfig:
    sketch_dim: int = 1024
    min_per_class: int = 4          # classes with fewer samples are not reported
    max_per_class: int = 200        # subsample large classes (compute budget)
    class_conditional: bool = True  # ablation: False = one signature over all data
    clip: float = 0.5               # per-sample clipping norm in sketch space (0 = none)
    dp_sigma: float = 0.0           # Gaussian noise multiplier (0 = no DP)
    dp_clip: float = 0.5            # per-sample clipping norm used with DP
    mode: str = "full"              # "full": gradient of all parameters; "head": final linear layer
    seed: int = 0


def _per_sample_sketched_grads(model, params, x, y, sketch, chunk=64):
    names = list(params.keys())

    def loss_fn(p, xb, yb):
        out = functional_call(model, p, (xb[None],))
        return F.cross_entropy(out, yb[None])

    g_fn = vmap(grad(loss_fn), in_dims=(None, 0, 0))
    outs = []
    for s in range(0, len(y), chunk):
        g = g_fn(params, x[s:s + chunk], y[s:s + chunk])
        flat = torch.cat([g[n].reshape(g[n].shape[0], -1) for n in names], 1)
        outs.append(sketch(flat))
    return torch.cat(outs)


@torch.no_grad()
def _per_sample_head_grads(model, x, y, sketch, chunk=256):
    """Per-sample gradient of the cross-entropy w.r.t. the final linear layer
    (weights and bias), in closed form: r (x) h and r, with r = softmax - onehot
    and h the penultimate features. Used for large models, where the gradient of
    all parameters is too expensive; the theory holds for any fixed map g(x, c)."""
    outs = []
    fc = model.fc2 if hasattr(model, "fc2") else model.fc
    for s in range(0, len(y), chunk):
        xb, yb = x[s:s + chunk], y[s:s + chunk]
        h = model.features(xb) if hasattr(model, "features") else F.relu(model.fc1(
            F.max_pool2d(F.relu(model.conv2(F.max_pool2d(F.relu(model.conv1(xb)), 2))), 2).flatten(1)))
        r = F.softmax(fc(h), 1) - F.one_hot(yb, fc.out_features).float()
        g = torch.cat([(r[:, :, None] * h[:, None, :]).flatten(1), r], 1)
        outs.append(sketch(g))
    return torch.cat(outs)


def client_signature(trainer, ref_vec, client, sketch, cfg: SignatureConfig, rng):
    """Returns {class: (a, b, n_class)} with a, b the sketched half-mean gradients."""
    trainer.set(ref_vec)
    model = trainer.model
    model.eval()
    params = {k: v.detach() for k, v in model.named_parameters()}
    y = client.y.cpu().numpy()
    groups = {int(c): np.where(y == c)[0] for c in np.unique(y)} if cfg.class_conditional \
        else {-1: np.arange(len(y))}
    sig = {}
    for c, idx in groups.items():
        if len(idx) < cfg.min_per_class:
            continue
        idx = rng.permutation(idx)[: cfg.max_per_class]
        h = len(idx) // 2
        halves = (idx[:h], idx[h:2 * h])
        vecs = []
        for part in halves:
            if cfg.mode == "head":
                g = _per_sample_head_grads(model, client.x[part], client.y[part], sketch)
            else:
                g = _per_sample_sketched_grads(model, params, client.x[part], client.y[part], sketch)
            if cfg.clip > 0 and cfg.dp_sigma <= 0:
                # bounded per-sample signatures: robust to heavy-tailed gradients;
                # the expected clipped gradient is still a functional of P(X|Y=c)
                norms = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
                g = g * torch.clamp(cfg.clip / norms, max=1.0)
            if cfg.dp_sigma > 0:
                norms = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
                g = g * torch.clamp(cfg.dp_clip / norms, max=1.0)
                v = g.mean(0)
                # replace-one sensitivity of a mean of h clipped vectors is 2C/h
                v = v + torch.randn(v.shape, device=v.device) * cfg.dp_sigma * 2 * cfg.dp_clip / len(part)
            else:
                v = g.mean(0)
            vecs.append(v.cpu())
        sig[c] = (vecs[0], vecs[1], len(idx))
    return sig


def gaussian_dp_epsilon(sigma, delta=1e-5):
    """Smallest eps such that the Gaussian mechanism with noise multiplier sigma
    (noise std = sigma * sensitivity) is (eps, delta)-DP, using the exact analytic
    characterisation of Balle & Wang (2018)."""
    if sigma <= 0:
        return math.inf
    from scipy.stats import norm

    def dlt(eps):
        return norm.cdf(0.5 / sigma - eps * sigma) - math.exp(eps) * norm.cdf(-0.5 / sigma - eps * sigma)

    lo, hi = 0.0, 1.0
    while dlt(hi) > delta:
        hi *= 2
        if hi > 1e4:
            return math.inf
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if dlt(mid) > delta else (lo, mid)
    return hi


# ----------------------------------------------------------------------------
# Server side: calibrated similarity
# ----------------------------------------------------------------------------
@dataclass
class SimilarityResult:
    S: np.ndarray            # (N, N) evidence-weighted disattenuated similarity
    E: np.ndarray            # (N, N) evidence (sum of weights)
    rho: np.ndarray          # (C, N, N) per-class disattenuated similarity (nan = missing)
    W: np.ndarray            # (C, N, N) per-class weights
    R: np.ndarray            # (N, C) reliabilities
    classes: list
    # half-level data used by the jackknife standard errors (None for old objects)
    Ah: np.ndarray | None = None       # (N, C, p) unit half vector a/||a|| (0 if ||a|| = 0 or class missing)
    Bh: np.ndarray | None = None       # (N, C, p) unit half vector b/||b||
    U: np.ndarray | None = None        # (N, C, p) unit mean (a+b)/(2||a+b||)
    r_half: np.ndarray | None = None   # (N, C) max(cos(a, b), 0); 0 when R = 0 or class missing


def compute_similarity(signatures, disattenuate=True, r_min=0.02):
    # Edge cases (explicit):
    # * classes a client does not report are missing: all of its entries stay 0 and
    #   W = 0 for every pair involving that class, so they carry no evidence;
    # * R <= 0 (non-positive split-half reliability, or below r_min) is treated as
    #   "no evidence" for that (client, class): R = 0, hence W = 0 on that class;
    # * a zero-norm half gives cos 0 (torch clamps the denominator), hence R = 0;
    #   its unit vector is set to the zero vector instead of dividing by zero.
    n = len(signatures)
    classes = sorted({c for s in signatures for c in s})
    ci = {c: k for k, c in enumerate(classes)}
    C = len(classes)
    p = next(iter(next(s for s in signatures if s).values()))[0].numel()
    U = torch.zeros(n, C, p)
    Ah = torch.zeros(n, C, p)
    Bh = torch.zeros(n, C, p)
    R = np.zeros((n, C))
    r_half = np.zeros((n, C))
    for i, sig in enumerate(signatures):
        for c, (a, b, _) in sig.items():
            k = ci[c]
            r = F.cosine_similarity(a, b, dim=0).item()
            rel = 2 * r / (1 + r) if r > 0 else 0.0
            m = (a + b) / 2
            U[i, k] = m / m.norm().clamp_min(1e-12)
            Ah[i, k] = a / a.norm().clamp_min(1e-12)
            Bh[i, k] = b / b.norm().clamp_min(1e-12)
            # R <= 0 (r <= 0 or rel < r_min): no evidence for this class
            R[i, k] = rel if rel >= r_min else 0.0
            r_half[i, k] = max(r, 0.0) if R[i, k] > 0 else 0.0
    rho = np.full((C, n, n), np.nan)
    W = np.zeros((C, n, n))
    for k in range(C):
        cos = (U[:, k] @ U[:, k].T).numpy()
        rk = R[:, k]
        present = rk > 0
        both = np.outer(present, present)
        if disattenuate:
            denom = np.sqrt(np.outer(rk, rk)).clip(1e-12)
            val = np.clip(cos / denom, -1.0, 1.0)
            w = np.outer(rk, rk)
        else:
            val = cos
            w = both.astype(float)
        rho[k][both] = val[both]
        W[k] = np.where(both, w, 0.0)
    for k in range(C):
        np.fill_diagonal(W[k], 0.0)
    E = W.sum(0)
    num = np.nansum(np.where(W > 0, W * rho, 0.0), axis=0)
    S = np.where(E > 0, num / np.maximum(E, 1e-12), np.nan)
    np.fill_diagonal(S, 1.0)
    return SimilarityResult(S, E, rho, W, R, classes, Ah=Ah.numpy().astype(np.float64),
                            Bh=Bh.numpy().astype(np.float64), U=U.numpy().astype(np.float64),
                            r_half=r_half)


# ----------------------------------------------------------------------------
# Server side: threshold clustering with class-wise bottleneck linkage
# ----------------------------------------------------------------------------
def _class_stats(sim):
    num = np.where(sim.W > 0, sim.W * np.nan_to_num(sim.rho), 0.0)
    return num, sim.W.copy()


def _linkage(num, den, lam, e_min, mode="min"):
    """Linkage from per-class accumulated statistics.

    rho_c = (num_c + lam) / (den_c + lam) shrinks each class estimate towards 1
    ("no evidence of a difference") in proportion to the missing evidence; the
    linkage is the bottleneck (minimum) over classes, because two groups share a
    concept only if *every* class-conditional gradient agrees. mode="mean" gives
    the evidence-weighted average used in the ablation.
    """
    tot = den.sum(0)
    if mode == "min":
        L = ((num + lam) / (den + lam)).min(0)
    else:
        L = num.sum(0) / np.maximum(tot, 1e-12)
    return np.where(tot >= e_min, L, -np.inf)


def threshold_cluster(sim, tau, active=None, lam=0.2, e_min=0.05, mode="min",
                      order="evidence"):
    """Agglomerative clustering on the calibrated similarity: repeatedly merge the
    pair of clusters with the highest linkage while it is >= tau. Pairs without
    shared evidence never merge directly. Returns labels and the audit log."""
    n = sim.S.shape[0]
    active = np.arange(n) if active is None else np.asarray(active)
    num, den = _class_stats(sim)
    alive = np.zeros(n, bool)
    alive[active] = True
    members = {int(i): [int(i)] for i in active}
    log = []
    while alive.sum() > 1:
        L = _linkage(num, den, lam, e_min, mode)
        mask = np.outer(alive, alive)
        np.fill_diagonal(mask, False)
        L = np.where(mask, L, -np.inf)
        eligible = L >= tau
        if not eligible.any():
            break
        # the bottleneck linkage decides *whether* two clusters may merge; the
        # order in which eligible merges happen follows the shared evidence, so
        # well-supported groups form first and ambiguous clients (little evidence
        # on the discriminative classes) join them instead of bridging groups
        prio = den.sum(0) if order == "evidence" else L
        a, b = np.unravel_index(np.argmax(np.where(eligible, prio, -np.inf)), L.shape)
        best = L[a, b]
        a, b = int(min(a, b)), int(max(a, b))
        pc = (num[:, a, b] + lam) / (den[:, a, b] + lam)
        log.append({"merge": [list(members[a]), list(members[b])], "linkage": float(best),
                    "bottleneck_class": int(sim.classes[int(np.argmin(pc))]),
                    "evidence": float(den[:, a, b].sum())})
        members[a] = members[a] + members.pop(b)
        for arr in (num, den):
            arr[:, a, :] += arr[:, b, :]
            arr[:, :, a] += arr[:, :, b]
            arr[:, b, :] = 0
            arr[:, :, b] = 0
            arr[:, a, a] = 0
        alive[b] = False
    labels = -np.ones(n, dtype=int)
    for new_id, (_, mem) in enumerate(sorted(members.items(), key=lambda kv: -len(kv[1]))):
        labels[mem] = new_id
    return labels, log


def client_to_cluster(sim: SimilarityResult, i, members, lam=0.2):
    """Bottleneck similarity of client i to a set of clients, the bottleneck class
    and a standard error computed from the spread over member pairs."""
    members = [j for j in members if j != i]
    if not members:
        return np.nan, np.nan, 0.0
    W = sim.W[:, i, members]
    rho = np.nan_to_num(sim.rho[:, i, members])
    num_c = (W * rho).sum(1)
    den_c = W.sum(1)
    if den_c.sum() <= 0:
        return np.nan, np.nan, 0.0
    vals = (num_c + lam) / (den_c + lam)
    c = int(np.argmin(vals))
    s = float(vals[c])
    if den_c[c] > 0 and (W[c] > 0).sum() > 1:
        w = W[c]
        mu = num_c[c] / den_c[c]
        se = math.sqrt((w ** 2 * (rho[c] - mu) ** 2).sum()) / den_c[c]
    else:
        se = np.nan
    return s, se, float(den_c.sum())


def _bottleneck_from(W, rho, lam):
    """W, rho: (C, m) per-class weights and similarities of one client to m others.
    Returns (s, class index, evidence); s = nan when there is no evidence at all."""
    num_c = (W * rho).sum(1)
    den_c = W.sum(1)
    if den_c.sum() <= 0:
        return np.nan, -1, 0.0
    vals = (num_c + lam) / (den_c + lam)
    c = int(np.argmin(vals))
    return float(vals[c]), c, float(den_c.sum())


def client_to_cluster_jk(sim: SimilarityResult, i, members, lam=0.2):
    """Bottleneck similarity of client i to `members` with a standard error that
    respects the dependence structure: all pairs (i, j) share client i.

    se_members : delete-one-member jackknife over j in members (the spread of the
                 estimate across members, with the pairs of j removed together);
    se_self    : uncertainty of client i's own signature. Client i is represented
                 by a single half (A, then B) with the single-half reliability
                 r_half[i, c]; |s_A - s_B| / 2 estimates the extra noise from i's
                 own estimation (the full-client s averages both halves);
    se         : sqrt(se_members^2 + se_self^2), ignoring NaN components.
    Classes are "missing" for a pair when r_half[i, c] <= 0 or R[j, c] <= 0.
    Returns a dict with s, se, se_members, se_self, evidence, bottleneck_class
    (the class label, as in client_to_cluster; None when undefined)."""
    if sim.Ah is None or sim.U is None or sim.r_half is None:
        raise ValueError("similarity object lacks half-level data; recompute with compute_similarity")
    nan = float("nan")
    out = {"s": nan, "se": nan, "se_members": nan, "se_self": nan, "evidence": 0.0,
           "bottleneck_class": None}
    members = np.asarray([j for j in members if j != i], dtype=int)
    if len(members) == 0:
        return out
    W = sim.W[:, i, members]                       # (C, m)
    rho = np.nan_to_num(sim.rho[:, i, members])    # (C, m)
    s, c, ev = _bottleneck_from(W, rho, lam)
    out["evidence"] = ev
    if np.isnan(s):
        return out
    out["s"] = s
    out["bottleneck_class"] = int(sim.classes[c])

    # (a) delete-one-member jackknife; leave-one-out values that are undefined
    # (no evidence left) are dropped and the factor uses the count actually used
    m = len(members)
    if m >= 2:
        loo = np.array([_bottleneck_from(W[:, np.arange(m) != t], rho[:, np.arange(m) != t], lam)[0]
                        for t in range(m)])
        loo = loo[np.isfinite(loo)]
        k = len(loo)
        if k >= 2:
            out["se_members"] = float(math.sqrt((k - 1) / k * ((loo - loo.mean()) ** 2).sum()))

    # (b) client i represented by one half only
    def half_s(H):                                  # H: (C, p) unit half vectors of client i
        cos_ij = np.einsum("cp,mcp->cm", H, sim.U[members])          # (C, m)
        r_i = sim.r_half[i][:, None]                                 # (C, 1)
        R_j = sim.R[members].T                                       # (C, m)
        ok = (r_i > 0) & (R_j > 0)                                   # class present for the pair
        denom = np.sqrt(np.where(ok, r_i * R_j, 1.0))
        rho_h = np.where(ok, np.clip(cos_ij / denom, -1.0, 1.0), 0.0)
        return _bottleneck_from(np.where(ok, W, 0.0), rho_h, lam)[0]

    sA, sB = half_s(sim.Ah[i]), half_s(sim.Bh[i])
    if np.isfinite(sA) and np.isfinite(sB):
        out["se_self"] = float(abs(sA - sB) / 2)

    comps = [x for x in (out["se_members"], out["se_self"]) if not np.isnan(x)]
    out["se"] = float(math.sqrt(sum(x * x for x in comps))) if comps else nan
    return out


# ----------------------------------------------------------------------------
# Full clustering procedure with FAT outputs
# ----------------------------------------------------------------------------
@dataclass
class DisCoResult:
    labels: np.ndarray
    sim: SimilarityResult
    audit_log: list
    certificates: list
    explanations: dict
    flags: dict = field(default_factory=dict)


def disco_cluster(signatures, tau=0.8, min_evidence=0.3, disattenuate=True, z=2.0,
                  lam=0.5, mode="min", order="evidence", small=1, attach=True):
    """attach=True (default): small clusters are attached to their closest cluster
    when not significantly separated, and low-evidence clients join their most
    similar cluster. attach=False: neither happens; every low-evidence client
    becomes its own singleton cluster (ids after the existing ones)."""
    sim = compute_similarity(signatures, disattenuate=disattenuate)
    n = len(signatures)
    evidence = sim.R.sum(1)                    # total reliability mass of the client
    low = np.where(evidence < min_evidence)[0]
    active = np.setdiff1d(np.arange(n), low)
    labels, log = threshold_cluster(sim, tau, active, lam=lam, mode=mode, order=order)
    if attach:
        labels, anomalies = _attach_small(sim, labels, tau, z, lam, small, log, active)
        attached_small = sorted({int(i) for e in log if "attach_small" in e for i in e["attach_small"]})
        # low-evidence clients are not allowed to shape the clusters; they are attached
        # to the most similar cluster (or the largest one if nothing can be said)
        K = labels.max() + 1
        for i in low:
            best, arg = -np.inf, 0
            for k in range(K):
                s, _, _ = client_to_cluster(sim, i, np.where(labels == k)[0], lam)
                if not np.isnan(s) and s > best:
                    best, arg = s, k
            labels[i] = arg
            log.append({"attach_low_evidence": int(i), "cluster": int(arg),
                        "similarity": None if best == -np.inf else float(best)})
        attached_low = [int(i) for i in low]
    else:
        anomalies, attached_small, attached_low = [], [], []
        K = labels.max() + 1
        for i in low:
            labels[i] = K
            log.append({"singleton_low_evidence": int(i), "cluster": int(K)})
            K += 1
    certs = certificates(sim, labels, tau, z, low, lam)
    expl = explain(sim, labels, tau, lam)
    sizes = np.bincount(labels)
    flags = {"low_evidence": low.tolist(), "anomalous": anomalies,
             "singletons": [int(np.where(labels == k)[0][0]) for k in range(len(sizes))
                            if sizes[k] == 1],
             "attached_small": attached_small, "attached_low": attached_low}
    return DisCoResult(labels, sim, log, certs, expl, flags)


def _relabel(labels):
    out = -np.ones_like(labels)
    ks = [k for k in np.unique(labels) if k >= 0]
    for new, k in enumerate(sorted(ks, key=lambda k: -(labels == k).sum())):
        out[labels == k] = new
    return out


def _attach_small(sim, labels, tau, z, lam, small, log, active):
    """Clusters with <= `small` members are attached to their closest cluster when
    the evidence does not *significantly* separate them from it (upper confidence
    bound >= tau); otherwise they are kept and flagged as anomalous."""
    anomalies = []
    if small <= 0:
        return labels, anomalies
    labels = labels.copy()
    active_set = set(int(i) for i in active)
    for k in sorted(set(labels[labels >= 0]), key=lambda k: (labels == k).sum()):
        mem = [int(i) for i in np.where(labels == k)[0] if int(i) in active_set]
        if not mem or len(mem) > small:
            continue
        best = (-np.inf, None, np.nan)
        for k2 in set(labels[labels >= 0]) - {k}:
            others = np.where(labels == k2)[0]
            if len(others) <= small:
                continue
            vals = [client_to_cluster_jk(sim, i, others, lam) for i in mem]
            s = np.nanmean([v["s"] for v in vals])
            se = np.nanmean([0.0 if np.isnan(v["se"]) else v["se"] for v in vals])
            if not np.isnan(s) and s > best[0]:
                best = (s, k2, se)
        s, k2, se = best
        if k2 is not None and s + z * se >= tau:
            labels[mem] = k2
            log.append({"attach_small": mem, "cluster": int(k2), "similarity": float(s),
                        "se": float(se)})
        else:
            anomalies.extend(mem)
            log.append({"anomalous": mem, "best_similarity": None if k2 is None else float(s)})
    return _relabel(labels), anomalies


def certificates(sim, labels, tau, z=2.0, low=(), lam=0.2):
    """Assignment certificate of each client: similarity to own cluster, to the
    closest other cluster, jackknife standard errors and a confidence verdict."""
    out = []
    K = labels.max() + 1
    for i in range(len(labels)):
        own = labels[i]
        r_own = client_to_cluster_jk(sim, i, np.where(labels == own)[0], lam)
        s_own, se_own, ev = r_own["s"], r_own["se"], r_own["evidence"]
        best_o, se_o, k_o = np.nan, np.nan, None
        for k in range(K):
            if k == own:
                continue
            r_k = client_to_cluster_jk(sim, i, np.where(labels == k)[0], lam)
            s, se = r_k["s"], r_k["se"]
            if not np.isnan(s) and (np.isnan(best_o) or s > best_o):
                best_o, se_o, k_o = s, se, k
        se_own_ = 0.0 if np.isnan(se_own) else se_own
        se_o_ = 0.0 if np.isnan(se_o) else se_o
        own_ok = np.isnan(s_own) or s_own - z * se_own_ >= tau
        other_ok = np.isnan(best_o) or best_o + z * se_o_ < tau
        verdict = "confident" if (own_ok and other_ok and i not in low) else "uncertain"
        if i in low:
            verdict = "low-evidence"
        out.append({"client": i, "cluster": int(own), "sim_own": _f(s_own), "se_own": _f(se_own),
                    "se_own_members": _f(r_own["se_members"]), "se_own_self": _f(r_own["se_self"]),
                    "closest_other": k_o, "sim_other": _f(best_o), "se_other": _f(se_o),
                    "margin": _f(s_own - best_o) if not (np.isnan(s_own) or np.isnan(best_o)) else None,
                    "evidence": float(ev), "verdict": verdict})
    return out


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or \
        (isinstance(x, np.floating) and np.isnan(x)) else float(x)


def explain(sim, labels, tau, lam=0.2):
    """Class-level explanation of the partition and heterogeneity-type diagnosis."""
    K = labels.max() + 1
    classes = sim.classes

    def per_class(A, B):
        res = {}
        for k, c in enumerate(classes):
            W = sim.W[k][np.ix_(A, B)]
            if W.sum() <= 0:
                continue
            rho = np.nan_to_num(sim.rho[k][np.ix_(A, B)])
            res[int(c)] = float(((W * rho).sum() + lam) / (W.sum() + lam))
        return res

    clusters = [np.where(labels == k)[0] for k in range(K)]
    within = {k: per_class(clusters[k], clusters[k]) for k in range(K)}
    pairs = {}
    for a in range(K):
        for b in range(a + 1, K):
            pc = per_class(clusters[a], clusters[b])
            div = sorted([c for c, v in pc.items() if v < tau])
            frac = len(div) / max(len(pc), 1)
            if not pc:
                kind = "undetermined (no shared classes)"
            elif frac >= 0.95:
                kind = "global concept shift (P(X|Y) differs for every shared class)"
            elif div:
                kind = "class-specific concept shift (P(Y|X) / P(X|Y) differs on classes %s)" % div
            else:
                kind = "no class-level divergence detected at the threshold"
            pairs[f"{a}-{b}"] = {"per_class": pc, "divergent_classes": div,
                                 "fraction_divergent": frac, "diagnosis": kind}
    if K == 1:
        overall = ("one cluster: no concept difference was detected at the reference model; "
                   "this is not evidence that no difference exists (see the identifiability diagnostic)")
    else:
        overall = f"{K} concept groups detected"
    return {"within": within, "between": pairs, "overall": overall}


def assign_newcomer(existing_signatures, labels, new_sig, tau, lam=0.2):
    """Assign a late-joining client using its signature at the same reference model.
    Returns (cluster, similarity) with cluster = -1 if it matches no cluster."""
    sigs = list(existing_signatures) + [new_sig]
    sim = compute_similarity(sigs)
    i = len(sigs) - 1
    best, arg = -np.inf, -1
    for k in range(labels.max() + 1):
        s, _, _ = client_to_cluster(sim, i, np.where(labels == k)[0], lam)
        if not np.isnan(s) and s > best:
            best, arg = s, k
    return (arg if best >= tau else -1), best
