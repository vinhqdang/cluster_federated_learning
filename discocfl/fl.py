"""Models and federated-training primitives shared by all algorithms."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=10, img=28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        s = ((img - 4) // 2 - 4) // 2
        self.fc1 = nn.Linear(32 * s * s, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


class Trainer:
    """Holds one model instance and moves flat parameter vectors in and out."""

    def __init__(self, fed, lr=0.05, batch_size=32, local_epochs=1, seed=0):
        c, h, _ = fed.in_shape
        torch.manual_seed(seed)
        self.arch = (c, fed.num_classes, h)
        self.model = SmallCNN(*self.arch)
        self.lr, self.bs, self.epochs = lr, batch_size, local_epochs
        self.rng = np.random.default_rng(seed + 12345)
        self.dim = self.get().numel()

    def get(self):
        return parameters_to_vector(self.model.parameters()).detach().clone()

    def set(self, vec):
        vector_to_parameters(vec, self.model.parameters())

    def init_vector(self, seed):
        """A fresh random initialisation (used by methods with several models)."""
        torch.manual_seed(seed)
        m = SmallCNN(*self.arch)
        return parameters_to_vector(m.parameters()).detach().clone()

    def local_train(self, vec, x, y, epochs=None):
        self.set(vec)
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.0)
        n = len(y)
        for _ in range(epochs or self.epochs):
            order = torch.from_numpy(self.rng.permutation(n))
            for s in range(0, n, self.bs):
                b = order[s:s + self.bs]
                opt.zero_grad()
                F.cross_entropy(self.model(x[b]), y[b]).backward()
                opt.step()
        return self.get()

    @torch.no_grad()
    def evaluate(self, vec, x, y):
        self.set(vec)
        self.model.eval()
        out = self.model(x)
        loss = F.cross_entropy(out, y).item()
        acc = (out.argmax(1) == y).float().mean().item()
        return acc, loss

    @torch.no_grad()
    def logits(self, vec, x):
        self.set(vec)
        self.model.eval()
        return self.model(x)

    def mean_grad(self, vec, x, y):
        """Gradient of the mean loss on (x, y) at parameters vec."""
        self.set(vec)
        self.model.eval()
        self.model.zero_grad()
        F.cross_entropy(self.model(x), y).backward()
        return torch.cat([p.grad.flatten() for p in self.model.parameters()]).clone()


def weighted_average(vecs, weights):
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()
    return (torch.stack(vecs) * w[:, None]).sum(0)


def participating(trainer, fed):
    """Clients taking part in this round (a random fraction when
    trainer.participation < 1, the same sampling for every method)."""
    frac = getattr(trainer, "participation", 1.0)
    if frac >= 1.0:
        return fed.clients
    m = max(1, int(round(frac * len(fed.clients))))
    idx = np.sort(trainer.rng.choice(len(fed.clients), m, replace=False))
    return [fed.clients[i] for i in idx]


def cluster_fedavg_round(trainer, fed, models, assign):
    """One synchronous round of FedAvg run independently inside each cluster.

    models: dict cluster_id -> parameter vector; assign: array client -> cluster.
    """
    updates = {}
    for c in participating(trainer, fed):
        k = int(assign[c.cid])
        updates.setdefault(k, []).append((trainer.local_train(models[k], c.x, c.y), c.n))
    new = dict(models)
    for k, lst in updates.items():
        new[k] = weighted_average([v for v, _ in lst], [n for _, n in lst])
    return new


def evaluate_assignment(trainer, fed, models, assign, prior_correction=False):
    """Per-client accuracy on local test data and on the concept-balanced test set.

    prior_correction: Bayes prior (logit) correction of the cluster model. For
    clients with the same class-conditionals the Bayes classifiers differ only by
    the label prior, so client i adds log p_i(y) - log p_k(y) to the logits of its
    cluster model k, where p_i is its own training label distribution and p_k the
    label distribution of the cluster's training data (a sum of label counts that
    can be obtained by secure aggregation). On the class-balanced concept test set
    the target prior is uniform, i.e. -log p_k(y) is added.
    """
    C = fed.num_classes
    counts = {c.cid: np.bincount(c.y.numpy(), minlength=C) + 1.0 for c in fed.clients}
    assign = np.asarray(assign)
    cl_counts = {}
    for c in fed.clients:
        k = int(assign[c.cid])
        cl_counts[k] = cl_counts.get(k, 0) + counts[c.cid] - 1.0
    log_pk = {k: torch.log(torch.tensor((v + 1.0) / (v + 1.0).sum(), dtype=torch.float32))
              for k, v in cl_counts.items()}
    local, balanced = [], []
    cache = {}
    for c in fed.clients:
        k = int(assign[c.cid])
        out = trainer.logits(models[k], c.x_test)
        if prior_correction:
            log_pi = torch.log(torch.tensor(counts[c.cid] / counts[c.cid].sum(), dtype=torch.float32))
            out = out + log_pi - log_pk[k]
        local.append((out.argmax(1) == c.y_test).float().mean().item())
        key = (k, c.group)
        if key not in cache:
            gx, gy = fed.group_test[c.group]
            og = trainer.logits(models[k], gx)
            if prior_correction:
                og = og - log_pk[k]
            cache[key] = (og.argmax(1) == gy).float().mean().item()
        balanced.append(cache[key])
    return np.array(local), np.array(balanced)
