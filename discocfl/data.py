"""Federated data generation with controlled non-IID heterogeneity.

Every client belongs to one latent *concept group*. A group is defined by a
feature transformation (image rotation, i.e. concept shift on features) and a
label permutation (concept shift on targets). On top of the group structure,
clients can additionally have target distribution skew (Dirichlet label
proportions) and quantity skew (log-normal dataset sizes). Only the concept
groups are the ground-truth clusters: label and quantity skew alone should not
split clients into different models.
"""
from __future__ import annotations

import gzip
import os
import pickle
import tarfile
from dataclasses import dataclass, field

import numpy as np
import torch

DATA_ROOT = os.environ.get("DISCO_DATA", "/home/user/data")

_MEAN_STD = {
    "mnist": ((0.1307,), (0.3081,)),
    "fmnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
}


def _read_idx(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def _load_idx_dataset(folder):
    def pick(stem):
        for suffix in ("", ".gz"):
            p = os.path.join(folder, "raw", stem + suffix)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(os.path.join(folder, "raw", stem))

    xtr = _read_idx(pick("train-images-idx3-ubyte"))
    ytr = _read_idx(pick("train-labels-idx1-ubyte"))
    xte = _read_idx(pick("t10k-images-idx3-ubyte"))
    yte = _read_idx(pick("t10k-labels-idx1-ubyte"))
    return xtr[:, None], ytr, xte[:, None], yte


def _load_cifar10(root):
    folder = os.path.join(root, "cifar-10-batches-py")
    if not os.path.isdir(folder):
        # extract to a private directory and rename atomically, so that parallel
        # workers never see a half-extracted folder
        tmp = os.path.join(root, f".cifar_extract_{os.getpid()}")
        with tarfile.open(os.path.join(root, "cifar-10-python.tar.gz")) as tar:
            tar.extractall(tmp)
        try:
            os.rename(os.path.join(tmp, "cifar-10-batches-py"), folder)
        except OSError:
            pass  # another worker finished first
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def batch(name):
        with open(os.path.join(folder, name), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32), np.array(d[b"labels"])

    parts = [batch(f"data_batch_{i}") for i in range(1, 6)]
    xtr = np.concatenate([p[0] for p in parts])
    ytr = np.concatenate([p[1] for p in parts])
    xte, yte = batch("test_batch")
    return xtr, ytr, xte, yte


def _load_svhn(root):
    from scipy.io import loadmat

    def part(name):
        d = loadmat(os.path.join(root, f"svhn_{name}_32x32.mat"))
        x = np.transpose(d["X"], (3, 2, 0, 1))
        y = d["y"].reshape(-1).astype(np.int64) % 10   # label 10 encodes digit 0
        return x, y

    xtr, ytr = part("train")
    xte, yte = part("test")
    return xtr, ytr, xte, yte


_CACHE: dict = {}


RAW_LOADERS = {}


def load_dataset(name: str):
    """Return normalised (x_train, y_train, x_test, y_test) torch tensors."""
    if name in _CACHE:
        return _CACHE[name]
    if name in RAW_LOADERS:               # raw uint8 images (featurised later)
        _CACHE[name] = RAW_LOADERS[name]()
        return _CACHE[name]
    if name == "mnist":
        xtr, ytr, xte, yte = _load_idx_dataset(os.path.join(DATA_ROOT, "MNIST"))
    elif name == "fmnist":
        xtr, ytr, xte, yte = _load_idx_dataset(os.path.join(DATA_ROOT, "FashionMNIST"))
    elif name == "cifar10":
        xtr, ytr, xte, yte = _load_cifar10(DATA_ROOT)
    elif name == "svhn":
        xtr, ytr, xte, yte = _load_svhn(DATA_ROOT)
    else:
        raise ValueError(name)
    mean, std = _MEAN_STD[name]
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)

    def prep(x):
        return (torch.from_numpy(np.array(x)).float() / 255.0 - mean) / std

    out = (prep(xtr), torch.from_numpy(ytr.astype(np.int64)),
           prep(xte), torch.from_numpy(yte.astype(np.int64)))
    _CACHE[name] = out
    return out


@dataclass
class Group:
    rotation: int = 0                 # multiples of 90 degrees
    perm: list | None = None          # label permutation (None = identity)
    angle: float = 0.0                # additional rotation in degrees (any value)

    def apply(self, x, y):
        if self.rotation % 4:
            x = torch.rot90(x, k=self.rotation % 4, dims=(2, 3))
        if self.angle:
            from torchvision.transforms.functional import rotate
            x = rotate(x, self.angle)
        if self.perm is not None:
            y = torch.as_tensor(self.perm, dtype=torch.long)[y]
        return x, y

    def key(self):
        """Feature transformation identifier (used to define conflicts)."""
        return (self.rotation % 4, round(float(self.angle), 6))


@dataclass
class Client:
    cid: int
    group: int
    x: torch.Tensor
    y: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    label_prop: np.ndarray = field(repr=False, default=None)

    @property
    def n(self):
        return len(self.y)


@dataclass
class Federation:
    clients: list
    groups: list
    group_test: list      # per group: (x, y) class-balanced test set
    num_classes: int
    in_shape: tuple
    dataset: str

    @property
    def true_labels(self):
        return np.array([c.group for c in self.clients])


def swap_perm(num_classes, pairs):
    perm = list(range(num_classes))
    for a, b in pairs:
        perm[a], perm[b] = perm[b], perm[a]
    return perm


def make_groups(kind: str, num_classes: int = 10):
    """Named concept-group configurations used in the experiments."""
    if kind == "rotation":
        return [Group(rotation=r) for r in range(4)]
    if kind == "swap":
        return [Group(), Group(perm=swap_perm(num_classes, [(0, 1), (2, 3)])),
                Group(perm=swap_perm(num_classes, [(4, 5), (6, 7)])),
                Group(perm=swap_perm(num_classes, [(8, 9), (1, 7)]))]
    if kind == "mixed":
        p = swap_perm(num_classes, [(0, 1), (2, 3)])
        return [Group(0), Group(0, p), Group(2), Group(2, p)]
    if kind == "single":
        return [Group()]
    if kind.startswith("angle"):          # angleXX: 4 groups rotated by 0, X, 2X, 3X degrees
        a = float(kind[5:])
        return [Group(angle=k * a) for k in range(4)]
    if kind.startswith("bigswap"):        # bigswapK: K groups, each swapping 10 random class pairs
        K = int(kind[7:])
        rng = np.random.default_rng(4321)
        gs = [Group()]
        for _ in range(K - 1):
            c = rng.permutation(num_classes)[:20]
            gs.append(Group(perm=swap_perm(num_classes, [(int(c[2 * i]), int(c[2 * i + 1]))
                                                         for i in range(10)])))
        return gs
    if kind.startswith("perm"):           # permK: K groups, each swapping two random class pairs
        K = int(kind[4:])
        rng = np.random.default_rng(1234)
        gs, seen = [Group()], set()
        while len(gs) < K:
            c = rng.permutation(num_classes)[:4]
            pairs = [tuple(sorted(c[:2])), tuple(sorted(c[2:]))]
            key = tuple(sorted(pairs))
            if key in seen:
                continue
            seen.add(key)
            gs.append(Group(perm=swap_perm(num_classes, [tuple(int(v) for v in pr) for pr in pairs])))
        return gs
    raise ValueError(kind)


def _sample_sizes(rng, n_clients, mean_size, quantity_sigma, min_size):
    if quantity_sigma <= 0:
        return np.full(n_clients, mean_size, dtype=int)
    raw = rng.lognormal(mean=0.0, sigma=quantity_sigma, size=n_clients)
    raw = raw / raw.mean() * mean_size
    return np.clip(raw.round().astype(int), min_size, 8 * mean_size)


def _draw(rng, pools, prop, n):
    counts = rng.multinomial(n, prop)
    idx = []
    for c, k in enumerate(counts):
        if k:
            idx.append(rng.choice(pools[c], size=k, replace=len(pools[c]) < k))
    return np.concatenate(idx) if idx else np.zeros(0, dtype=int)


def make_federation(dataset="fmnist", groups="rotation", clients_per_group=10,
                    mean_size=400, alpha=0.3, quantity_sigma=0.0, min_size=30,
                    test_size=300, group_test_size=2000, seed=0,
                    group_counts=None):
    """Build a federation.

    alpha: Dirichlet concentration of per-client label proportions (None = uniform).
    quantity_sigma: log-normal sigma of client sizes (0 = equal sizes).
    group_counts: optional explicit number of clients per group (imbalanced groups).
    """
    rng = np.random.default_rng(seed)
    xtr, ytr, xte, yte = load_dataset(dataset)
    num_classes = int(ytr.max()) + 1
    gs = make_groups(groups, num_classes) if isinstance(groups, str) else groups
    counts = group_counts or [clients_per_group] * len(gs)
    ytr_np, yte_np = ytr.numpy(), yte.numpy()
    tr_pools = [np.where(ytr_np == c)[0] for c in range(num_classes)]
    te_pools = [np.where(yte_np == c)[0] for c in range(num_classes)]
    n_clients = sum(counts)
    sizes = _sample_sizes(rng, n_clients, mean_size, quantity_sigma, min_size)

    clients = []
    cid = 0
    for g, (grp, cnt) in enumerate(zip(gs, counts)):
        for _ in range(cnt):
            if alpha is None:
                prop = np.full(num_classes, 1.0 / num_classes)
            else:
                prop = rng.dirichlet(np.full(num_classes, alpha))
            itr = _draw(rng, tr_pools, prop, int(sizes[cid]))
            ite = _draw(rng, te_pools, prop, test_size)
            x, y = grp.apply(xtr[itr], ytr[itr])
            xt, yt = grp.apply(xte[ite], yte[ite])
            clients.append(Client(cid, g, x, y, xt, yt, prop))
            cid += 1

    group_test = []
    for grp in gs:
        idx = rng.choice(len(yte_np), size=min(group_test_size, len(yte_np)), replace=False)
        group_test.append(grp.apply(xte[idx], yte[idx]))
    return Federation(clients, gs, group_test, num_classes, tuple(xtr.shape[1:]), dataset)
