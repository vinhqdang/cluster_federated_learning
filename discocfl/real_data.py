"""Realistic federations: natural domain shift (PACS, Office-Home), CIFAR-100 with
synthetic concept groups, and FEMNIST with natural writers.

Images are resized once to ``size`` x ``size`` and stored as uint8. For the
ResNet-18 benchmarks every client's images are then mapped through the frozen part
of the network (see :mod:`discocfl.models`), so that all methods train the same
8.4M-parameter head on cached layer-3 feature maps.
"""
from __future__ import annotations

import io
import os
import tarfile
import pickle
import urllib.request

import numpy as np
import torch

from . import data as D
from .data import Client, Federation

ROOT = D.DATA_ROOT
HF = "https://huggingface.co/datasets/flwrlabs/{name}/resolve/main/data/{f}"
HF_FILES = {"pacs": ["train-00000-of-00001.parquet"],
            "office-home": [f"train-0000{i}-of-00003.parquet" for i in range(3)],
            "femnist": ["train-00000-of-00001.parquet"]}


def _download(url, path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def _resize(img, size):
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((l, t, l + size, t + size)))


def load_domain_dataset(name, size=112):
    """PACS / Office-Home as (x uint8 (N,3,S,S), y, domain id, class names, domain names)."""
    cache = os.path.join(ROOT, f"{name}_{size}.npz")
    if not os.path.exists(cache):
        import pyarrow.parquet as pq
        from PIL import Image
        xs, ys, ds = [], [], []
        for f in HF_FILES[name]:
            path = _download(HF.format(name=name, f=f), os.path.join(ROOT, name, f))
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=256):
                cols = batch.to_pydict()
                for im, lab, dom in zip(cols["image"], cols["label"], cols["domain"]):
                    xs.append(_resize(Image.open(io.BytesIO(im["bytes"])), size))
                    ys.append(lab)
                    ds.append(dom)
        doms = sorted(set(ds))
        np.savez(cache + ".tmp.npz", x=np.stack(xs).transpose(0, 3, 1, 2), y=np.array(ys),
                 d=np.array([doms.index(v) for v in ds]), domains=np.array(doms))
        os.replace(cache + ".tmp.npz", cache)
    z = np.load(cache)
    return torch.from_numpy(z["x"]), torch.from_numpy(z["y"].astype(np.int64)), \
        z["d"], [str(v) for v in z["domains"]]


class DomainGroup:
    """A natural domain; two clients conflict iff they come from different domains."""
    perm = None

    def __init__(self, name):
        self.name = name

    def key(self):
        return self.name


def _backbone(device, pretrained):
    from .models import FrozenBackbone
    return FrozenBackbone(pretrained).to(device)


def make_domain_federation(name="pacs", clients_per_domain=10, mean_size=200, alpha=0.3,
                           seed=0, size=112, device="cpu", pretrained=True, test_size=100,
                           group_test_per_class=40, domains=None, group_counts=None,
                           quantity_sigma=0.0, min_size=30):
    """One concept group per domain; Dirichlet label skew inside every client."""
    rng = np.random.default_rng(seed)
    x, y, dom, dnames = load_domain_dataset(name, size)
    keep = list(range(len(dnames))) if domains is None else [dnames.index(d) for d in domains]
    num_classes = int(y.max()) + 1
    y_np = y.numpy()
    bb = _backbone(device, pretrained)
    # per (domain, class): 80% train pool, 20% test pool
    train_pool, test_pool = {}, {}
    for d in keep:
        for c in range(num_classes):
            idx = rng.permutation(np.where((dom == d) & (y_np == c))[0])
            k = max(1, int(round(0.2 * len(idx)))) if len(idx) > 1 else 0
            test_pool[d, c], train_pool[d, c] = idx[:k], idx[k:]
    counts = group_counts or [clients_per_domain] * len(keep)
    sizes = D._sample_sizes(rng, sum(counts), mean_size, quantity_sigma, min_size)
    clients, cid = [], 0
    for g, (d, cnt) in enumerate(zip(keep, counts)):
        present = [c for c in range(num_classes) if len(train_pool[d, c]) and len(test_pool[d, c])]
        for _ in range(cnt):
            prop = np.zeros(num_classes)
            prop[present] = rng.dirichlet(np.full(len(present), alpha))
            itr = D._draw(rng, [train_pool[d, c] for c in range(num_classes)], prop, int(sizes[cid]))
            ite = D._draw(rng, [test_pool[d, c] for c in range(num_classes)], prop, test_size)
            clients.append(Client(cid, g, x[itr], y[itr], x[ite], y[ite], prop))
            cid += 1
    group_test = []
    for d in keep:
        idx = np.concatenate([rng.choice(test_pool[d, c], min(group_test_per_class, len(test_pool[d, c])),
                                         replace=False) for c in range(num_classes) if len(test_pool[d, c])])
        group_test.append((x[idx], y[idx]))
    fed = Federation(clients, [DomainGroup(dnames[d]) for d in keep], group_test, num_classes,
                     tuple(x.shape[1:]), name)
    return featurize_federation(fed, bb, device)


def featurize_federation(fed, bb, device):
    """Replace the raw uint8 images of every client by frozen layer-3 feature maps
    (the raw 32x32 version is kept for PACFL, which needs input-space subspaces)."""
    import torch.nn.functional as F
    for c in fed.clients:
        c.x_raw = F.adaptive_avg_pool2d(c.x.float(), 16).to(torch.uint8)
        c.x, c.x_test = bb.featurize(c.x), bb.featurize(c.x_test)
        c.y, c.y_test = c.y.to(device), c.y_test.to(device)
    fed.group_test = [(bb.featurize(gx), gy.to(device)) for gx, gy in fed.group_test]
    fed.in_shape = tuple(fed.clients[0].x.shape[1:])
    return fed


# ---------------------------------------------------------------------------
# CIFAR-100 (raw uint8, upsampled) -> reuse data.make_federation + featurize
# ---------------------------------------------------------------------------
def _load_cifar100_raw(size=96):
    import torch.nn.functional as F
    folder = os.path.join(ROOT, "cifar-100-python")
    if not os.path.isdir(folder):
        tgz = _download("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
                        os.path.join(ROOT, "cifar-100-python.tar.gz"))
        tmp = os.path.join(ROOT, f".c100_{os.getpid()}")
        with tarfile.open(tgz) as t:
            t.extractall(tmp)
        try:
            os.rename(os.path.join(tmp, "cifar-100-python"), folder)
        except OSError:
            pass

    def part(n):
        with open(os.path.join(folder, n), "rb") as f:
            d = pickle.load(f, encoding="latin1")
        raw = torch.from_numpy(d["data"].reshape(-1, 3, 32, 32))
        out = torch.empty(len(raw), 3, size, size, dtype=torch.uint8)
        for s0 in range(0, len(raw), 1000):       # chunked: the full float array needs > 5 GB
            chunk = F.interpolate(raw[s0:s0 + 1000].float(), size=size, mode="bilinear", align_corners=False)
            out[s0:s0 + 1000] = chunk.round().clamp(0, 255).to(torch.uint8)
        return out, torch.tensor(d["fine_labels"], dtype=torch.long)

    xtr, ytr = part("train")
    xte, yte = part("test")
    return xtr, ytr, xte, yte


D.RAW_LOADERS["cifar100"] = _load_cifar100_raw


def make_cifar100_federation(groups="rotation", device="cpu", pretrained=True, seed=0, **kw):
    fed = D.make_federation(dataset="cifar100", groups=groups, seed=seed, **kw)
    return featurize_federation(fed, _backbone(device, pretrained), device)


# ---------------------------------------------------------------------------
# FEMNIST: natural writers, no ground-truth groups
# ---------------------------------------------------------------------------
def make_femnist_federation(n_writers=100, min_samples=150, max_samples=400, seed=0, device="cpu"):
    import pyarrow.parquet as pq
    from PIL import Image
    cache = os.path.join(ROOT, "femnist_sub.npz")
    if not os.path.exists(cache):
        path = _download(HF.format(name="femnist", f=HF_FILES["femnist"][0]),
                         os.path.join(ROOT, "femnist", HF_FILES["femnist"][0]))
        t = pq.read_table(path, columns=["writer_id", "character"]).to_pandas()
        keep = t.groupby("writer_id").size()
        keep = keep[keep >= min_samples].index.tolist()
        rng = np.random.default_rng(0)
        chosen = set(rng.permutation(keep)[:600].tolist())
        xs, ys, ws = [], [], []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            cols = batch.to_pydict()
            for im, ch, w in zip(cols["image"], cols["character"], cols["writer_id"]):
                if w in chosen:
                    a = np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("L").resize((28, 28)))
                    xs.append(255 - a if a.mean() > 127 else a)
                    ys.append(ch)
                    ws.append(w)
        names = sorted(set(ws))
        np.savez(cache + ".tmp.npz", x=np.stack(xs), y=np.array(ys), w=np.array([names.index(v) for v in ws]))
        os.replace(cache + ".tmp.npz", cache)
    z = np.load(cache)
    x, y, w = z["x"], z["y"].astype(np.int64), z["w"]
    rng = np.random.default_rng(seed)
    writers = rng.permutation(np.unique(w))[:n_writers]
    clients = []
    for cid, wid in enumerate(writers):
        idx = rng.permutation(np.where(w == wid)[0])[:max_samples]
        nt = max(20, len(idx) // 5)
        te, tr = idx[:nt], idx[nt:]
        f = lambda i: ((torch.from_numpy(x[i]).float() / 255.0 - 0.1307) / 0.3081)[:, None].to(device)
        clients.append(Client(cid, 0, f(tr), torch.from_numpy(y[tr]).to(device),
                              f(te), torch.from_numpy(y[te]).to(device)))
    gx = torch.cat([c.x_test for c in clients[:20]])
    gy = torch.cat([c.y_test for c in clients[:20]])
    return Federation(clients, [DomainGroup("writers")], [(gx, gy)], 62, (1, 28, 28), "femnist")
