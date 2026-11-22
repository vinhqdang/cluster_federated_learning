"""Figure: clustering quality of DisCo-CFL against the estimated separation of the
true groups, one point per scenario (mean over seeds), with the individual runs in the
background. Writes results/figures/separation.pdf/.png and prints summary statistics.

usage: python experiments/fig_separation.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "results", "figures")
TAU = 0.9

PRETRAINED = {"pacs", "pacs_qs", "officehome", "c100_rot", "c100_swap", "c10r_rot", "c10r_swap"}
TEXT = {"agnews_swap"}
# label text and offset in points
LABELS = {"svhn_swap": ("SVHN (rot. and swap)", (6, 6)), "cifar_rot": ("CIFAR-10 rot.", (6, -2)),
          "cifar_swap": ("CIFAR-10 swap", (-8, 7)), "officehome": ("Office-Home", (5, 4)),
          "c100_rot": ("CIFAR-100 rot.", (6, 7)), "c10r_rot": ("CIFAR-10 rot. (ResNet)", (-12, 15)),
          "agnews_swap": ("AG News", (5, -10)), "emnist_swap": ("EMNIST swap", (5, -10)),
          "angle15": (r"Rotation 15$^\circ$", (5, 4)), "c10r_swap": ("CIFAR-10 swap (ResNet)", (-30, -14)),
          "c100_swap": ("CIFAR-100 swap", (-66, -12)), "pacs": ("PACS", (5, 3))}


def category(s):
    return "text" if s in TEXT else ("pretrained" if s in PRETRAINED else "small")


def load():
    runs = defaultdict(list)
    for d in ("raw", "raw_real"):
        for f in glob.glob(os.path.join(ROOT, "results", "r1", d, "*__DisCo__s*.json")):
            r = json.load(open(f))
            dg = r.get("diag") or {}
            dh = dg.get("delta_hat")
            if dh is None or not np.isfinite(dh) or r.get("k_true", 1) < 2:
                continue
            runs[r["scenario"]].append((float(dh), float(r["ari"])))
    return runs


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = load()
    style = {"small": dict(c="#2a78d6", marker="o", label="small CNN, trained from scratch"),
             "pretrained": dict(c="#eb6834", marker="s", label="pretrained ResNet-18"),
             "text": dict(c="#1baf7a", marker="^", label="text (frozen sentence encoder)")}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.axvline(1 - TAU, color="#52514e", lw=1.0, ls="--", zorder=1)
    ax.text(1 - TAU + 0.006, 1.0, r"$1-\tau$", color="#52514e", fontsize=8, va="top")
    means = {}
    for s, v in runs.items():
        a = np.array(v)
        st = style[category(s)]
        ax.scatter(a[:, 0], a[:, 1], s=8, c=st["c"], alpha=0.18, lw=0, marker=st["marker"], zorder=2)
        means[s] = a.mean(0)
    for cat, st in style.items():
        pts = np.array([m for s, m in means.items() if category(s) == cat])
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=34, c=st["c"], marker=st["marker"], edgecolors="#0b0b0b",
                       linewidths=0.6, label=st["label"], zorder=3)
    for s, (x, y) in means.items():
        if s in LABELS:
            text, off = LABELS[s]
            ax.annotate(text, (x, y), xytext=off, textcoords="offset points", fontsize=6.5, color="#0b0b0b")
    ax.set_xlabel(r"estimated separation $\hat\Delta$ of the true groups")
    ax.set_ylabel("ARI of DisCo-CFL")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.02, max(m[0] for m in means.values()) + 0.05)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "separation.pdf"))
    fig.savefig(os.path.join(FIG, "separation.png"), dpi=200)
    # summary statistics
    from scipy.stats import spearmanr
    xs = np.array([m[0] for m in means.values()])
    ys = np.array([m[1] for m in means.values()])
    rho, p = spearmanr(xs, ys)
    print(f"{len(means)} scenarios, {sum(len(v) for v in runs.values())} runs; Spearman rho={rho:.2f} (p={p:.1e})")
    for thr in (0.1, 0.15, 0.2):
        pred = xs > thr
        ok = ys >= 0.5
        print(f"rule 'delta_hat > {thr}' predicts ARI>=0.5 in {np.mean(pred == ok):.2f} of scenarios; "
              f"misclassified: {sorted(s for s, m in means.items() if (m[0] > thr) != (m[1] >= 0.5))}")


if __name__ == "__main__":
    main()
