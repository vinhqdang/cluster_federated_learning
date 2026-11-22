"""Calibration of the certificate standard errors on a simulation grid
(PREREGISTRATION_R1_ROUND3.md, section 3).

For every cell (h samples per half, M cluster members, C classes, case) we draw
synthetic signatures with a known population similarity and compare the
jackknife standard error of the client-to-cluster bottleneck similarity with
the spread of the estimate over draws.

usage: python experiments/cert_grid.py [--draws 200] [--workers 4]
"""
from __future__ import annotations

import argparse
import itertools
import json
import zlib
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from discocfl.disco import client_to_cluster_jk, compute_similarity  # noqa: E402
from discocfl.synthetic import concept_vectors, synthetic_signatures  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "r1", "clustering", "cert_grid.json")
P, LAM = 512, 0.5
H = [10, 20, 50, 200]
M = [3, 6, 12]
C = [2, 5, 10, 20]
CASES = [("same", 0.0), ("diff", 0.1), ("diff", 0.3)]


def cell(args):
    h, m, c, case, delta, draws = args
    s, se, truth = [], [], []
    for d in range(draws):
        seed = zlib.crc32(repr((h, m, c, case, delta, d)).encode())
        rng = np.random.default_rng(seed)
        if case == "same":
            mu = concept_vectors(1, c, P, 0.0, {}, rng)
            groups = [0] * (m + 1)
        else:
            mu = concept_vectors(2, c, P, delta, {1: [0]}, rng)
            groups = [1] + [0] * m
        cs = [list(range(c))] * (m + 1)
        sig = synthetic_signatures(groups, cs, h, mu, 1.0, rng)
        ref = synthetic_signatures(groups, cs, 10 ** 6, mu, 1.0, np.random.default_rng(seed + 1))
        r = client_to_cluster_jk(compute_similarity(sig), 0, range(1, m + 1), LAM)
        t = client_to_cluster_jk(compute_similarity(ref), 0, range(1, m + 1), LAM)["s"]
        if np.isfinite(r["se"]):
            s.append(r["s"]), se.append(r["se"]), truth.append(t)
    s, se, truth = map(np.array, (s, se, truth))
    err = s - truth
    bias = err.mean()
    return dict(h=h, m=m, c=c, case=case, delta=delta, n=int(len(s)),
                sd_over_se=float(s.std(ddof=1) / se.mean()),
                bias_in_se=float(bias / se.mean()),
                coverage=float(np.mean(np.abs(err) <= 2 * se)),
                coverage_debiased=float(np.mean(np.abs(err - bias) <= 2 * se)),
                mean_se=float(se.mean()), mean_truth=float(truth.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(h, m, c, case, delta, a.draws)
            for h, m, c, (case, delta) in itertools.product(H, M, C, CASES)]
    with Pool(a.workers) as p:
        rows = p.map(cell, jobs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=1)
    ratio = np.array([r["sd_over_se"] for r in rows])
    cov = np.array([r["coverage"] for r in rows])
    print(f"cells: {len(rows)}")
    print(f"H-S1 share of cells with SD/se in [0.7, 1.5]: {np.mean((ratio >= 0.7) & (ratio <= 1.5)):.2f}"
          f" (holds if >= 0.8)")
    print(f"H-S2 share of cells with coverage >= 0.9: {np.mean(cov >= 0.9):.2f} (holds if >= 0.8)")


if __name__ == "__main__":
    main()
