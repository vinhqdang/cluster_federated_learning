"""Effect of the warm-up length (quality of the reference model) on clustering.

usage: python experiments/warmup_study.py  ->  results/clustering/warmup.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))


def job(args):
    scen, warm, seed = args
    import clustering_study as cs
    from run import SCENARIOS
    from discocfl.disco import disco_cluster
    cs._setup()
    fed, tr, w = cs.prepare(cs.base(**SCENARIOS[scen]), seed, warmup=warm)
    ref_acc = float(np.mean([tr.evaluate(w, c.x_test, c.y_test)[0] for c in fed.clients]))
    r = disco_cluster(cs.signatures(fed, tr, w, seed))
    return dict(scenario=scen, warmup=warm, seed=seed, ref_acc=ref_acc, **cs.cluster_scores(fed, r.labels))


if __name__ == "__main__":
    jobs = [(s, wu, sd) for s in ["cifar_rot"] for wu in [10, 20, 30] for sd in range(3)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as p:
        rows = p.map(job, jobs)
    out = os.path.join(HERE, "..", "results", "clustering", "warmup.json")
    json.dump(rows, open(out, "w"), indent=1)
    for r in rows:
        print(r)
