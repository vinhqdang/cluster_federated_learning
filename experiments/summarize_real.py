"""Tables for the realistic benchmarks (results/raw_real -> paper/table_real_*.tex)."""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "results", "raw_real")
sys.path.insert(0, os.path.dirname(__file__))

ORDER = ["FedAvg", "Local", "FedAvg-FT", "IFCA", "FeSEM", "MTCFL", "FL+HC", "PACFL", "DisCo", "Oracle"]
NAMES = {"pacs": "PACS (4 domains, 7 classes)", "pacs_qs": "PACS + quantity skew",
         "officehome": "Office-Home (4 domains, 65 classes)",
         "c100_rot": "CIFAR-100, rotations", "c100_swap": "CIFAR-100, label swaps",
         "femnist": "FEMNIST (100 writers)"}
COLS = [("pc_acc_mean", "Acc", True), ("pc_acc_p10", "Acc@10\\%", True), ("pc_worst_group_acc", "Worst-grp", True),
        ("pc_acc_concept_mean", "Concept", True), ("ari", "ARI", False), ("k_found", "$K$", False),
        ("conflict_merge_rate", "Conflict\\%", True)]


def load():
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RAW, "*.json")))]


def agg(rs, key):
    v = np.array([r[key] for r in rs], float)
    return v.mean(), v.std()


def table(rows, scenarios, name, caption_cols=COLS):
    by = defaultdict(list)
    for r in rows:
        by[(r["scenario"], r["method"])].append(r)
    methods = [m for m in ORDER if any((s, m) in by for s in scenarios)]
    lines = []
    for s in scenarios:
        if not any((s, m) in by for m in methods):
            continue
        lines.append(r"\multicolumn{%d}{l}{\textit{%s}}\\" % (len(caption_cols) + 1, NAMES.get(s, s)))
        best = {}
        for key, _, pct in caption_cols:
            if key in ("k_found",):
                continue
            vals = {m: round((100 if pct else 1) * agg(by[(s, m)], key)[0], 1 if pct else 2)
                    for m in methods if (s, m) in by and m != "Oracle"}
            if vals:
                b = (min if key == "conflict_merge_rate" else max)(vals.values())
                best[key] = {m for m, v in vals.items() if v == b}
        for m in methods:
            if (s, m) not in by:
                continue
            cells = []
            for key, _, pct in caption_cols:
                mu, sd = agg(by[(s, m)], key)
                if key == "k_found":
                    c = f"{mu:.1f}"
                elif pct:
                    c = f"{100 * mu:.1f}"
                else:
                    c = f"{mu:.2f}"
                if m in best.get(key, ()):
                    c = r"\textbf{%s}" % c
                cells.append(c)
            n = len(by[(s, m)])
            lines.append(f"{m} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    head = ("\\scriptsize\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{l" + "c" * len(caption_cols) + "}\n\\toprule\n"
            "Method & " + " & ".join(c[1] for c in caption_cols) + " \\\\\n\\midrule\n")
    open(os.path.join(ROOT, "paper", f"table_{name}.tex"), "w").write(
        head + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def significance(rows, scenarios):
    from scipy.stats import wilcoxon
    by = {(r["scenario"], r["method"], r["seed"]): r for r in rows if r["scenario"] in scenarios}
    out = []
    for m in ORDER:
        if m in ("DisCo", "Oracle"):
            continue
        for key, lab in [("pc_acc_mean", "Acc"), ("pc_acc_concept_mean", "Concept"), ("pc_worst_group_acc", "Worst-grp")]:
            d = [by[k][key] - by[(k[0], m, k[2])][key] for k in by
                 if k[1] == "DisCo" and (k[0], m, k[2]) in by]
            if len(d) >= 5 and any(x != 0 for x in d):
                w = sum(x > 0.0005 for x in d); l = sum(x < -0.0005 for x in d)
                p = wilcoxon(d, alternative="greater").pvalue
                out.append((m, lab, len(d), w, len(d) - w - l, l, p))
    return out


if __name__ == "__main__":
    rows = load()
    print(len(rows), "runs")
    table(rows, ["pacs", "officehome"], "real_domains")
    table(rows, ["c100_swap", "c100_rot", "pacs_qs"], "real_c100")
    table(rows, ["femnist"], "real_femnist")
    for t in significance(rows, {"pacs", "officehome", "c100_swap", "c100_rot", "pacs_qs"}):
        print(t)
