"""Tables for the realistic benchmarks (results/raw_real -> paper/table_real_*.tex)."""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "results", "r1", "raw_real")
sys.path.insert(0, os.path.dirname(__file__))

ORDER = ["FedAvg", "Local", "FedAvg-FT", "IFCA", "FeSEM", "MTCFL", "FL+HC", "PACFL",
         "StoCFL", "StoCFL-W", "FedCCFA", "FedDAG-C", "DisCo-noAttach", "DisCo", "Oracle"]
VARIANT_ONLY = ("DisCo", "Oracle", "DisCo-noAttach")   # not compared as baselines in the tests
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
        if m in VARIANT_ONLY:
            continue
        for key, lab in [("pc_acc_mean", "Acc"), ("pc_acc_concept_mean", "Concept"), ("pc_worst_group_acc", "Worst-grp")]:
            d = [by[k][key] - by[(k[0], m, k[2])][key] for k in by
                 if k[1] == "DisCo" and (k[0], m, k[2]) in by]
            if len(d) >= 5 and any(x != 0 for x in d):
                w = sum(x > 0.0005 for x in d); l = sum(x < -0.0005 for x in d)
                p = wilcoxon(d, alternative="greater").pvalue
                out.append((m, lab, len(d), w, len(d) - w - l, l, p))
    return out



def signif_table(rows, scenarios, name):
    out = significance(rows, scenarios)
    by = defaultdict(dict)
    for m, lab, n, w, t, l, p in out:
        by[m][lab] = (w, t, l, p)
    lines = []
    for m in ORDER:
        if m not in by:
            continue
        cells = []
        for lab in ("Acc", "Concept", "Worst-grp"):
            w, t, l, p = by[m][lab]
            ps = "<0.001" if p < 0.001 else f"{p:.3f}"
            cells.append(f"{w}/{t}/{l} & {ps}")
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    head = ("\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lcccccc}\n\\toprule\n"
            " & \\multicolumn{2}{c}{Acc} & \\multicolumn{2}{c}{Concept} & \\multicolumn{2}{c}{Worst-grp} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "Baseline & W/T/L & $p$ & W/T/L & $p$ & W/T/L & $p$ \\\\\n\\midrule\n")
    open(os.path.join(ROOT, "paper", f"table_{name}.tex"), "w").write(
        head.replace("\\\\", "\\") + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def _mean_rows(rows, seeds):
    """Average the reported metrics of each (scenario, method) over the given seeds."""
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        if r["seed"] in seeds:
            g[(r["scenario"], r["method"])].append(r)
    out = {}
    for k, lst in g.items():
        out[k] = {m: float(np.mean([x[m] for x in lst])) for m in
                  ("pc_acc_mean", "pc_acc_concept_mean", "ari", "k_found", "conflict_merge_rate")}
        out[k]["n"] = len(lst)
    return out


def variants_table(rows, name, seeds=(0, 1, 2)):
    by = _mean_rows(rows, seeds)
    meths = [("FedAvg", "FedAvg"), ("FeSEM", "FeSEM"), ("Oracle", "Oracle"), ("DisCo", "DisCo-CFL (default)"),
             ("DisCo-full", "DisCo-CFL, full-layer signature"), ("DisCo-T30", "DisCo-CFL, 30 warm-up rounds")]
    lines = []
    for sc in ("officehome", "c100_rot"):
        lines.append(r"\multicolumn{6}{l}{\textit{%s}}\\" % NAMES[sc])
        for key, lab in meths:
            r = by.get((sc, key))
            if r:
                lines.append(f"{lab} & {100*r['pc_acc_mean']:.1f} & {100*r['pc_acc_concept_mean']:.1f} & "
                             f"{r['ari']:.2f} & {r['k_found']:.1f} & {100*r['conflict_merge_rate']:.1f} \\\\")
        lines.append(r"\midrule")
    head = ("\\scriptsize\n\\begin{tabular}{lccccc}\n\\toprule\n"
            "Method & Acc & Concept & ARI & $K$ & Conflict\\% \\\\\n\\midrule\n")
    open(os.path.join(ROOT, "paper", f"table_{name}.tex"), "w").write(
        head.replace("\\\\", "\\") + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def ablation_table(rows, name, scenario="c100_swap", seeds=(0, 1, 2)):
    by = {m: v for (sc, m), v in _mean_rows(rows, seeds).items() if sc == scenario}
    names = [("DisCo", "DisCo-CFL (full)"), ("DisCo-noDisatt", "no disattenuation (raw cosine)"),
             ("DisCo-noClass", "no class conditioning"), ("DisCo-mean", "mean instead of bottleneck linkage"),
             ("DisCo-linkorder", "linkage-ordered merging"), ("DisCo-noClip", "no per-sample clipping"),
             ("DisCo-noAttach", "no attachment stage")]
    lines = []
    for key, lab in names:
        r = by.get(key)
        if r:
            lines.append(f"{lab} & {100*r['pc_acc_mean']:.1f} & {100*r['pc_acc_concept_mean']:.1f} & "
                         f"{r['ari']:.2f} & {r['k_found']:.1f} & {100*r['conflict_merge_rate']:.1f} \\\\")
    head = ("\\scriptsize\n\\begin{tabular}{lccccc}\n\\toprule\n"
            "Variant & Acc & Concept & ARI & $K$ & Conflict\\% \\\\\n\\midrule\n")
    open(os.path.join(ROOT, "paper", f"table_{name}.tex"), "w").write(
        head.replace("\\\\", "\\") + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def _fix_headers():
    """The header rows of the generated tables must end with a double backslash."""
    for f in glob.glob(os.path.join(ROOT, "paper", "table_real_*.tex")):
        t = open(f).read()
        t = "\n".join(l + "\\" if l.endswith(" \\") and not l.endswith("\\\\") else l for l in t.split("\n"))
        open(f, "w").write(t)


if __name__ == "__main__":
    rows = load()
    print(len(rows), "runs")
    # the per-benchmark results are in paper/table_r1_ci_real.tex (experiments/stats_r1.py)
    fem = [r for r in rows if r["scenario"] == "femnist"]
    if fem:
        table(rows, ["femnist"], "real_femnist", caption_cols=[("pc_acc_mean", "Acc", True), ("pc_acc_p10", "Acc@10\\%", True),
                                                               ("pc_acc_concept_mean", "Pooled", True), ("k_found", "$K$", False)])
    variants_table(rows, "real_variants")
    ablation_table(rows, "real_ablation")
    _fix_headers()
