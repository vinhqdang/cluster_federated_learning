"""Compact tables for the two-column TNNLS version (reads ../results/raw)."""
import collections
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "..", "results", "raw", "*.json"))]
by = collections.defaultdict(list)
for r in rows:
    by[(r["scenario"], r["method"])].append(r)

M = ["FedAvg", "Local", "FedAvg-FT", "IFCA", "FeSEM", "MTCFL", "FL+HC", "PACFL", "DisCo"]
NAME = {"MTCFL": "CFL", "DisCo": r"\textbf{DisCo-CFL}"}


def mean(sc, m, k):
    v = by[(sc, m)]
    return float(np.mean([x[k] for x in v])) if v else None


def acc_concept_table(scens, heads, fname, extra=()):
    cols = "l" + "cc" * len(scens)
    out = [r"\begin{tabular}{" + cols + "}", r"\toprule",
           " & " + " & ".join(r"\multicolumn{2}{c}{" + h + "}" for h in heads) + r" \\",
           "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(len(scens))),
           "Method & " + " & ".join(["Acc", "Con."] * len(scens)) + r" \\", r"\midrule"]
    methods = M + list(extra) + ["Oracle"]
    best = {}
    for sc in scens:
        for k in ("pc_acc_mean", "pc_acc_concept_mean"):
            vals = {m: round(100 * mean(sc, m, k), 1) for m in M + list(extra) if mean(sc, m, k) is not None}
            best[(sc, k)] = max(vals.values())
    for m in methods:
        if m == "Oracle":
            out.append(r"\midrule")
        cells = []
        for sc in scens:
            for k in ("pc_acc_mean", "pc_acc_concept_mean"):
                v = mean(sc, m, k)
                if v is None:
                    cells.append("--")
                    continue
                s = f"{100 * v:.1f}"
                if m != "Oracle" and round(100 * v, 1) == best[(sc, k)]:
                    s = r"\textbf{" + s + "}"
                cells.append(s)
        out.append(NAME.get(m, m) + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(HERE, fname), "w").write("\n".join(out) + "\n")


def clustering_table(fname):
    groups = [("FMNIST", ["rot", "swap", "rot_qs", "mixed_qs", "minority"]),
              ("MNIST", ["mnist_rot", "mnist_swap"]), ("SVHN", ["svhn_rot", "svhn_swap"]),
              ("CIFAR-10", ["cifar_rot", "cifar_swap"])]
    meths = ["IFCA", "FeSEM", "MTCFL", "FL+HC", "PACFL", "DisCo"]
    out = [r"\begin{tabular}{l" + "cc" * len(groups) + "}", r"\toprule",
           " & " + " & ".join(r"\multicolumn{2}{c}{" + g + "}" for g, _ in groups) + r" \\",
           "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(len(groups))),
           "Method & " + " & ".join(["ARI", "Conf."] * len(groups)) + r" \\", r"\midrule"]
    for m in meths:
        cells = []
        for _, scs in groups:
            ari = np.mean([x["ari"] for s in scs for x in by[(s, m)]])
            cf = np.mean([x["conflict_merge_rate"] for s in scs for x in by[(s, m)]])
            cells += [f"{ari:.2f}", f"{100 * cf:.1f}"]
        out.append(NAME.get(m, m) + " & " + " & ".join(cells) + r" \\")
    out.append(r"DisCo-T30 & -- & -- & -- & -- & -- & -- & " +
               f"{np.mean([x['ari'] for s in ['cifar_rot', 'cifar_swap'] for x in by[(s, 'DisCo-T30')]]):.2f} & "
               f"{100 * np.mean([x['conflict_merge_rate'] for s in ['cifar_rot', 'cifar_swap'] for x in by[(s, 'DisCo-T30')]]):.1f}" + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(HERE, fname), "w").write("\n".join(out) + "\n")


def sweep_table(fname):
    S = [(r"$\theta{=}15^\circ$", "angle15"), (r"$\theta{=}30^\circ$", "angle30"), (r"$\theta{=}45^\circ$", "angle45"),
         ("$K{=}2$", "perm2"), ("$K{=}8$", "perm8"), ("part.\\ rot.", "rot_part30"), ("part.\\ swap", "swap_part30"),
         ("$N{=}200$", "rot_n200")]
    base = ["FedAvg", "IFCA", "FeSEM", "MTCFL", "FL+HC", "PACFL", "FedAvg-FT", "Local"]
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r"Setting & ARI & $\hat K$ & Acc & Con. & Best base.\ Acc / Con. & Oracle \\", r"\midrule"]
    for lab, sc in S:
        av = [m for m in base if by[(sc, m)]]
        ba = max(mean(sc, m, "pc_acc_mean") for m in av)
        bc = max(mean(sc, m, "pc_acc_concept_mean") for m in av)
        out.append(f"{lab} & {mean(sc, 'DisCo', 'ari'):.2f} & {mean(sc, 'DisCo', 'k_found'):.1f} & "
                   f"{100 * mean(sc, 'DisCo', 'pc_acc_mean'):.1f} & {100 * mean(sc, 'DisCo', 'pc_acc_concept_mean'):.1f} & "
                   f"{100 * ba:.1f} / {100 * bc:.1f} & {100 * mean(sc, 'Oracle', 'pc_acc_mean'):.1f}" + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(HERE, fname), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    acc_concept_table(["rot", "swap", "rot_qs", "mixed_qs", "label", "minority"],
                      ["Rotation", "Label swap", "Rot.+QS", "Mixed+QS", "Label only", "Minority"], "tab_fmnist.tex")
    acc_concept_table(["mnist_rot", "mnist_swap", "svhn_rot", "svhn_swap", "cifar_rot", "cifar_swap"],
                      ["MNIST rot.", "MNIST swap", "SVHN rot.", "SVHN swap", "CIFAR rot.", "CIFAR swap"],
                      "tab_other.tex", extra=("DisCo-T30",))
    clustering_table("tab_cluster.tex")
    sweep_table("tab_sweep.tex")
    print("ok")
