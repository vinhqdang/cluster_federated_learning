"""Table: DisCo-CFL with the fixed 10-round warm-up against DisCo-AS (label-free stable
warm-up, pre-registered in experiments/PREREGISTRATION_R1_ROUND2.md), per scenario.
Writes paper/table_as.tex and prints the checks of the pre-registered hypotheses.

usage: python experiments/table_as.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SCEN = [("rot", "FMNIST rotation"), ("swap", "FMNIST label swap"), ("rot_qs", "FMNIST rotation + QS"),
        ("mixed_qs", "FMNIST mixed + QS"), ("label", "FMNIST label skew only"), ("minority", "FMNIST minority"),
        ("mnist_rot", "MNIST rotation"), ("mnist_swap", "MNIST label swap"),
        ("kmnist_rot", "KMNIST rotation"), ("kmnist_swap", "KMNIST label swap"),
        ("emnist_rot", "EMNIST rotation"), ("emnist_swap", "EMNIST label swap"),
        ("svhn_rot", "SVHN rotation"), ("svhn_swap", "SVHN label swap"),
        ("cifar_rot", "CIFAR-10 rotation"), ("cifar_swap", "CIFAR-10 label swap"),
        ("angle15", r"Rotation $15^\circ$"), ("angle30", r"Rotation $30^\circ$"), ("angle45", r"Rotation $45^\circ$"),
        ("perm2", "2 label permutations"), ("perm8", "8 label permutations"),
        ("rot_part30", "Rotation, 30\\% part."), ("swap_part30", "Label swap, 30\\% part."),
        ("rot_n200", "Rotation, 200 clients")]
NO_HARM = {"rot", "swap", "rot_qs", "mixed_qs", "label", "minority", "mnist_rot", "mnist_swap",
           "kmnist_rot", "kmnist_swap", "emnist_rot"}   # H-AS1 (F-MNIST, MNIST, KMNIST, EMNIST rotations)
HARD = {"svhn_rot", "svhn_swap", "cifar_rot", "cifar_swap"}   # H-AS2


def load(method):
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(ROOT, "results", "r1", "raw", f"*__{method}__s*.json")):
        r = json.load(open(f))
        out[r["scenario"]][r["seed"]] = r
    return out


def main():
    d, a = load("DisCo"), load("DisCo-AS")
    rows, checks = [], []
    for s, name in SCEN:
        seeds = sorted(set(d[s]) & set(a[s]))
        if not seeds:
            continue
        acc_d = np.array([100 * d[s][k]["pc_acc_mean"] for k in seeds])
        acc_a = np.array([100 * a[s][k]["pc_acc_mean"] for k in seeds])
        ari_d = np.array([d[s][k]["ari"] for k in seeds])
        ari_a = np.array([a[s][k]["ari"] for k in seeds])
        t = [a[s][k]["extra"].get("warmup_used") for k in seeds]
        diff = acc_a - acc_d
        rows.append(f"{name} & {len(seeds)} & {acc_d.mean():.1f} & {acc_a.mean():.1f} & {diff.mean():+.1f} & "
                    f"{ari_d.mean():.2f} & {ari_a.mean():.2f} & {np.mean(t):.0f} ({min(t)}--{max(t)}) \\\\")
        checks.append((s, diff.mean(), ari_a.mean() - ari_d.mean()))
    head = ("\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrcccccc}\n\\toprule\n"
            " & & \\multicolumn{3}{c}{Accuracy (\\%)} & \\multicolumn{2}{c}{ARI} & \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n"
            "Scenario & $n$ & DisCo & DisCo-AS & $\\Delta$ & DisCo & DisCo-AS & $T_0$ chosen (range) \\\\\n\\midrule\n")
    open(os.path.join(ROOT, "paper", "table_as.tex"), "w").write(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    harm = [(s, round(dacc, 2)) for s, dacc, _ in checks if s in NO_HARM]
    print("H-AS1 (|accuracy difference| <= 0.5 where the default works):")
    print("   violations:", [(s, x) for s, x in harm if x < -0.5], "| all:", harm)
    hard = [(s, round(dacc, 2), round(dari, 2)) for s, dacc, dari in checks if s in HARD]
    print("H-AS2 (higher mean ARI and accuracy on SVHN and CIFAR-10):", hard,
          "holds:", all(x > 0 and y > 0 for _, x, y in hard))


if __name__ == "__main__":
    main()
