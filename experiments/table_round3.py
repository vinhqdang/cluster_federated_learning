"""Tables and hypothesis checks for PREREGISTRATION_R1_ROUND3.md.

Writes paper/table_l02.tex, paper/table_refmodel.tex and paper/table_certgrid.tex
(each only when its results exist) and prints the checks of H-L1, H-L2, H-R1, H-R2,
H-S1 and H-S2.

usage: python experiments/table_round3.py
"""
import json
import os
from collections import defaultdict

import numpy as np

from table_as import ROOT, SCEN, load

CL = os.path.join(ROOT, "results", "r1", "clustering")
REF_SCEN = [("rot", "FMNIST rotation"), ("swap", "FMNIST label swap"), ("mixed_qs", "FMNIST mixed + QS"),
            ("label", "FMNIST label skew only"), ("mnist_rot", "MNIST rotation"),
            ("mnist_swap", "MNIST label swap")]


def write(name, body):
    open(os.path.join(ROOT, "paper", name), "w").write(body)


def lambda_table():
    d, l = load("DisCo"), load("DisCo-L02")
    rows, h1, dari = [], [], []
    for s, name in SCEN:
        seeds = sorted(set(d[s]) & set(l[s]))
        if not seeds:
            continue
        acc_d = np.array([100 * d[s][k]["pc_acc_mean"] for k in seeds])
        acc_l = np.array([100 * l[s][k]["pc_acc_mean"] for k in seeds])
        ari_d = np.array([d[s][k]["ari"] for k in seeds])
        ari_l = np.array([l[s][k]["ari"] for k in seeds])
        k_d = np.mean([d[s][k]["k_found"] for k in seeds])
        k_l = np.mean([l[s][k]["k_found"] for k in seeds])
        rows.append(f"{name} & {len(seeds)} & {acc_d.mean():.1f} & {acc_l.mean():.1f} & "
                    f"{acc_l.mean() - acc_d.mean():+.1f} & {ari_d.mean():.2f} & {ari_l.mean():.2f} & "
                    f"{k_d:.1f} & {k_l:.1f} \\\\")
        dari.append(abs(ari_l.mean() - ari_d.mean()))
        if ari_d.mean() >= 0.8:
            h1.append((s, round(acc_l.mean() - acc_d.mean(), 2)))
    if not rows:
        return
    head = ("\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrccccccc}\n\\toprule\n"
            " & & \\multicolumn{3}{c}{Accuracy (\\%)} & \\multicolumn{2}{c}{ARI} & \\multicolumn{2}{c}{$K$} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
            "Scenario & $n$ & $\\lambda=0.5$ & $\\lambda=0.02$ & $\\Delta$ & $\\lambda=0.5$ & $\\lambda=0.02$ & "
            "$\\lambda=0.5$ & $\\lambda=0.02$ \\\\\n\\midrule\n")
    write("table_l02.tex", head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print("H-L1 (|accuracy difference| <= 0.5 where the default recovers the groups):",
          "violations", [x for x in h1 if abs(x[1]) > 0.5], "| all", h1)
    print(f"H-L2 (mean |ARI difference| <= 0.05): {np.mean(dari):.3f} over {len(dari)} scenarios,",
          "holds" if np.mean(dari) <= 0.05 else "fails")


def refmodel_table():
    p = os.path.join(CL, "refmodel.json")
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["scenario"], r["ref"])].append(r)

    def m(s, ref, key):
        v = [r[key] for r in agg.get((s, ref), []) if r[key] is not None]
        return np.mean(v) if v else float("nan")

    out, h1, h2 = [], [], []
    for s, name in REF_SCEN:
        f = lambda x: "--" if not np.isfinite(x) else f"{x:.2f}"
        cells = [f(m(s, "full", "ari")), f(m(s, "half", "ari")), f(m(s, "half", "ari_in")),
                 f(m(s, "half", "ari_out")), f(m(s, "half", "k")),
                 f(m(s, "drop_group", "ari")), f(m(s, "drop_group", "k")),
                 "--" if not agg.get((s, "drop_group")) else f"{100 * m(s, 'drop_group', 'conflict_merge_group0'):.1f}"]
        out.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if s == "label":
            h1.append((s, "K_half", round(m(s, "half", "k"), 2)))
        elif m(s, "full", "ari") >= 0.8:
            h1.append((s, round(m(s, "half", "ari") - m(s, "full", "ari"), 3)))
        if agg.get((s, "drop_group")):
            h2.append((s, round(100 * m(s, "drop_group", "conflict_merge_group0"), 1)))
    head = ("\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lcccccccc}\n\\toprule\n"
            " & All clients & \\multicolumn{4}{c}{Random half} & \\multicolumn{3}{c}{Without group 0} \\\\\n"
            "\\cmidrule(lr){2-2}\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}\n"
            "Scenario & ARI & ARI & ARI (in) & ARI (out) & $K$ & ARI & $K$ & Merge g0 (\\%) \\\\\n\\midrule\n")
    write("table_refmodel.tex", head + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n")
    print("H-R1 (half: ARI within 0.05 of full; label-only keeps K=1):", h1)
    print("H-R2 (drop_group: conflict merges involving group 0 <= 5%):", h2,
          "holds" if all(x <= 5 for _, x in h2) else "fails")


def cert_table():
    p = os.path.join(CL, "cert_grid.json")
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    ratio = np.array([r["sd_over_se"] for r in rows])
    cov = np.array([r["coverage"] for r in rows])
    print(f"H-S1 share of cells with SD/se in [0.7, 1.5]: {np.mean((ratio >= 0.7) & (ratio <= 1.5)):.2f}")
    print(f"H-S2 share of cells with coverage >= 0.9: {np.mean(cov >= 0.9):.2f}")
    out = []
    for case, delta, lab in [("same", 0.0, "Same concept"), ("diff", 0.1, r"Different, $\Delta=0.1$"),
                             ("diff", 0.3, r"Different, $\Delta=0.3$")]:
        for h in sorted({r["h"] for r in rows}):
            sel = [r for r in rows if r["case"] == case and r["delta"] == delta and r["h"] == h]
            q = lambda k: np.array([r[k] for r in sel])
            out.append(f"{lab if h == min(r['h'] for r in rows) else ''} & {h} & "
                       f"{np.median(q('sd_over_se')):.2f} [{q('sd_over_se').min():.2f}, {q('sd_over_se').max():.2f}] & "
                       f"{np.median(q('bias_in_se')):+.2f} & {100 * np.median(q('coverage')):.0f} & "
                       f"{100 * np.median(q('coverage_debiased')):.0f} \\\\")
    head = ("\\scriptsize\n\\begin{tabular}{lrcccc}\n\\toprule\n"
            "Case & $h$ & SD/se, median [range] & Bias / se & Coverage (\\%) & Debiased coverage (\\%) \\\\\n\\midrule\n")
    write("table_certgrid.tex", head + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    lambda_table()
    refmodel_table()
    cert_table()
