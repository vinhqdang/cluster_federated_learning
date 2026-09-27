"""Aggregate results into tables (results/tables) and figures (results/figures)."""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RAW = os.path.join(ROOT, "results", "raw")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")

# categorical palette (fixed order); DisCo always slot 1
C1, C2, C3, C4, C5, C6, C7, C8 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                  "#008300", "#4a3aa7", "#e34948")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

METHOD_ORDER = ["FedAvg", "Local", "IFCA", "MTCFL", "FL+HC", "PACFL", "DisCo", "Oracle"]
SCEN_NAMES = {"rot": "Rotation", "swap": "Label swap", "rot_qs": "Rotation + QS",
              "mixed_qs": "Mixed + QS", "label": "Label skew only", "minority": "Minority groups",
              "mnist_rot": "MNIST rotation", "mnist_swap": "MNIST label swap",
              "cifar_rot": "CIFAR-10 rotation"}


def load_raw():
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(RAW, "*.json"))]
    return rows


def agg(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return v.mean(), v.std()


def fmt(m, s, pct=True, d=1):
    if pct:
        return f"{100 * m:.{d}f}±{100 * s:.{d}f}"
    return f"{m:.2f}±{s:.2f}"


def main_tables(rows, scenarios, name):
    by = defaultdict(list)
    for r in rows:
        by[(r["scenario"], r["method"])].append(r)
    methods = [m for m in METHOD_ORDER if any((s, m) in by for s in scenarios)] + \
        sorted({r["method"] for r in rows if r["method"] not in METHOD_ORDER and r["scenario"] in scenarios})
    cols = [("acc_mean", "Acc", True), ("acc_p10", "Acc@10%", True), ("worst_group_acc", "Worst-grp", True),
            ("acc_concept_mean", "Concept acc", True), ("jain", "Jain", False), ("ari", "ARI", False),
            ("k_found", "K", False), ("conflict_merge_rate", "Conflict", True)]
    md, tex = [], []
    for s in scenarios:
        if not any((s, m) in by for m in methods):
            continue
        md.append(f"\n### {SCEN_NAMES.get(s, s)}\n")
        md.append("| Method | " + " | ".join(c[1] for c in cols) + " |")
        md.append("|---" * (len(cols) + 1) + "|")
        best = {}
        for key, _, _ in cols:
            vals = {m: agg(by[(s, m)], key)[0] for m in methods if (s, m) in by and m not in ("Oracle",)}
            if vals and key not in ("k_found",):
                best[key] = (min if key == "conflict_merge_rate" else max)(vals, key=vals.get)
        tex.append(r"\multicolumn{%d}{l}{\textit{%s}}\\" % (len(cols) + 1, SCEN_NAMES.get(s, s)))
        for m in methods:
            if (s, m) not in by:
                continue
            cells, tcells = [], []
            for key, _, pct in cols:
                mu, sd = agg(by[(s, m)], key)
                if key == "k_found":
                    c = f"{mu:.1f}"
                elif key in ("jain", "ari"):
                    c = f"{mu:.2f}"
                else:
                    c = fmt(mu, sd, pct) if key in ("acc_mean",) else f"{100 * mu:.1f}"
                t = c.replace("±", r"$\pm$")
                if best.get(key) == m:
                    c, t = f"**{c}**", r"\textbf{%s}" % t
                cells.append(c)
                tcells.append(t)
            md.append(f"| {m} | " + " | ".join(cells) + " |")
            tex.append(f"{m} & " + " & ".join(tcells) + r" \\")
        tex.append(r"\midrule")
    os.makedirs(TAB, exist_ok=True)
    open(os.path.join(TAB, f"{name}.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(TAB, f"{name}.tex"), "w").write("\n".join(tex[:-1]) + "\n")
    return "\n".join(md)


# ----------------------------------------------------------------------------
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelcolor": INK, "axes.titlesize": 9,
                         "savefig.bbox": "tight", "figure.dpi": 150})
    os.makedirs(FIG, exist_ok=True)
    th = os.path.join(ROOT, "results", "theory")
    cl = os.path.join(ROOT, "results", "clustering")

    # --- theory: estimation bias / rmse
    f = os.path.join(th, "estimation.json")
    if os.path.exists(f):
        rows = json.load(open(f))
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
        for ax, truth, title in zip(axs, [1.0, 0.5], ["same concept (true cos = 1)", "different concept (true cos = 0.5)"]):
            rr = [r for r in rows if r["truth"] == truth]
            h = [r["h"] for r in rr]
            ax.plot(h, [r["bias_raw"] for r in rr], "-o", color=C2, lw=2, ms=4, label="raw cosine")
            ax.plot(h, [r["bias_disatt"] for r in rr], "-o", color=C1, lw=2, ms=4, label="disattenuated")
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("samples per half-class h")
            ax.set_title(title, color=INK)
            _style(ax)
        axs[0].set_ylabel("bias of similarity")
        axs[0].legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(FIG, "theory_estimation.pdf"))
        fig.savefig(os.path.join(FIG, "theory_estimation.png"))
        plt.close(fig)

    # --- theory: recovery vs h
    f = os.path.join(th, "recovery.json")
    if os.path.exists(f):
        rows = json.load(open(f))
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
        cols = [C1, C2, C3, C7]
        for ax, full, title in zip(axs, [False, True], ["random class supports: safe & maximal",
                                                          "full support: exact recovery"]):
            key = "p_exact" if full else "p_safe_maximal"
            for col, d in zip(cols, [0.2, 0.35, 0.5, 0.7]):
                rr = [r for r in rows if r["full_support"] == full and r["delta"] == d]
                ax.plot([r["h"] for r in rr], [r[key] for r in rr], "-o", color=col, lw=2, ms=4,
                        label=f"Δ={d}")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("samples per half-class h")
            ax.set_title(title, color=INK)
            _style(ax)
        axs[0].set_ylabel("probability")
        axs[1].legend(frameon=False, fontsize=8, loc="lower right")
        fig.savefig(os.path.join(FIG, "theory_recovery.pdf"))
        fig.savefig(os.path.join(FIG, "theory_recovery.png"))
        plt.close(fig)

    # --- theory: minority independence
    f = os.path.join(th, "minority.json")
    if os.path.exists(f):
        rows = json.load(open(f))
        fig, ax = plt.subplots(figsize=(3.2, 2.4))
        for col, m in zip([C1, C2, C3, C7], [1, 2, 4, 8]):
            rr = [r for r in rows if r["minority_size"] == m]
            ax.plot([r["h"] for r in rr], [r["p_exact"] for r in rr], "-o", color=col, lw=2, ms=4,
                    label=f"minority size {m}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("samples per half-class h")
        ax.set_ylabel("P(exact recovery)")
        ax.legend(frameon=False, fontsize=7)
        _style(ax)
        fig.savefig(os.path.join(FIG, "theory_minority.pdf"))
        fig.savefig(os.path.join(FIG, "theory_minority.png"))
        plt.close(fig)

    # --- tau sensitivity
    f = os.path.join(cl, "tau.json")
    if os.path.exists(f):
        rows = json.load(open(f))
        scen = ["rot", "swap", "rot_qs", "mixed_qs", "label", "minority"]
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
        cols = [C1, C2, C3, C4, C5, C7]
        for ax, dis, title in zip(axs, [True, False], ["disattenuated (DisCo)", "raw cosine (ablation)"]):
            for col, s in zip(cols, scen):
                taus = sorted({r["tau"] for r in rows})
                # label scenario: ARI undefined (single group), plot 1-compatible split rate
                y = []
                for t in taus:
                    rr = [r for r in rows if r["scenario"] == s and r["disattenuate"] == dis and r["tau"] == t]
                    y.append(np.mean([(1.0 if r["k"] == 1 else 0.0) if s == "label" else r["ari"] for r in rr]))
                ax.plot(taus, y, "-o", color=col, lw=2, ms=3, label=SCEN_NAMES[s])
            ax.axvline(0.8, color=INK2, lw=0.8, ls="--")
            ax.set_xlabel("threshold τ")
            ax.set_title(title, color=INK)
            _style(ax)
        axs[0].set_ylabel("ARI (label skew: P(K=1))")
        axs[1].legend(frameon=False, fontsize=7, loc="lower left")
        fig.savefig(os.path.join(FIG, "tau_sensitivity.pdf"))
        fig.savefig(os.path.join(FIG, "tau_sensitivity.png"))
        plt.close(fig)

    # --- sample size
    f = os.path.join(cl, "size.json")
    if os.path.exists(f):
        rows = json.load(open(f))
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
        for ax, g in zip(axs, ["rotation", "swap"]):
            sizes = sorted({r["mean_size"] for r in rows})
            for dis, col, lab in [(True, C1, "disattenuated"), (False, C2, "raw cosine")]:
                y = [np.mean([r["same_group_sim"] for r in rows if r["groups"] == g and r["mean_size"] == n
                              and r["disattenuate"] == dis]) for n in sizes]
                a = [np.mean([r["ari"] for r in rows if r["groups"] == g and r["mean_size"] == n
                              and r["disattenuate"] == dis]) for n in sizes]
                ax.plot(sizes, y, "-o", color=col, lw=2, ms=4, label=f"same-group sim ({lab})")
                ax.plot(sizes, a, "--s", color=col, lw=1.5, ms=4, label=f"ARI ({lab})")
            ax.set_xscale("log")
            ax.set_xlabel("samples per client")
            ax.set_title(SCEN_NAMES["rot" if g == "rotation" else "swap"], color=INK)
            _style(ax)
        axs[0].legend(frameon=False, fontsize=6.5)
        fig.savefig(os.path.join(FIG, "sample_size.pdf"))
        fig.savefig(os.path.join(FIG, "sample_size.png"))
        plt.close(fig)

    # --- differential privacy
    f = os.path.join(cl, "dp.json")
    if os.path.exists(f):
        rows = json.load(open(f))
        sizes = sorted({r["mean_size"] for r in rows})
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
        for ax, s in zip(axs, ["rot", "swap"]):
            sig = sorted({r["sigma"] for r in rows})
            for n, col in zip(sizes, [C2, C1]):
                for dis, ls in [(True, "-o"), (False, "--s")]:
                    y = [np.mean([r["ari"] for r in rows if r["scenario"] == s and r["sigma"] == g
                                  and r["mean_size"] == n and r["disattenuate"] == dis]) for g in sig]
                    ax.plot(range(len(sig)), y, ls, color=col, lw=2 if dis else 1.2, ms=4,
                            label=f"n={n}, {'disatt.' if dis else 'raw'}")
            eps = []
            for g in sig:
                e = [r["epsilon"] for r in rows if r["sigma"] == g][0]
                eps.append("∞" if not np.isfinite(e) else f"{e:.1f}")
            ax.set_xticks(range(len(sig)))
            ax.set_xticklabels([f"σ={g}\nε={e}" for g, e in zip(sig, eps)], fontsize=6.5)
            ax.set_title(SCEN_NAMES[s], color=INK)
            _style(ax)
        axs[0].set_ylabel("ARI")
        axs[1].legend(frameon=False, fontsize=6.5)
        fig.savefig(os.path.join(FIG, "dp.pdf"))
        fig.savefig(os.path.join(FIG, "dp.png"))
        plt.close(fig)


def study_tables():
    cl = os.path.join(ROOT, "results", "clustering")
    out = []
    f = os.path.join(cl, "minority.json")
    if os.path.exists(f):
        rows = json.load(open(f))
        out.append("\n### Minority-group recovery (clustering only, groups 20/10/10/m)\n")
        out.append("| Method | m | Minority recall | Minority purity | ARI | Conflict merge % |")
        out.append("|---|---|---|---|---|---|")
        for meth in ["DisCo", "FL+HC", "PACFL"]:
            for m in sorted({r["minority_size"] for r in rows}):
                rr = [r for r in rows if r["method"] == meth and r["minority_size"] == m]
                out.append(f"| {meth} | {m} | {np.mean([r['minority_recall'] for r in rr]):.2f} | "
                           f"{np.mean([r['minority_purity'] for r in rr]):.2f} | "
                           f"{np.mean([r['ari'] for r in rr]):.2f} | "
                           f"{100 * np.mean([r['conflict_merge_rate'] for r in rr]):.1f} |")
    f = os.path.join(cl, "explain.json")
    if os.path.exists(f):
        rows = json.load(open(f))
        ex = [r for r in rows if r["type"] == "explain"]
        ce = [r for r in rows if r["type"] == "cert"]
        out.append("\n### Explanation fidelity (divergent classes between clusters)\n")
        out.append("| Scenario | pairs | precision | recall | exact set | diagnosis correct |")
        out.append("|---|---|---|---|---|---|")
        for s in sorted({r["scenario"] for r in ex}):
            rr = [r for r in ex if r["scenario"] == s]
            out.append(f"| {SCEN_NAMES.get(s, s)} | {len(rr)} | {np.mean([r['precision'] for r in rr]):.2f} | "
                       f"{np.mean([r['recall'] for r in rr]):.2f} | {np.mean([r['exact'] for r in rr]):.2f} | "
                       f"{np.mean([r['diagnosis_ok'] for r in rr]):.2f} |")
        out.append("\n### Assignment certificates (accountability)\n")
        out.append("| Scenario | clients | confident | correct among confident | correct among uncertain |")
        out.append("|---|---|---|---|---|")
        for s in sorted({r["scenario"] for r in ce}):
            rr = [r for r in ce if r["scenario"] == s]
            conf = [r for r in rr if r["verdict"] == "confident"]
            unc = [r for r in rr if r["verdict"] != "confident"]
            out.append(f"| {SCEN_NAMES.get(s, s)} | {len(rr)} | {len(conf) / len(rr):.2f} | "
                       f"{np.mean([r['correct'] for r in conf]) if conf else float('nan'):.3f} | "
                       f"{np.mean([r['correct'] for r in unc]) if unc else float('nan'):.3f} |")
    f = os.path.join(cl, "newcomer.json")
    if os.path.exists(f):
        rows = json.load(open(f))
        out.append("\n### Newcomers (assigned with the frozen reference model)\n")
        out.append("| Scenario | known-concept newcomers correctly assigned | novel-concept newcomers flagged |")
        out.append("|---|---|---|")
        for s in sorted({r["scenario"] for r in rows}):
            k = [r["correct"] for r in rows if r["scenario"] == s and r["kind"] == "known"]
            n = [r["correct"] for r in rows if r["scenario"] == s and r["kind"] == "novel"]
            out.append(f"| {SCEN_NAMES.get(s, s)} | {np.mean(k):.2f} ({len(k)}) | {np.mean(n):.2f} ({len(n)}) |")
    f = os.path.join(cl, "scale.json")
    if os.path.exists(f):
        rows = json.load(open(f))
        out.append("\n### Scalability\n")
        out.append("| Clients | signature time / client (s) | server clustering time (s) | upload floats / client | model dim | ARI |")
        out.append("|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: r["n_clients"]):
            out.append(f"| {r['n_clients']} | {r['sig_time_per_client_s']:.2f} | {r['server_time_s']:.2f} | "
                       f"{r['upload_floats_per_client']:.0f} | {r['model_dim']} | {r['ari']:.2f} |")
    f = os.path.join(cl, "dp.json")
    if os.path.exists(f):
        rows = json.load(open(f))
        out.append("\n### Differential privacy (ARI, mean over 3 seeds)\n")
        out.append("| Scenario | samples/client | σ | ε (δ=1e-5) | DisCo | raw-cosine ablation |")
        out.append("|---|---|---|---|---|---|")
        for s in ["rot", "swap"]:
            for n in sorted({r["mean_size"] for r in rows}):
                for g in sorted({r["sigma"] for r in rows}):
                    rr = [r for r in rows if r["scenario"] == s and r["mean_size"] == n and r["sigma"] == g]
                    if not rr:
                        continue
                    e = rr[0]["epsilon"]
                    a1 = np.mean([r["ari"] for r in rr if r["disattenuate"]])
                    a0 = np.mean([r["ari"] for r in rr if not r["disattenuate"]])
                    out.append(f"| {SCEN_NAMES[s]} | {n} | {g} | {'∞' if not np.isfinite(e) else f'{e:.2f}'} | "
                               f"{a1:.2f} | {a0:.2f} |")
    text = "\n".join(out) + "\n"
    os.makedirs(TAB, exist_ok=True)
    open(os.path.join(TAB, "studies.md"), "w").write(text)
    return text


if __name__ == "__main__":
    rows = load_raw()
    print(main_tables(rows, ["rot", "swap", "rot_qs", "mixed_qs", "label", "minority"], "main"))
    print(main_tables(rows, ["mnist_rot", "mnist_swap", "cifar_rot"], "other_datasets"))
    abl = [r for r in rows if r["method"].startswith("DisCo")]
    print(main_tables(abl, ["rot", "swap", "rot_qs", "mixed_qs", "label"], "ablation"))
    print(study_tables())
    figures()
