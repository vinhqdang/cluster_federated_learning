#!/usr/bin/env python3
"""Statistics and LaTeX/Markdown tables for the round-1 DisCo revision.

Reads per-run JSON results from ``results/raw`` (small-CNN benchmarks) and
``results/raw_real`` (realistic benchmarks) and writes:

* ``paper/table_r1_ci_{main,other,real}.tex``  per-scenario mean +- 95% t-interval
* ``paper/table_r1_failure.tex``               failure rates (k_true > 1 and ARI < 0.5)
* ``paper/table_r1_hier.tex``                  paired DisCo-vs-baseline comparison with
                                               hierarchical bootstrap CIs and Wilcoxon tests
* ``paper/table_r1_cost.tex``                  wall-clock time and upload cost
* ``paper/table_r1_decomp.tex``                prior-correction vs. clustering decomposition
* ``paper/table_r1_diag.tex``                  DisCo diagnostics (only if any run has ``diag``)

Every table is also written as a Markdown copy to ``results/tables/``, and a
plain-text summary of all tables is printed to stdout.

With ``--heldout`` only the CI and hierarchical tables are recomputed after
excluding the development runs in ``DEV_RUNS``; they are written with the
suffix ``_heldout``.

Only numpy, scipy and the standard library are used. The script is
deterministic: the bootstrap uses ``numpy.random.default_rng(0)``.
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIRS = [os.path.join(ROOT, "results", "r1", "raw"), os.path.join(ROOT, "results", "r1", "raw_real")]
PAPER_DIR = os.path.join(ROOT, "paper")
MD_DIR = os.path.join(ROOT, "results", "tables")

# ---------------------------------------------------------------------------
# Scenario and method constants
# ---------------------------------------------------------------------------

SMALL_MAIN = ["rot", "swap", "rot_qs", "mixed_qs", "label", "minority"]
SMALL_OTHER = ["mnist_rot", "mnist_swap", "svhn_rot", "svhn_swap", "cifar_rot", "cifar_swap"]
SMALL_SWEEP = ["angle15", "angle30", "angle45", "perm2", "perm8",
               "rot_part30", "swap_part30", "rot_n200"]
REAL = ["pacs", "pacs_qs", "officehome", "c100_swap", "c100_rot"]
FEMNIST = ["femnist"]
REAL_GROUP = REAL + FEMNIST
# pre-registered additional benchmarks (experiments/PREREGISTRATION_R1_EXTRA.md)
EXTRA_GROUP = ["kmnist_rot", "kmnist_swap", "emnist_rot", "emnist_swap",
               "c10r_rot", "c10r_swap", "agnews_swap"]

DEV_RUNS = {("rot", 0), ("swap", 0), ("mixed_qs", 0), ("label", 0), ("pacs", 0)}

# Scenarios excluded from the hierarchical comparison (k_true == 1).
HIER_EXCLUDE = {"label", "femnist"}

SCEN_NAMES = {
    "rot": "Rotation",
    "swap": "Label swap",
    "rot_qs": "Rotation + QS",
    "mixed_qs": "Mixed + QS",
    "label": "Label skew only",
    "minority": "Minority groups",
    "mnist_rot": "MNIST rotation",
    "mnist_swap": "MNIST label swap",
    "svhn_rot": "SVHN rotation",
    "svhn_swap": "SVHN label swap",
    "cifar_rot": "CIFAR-10 rotation",
    "cifar_swap": "CIFAR-10 label swap",
    "angle15": "Rotation 15°",
    "angle30": "Rotation 30°",
    "angle45": "Rotation 45°",
    "perm2": "Permutation (2 concepts)",
    "perm8": "Permutation (8 concepts)",
    "rot_part30": "Rotation, 30% participation",
    "swap_part30": "Label swap, 30% participation",
    "rot_n200": "Rotation, 200 clients",
    "pacs": "PACS",
    "pacs_qs": "PACS + QS",
    "officehome": "Office-Home",
    "c100_swap": "CIFAR-100 label swaps",
    "c100_rot": "CIFAR-100 rotations",
    "femnist": "FEMNIST",
    "kmnist_rot": "KMNIST rotation",
    "kmnist_swap": "KMNIST label swap",
    "emnist_rot": "EMNIST rotation",
    "emnist_swap": "EMNIST label swap",
    "c10r_rot": "CIFAR-10 (ResNet) rotation",
    "c10r_swap": "CIFAR-10 (ResNet) label swap",
    "agnews_swap": "AG News label swap",
}

METHOD_ORDER = ["DisCo", "DisCo-noAttach", "DisCo-T30", "FedAvg", "Local", "FedAvg-FT", "IFCA", "FeSEM",
                "MTCFL", "FL+HC", "PACFL", "StoCFL", "StoCFL-W", "FedCCFA", "FedDAG-C", "Oracle"]
FAIL_METHODS = ["DisCo", "DisCo-noAttach", "IFCA", "FeSEM", "MTCFL", "FL+HC",
                "PACFL", "StoCFL", "StoCFL-W", "FedCCFA", "FedDAG-C"]
HIER_BASELINES = ["FedAvg", "Local", "FedAvg-FT", "IFCA", "FeSEM", "MTCFL", "FL+HC",
                  "PACFL", "StoCFL", "StoCFL-W", "FedCCFA", "FedDAG-C"]
DECOMP_METHODS = ["FedAvg", "IFCA", "FeSEM", "FL+HC", "PACFL", "MTCFL", "StoCFL",
                  "FedDAG-C", "DisCo", "DisCo-noAttach", "Oracle"]

GROUPS_CI = {"main": SMALL_MAIN, "other": SMALL_OTHER, "real": REAL_GROUP, "extra": EXTRA_GROUP}
GROUPS_FAIL = {"Main": SMALL_MAIN, "Other": SMALL_OTHER, "Sweep": SMALL_SWEEP, "Real": REAL_GROUP,
               "Extra": EXTRA_GROUP}
GROUPS_DECOMP = {"main": SMALL_MAIN, "other": SMALL_OTHER, "real": REAL_GROUP}

REF = "DisCo"
TIE_PP = 0.05          # W/T/L tie band, in percentage points (see report)
BOOT_B = 10000
ARI_FAIL = 0.5
# per-dataset failure rates (multi-group scenarios of each dataset; the FMNIST column includes the sweeps)
GROUPS_FAIL_DS = {"FMNIST": SMALL_MAIN + SMALL_SWEEP, "MNIST": ["mnist_rot", "mnist_swap"],
                  "KMNIST": ["kmnist_rot", "kmnist_swap"], "EMNIST": ["emnist_rot", "emnist_swap"],
                  "SVHN": ["svhn_rot", "svhn_swap"], "CIFAR-10": ["cifar_rot", "cifar_swap"],
                  "PACS": ["pacs", "pacs_qs"], "Office-Home": ["officehome"],
                  "CIFAR-100 (R)": ["c100_swap", "c100_rot"], "CIFAR-10 (R)": ["c10r_rot", "c10r_swap"],
                  "AG News": ["agnews_swap"]}


# ---------------------------------------------------------------------------
# Loading and filtering
# ---------------------------------------------------------------------------

def load(dirs):
    """Load every per-run JSON file found in the given directories.

    Parameters
    ----------
    dirs : iterable of str
        Directories containing ``{scenario}__{method}__s{seed}.json`` files.

    Returns
    -------
    list of dict
        One dict per run, as stored on disk. Unreadable files are skipped
        with a warning on stderr.
    """
    runs = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                with open(path) as fh:
                    run = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"warning: skipping {path}: {exc}", file=sys.stderr)
                continue
            if not isinstance(run, dict):
                continue
            run.setdefault("scenario", os.path.basename(path).split("__")[0])
            run["_path"] = path
            runs.append(run)
    return runs


def is_dev(run):
    """Return True if the run is one of the development runs in ``DEV_RUNS``."""
    return (run.get("scenario"), run.get("seed")) in DEV_RUNS


def filter_runs(runs, heldout=False):
    """Drop development runs when ``heldout`` is requested."""
    if not heldout:
        return list(runs)
    return [r for r in runs if not is_dev(r)]


def by_method(runs, method):
    """Return the runs of one method."""
    return [r for r in runs if r.get("method") == method]


def finite_values(runs, key, scale=1.0):
    """Collect finite numeric values of ``key`` across runs, scaled by ``scale``.

    Runs missing the key (or holding a non-numeric / NaN value) are skipped.
    """
    out = []
    for r in runs:
        v = r.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if not math.isfinite(v):
            continue
        out.append(float(v) * scale)
    return np.asarray(out, dtype=float)


def scen_k_true(run):
    """Return the true number of concepts of a run, or None if unknown."""
    k = run.get("k_true")
    return k if isinstance(k, int) and not isinstance(k, bool) else None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_mean_hw(vals, nd):
    """Format a sample as ``mean +- half-width`` of the 95% t-interval.

    Shows only the mean when fewer than two values are available, and "--"
    when there are none. Uses ``sd`` with ddof=1.
    """
    n = len(vals)
    if n == 0:
        return "--"
    m = float(np.mean(vals))
    if n < 2:
        return f"{m:.{nd}f}"
    sd = float(np.std(vals, ddof=1))
    hw = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return f"{m:.{nd}f} ± {hw:.{nd}f}"


def fmt_mean(vals, nd):
    """Format the mean of a sample, or "--" if empty."""
    if len(vals) == 0:
        return "--"
    return f"{float(np.mean(vals)):.{nd}f}"


def fmt_signed(x, nd=1):
    """Format a signed number with an explicit sign."""
    if round(x, nd) == 0:
        x = 0.0  # avoid printing "-0.0"
    return f"{x:+.{nd}f}"


def fmt_sci(v):
    """Format a non-negative count as a two-significant-digit scientific string (e.g. 1.2e4)."""
    if v == 0:
        return "0"
    mant, exp = f"{v:.1e}".split("e")
    return f"{mant}e{int(exp)}"


def scen_name(s):
    """Return the display name of a scenario (falls back to the raw key)."""
    return SCEN_NAMES.get(s, s)


_TEX_ESCAPE = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
]
_TEX_UNICODE = {"±": r"$\pm$", "Δ": r"$\Delta$", "°": r"$^\circ$"}


def to_tex(text):
    """Escape a plain-text cell for LaTeX and map the unicode symbols used here."""
    text = str(text)
    for ch, rep in _TEX_ESCAPE:
        text = text.replace(ch, rep)
    for ch, rep in _TEX_UNICODE.items():
        text = text.replace(ch, rep)
    # Use a math minus for negative numbers that are not part of a name.
    text = re.sub(r"(?<![\w])-(?=\d)", r"$-$", text)
    text = re.sub(r"(?<![$\\])<(?=\d)", r"$<$", text)
    return text


# ---------------------------------------------------------------------------
# Generic table rendering
# ---------------------------------------------------------------------------
# A table is described by:
#   colspec : LaTeX column specification, e.g. "l c c"
#   header  : list of header rows; a cell is a str or (str, span)
#   body    : list of items: ("section", text) | ("row", [cells]) |
#             ("midrule",) | ("space",)
#   scriptsize : wrap the tabular in {\scriptsize ...}

def render_tex(colspec, header, body, scriptsize=True):
    """Render a booktabs tabular from the generic table description."""
    lines = []
    ncols = len(colspec.replace("|", "").split())
    if scriptsize:
        lines.append("{\\scriptsize")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    for i, hrow in enumerate(header):
        cells, col, spans = [], 1, []
        for cell in hrow:
            text, span = (cell, 1) if isinstance(cell, str) else cell
            if span > 1:
                cells.append(f"\\multicolumn{{{span}}}{{c}}{{{to_tex(text)}}}")
                spans.append((col, col + span - 1))
            else:
                cells.append(to_tex(text))
            col += span
        lines.append(" & ".join(cells) + " \\\\")
        if len(header) > 1 and i == 0:
            for a, b in spans:
                lines.append(f"\\cmidrule(lr){{{a}-{b}}}")
    lines.append("\\midrule")
    for item in body:
        kind = item[0]
        if kind == "section":
            lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{to_tex(item[1])}}}}} \\\\")
        elif kind == "row":
            lines.append(" & ".join(to_tex(c) for c in item[1]) + " \\\\")
        elif kind == "midrule":
            lines.append("\\midrule")
        elif kind == "space":
            lines.append("\\addlinespace")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if scriptsize:
        lines.append("}")
    return "\n".join(lines) + "\n"


def render_md(title, header, body):
    """Render the same table description as a Markdown pipe table."""
    flat_header = _flatten_header(header)
    ncols = len(flat_header)
    out = [f"### {title}", ""]
    out.append("| " + " | ".join(flat_header) + " |")
    out.append("|" + "|".join(["---"] * ncols) + "|")
    for item in body:
        kind = item[0]
        if kind == "section":
            out.append("| **" + item[1] + "**" + " |" * ncols)
        elif kind == "row":
            out.append("| " + " | ".join(str(c) for c in item[1]) + " |")
    return "\n".join(out) + "\n"


def _flatten_header(header):
    """Flatten a multi-row header into one Markdown header row."""
    if len(header) == 1:
        return [c if isinstance(c, str) else c[0] for c in header[0]]
    # Prefix each column of the last row with the spanning label above it.
    last = [c if isinstance(c, str) else c[0] for c in header[-1]]
    labels = []
    for ci in range(len(last)):
        label = ""
        col = 0
        for hrow in header[:-1]:
            col = 0
            for cell in hrow:
                text, span = (cell, 1) if isinstance(cell, str) else cell
                if col <= ci < col + span:
                    label = text
                col += span
        labels.append(f"{label} {last[ci]}".strip())
    return labels


def write_table(stem, tex_body, md_text, mdir_name=None):
    """Write the ``.tex`` file to ``paper/`` and the Markdown copy to ``results/tables/``."""
    os.makedirs(PAPER_DIR, exist_ok=True)
    os.makedirs(MD_DIR, exist_ok=True)
    tex_path = os.path.join(PAPER_DIR, stem + ".tex")
    md_path = os.path.join(MD_DIR, stem + ".md")
    with open(tex_path, "w") as fh:
        fh.write(tex_body)
    with open(md_path, "w") as fh:
        fh.write(md_text)
    return tex_path, md_path


def plain_text(title, header, body):
    """Return a compact plain-text rendering of a table for stdout."""
    rows = [_flatten_header(header)]
    for item in body:
        if item[0] == "section":
            rows.append([f"[{item[1]}]"])
        elif item[0] == "row":
            rows.append([str(c) for c in item[1]])
    ncols = max(len(r) for r in rows)
    widths = [0] * ncols
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    out = [f"== {title} =="]
    for r in rows:
        padded = [c.ljust(widths[i]) for i, c in enumerate(r)] + [""] * (ncols - len(r))
        out.append("  ".join(padded).rstrip())
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Table 1: per-scenario confidence intervals
# ---------------------------------------------------------------------------

CI_HEADER = [["Method", "n", "Acc (%)", "Concept (%)", "Worst-grp (%)", "ARI", "K", "Conflict (%)"]]
CI_COLSPEC = "l r c c c c c c"


def ci_row(runs, method):
    """Build the cells of one CI-table row for a method over a set of runs."""
    rs = by_method(runs, method)
    acc = finite_values(rs, "pc_acc_mean", 100)
    con = finite_values(rs, "pc_acc_concept_mean", 100)
    wst = finite_values(rs, "pc_worst_group_acc", 100)
    ari = finite_values(rs, "ari")
    kf = finite_values(rs, "k_found")
    cnf = finite_values(rs, "conflict_merge_rate", 100)
    return [
        method,
        str(len(acc)),
        fmt_mean_hw(acc, 1),
        fmt_mean_hw(con, 1),
        fmt_mean_hw(wst, 1),
        fmt_mean_hw(ari, 2),
        fmt_mean(kf, 1),
        fmt_mean(cnf, 1),
    ]


def build_ci_table(group_name, runs):
    """Build the CI table for one scenario group.

    Returns (title, header, body) or None if no scenario of the group has runs.
    """
    body = []
    scen_list = [s for s in GROUPS_CI[group_name] if any(r.get("scenario") == s for r in runs)]
    for si, s in enumerate(scen_list):
        srs = [r for r in runs if r.get("scenario") == s]
        if si > 0:
            body.append(("space",))
        body.append(("section", scen_name(s)))
        for m in METHOD_ORDER:
            if not by_method(srs, m):
                continue
            body.append(("row", ci_row(srs, m)))
    if not body:
        return None
    title = f"CI table, group {group_name}"
    return title, CI_HEADER, body


def ci_table_tex(header, body):
    """Render a CI table body with the agreed column specification."""
    return render_tex(CI_COLSPEC, header, body)


# ---------------------------------------------------------------------------
# Table 2: failure rates
# ---------------------------------------------------------------------------

def failure_stats(runs, scenarios, method):
    """Return (failures, n) for a method over the given scenarios.

    Only runs with k_true > 1 and a finite ARI count; a run fails if ARI < 0.5.
    """
    rs = [r for r in by_method(runs, method)
          if r.get("scenario") in scenarios and scen_k_true(r) is not None
          and scen_k_true(r) > 1 and finite_values([r], "ari").size == 1]
    fails = sum(1 for r in rs if float(r["ari"]) < ARI_FAIL)
    return fails, len(rs)


def build_failure_table(runs):
    """Build the failure-rate table: rows are methods, columns are scenario groups."""
    header = [["Method"] + list(GROUPS_FAIL.keys())]
    body = []
    for m in FAIL_METHODS:
        row = [m]
        for gname, scen in GROUPS_FAIL.items():
            f, n = failure_stats(runs, scen, m)
            row.append("--" if n == 0 else f"{100.0 * f / n:.1f}% ({n})")
        body.append(("row", row))
    return "Failure rates (k_true > 1, ARI < 0.5)", header, body


def build_failure_ds_table(runs):
    """Failure rates per dataset: rows are methods, columns are datasets (percent; runs in the caption)."""
    header = [["Method"] + list(GROUPS_FAIL_DS.keys())]
    body = []
    for m in FAIL_METHODS:
        row = [m]
        for scen in GROUPS_FAIL_DS.values():
            f, n = failure_stats(runs, scen, m)
            row.append("--" if n == 0 else f"{100.0 * f / n:.0f}")
        body.append(("row", row))
    return "Failure rates per dataset (k_true > 1, ARI < 0.5)", header, body


# ---------------------------------------------------------------------------
# Table 3: hierarchical comparison
# ---------------------------------------------------------------------------

def paired_scenario_diffs(runs, baseline, metric):
    """Compute per-scenario paired differences DisCo minus baseline, in percentage points.

    Runs are paired by (scenario, seed). Only runs with k_true > 1 and
    scenarios outside ``HIER_EXCLUDE`` enter.

    Returns
    -------
    dict
        Maps scenario -> numpy array of per-seed differences (pp).
    """
    def eligible(r):
        k = scen_k_true(r)
        return k is not None and k > 1 and r.get("scenario") not in HIER_EXCLUDE

    ref = {}
    base = {}
    for r in runs:
        if not eligible(r):
            continue
        v = finite_values([r], metric)
        if v.size != 1:
            continue
        key = (r["scenario"], r.get("seed"))
        if r.get("method") == REF:
            ref[key] = float(v[0])
        elif r.get("method") == baseline:
            base[key] = float(v[0])
    out = {}
    for key in sorted(set(ref) & set(base), key=lambda k: (k[0], str(k[1]))):
        out.setdefault(key[0], []).append(100.0 * (ref[key] - base[key]))
    return {s: np.asarray(v, dtype=float) for s, v in out.items()}


def hier_test(diffs_by_scen, b_reps=BOOT_B):
    """Run the hierarchical bootstrap and the one-sided Wilcoxon test.

    Parameters
    ----------
    diffs_by_scen : dict
        Scenario -> per-seed differences (pp).
    b_reps : int
        Number of bootstrap replicates.

    Returns
    -------
    dict with keys: n_scen, mean, lo, hi, wtl (tuple), p (float or None).
    """
    scen = sorted(diffs_by_scen)
    if not scen:
        return {"n_scen": 0}
    d_s = np.array([float(np.mean(diffs_by_scen[s])) for s in scen])
    point = float(np.mean(d_s))
    wins = int(np.sum(d_s > TIE_PP))
    ties = int(np.sum(np.abs(d_s) <= TIE_PP))
    losses = int(np.sum(d_s < -TIE_PP))

    rng = np.random.default_rng(0)
    n_s = len(scen)
    # Within-scenario bootstrap of seeds, one column per scenario.
    seed_means = np.empty((b_reps, n_s))
    for j, s in enumerate(scen):
        x = diffs_by_scen[s]
        idx = rng.integers(0, len(x), size=(b_reps, len(x)))
        seed_means[:, j] = x[idx].mean(axis=1)
    # Resample scenarios with replacement, then average over sampled scenarios.
    sidx = rng.integers(0, n_s, size=(b_reps, n_s))
    boot = np.take_along_axis(seed_means, sidx, axis=1).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    p = None
    if n_s >= 5 and np.any(d_s != 0):
        try:
            p = float(stats.wilcoxon(d_s, alternative="greater").pvalue)
        except ValueError:
            p = None
    return {"n_scen": n_s, "mean": point, "lo": float(lo), "hi": float(hi),
            "wtl": (wins, ties, losses), "p": p}


def fmt_p(p):
    """Format a p-value, using "--" when unavailable."""
    if p is None:
        return "--"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


HIER_SECTIONS = [
    ("(a) Small-CNN benchmarks (main, other, sweep scenarios pooled)",
     SMALL_MAIN + SMALL_OTHER + SMALL_SWEEP),
    ("(b) Realistic benchmarks", REAL_GROUP),
    ("(c) Additional benchmarks", EXTRA_GROUP),
]
HIER_METRICS = [("Accuracy", "pc_acc_mean"),
                ("Concept acc.", "pc_acc_concept_mean"),
                ("Worst group", "pc_worst_group_acc")]


def build_hier_table(runs):
    """Build the hierarchical comparison table (sections a and b)."""
    header = [
        [("Baseline", 1)] + [(name, 3) for name, _ in HIER_METRICS],
        [""] + ["Δ [95% CI]", "W/T/L", "p"] * len(HIER_METRICS),
    ]
    body = []
    for si, (sec_title, scen_list) in enumerate(HIER_SECTIONS):
        sec_runs = [r for r in runs if r.get("scenario") in scen_list]
        if si > 0:
            body.append(("space",))
        body.append(("section", sec_title))
        for base in HIER_BASELINES:
            row = [base]
            for _, metric in HIER_METRICS:
                diffs = paired_scenario_diffs(sec_runs, base, metric)
                res = hier_test(diffs)
                if res.get("n_scen", 0) == 0:
                    row += ["--", "--", "--"]
                    continue
                w, t, l = res["wtl"]
                row.append(f"{fmt_signed(res['mean'])} [{fmt_signed(res['lo'])}, {fmt_signed(res['hi'])}]")
                row.append(f"{w}/{t}/{l}")
                row.append(fmt_p(res["p"]))
            body.append(("row", row))
    return "Hierarchical comparison of DisCo vs baselines (pp)", header, body


def hier_table_tex(header, body):
    """Render the hierarchical table with its column specification."""
    colspec = "l " + " ".join(["c c c"] * len(HIER_METRICS))
    return render_tex(colspec, header, body)


# ---------------------------------------------------------------------------
# Table 4: cost
# ---------------------------------------------------------------------------

def build_cost_table(runs):
    """Build the cost table: wall-clock minutes and upload floats per client."""
    present = {r.get("method") for r in runs if r.get("method")}
    methods = [m for m in METHOD_ORDER if m in present] + sorted(present - set(METHOD_ORDER) - {"DisCo-L02"})
    main_set, real_set = set(SMALL_MAIN), set(REAL_GROUP)
    header = [["Method", "Time, small-CNN main (min)", "Time, realistic (min)",
               "Upload floats / client"]]
    body = []
    for m in methods:
        rs = by_method(runs, m)
        t_main = finite_values([r for r in rs if r.get("scenario") in main_set], "time_s", 1 / 60)
        t_real = finite_values([r for r in rs if r.get("scenario") in real_set], "time_s", 1 / 60)
        ups = []
        for r in rs:
            v = finite_values([r], "cluster_upload_floats")
            n = r.get("n_clients")
            if v.size == 1 and isinstance(n, int) and n > 0:
                ups.append(float(v[0]) / n)
        ups = np.asarray(ups, dtype=float)
        if ups.size:
            up = fmt_sci(float(np.mean(ups)))
        else:
            up = "--"
        body.append(("row", [m, fmt_mean(t_main, 1), fmt_mean(t_real, 1), up]))
    return "Cost per run", header, body


# ---------------------------------------------------------------------------
# Table 5: prior correction decomposition
# ---------------------------------------------------------------------------

def build_decomp_table(runs):
    """Build the decomposition table: raw vs prior-corrected accuracy per group."""
    header = [["Group", "Method", "Acc raw (%)", "Acc PC (%)", "Gain (pp)",
               "Concept raw (%)", "Concept PC (%)"]]
    body = []
    for gname, scen in GROUPS_DECOMP.items():
        gruns = [r for r in runs if r.get("scenario") in scen]
        for m in DECOMP_METHODS:
            rs = by_method(gruns, m)
            raw = finite_values(rs, "acc_mean", 100)
            pc = finite_values(rs, "pc_acc_mean", 100)
            craw = finite_values(rs, "acc_concept_mean", 100)
            cpc = finite_values(rs, "pc_acc_concept_mean", 100)
            # Gain is the mean per-run difference, over runs that carry both values.
            pairs = [(float(r["pc_acc_mean"]) - float(r["acc_mean"])) * 100
                     for r in rs if finite_values([r], "acc_mean").size == 1
                     and finite_values([r], "pc_acc_mean").size == 1]
            gain = fmt_signed(float(np.mean(pairs))) if pairs else "--"
            body.append(("row", [gname, m, fmt_mean(raw, 1), fmt_mean(pc, 1), gain,
                                 fmt_mean(craw, 1), fmt_mean(cpc, 1)]))
    return "Prior correction vs clustering", header, body


# ---------------------------------------------------------------------------
# Table 6: DisCo diagnostics
# ---------------------------------------------------------------------------

DIAG_FIELDS = [("delta_hat", "delta_hat", 2), ("within_min", "within_min", 2),
               ("coverage", "coverage", 2), ("otsu_tau", "otsu_tau", 2),
               ("otsu_eta", "otsu_eta", 2), ("ari_otsu", "ari_otsu", 2)]
DIAG_TOP_RATES = [("conflict_rate_merge_stage", "Conflict merge (%)", 100),
                  ("conflict_rate_attach_stage", "Conflict attach (%)", 100),
                  ("n_attached", "n attached", 1)]


def build_diag_table(runs):
    """Build the DisCo diagnostics table, or return None if no run has ``diag``."""
    drs = [r for r in by_method(runs, REF) if isinstance(r.get("diag"), dict)]
    if not drs:
        return None
    header = [["Scenario", "n", "delta hat", "within min", "coverage", "Otsu tau", "Otsu eta",
               "ARI (Otsu)", "ARI (DisCo)", "Conflict merge (%)", "Conflict attach (%)",
               "n attached"]]
    body = []
    scen_list = [s for s in SMALL_MAIN + SMALL_OTHER + SMALL_SWEEP + REAL_GROUP + EXTRA_GROUP
                 if any(r.get("scenario") == s for r in drs)]
    for s in scen_list:
        srs = [r for r in drs if r.get("scenario") == s]
        row = [scen_name(s), str(len(srs))]
        for key, _, nd in DIAG_FIELDS:
            # Diagnostics live inside ``diag``; reuse the numeric filter on a wrapper.
            vals = finite_values([{"v": r["diag"].get(key)} for r in srs], "v")
            row.append(fmt_mean(vals, nd))
        row.append(fmt_mean(finite_values(srs, "ari"), 2))
        # Stage-wise conflict statistics are top-level fields, shown only when present.
        for key, _, scale in DIAG_TOP_RATES:
            row.append(fmt_mean(finite_values(srs, key, scale), 1))
        body.append(("row", row))
    return "DisCo diagnostics per scenario", header, body


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

TABLE_COLSPECS = {
    "table_r1_failure": "l c c c c c",
    "table_r1_failure_ds": "l c c c c c c c c c c c",
    "table_r1_cost": "l c c c",
    "table_r1_decomp": "l l c c c c c",
    "table_r1_diag": "l r " + "c " * 10 + "c",
}


def emit(stem, title, header, body, summary):
    """Write one table (tex + md) and append its plain-text rendering to ``summary``."""
    tex = render_tex(TABLE_COLSPECS[stem], header, body)
    write_table(stem, tex, render_md(title, header, body))
    summary.append(plain_text(title, header, body))


def run(heldout):
    """Compute all tables, write the files, and return the stdout summary text."""
    raw = load(RAW_DIRS)
    print(f"Loaded {len(raw)} runs from {len(RAW_DIRS)} directories.")
    runs = filter_runs(raw, heldout=heldout)
    if heldout:
        print(f"Held-out mode: excluded {len(raw) - len(runs)} development runs.")
    suffix = "_heldout" if heldout else ""
    summary = []

    # Table 1: CI tables per group.
    for gname in GROUPS_CI:
        built = build_ci_table(gname, runs)
        stem = f"table_r1_ci_{gname}{suffix}"
        if built is None:
            print(f"note: no runs for CI group {gname}; {stem}.tex not written.")
            continue
        title, header, body = built
        tex = ci_table_tex(header, body)
        write_table(stem, tex, render_md(title, header, body))
        summary.append(plain_text(title, header, body))

    # Table 3: hierarchical comparison (also produced for --heldout).
    title, header, body = build_hier_table(runs)
    write_table(f"table_r1_hier{suffix}", hier_table_tex(header, body),
                render_md(title, header, body))
    summary.append(plain_text(title, header, body))

    if not heldout:
        title, header, body = build_failure_table(runs)
        emit("table_r1_failure", title, header, body, summary)
        title, header, body = build_failure_ds_table(runs)
        emit("table_r1_failure_ds", title, header, body, summary)
        title, header, body = build_cost_table(runs)
        emit("table_r1_cost", title, header, body, summary)
        title, header, body = build_decomp_table(runs)
        emit("table_r1_decomp", title, header, body, summary)
        diag = build_diag_table(runs)
        if diag is not None:
            title, header, body = diag
            emit("table_r1_diag", title, header, body, summary)

    return "\n\n".join(summary) + "\n"


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--heldout", action="store_true",
                        help="recompute the CI and hierarchical tables excluding DEV_RUNS, "
                             "writing files with the suffix _heldout")
    args = parser.parse_args(argv)
    text = run(args.heldout)
    if not args.heldout:
        split_main_table()
    print(text, end="")


def split_table(stem, per_part=3):
    """Split paper/<stem>.tex into parts of at most ``per_part`` scenario blocks
    (<stem>_a.tex, <stem>_b.tex, ...), each of which fits on one page."""
    path = os.path.join(PAPER_DIR, f"{stem}.tex")
    if not os.path.exists(path):
        return
    t = open(path).read().split("\n")
    idx = [i for i, l in enumerate(t) if l.startswith("\\multicolumn{8}{l}{\\textit")]
    if len(idx) <= per_part:
        return
    end = max(i for i, l in enumerate(t) if l.startswith("\\bottomrule"))
    head, foot = t[:idx[0]], t[end:]
    starts = idx[::per_part] + [end]
    for k in range(len(starts) - 1):
        with open(os.path.join(PAPER_DIR, f"{stem}_{'abcdefgh'[k]}.tex"), "w") as f:
            f.write("\n".join(head + t[starts[k]:starts[k + 1]] + foot) + "\n")


def split_main_table():
    """Split the per-dataset tables that do not fit on one page."""
    for stem in ("table_r1_ci_main", "table_r1_ci_other", "table_r1_ci_real", "table_r1_ci_extra"):
        split_table(stem)


if __name__ == "__main__":
    main()
