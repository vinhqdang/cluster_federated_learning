"""Assemble the flat Springer Nature (sn-jnl) submission directory from ../paper.

Springer requires all source files in one folder (no subfolders), so every input,
figure and style file is copied here. Re-run after changing ../paper or ../results.
"""
import glob, os, re, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "paper")
FIG = os.path.join(HERE, "..", "results", "figures")

body = open(os.path.join(SRC, "main.tex")).read()
start = body.index(r"\section{Introduction}")
main_end = body.index(r"\bibliographystyle")
app_start = body.index(r"\appendix")
main_part = body[start:main_end]
app_part = body[app_start + len(r"\appendix"):].split(r"\end{document}")[0]
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1).strip()

WIDE = {"table_main_a.tex", "table_main_b.tex", "table_other_a.tex", "table_other_b.tex",
        "table_r1_ci_main_a.tex", "table_r1_ci_main_b.tex", "table_r1_ci_other.tex",
        "table_r1_ci_real.tex", "table_r1_hier.tex", "table_r1_diag.tex", "table_r1_decomp.tex",
        "table_support.tex", "table_r1_ci_extra.tex", "table_r1_ci_other_a.tex", "table_r1_ci_other_b.tex",
        "table_r1_ci_real_a.tex", "table_r1_ci_real_b.tex", "table_r1_ci_extra_a.tex",
        "table_r1_ci_extra_b.tex", "table_r1_ci_extra_c.tex", "table_r1_hier_heldout.tex", "table_r1_cost.tex",
        "table_l02.tex", "table_certgrid.tex", "table_refmodel.tex"}


def flat(s):
    s = s.replace("../results/figures/", "")
    # sn-jnl wraps tables in center+threeparttable, which breaks \resizebox; such tables use the
    # original float environment instead
    def fix_table(m):
        blk = m.group(0)
        if "resizebox" in blk or "table_sweeps" in blk or "table_main_" in blk or "table_other_" in blk or "table_r1_" in blk \
                or "table_support" in blk or "table_real_" in blk or "table_l02" in blk or "table_certgrid" in blk or "table_refmodel" in blk \
                or "Variant" in blk:
            blk = blk.replace("\\begin{table}", "\\begin{tableorg}", 1).replace("\\end{table}", "\\end{tableorg}")
            if "Variant" in blk and "resizebox" not in blk:
                blk = blk.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
                blk = blk.replace("\\end{tabular}", "\\end{tabular}}", 1)
        return blk
    s = re.sub(r"\\begin\{table\}.*?\\end\{table\}", fix_table, s, flags=re.S)
    s = re.sub(r"\\caption\*\{(Table~\\ref\{[^}]*\} \(continued\)\.)\}", r"\\noindent\\textit{\1}\\par\\smallskip", s)
    s = s.replace("p{0.62\\linewidth}", ">{\\raggedright\\arraybackslash}p{0.62\\linewidth}")
    return s

# copy inputs and figures flat
for f in glob.glob(os.path.join(SRC, "*.tex")) + glob.glob(os.path.join(SRC, "*.bib")):
    if os.path.basename(f) == "main.tex":
        continue
    txt = open(f).read() if f.endswith(".tex") else None
    if txt is not None:
        out_txt = flat(txt)
        if os.path.basename(f) in WIDE:
            out_txt = out_txt.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
            out_txt = out_txt.replace("\\end{tabular}", "\\end{tabular}}", 1)
        open(os.path.join(HERE, os.path.basename(f)), "w").write(out_txt)
    else:
        shutil.copy(f, HERE)
for f in glob.glob(os.path.join(FIG, "*.pdf")):
    shutil.copy(f, HERE)

tmpl = open(os.path.join(HERE, "template_head.tex")).read()
out = tmpl.replace("%%ABSTRACT%%", abstract).replace("%%MAIN%%", flat(main_part)).replace("%%APPENDIX%%", flat(app_part))
open(os.path.join(HERE, "main.tex"), "w").write(out)
print("built main.tex")
