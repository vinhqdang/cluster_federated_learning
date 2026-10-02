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

WIDE = {"table_main_a.tex", "table_main_b.tex", "table_other_a.tex", "table_other_b.tex"}


def flat(s):
    s = s.replace("../results/figures/", "")
    # sn-jnl wraps tables in center+threeparttable, which breaks \resizebox; such tables use the
    # original float environment instead
    def fix_table(m):
        blk = m.group(0)
        if "resizebox" in blk or "table_sweeps" in blk or "table_main_" in blk or "table_other_" in blk \
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
