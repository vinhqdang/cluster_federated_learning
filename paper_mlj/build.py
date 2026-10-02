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
app_part = body[app_start + len(r"\appendix"):]
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1).strip()

def flat(s):
    s = s.replace("../results/figures/", "")
    s = re.sub(r"\\caption\*\{([^}]*)\}", r"\\captionsetup{labelformat=empty}\\caption{\1}\\captionsetup{labelformat=default}", s)
    return s

# copy inputs and figures flat
for f in glob.glob(os.path.join(SRC, "*.tex")) + glob.glob(os.path.join(SRC, "*.bib")):
    if os.path.basename(f) == "main.tex":
        continue
    txt = open(f).read() if f.endswith(".tex") else None
    if txt is not None:
        open(os.path.join(HERE, os.path.basename(f)), "w").write(flat(txt))
    else:
        shutil.copy(f, HERE)
for f in glob.glob(os.path.join(FIG, "*.pdf")):
    shutil.copy(f, HERE)

tmpl = open(os.path.join(HERE, "template_head.tex")).read()
out = tmpl.replace("%%ABSTRACT%%", abstract).replace("%%MAIN%%", flat(main_part)).replace("%%APPENDIX%%", flat(app_part))
open(os.path.join(HERE, "main.tex"), "w").write(out)
print("built main.tex")
