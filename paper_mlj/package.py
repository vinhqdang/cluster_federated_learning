"""Zip only the files that main.tex really uses (flat, as Springer requires)."""
import os, re, zipfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "submission_mlj")
used = {"main.tex", "sn-jnl.cls", "sn-apacite.bst", "refs.bib", "main.bbl"}
todo = ["main.tex"]
seen = set()
while todo:
    f = todo.pop()
    if f in seen or not os.path.exists(os.path.join(HERE, f)):
        continue
    seen.add(f)
    t = open(os.path.join(HERE, f), encoding="latin1").read()
    for m in re.finditer(r"\\(?:input|includegraphics(?:\[[^\]]*\])?)\{([^}]+)\}", t):
        n = m.group(1)
        if not os.path.splitext(n)[1]:
            n += ".tex"
        used.add(n)
        if n.endswith(".tex"):
            todo.append(n)
missing = [u for u in used if not os.path.exists(os.path.join(HERE, u))]
assert not missing, missing
with zipfile.ZipFile(os.path.join(OUT, "manuscript_source.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for u in sorted(used):
        z.write(os.path.join(HERE, u), u)
shutil.copy(os.path.join(HERE, "main.pdf"), os.path.join(OUT, "manuscript.pdf"))
print(len(used), "files:", ", ".join(sorted(used)))
