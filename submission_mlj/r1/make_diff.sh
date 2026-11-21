#!/usr/bin/env bash
# Build the marked-up manuscript (changes against the submitted version, commit 0f79ff2).
# usage: bash submission_mlj/r1/make_diff.sh   (run from the repository root)
set -euo pipefail
ROOT=$(pwd)
TMP=$(mktemp -d)
mkdir -p "$TMP/old" "$TMP/new"
git archive 0f79ff2 paper_mlj | tar x -C "$TMP/old"
cp paper_mlj/*.tex paper_mlj/*.bib paper_mlj/*.bst paper_mlj/*.cls paper_mlj/*.pdf "$TMP/new/" 2>/dev/null || true
rm -f "$TMP"/old/paper_mlj/*.bbl "$TMP"/new/*.bbl "$TMP"/new/main.pdf
cd "$TMP/new"
latexdiff --flatten --math-markup=whole --disable-citation-markup \
  "$TMP/old/paper_mlj/main.tex" main.tex > diff_main.tex
# latexdiff marks up the arguments of \cmidrule in new tables; restore them
python3 - <<'PY'
import re
s = open("diff_main.tex").read()
s = re.sub(r"\\cmidrule\\DIF(?:add|del)FL\{(\([a-z]*\))\}\{\\DIF(?:add|del)FL\{([0-9]+-[0-9]+)\}\}",
           lambda m: "\\cmidrule" + m.group(1) + "{" + m.group(2) + "}", s)
open("diff_main.tex", "w").write(s)
PY
for step in 1 2 3; do
  pdflatex -interaction=nonstopmode diff_main.tex > /dev/null || true
  [ "$step" = 1 ] && bibtex diff_main > /dev/null || true
done
cp diff_main.pdf "$ROOT/submission_mlj/r1/manuscript_marked.pdf"
echo "written submission_mlj/r1/manuscript_marked.pdf"
