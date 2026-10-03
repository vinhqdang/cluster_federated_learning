# Submission log

## Machine Learning (Springer)

- **Submitted:** 3 October 2026 (Vietnam time)
- **Type:** Original research article, single-blind (author names on the title page)
- **Manuscript ID:** _(fill in from the confirmation e-mail)_
- **Status:** Submitted

### Files submitted

| Item | File |
|---|---|
| Manuscript (PDF, 45 pages, Springer Nature template) | `manuscript.pdf` |
| LaTeX source (flat folder, 36 files) | `manuscript_source.zip` |
| MLJ Contribution Information Sheet | `contribution_sheet.pdf` |
| Cover letter | `cover_letter.pdf` (also `cover_letter_with_information_sheet.pdf`) |

### Notes

- Source of the manuscript: `paper/` (long version); `paper_mlj/build.py` assembles the flat Springer folder and `paper_mlj/package.py` writes the zip.
- Structure: six sections (Introduction, Related work, Method, Theory, Experiments, Discussion and conclusion) and two appendices.
- Realistic benchmarks (pretrained ResNet-18, 150 runs; FEMNIST, 24 runs) are in `results/raw_real`. Known failures are reported in the paper: Office-Home and rotated CIFAR-100 (groups not separated), FEMNIST (over-splitting).
- Variants and ablations on the realistic benchmarks use one seed only.
- The code repository must stay public, since the manuscript links to it.
- Earlier submission: IEEE TNNLS (TNNLS-2026-P-51613), returned without review on 1 October 2026. Its files were removed from the repository (still in the git history).
