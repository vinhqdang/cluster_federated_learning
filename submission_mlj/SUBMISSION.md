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

## Revision 1 (major revision, decision received 20 November 2026; recommended deadline 20 January 2027)

Submission ID 0be006c7-a304-4145-9451-f6289c9e780d. Status: ready to submit. Title unchanged (no reviewer asked to change it).

Files to upload (all in `submission_mlj/r1/`):

| Item in the submission system | File |
|---|---|
| Revised manuscript (PDF, 72 pages) | `r1/manuscript.pdf` |
| LaTeX source (flat folder; compiles standalone with pdflatex + bibtex) | `r1/manuscript_source.zip` |
| Response to reviewers (9 pages) | `r1/response_letter.pdf` (source `r1/response_letter.tex`) |
| Manuscript with changes marked against the submitted version (commit 0f79ff2) | `r1/manuscript_marked.pdf` (built by `r1/make_diff.sh`) |

Checks done before submission:

- Every reviewer request (Reviewer 1, points 1-7; Reviewer 2, points 1-4) is answered in the letter, and every section, table, figure and remark number cited in the letter matches the final manuscript.
- Numbers in the abstract, introduction, discussion, conclusion and letter were checked against the generated tables; all tables and figures are cited in the text; all abbreviations are defined at first use.
- The manuscript is self-contained: no references to code, files or the repository outside the Code availability statement.
- Abstract: 250 words.
- References: full author lists (style shows up to 20 authors); DOIs verified against Crossref/DataCite for 59 of 92 references (including three related works of the author in Section 2 and 5.7); the other 33 are venues without DOIs (NeurIPS, ICML, ICLR, AISTATS, MLSys, KDD, JMLR, technical reports).

What changed:

- An implementation error in local training (a shared model vector was modified in place) was found and fixed; all experiments were rerun (1,801 small-CNN runs, 703 runs with pretrained networks, FEMNIST and text, all clustering studies; raw results in `results/r1/`). Regression test: `tests/test_regression_r1.py`. The error is mentioned in the response letter only.
- Threshold re-selected on the same pilot runs: tau = 0.9 (was 0.8).
- New baselines: StoCFL, StoCFL-W, FedCCFA, FedDAG-C (clustering stage).
- Five seeds, 95% intervals, failure rates (per group and per dataset), hierarchical bootstrap, held-out comparison, cost table, separation diagnostics, edge-case and class-support studies.
- Section 5 rewritten: agenda (Q1-Q5), all benchmarks in 5.1, studies specified in advance listed once.
- Studies specified in advance (records in `experiments/PREREGISTRATION_R1_*.md`): additional benchmarks (KMNIST, EMNIST, CIFAR-10 with ResNet-18, AG News), label-free warm-up (DisCo-AS), drift test, lambda = 0.02 rerun, reference-model dependence, certificate calibration grid. Exploratory: malicious clients, certificates on further datasets.
- Changed conclusions: DisCo-CFL fails on SVHN and CIFAR-10 with the 10-round warm-up (up to 17 points below the oracle); it equals FedAvg on FEMNIST; PACS succeeds; certificates cannot detect a reference model that does not separate the groups.
- Colab compute used: about 59 compute units.
- Pipeline: `experiments/stats_r1.py`, `experiments/table_as.py`, `experiments/table_round3.py`, `experiments/summarize.py`, `summarize_real.py`, `paper_mlj/build.py`, `paper_mlj/package.py`.
- Still to do by the author: upload the four files above in the journal system before 20 January 2027.
