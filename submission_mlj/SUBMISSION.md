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

Submission ID 0be006c7-a304-4145-9451-f6289c9e780d. Files in `submission_mlj/r1/`:

| Item | File |
|---|---|
| Revised manuscript (PDF, 58 pages) | `r1/manuscript.pdf` |
| LaTeX source (flat folder) | `r1/manuscript_source.zip` |
| Response to the reviewers (7 pages) | `r1/response_letter.pdf` (source `r1/response_letter.tex`) |
| Manuscript with changes marked against the submitted version (commit 0f79ff2) | `r1/manuscript_marked.pdf` (built by `r1/make_diff.sh`) |

What changed:

- An implementation error in local training (a shared model vector was modified in place) was found and fixed; all experiments were rerun (1,293 small-CNN runs, 488 realistic/FEMNIST runs, all clustering studies; raw results in `results/r1/`). A regression test is in `tests/test_regression_r1.py`.
- The threshold was re-selected on the same pilot runs: tau = 0.9 (was 0.8).
- New baselines: StoCFL, StoCFL-W, FedCCFA, FedDAG-C (clustering stage). FedCCFA's evaluation was corrected once during the rerun and its 114 runs repeated.
- Five seeds, 95% intervals, failure rates, hierarchical bootstrap, held-out comparison, cost table, separation diagnostics, edge-case and class-support studies.
- Changed conclusions: DisCo-CFL fails on SVHN and CIFAR-10 with the 10-round warm-up (up to 17 points below the oracle); it no longer over-splits FEMNIST (it equals FedAvg there); PACS now succeeds.
- Pipeline: `experiments/stats_r1.py` (statistics tables), `experiments/summarize.py` and `summarize_real.py` (other tables and figures), `paper_mlj/build.py` and `paper_mlj/package.py` (journal version, writes to `submission_mlj/r1/`).
- Colab compute used for the rerun: about 44 compute units; the pre-registered additional benchmarks (515 runs) used about 15 more.
- Additional benchmarks: pre-registered in `experiments/PREREGISTRATION_R1_EXTRA.md` (KMNIST, EMNIST, CIFAR-10 with a pretrained network, AG News); all outcomes reported in Section 5.4.
- Still to do by the author: upload the four files above in the journal system before the deadline.
