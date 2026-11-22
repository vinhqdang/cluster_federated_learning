# Revision, round 3: pre-registration

Committed to the repository before any of the runs below. Three studies answer points that
remain open after the second internal review: the theory does not cover the default
shrinkage, the dependence on the reference model, and the calibration of the certificate
standard errors. Hyperparameters not named here are those of the paper.

## 1. DisCo-CFL with lambda = 0.02 (theorem-covered configuration)

Remark 2 shows that the evidence condition of Corollary 1 cannot hold at lambda = 0.5;
it can at lambda = 0.02. We rerun the full pipeline (warm-up, clustering, cluster
training, evaluation) with lambda = 0.02 and everything else unchanged.

- Variant `DisCo-L02` (lambda = 0.02).
- Suites `r1_l02`: the 16 small-CNN scenarios of `r1_as`, seeds 0-4 (80 runs);
  `r1_l02_sweeps`: the 8 sweep scenarios, seeds 0-2 (24 runs).
- H-L1: in every scenario where the default (lambda = 0.5) recovers the groups (mean ARI
  >= 0.8), the mean accuracy of DisCo-L02 is within 0.5 points of the default.
- H-L2: pooled over all scenarios, the mean absolute difference in ARI between
  DisCo-L02 and the default is at most 0.05.
- Reported for every scenario regardless of outcome.

## 2. Dependence on the reference model

The reference model is trained on the same clients that are clustered. We test whether
the clustering depends on which clients trained it.

- Study `refmodel` (clustering level, no cluster training): scenarios rot, swap,
  mixed_qs, label (Fashion-MNIST), mnist_rot, mnist_swap; seeds 0-2.
- Reference models, all 10 rounds of FedAvg from the same initialisation:
  (a) `full`: all clients (the default);
  (b) `half`: a random half of the clients (seeded, independent of the groups);
  (c) `drop_group`: only the clients of groups 1..K-1, i.e. one concept group (group 0)
      never contributes to the reference model (not run for the label-only scenario).
- All clients compute signatures at the chosen model and are clustered together.
- Reported: ARI, K, conflict-merge rate overall; ARI restricted to the clients that did and
  that did not train the reference model; for (c), the conflict-merge rate of pairs that
  involve group 0.
- H-R1: with (b), mean ARI is within 0.05 of (a) in each scenario where (a) has mean ARI
  >= 0.8, and the label-only federation keeps K = 1.
- H-R2: with (c), the excluded group is still separated: the conflict-merge rate of pairs
  involving group 0 is at most 5% in each scenario.

## 3. Calibration of the certificate standard errors

The paper reports one simulation (6 same-concept clients, 5 classes, 20 samples per
half). We repeat it on a grid with known population similarity.

- Synthetic signatures (discocfl.synthetic), p = 512, isotropic noise sigma = 1.
- Grid: samples per half h in {10, 20, 50, 200}; cluster size M in {3, 6, 12}; classes
  C in {2, 5, 10, 20}; two cases: same concept (population similarity 1) and a
  different concept that differs on one class with cosine 1 - Delta, Delta in {0.1, 0.3}
  (population value from signatures with 10^6 samples per half). 200 draws per cell.
- Reported per cell: ratio of the standard deviation of s over draws to the mean
  standard error; bias of s in units of the mean standard error; coverage of the
  +-2 se interval; coverage after removing the mean bias.
- H-S1: the ratio SD / mean se lies in [0.7, 1.5] in at least 80% of the cells.
- H-S2 (expected to fail, i.e. we expect the bias reported in the paper to persist):
  the +-2 se interval covers the population value in at least 90% of the draws in at
  least 80% of the cells.
