# Additional benchmarks for the revision: pre-registration

Written before any of the runs below was executed. All results of these runs will be
reported in the manuscript, whatever their outcome. Hyperparameters are those of the
existing benchmarks (tau = 0.9, lambda = 0.5, T0 = 10, p = 1024, the same baselines and
budgets); nothing is tuned on these datasets.

## Selection criterion

The manuscript claims that DisCo-CFL recovers concept groups when the reference model
separates them (estimated separation clearly above 1 - tau) and fails otherwise. The
additional benchmarks test this claim on (A) a dataset where the original failure was
attributed to an uninformative reference model, (B) a new modality, and (C) two further
standard image benchmarks of the CFL literature.

## Benchmarks and hypotheses

A. CIFAR-10 with the pretrained ResNet-18 of the realistic benchmarks (same protocol as
   CIFAR-100: layer-4 block and classifier trained, 96x96 inputs, 40 clients, 4 groups,
   Dirichlet alpha = 0.5, mean 300 samples per client).
   - `c10r_swap` (label swaps, the 4 groups of the small-CNN label-swap scenario).
     H-A1: DisCo-CFL recovers the groups (mean ARI >= 0.8) and its accuracy is within
     1 point of the oracle.
   - `c10r_rot` (rotations by 0/90/180/270 degrees).
     H-A2: no prediction of success; on rotated CIFAR-100 the pretrained features did not
     separate rotations (estimated separation 0.05), and we expect the same here.

B. Text: AG News (4 classes) with a frozen pretrained sentence encoder and a trainable
   two-layer classifier head; 40 clients, 4 groups, Dirichlet alpha = 0.5.
   - `agnews_swap` (groups: identity labels and three different swaps of two classes).
     H-B1: DisCo-CFL recovers the groups (mean ARI >= 0.8).

C. Small CNN, trained from scratch as for MNIST and Fashion-MNIST:
   - `kmnist_rot`, `kmnist_swap` (Kuzushiji-MNIST, 10 classes).
   - `emnist_rot`, `emnist_swap` (EMNIST balanced, 47 classes; swaps of two class pairs per group).
     H-C1: DisCo-CFL recovers the groups (mean ARI >= 0.8) in all four scenarios.

## Protocol

Methods: FedAvg, Local, FedAvg-FT, IFCA, FeSEM, MTCFL, FL+HC, PACFL (image benchmarks
only), StoCFL, StoCFL-W, FedCCFA, FedDAG-C (image benchmarks only), DisCo-CFL,
DisCo-noAttach, Oracle. Five seeds (0-4). Metrics, failure rate, diagnostics and the
hierarchical statistics are computed exactly as for the existing benchmarks, and the new
scenarios enter the pooled hierarchical comparison as a separate block.

## Amendment (before any AG News run with the methods)

The text model is new, so it has no learning rate from the earlier benchmarks. Smoke tests
showed that a classifier on unit-norm sentence embeddings does not train with SGD
(coordinates of order 0.05; FedAvg stays at the majority-class accuracy for 20 rounds).
We therefore standardise every embedding dimension with the training-set statistics and
choose the learning rate among {0.01, 0.05, 0.1} with FedAvg on a single-concept AG News
federation (40 clients, seed 0, 50 rounds), the same procedure as for PACS. Accuracy after
10/20/50 rounds: lr 0.01: 0.699/0.793/0.859; lr 0.05: 0.857/0.874/0.889; lr 0.1:
0.874/0.884/0.895. We use lr 0.1. Apart from 3-round smoke tests that only checked that the code runs, no DisCo-CFL or
baseline run on AG News was made before this choice.
