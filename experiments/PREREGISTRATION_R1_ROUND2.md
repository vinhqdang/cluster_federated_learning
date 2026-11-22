# Second set of revision experiments: pre-registration

Written before any of the runs below was executed. All outcomes will be reported.
Defaults (tau = 0.9, lambda = 0.5, p = 1024, 50 rounds, baselines) are unchanged.

## 1. Label-free adaptive warm-up (variant DisCo-AS)

Motivation, stated openly: with the fixed 10-round warm-up DisCo-CFL fails on SVHN and
CIFAR-10 because the reference model has not started to learn (training loss close to
log C). A rule that stops when the training loss plateaus failed for the same reason
(the loss is flat before learning starts). The rule below was designed after observing
these failures, from two principles, and its constants are round numbers fixed now; it is
not tuned on any outcome.

Rule. Checkpoints t = 10, 15, 20, 25, 30 rounds of FedAvg. At each checkpoint the clients
report the loss of the global model on their training data (one scalar each).
- Learning condition: the data-weighted loss is at most 0.75 log C (25% below the loss of
  a uniform prediction).
- Stability condition: at the checkpoint and at the previous checkpoint the learning
  condition holds, and the DisCo-CFL partitions computed at the two checkpoints agree
  (adjusted Rand index between them >= 0.9). No labels of groups are used.
Warm-up stops at the first checkpoint that satisfies both conditions, and at 30 rounds at
the latest; the remaining rounds of the 50-round budget train the clusters.

Evaluation: all small-CNN scenarios with five seeds (Fashion-MNIST main, MNIST, SVHN,
CIFAR-10, KMNIST, EMNIST) and the sweeps with three seeds.
Hypotheses: H-AS1: on Fashion-MNIST, MNIST, KMNIST and EMNIST rotations, DisCo-AS is within
0.5 points of DisCo-CFL in accuracy (no harm where the default works). H-AS2: on SVHN and
CIFAR-10, DisCo-AS has a higher mean ARI and accuracy than DisCo-CFL. No prediction of
full recovery on SVHN rotations and CIFAR-10, where 30 rounds were not enough before.

## 2. Concept drift after clustering

Fashion-MNIST, Rotation and Label swap, 3 seeds, clustering level. After clustering at the
reference model, a fraction q in {0.1, 0.2} of the clients switch to another concept group
(their data are transformed with the other group's rotation or permutation). The server
recomputes their signatures at the same reference model and applies the newcomer test to
every client: a client is flagged as drifted if its similarity to its own cluster is below
tau, and reassigned to the most similar cluster if that reaches tau.
Reported: detection rate of drifted clients, false-alarm rate on unchanged clients, and the
fraction of drifted clients reassigned to the cluster of their new group.
Hypothesis H-D1: detection rate >= 0.9 and false-alarm rate <= 0.05 on both scenarios.

## 3. Malicious clients

Fashion-MNIST, Rotation and Label swap, 3 seeds, clustering level, m in {1, 2, 4} malicious
clients out of 40. Attacks on the signature: (a) random vectors with the norm of honest
signatures; (b) a copy of an honest client's signature from another group (mimicry);
(c) the sign-flipped signature of an honest client.
Reported: ARI and conflict-merge rate among honest clients, and whether malicious clients
are isolated, flagged anomalous or low-evidence, or merged into an honest cluster.
No hypothesis of robustness is made: DisCo-CFL has no defence against adversarial clients,
and the study measures what happens.
