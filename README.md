# DisCo-CFL — Disattenuated Class-Conditional Clustered Federated Learning

DisCo-CFL is a clustered federated learning (CFL) algorithm. It groups clients by
**concept** (the class-conditional distribution P(X | Y)), not by label distribution or
dataset size. It determines the number of clusters automatically with one threshold on a
**calibrated** similarity scale, and it comes with provable guarantees and with fairness,
accountability, transparency and privacy properties that go beyond accuracy.

The design targets open problems identified in *A Survey on Clustered Federated Learning:
Taxonomy, Analysis and Applications* (Ben Ali et al., ACM Computing Surveys 58(16), 2026):
unstable high-dimensional clustering features, grouping by label similarity, automatic
choice of K, quantity skew, privacy of the clustering step, and evaluation that looks only
at performance.

## Key idea

At a frozen reference model `w` (after a short FedAvg warm-up), each client uploads **once**,
for every class `c` it holds, the count-sketched mean gradient of the loss on **two random
halves** of its class-`c` samples: `(a_c, b_c)`, each of dimension `p = 1024`.

* The expected class-conditional gradient `mu_c = E[grad l(w; x, c) | x ~ P_i(X | Y=c)]`
  depends on a client **only through P_i(X | Y=c)**. It is therefore exactly invariant to
  label skew and quantity skew, and it changes under concept shift on features (e.g.
  rotation) and on targets (e.g. label swap).
* Estimated signatures are noisy, and noise *attenuates* cosine similarities. The two halves
  give the reliability `R_c = 2 r_c / (1 + r_c)` with `r_c = cos(a_c, b_c)`
  (Spearman–Brown). **Spearman's correction for attenuation** then gives the calibrated
  similarity

  `rho_ij^c = cos(m_i^c, m_j^c) / sqrt(R_i^c R_j^c)`,   with `m = (a + b) / 2`.

  This is a consistent estimator of the population cosine, so `1` means "same concept" for
  every client, whatever its sample size or DP noise level.
* Clustering uses agglomerative merging with a **class-wise bottleneck (min) linkage**,
  because two groups share a concept only if *every* class agrees. Merges are ordered by
  evidence and stop at a fixed threshold `tau = 0.8`. K is not an input.

## Properties beyond accuracy

| Property | What DisCo-CFL provides | Guarantee / measurement |
|---|---|---|
| **Fairness (group size)** | minority concept groups are not absorbed by large groups | recovery sample complexity independent of group size (Cor. 1); minority recall / purity, worst-group accuracy |
| **Fairness (data size)** | small clients are not split off by noise | disattenuation removes the sample-size bias (Thm. 1); low-evidence clients are attached, not isolated |
| **Transparency** | class-level explanation of why clusters differ, automatic heterogeneity diagnosis (feature shift vs. class-specific target shift vs. label/quantity skew only) | explanation fidelity (Prop. 3); precision/recall of the divergent classes |
| **Accountability** | per-client assignment certificate (similarity to own/closest cluster, standard errors, verdict), audit log of every merge, low-evidence and anomaly flags, newcomer and novel-concept detection | certificate calibration |
| **Privacy** | optional (ε, δ)-DP signatures (per-sample clipping + Gaussian noise); one-shot, low-dimensional upload | DP by the analytic Gaussian mechanism; the disattenuation stays consistent under DP noise (Prop. 2) |
| **Theory** | invariance, concentration, safety / maximality / exact recovery, minimax lower bound | Prop. 1, Thm. 1–3, Cor. 1 (proofs in `paper/`) |

## Theoretical results (summary)

* **Invariance.** `mu_i^c` depends only on `P_i(X | Y=c)`, so label skew and quantity skew
  leave the population similarity at exactly 1.
* **Concentration.** `|rho_hat − rho| ≤ C (sqrt(κ/r_s) + κ/sqrt(r_e)) sqrt(log(1/δ))`,
  where κ is the inverse SNR of a half-mean and r_s, r_e are effective ranks of the
  gradient noise. The raw cosine is biased by the factor `sqrt(R_i R_j)`.
* **Safety / maximality / exact recovery.** No two clients with conflicting concepts are
  ever merged. No two output clusters could be merged without mixing concepts. Under full
  class support the ground-truth partition is recovered and K̂ = K. Required samples per
  half-class: `h* = O(σ² log(N²C/δ) / (a r_s γ²))`, independent of cluster sizes.
* **Lower bound.** Any test needs `n ≥ σ² log(1/(4δ)) / (a r_s Δ)` samples. This matches
  the upper bound in σ², a and r_s, up to a factor 1/Δ and a log N term.
* **DP.** Signatures are (ε, δ)-DP. The required per-class sample size grows linearly in
  1/ε, while the raw cosine collapses under DP noise.

## Repository layout

```
discocfl/
  disco.py        # signatures, disattenuated similarity, clustering, certificates, explanations, DP, newcomers
  algorithms.py   # DisCo-CFL and baselines: FedAvg, Local, IFCA, CFL/MTCFL, FL+HC, PACFL, Oracle
  data.py         # federated non-IID generation: rotation / label swap / label skew / quantity skew / minority groups
  fl.py           # model (small CNN) and FedAvg primitives
  synthetic.py    # synthetic signatures with known ground truth (tests, theory validation)
experiments/
  run.py              # accuracy benchmark (scenarios x methods x seeds)
  clustering_study.py # tau / DP / sample size / minority / explanations / certificates / newcomers / scalability
  theory_sim.py       # synthetic validation of the theorems
  summarize.py        # tables (results/tables) and figures (results/figures)
paper/            # LaTeX manuscript with full proofs
tests/            # unit tests
results/          # raw results, tables, figures
```

## Reproducing

```bash
pip install -r requirements.txt
export DISCO_DATA=/path/to/data          # MNIST/, FashionMNIST/ (raw idx files), cifar-10-python.tar.gz
python -m pytest -q tests
python experiments/theory_sim.py
python experiments/run.py --suite main --workers 4
python experiments/run.py --suite ablation --workers 4
python experiments/run.py --suite mnist --workers 4
python experiments/clustering_study.py --study all --workers 4
python experiments/summarize.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Results

RESULTS_PLACEHOLDER
