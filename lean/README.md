# Lean 4 verification of the DisCo-CFL proofs

This Lean 4 project (Mathlib) machine-checks the deterministic core of every result in the
paper. The probabilistic inputs (sub-Gaussian and Hanson–Wright concentration, the
Bretagnolle–Huber inequality, the analytic Gaussian mechanism) are standard published results
that the paper cites. They enter the formal statements as hypotheses: "on the good event"
or "given the concentration bound".

```bash
curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
cd lean
lake exe cache get      # precompiled Mathlib
lake build              # checks all proofs
lake env lean AxiomsCheck.lean   # every theorem depends only on propext, Classical.choice, Quot.sound
```

No proof uses `sorry`. `AxiomsCheck.lean` prints the axioms of all 25 theorems.

| Paper result | Lean theorem(s) | File | What is verified |
|---|---|---|---|
| Lemma 1 (population calibration) | `spearman_brown`, `population_calibration`, `raw_cosine_attenuated`, `attenuation_le_one` | `Calibration.lean` | Spearman–Brown extension; the disattenuated population cosine equals the cosine of the class means for any noise levels; the raw cosine is attenuated by `√(R_i R_j) ≤ 1` |
| Theorem 1 (clipping step) | `clip_reduces_error` | `Calibration.lean` | clipping the estimate to [-1, 1] never increases its error |
| Theorem 2, Claim 1 | `claim1_conflict_not_linked` | `Linkage.lean` | conflicting clusters with evidence above `w₀` have linkage below τ |
| Theorem 2, Claim 2 | `claim2_compatible_linked`, `claim2_lower_bound` | `Linkage.lean` | compatible clusters have linkage above τ (at least (1+τ)/2) |
| Theorem 2(a) safety | `singleton_conflict_free`, `merge_preserves_conflict_free`, `safety_all_merges` | `Clustering.lean` | every state reachable by **any** sequence of allowed merges is conflict-free |
| Theorem 2(b) maximality | `maximality` | `Clustering.lean` | at termination, no two output clusters can be merged without a conflict |
| Theorem 2(c) exact recovery | `exact_recovery` | `Clustering.lean` | a conflict-free maximal partition equals the group partition under full support |
| Corollary 1 (arithmetic) | `sample_size_sufficient`, `evidence_condition` | `Linkage.lean` | the sample-size condition implies the concentration requirement; the λ condition implies `W > w₀` |
| Theorem 3 (lower bound) | `lower_bound_construction`, `lower_bound_cosine`, `kl_value`, `lower_bound_sample_size` | `LowerBound.lean` | geometry of the two hypotheses, KL value `n a Δ r_s/σ²`, and the resulting bound on `n` |
| Proposition 2(i) (DP sensitivity) | `norm_clip_le`, `clipped_mean_sensitivity`, `clipped_signature_sensitivity` | `Privacy.lean` | clipped vectors have norm ≤ C; replace-one sensitivity of the clipped mean is `2C/h` |
| Proposition 3 (explanation fidelity) | `explanation_fidelity` | `Linkage.lean` | a class is reported as divergent iff the concepts differ on it |
| Proposition 4 (prior correction) | `prior_correction_proportional`, `prior_correction_argmax` | `PriorCorrection.lean` | the corrected cluster model ranks classes exactly as the client's Bayes posterior |

**Not formalized.** The concentration bound of Theorem 1 (sub-Gaussian and Hanson–Wright
tail bounds and the smoothness argument), the Bretagnolle–Huber inequality, and the
(ε, δ) accounting of the analytic Gaussian mechanism. These are cited results, and Mathlib
does not yet provide them in the required form.
