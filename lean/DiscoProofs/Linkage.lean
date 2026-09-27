/-
The deterministic core of Theorem 2 (Claims 1 and 2), Proposition 3 (explanation
fidelity) and the arithmetic of Corollary 1.

All statements are about the shrunk pooled class similarity
  `(num + λ) / (den + λ)`,
where `num = Σ W ρ̂` and `den = Σ W` over the client pairs of two clusters.
-/
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace DiscoProofs

/-- Claim 1: if every pair behind the pooled statistic conflicts (so on the good event
`num ≤ (1 - Δ + γ/2) den`), and the evidence exceeds `w₀ = 2λ(1-τ)/γ`, then the shrunk
pooled similarity is below the threshold `τ`, hence the two clusters are never linked. -/
theorem claim1_conflict_not_linked (num den lam tau Δ γ : ℝ)
    (hlam : 0 < lam) (htau : tau < 1) (hγ : 0 < γ) (hγΔ : γ ≤ tau - 1 + Δ)
    (hnum : num ≤ (1 - Δ + γ / 2) * den)
    (hden : 2 * lam * (1 - tau) / γ < den) :
    (num + lam) / (den + lam) < tau := by
  have hden0 : 0 < den := lt_trans (by positivity) hden
  have hpos : 0 < den + lam := by linarith
  rw [div_lt_iff₀ hpos]
  have h1 : 2 * lam * (1 - tau) < den * γ := by
    rwa [div_lt_iff₀ hγ] at hden
  nlinarith

/-- Claim 2: if all pairs behind the pooled statistic are compatible (so on the good
event `num ≥ (1 - γ/2) den`) and `γ ≤ 1 - τ`, the shrunk pooled similarity exceeds `τ`
(it is at least `(1 + τ)/2`). This also covers classes without evidence (`den = 0`). -/
theorem claim2_compatible_linked (num den lam tau γ : ℝ)
    (hlam : 0 < lam) (hden : 0 ≤ den) (hγ : 0 ≤ γ) (hγτ : γ ≤ 1 - tau) (htau : tau < 1)
    (hnum : (1 - γ / 2) * den ≤ num) :
    tau < (num + lam) / (den + lam) := by
  have hpos : 0 < den + lam := by linarith
  rw [lt_div_iff₀ hpos]
  nlinarith

/-- The pooled value of a compatible pair is at least `(1+τ)/2`. -/
theorem claim2_lower_bound (num den lam tau γ : ℝ)
    (hlam : 0 < lam) (hden : 0 ≤ den) (hγτ : γ ≤ 1 - tau) (htau : tau < 1)
    (hnum : (1 - γ / 2) * den ≤ num) :
    (1 + tau) / 2 ≤ (num + lam) / (den + lam) := by
  have hpos : 0 < den + lam := by linarith
  rw [le_div_iff₀ hpos]
  nlinarith

/-- Proposition 3 (explanation fidelity), deterministic core: on the good event, the
class is reported as divergent (pooled value `< τ`) exactly when the two concepts differ
on it. -/
theorem explanation_fidelity (differ : Prop) (num den lam tau Δ γ : ℝ)
    (hlam : 0 < lam) (htau : tau < 1) (hγ : 0 < γ) (hγΔ : γ ≤ tau - 1 + Δ)
    (hγτ : γ ≤ 1 - tau) (hden : 2 * lam * (1 - tau) / γ < den)
    (hdiff : differ → num ≤ (1 - Δ + γ / 2) * den)
    (hsame : ¬ differ → (1 - γ / 2) * den ≤ num) :
    (num + lam) / (den + lam) < tau ↔ differ := by
  have hden0 : 0 ≤ den := le_of_lt (lt_trans (by positivity) hden)
  constructor
  · intro hlt
    by_contra hnd
    have := claim2_compatible_linked num den lam tau γ hlam hden0 hγ.le hγτ htau (hsame hnd)
    linarith
  · intro hd
    exact claim1_conflict_not_linked num den lam tau Δ γ hlam htau hγ hγΔ (hdiff hd) hden

/-- Corollary 1, evidence condition: if both reliabilities are at least `2/3 - γ/2`
with `γ ≤ 1/3`, the pair weight `W = R_i R_j` is at least `1/4`; if moreover
`λ < γ / (8 (1 - τ))`, then `W > w₀ = 2λ(1-τ)/γ`. -/
theorem evidence_condition (Ri Rj γ lam tau : ℝ)
    (hγ : 0 < γ) (hγ3 : γ ≤ 1 / 3) (htau : tau < 1)
    (hRi : 2 / 3 - γ / 2 ≤ Ri) (hRj : 2 / 3 - γ / 2 ≤ Rj)
    (hsmall : lam < γ / (8 * (1 - tau))) :
    2 * lam * (1 - tau) / γ < Ri * Rj := by
  have hRi' : 1 / 2 ≤ Ri := by linarith
  have hRj' : 1 / 2 ≤ Rj := by linarith
  have hW : 1 / 4 ≤ Ri * Rj := by nlinarith
  have h1τ : 0 < 1 - tau := by linarith
  have hlam' : lam * (8 * (1 - tau)) < γ := by
    rwa [lt_div_iff₀ (by positivity)] at hsmall
  have : 2 * lam * (1 - tau) / γ < 1 / 4 := by
    rw [div_lt_iff₀ hγ]; nlinarith
  linarith

/-- Corollary 1, sample size: with `κ = σ²/(a h)`, the two requirements of the
concentration bound, `√(κ/r_s)·√L ≤ γ/(4C)` and `(κ/√r_e)·√L ≤ γ/(4C)`, hold as soon as
`h ≥ 16 C² σ² L / (a r_s γ²)` and `h ≥ 4 C σ² √L / (a √r_e γ)`. -/
theorem sample_size_sufficient (σ2 a h rs re L C γ : ℝ)
    (hσ : 0 < σ2) (ha : 0 < a) (hh : 0 < h) (hrs : 0 < rs) (hre : 0 < re)
    (hC : 0 < C) (hγ : 0 < γ)
    (h1 : 16 * C ^ 2 * σ2 * L / (a * rs * γ ^ 2) ≤ h)
    (h2 : 4 * C * σ2 * Real.sqrt L / (a * Real.sqrt re * γ) ≤ h) :
    Real.sqrt (σ2 / (a * h) / rs) * Real.sqrt L ≤ γ / (4 * C) ∧
      σ2 / (a * h) / Real.sqrt re * Real.sqrt L ≤ γ / (4 * C) := by
  have hsre : 0 < Real.sqrt re := Real.sqrt_pos.mpr hre
  have hγC : 0 < γ / (4 * C) := by positivity
  constructor
  · -- square both sides
    rw [← Real.sqrt_mul (by positivity)]
    rw [Real.sqrt_le_left (le_of_lt hγC)]
    have : 16 * C ^ 2 * σ2 * L ≤ h * (a * rs * γ ^ 2) := by
      rwa [div_le_iff₀ (by positivity)] at h1
    rw [div_pow, show (4 * C) ^ 2 = 16 * C ^ 2 by ring]
    rw [le_div_iff₀ (by positivity)]
    rw [div_div, div_mul_eq_mul_div, div_mul_eq_mul_div, div_le_iff₀ (by positivity)]
    nlinarith [sq_nonneg γ]
  · have : 4 * C * σ2 * Real.sqrt L ≤ h * (a * Real.sqrt re * γ) := by
      rwa [div_le_iff₀ (by positivity)] at h2
    rw [le_div_iff₀ (by positivity)]
    rw [div_div, div_mul_eq_mul_div, div_mul_eq_mul_div, div_le_iff₀ (by positivity)]
    nlinarith

end DiscoProofs
