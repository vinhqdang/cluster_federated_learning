/-
Theorem 3 (minimax lower bound): the two-point construction and the value of the
Kullback–Leibler divergence. The information-theoretic step (Bretagnolle–Huber) is
standard and cited; here we verify the geometry of the two hypotheses and the algebra
that turns the KL divergence into `n a Δ r_s / σ²`.
-/
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace DiscoProofs

open Real
open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Two class means `u, v` of squared norm `a`, symmetric around `e₂` along the top
eigenvector `e₁` of the noise covariance: they have equal norms, inner product
`a (1 - 2 s²)` (so cosine `1 - Δ` for `s² = Δ/2`) and difference `2√a s e₁`. -/
theorem lower_bound_construction (e₁ e₂ : E) (h₁ : ‖e₁‖ = 1) (h₂ : ‖e₂‖ = 1)
    (h₁₂ : ⟪e₁, e₂⟫ = 0) (a s : ℝ) (ha : 0 ≤ a) (hs : s ^ 2 ≤ 1) :
    let u := Real.sqrt a • (Real.sqrt (1 - s ^ 2) • e₂ + s • e₁)
    let v := Real.sqrt a • (Real.sqrt (1 - s ^ 2) • e₂ - s • e₁)
    ⟪u, u⟫ = a ∧ ⟪v, v⟫ = a ∧ ⟪u, v⟫ = a * (1 - 2 * s ^ 2) ∧
      u - v = (2 * Real.sqrt a * s) • e₁ := by
  intro u v
  have e11 : ⟪e₁, e₁⟫ = 1 := by rw [real_inner_self_eq_norm_sq, h₁]; norm_num
  have e22 : ⟪e₂, e₂⟫ = 1 := by rw [real_inner_self_eq_norm_sq, h₂]; norm_num
  have e21 : ⟪e₂, e₁⟫ = 0 := by rw [real_inner_comm]; exact h₁₂
  have hsa : Real.sqrt a * Real.sqrt a = a := Real.mul_self_sqrt ha
  have hs1 : Real.sqrt (1 - s ^ 2) * Real.sqrt (1 - s ^ 2) = 1 - s ^ 2 :=
    Real.mul_self_sqrt (by linarith)
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp only [u, inner_smul_left, inner_smul_right, inner_add_left, inner_add_right,
      e11, e22, h₁₂, e21, RCLike.conj_to_real]
    nlinarith [hsa, hs1]
  · simp only [v, inner_smul_left, inner_smul_right, inner_sub_left, inner_sub_right,
      e11, e22, h₁₂, e21, RCLike.conj_to_real]
    nlinarith [hsa, hs1]
  · simp only [u, v, inner_smul_left, inner_smul_right, inner_add_left, inner_sub_right,
      e11, e22, h₁₂, e21, RCLike.conj_to_real]
    nlinarith [hsa, hs1]
  · simp only [u, v, ← smul_sub]
    rw [show Real.sqrt (1 - s ^ 2) • e₂ + s • e₁ - (Real.sqrt (1 - s ^ 2) • e₂ - s • e₁)
        = (2 * s) • e₁ by
      rw [add_sub_sub_cancel, ← two_smul ℝ (s • e₁), smul_smul]]
    rw [smul_smul]
    congr 1
    ring

/-- With `s² = Δ/2`, the cosine of the two hypotheses is `1 - Δ`. -/
theorem lower_bound_cosine (a Δ : ℝ) (ha : 0 < a) :
    a * (1 - 2 * (Δ / 2)) / a = 1 - Δ := by
  field_simp
  try ring

/-- The KL divergence between `N(u,Σ)^{⊗n}` and `N(v,Σ)^{⊗n}` with `u - v` along the
top eigenvector (eigenvalue `λ₁ = σ²/r_s`) is `(n/2)·4 a s²/λ₁`; with `s² = Δ/2` this
equals `n a Δ r_s / σ²`. -/
theorem kl_value (n a Δ σ2 rs : ℝ) (hσ : 0 < σ2) (hrs : 0 < rs) :
    n / 2 * (4 * a * (Δ / 2) / (σ2 / rs)) = n * a * Δ * rs / σ2 := by
  field_simp
  try ring

/-- The resulting sample-size requirement: if the error probability must be at most
`δ` and `(1/4) exp(-n a Δ r_s / σ²) ≤ δ` is necessary, then
`n ≥ σ²/(a r_s Δ) · log(1/(4δ))`. -/
theorem lower_bound_sample_size (n a Δ σ2 rs δ : ℝ) (ha : 0 < a) (hΔ : 0 < Δ)
    (hσ : 0 < σ2) (hrs : 0 < rs) (hδ : 0 < δ)
    (h : 1 / 4 * Real.exp (-(n * a * Δ * rs / σ2)) ≤ δ) :
    σ2 / (a * rs * Δ) * Real.log (1 / (4 * δ)) ≤ n := by
  have hk : 0 < a * rs * Δ := by positivity
  have h1 : Real.exp (-(n * a * Δ * rs / σ2)) ≤ 4 * δ := by linarith
  have h2 : -(n * a * Δ * rs / σ2) ≤ Real.log (4 * δ) := by
    rw [← Real.exp_le_exp, Real.exp_log (by positivity)]; exact h1
  have h3 : Real.log (1 / (4 * δ)) = - Real.log (4 * δ) := by
    rw [one_div, Real.log_inv]
  rw [h3]
  have h4 : n * a * Δ * rs / σ2 = n * (a * rs * Δ) / σ2 := by ring
  rw [h4] at h2
  rw [div_mul_eq_mul_div, div_le_iff₀ hk]
  have : -Real.log (4 * δ) ≤ n * (a * rs * Δ) / σ2 := by linarith
  rw [le_div_iff₀ hσ] at this
  nlinarith

end DiscoProofs
