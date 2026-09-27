/-
Population-level calibration of the disattenuated similarity (Lemma 1)
and elementary facts used in Theorem 1.
-/
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace DiscoProofs

open Real

/-- Spearman–Brown: if the split-half reliability is `r = a / (a + t)` (signal `a`,
noise `t` of a half-mean), then `2r/(1+r) = a / (a + t/2)`, the reliability of the
full mean (whose noise is `t/2`). -/
theorem spearman_brown (a t : ℝ) (ha : 0 < a) (ht : 0 ≤ t) :
    2 * (a / (a + t)) / (1 + a / (a + t)) = a / (a + t / 2) := by
  have h1 : 0 < a + t := by linarith
  have h2 : 0 < a + t / 2 := by linarith
  have h3 : (1 + a / (a + t)) = (2 * a + t) / (a + t) := by
    field_simp
    try ring
  rw [h3]
  field_simp
  try ring

/-- Population calibration (Lemma 1): with population norms `‖m_i‖² = a_i + t_i/2`,
cross inner product `ip = ⟨μ_i, μ_j⟩` and reliabilities `R = a/(a+t/2)`,
the disattenuated cosine equals the cosine of the population class means,
`ip / √(a_i a_j)`, whatever the noise levels `t_i, t_j`. -/
theorem population_calibration (ai aj ti tj ip : ℝ)
    (hai : 0 < ai) (haj : 0 < aj) (hti : 0 ≤ ti) (htj : 0 ≤ tj) :
    (ip / Real.sqrt ((ai + ti / 2) * (aj + tj / 2))) /
        Real.sqrt ((ai / (ai + ti / 2)) * (aj / (aj + tj / 2)))
      = ip / Real.sqrt (ai * aj) := by
  have hi : 0 < ai + ti / 2 := by linarith
  have hj : 0 < aj + tj / 2 := by linarith
  have hprod : 0 < (ai + ti / 2) * (aj + tj / 2) := mul_pos hi hj
  have key : (ai / (ai + ti / 2)) * (aj / (aj + tj / 2))
      = (ai * aj) / ((ai + ti / 2) * (aj + tj / 2)) := by
    field_simp
  rw [key, Real.sqrt_div' _ hprod.le]
  have hs : 0 < Real.sqrt ((ai + ti / 2) * (aj + tj / 2)) := Real.sqrt_pos.mpr hprod
  have hs2 : 0 < Real.sqrt (ai * aj) := Real.sqrt_pos.mpr (mul_pos hai haj)
  field_simp

/-- The raw (uncorrected) population cosine is the calibrated one multiplied by the
attenuation factor `√(R_i R_j) ≤ 1`: noise biases the raw cosine towards zero. -/
theorem raw_cosine_attenuated (ai aj ti tj ip : ℝ)
    (hai : 0 < ai) (haj : 0 < aj) (hti : 0 ≤ ti) (htj : 0 ≤ tj) :
    ip / Real.sqrt ((ai + ti / 2) * (aj + tj / 2))
      = (ip / Real.sqrt (ai * aj)) *
          Real.sqrt ((ai / (ai + ti / 2)) * (aj / (aj + tj / 2))) := by
  have h := population_calibration ai aj ti tj ip hai haj hti htj
  have hi : 0 < ai + ti / 2 := by linarith
  have hj : 0 < aj + tj / 2 := by linarith
  have hpos : 0 < Real.sqrt ((ai / (ai + ti / 2)) * (aj / (aj + tj / 2))) :=
    Real.sqrt_pos.mpr (mul_pos (div_pos hai hi) (div_pos haj hj))
  rw [← h]
  field_simp

/-- The attenuation factor is at most one. -/
theorem attenuation_le_one (a t : ℝ) (ha : 0 < a) (ht : 0 ≤ t) :
    a / (a + t / 2) ≤ 1 := by
  rw [div_le_one (by linarith)]
  linarith

/-- Clipping an estimate to `[-1, 1]` never increases its distance to a target
that lies in `[-1, 1]`. -/
theorem clip_reduces_error (x ρ : ℝ) (h1 : -1 ≤ ρ) (h2 : ρ ≤ 1) :
    |max (-1) (min 1 x) - ρ| ≤ |x - ρ| := by
  rcases le_total x 1 with hx | hx
  · rw [min_eq_right hx]
    rcases le_total (-1) x with hy | hy
    · rw [max_eq_right hy]
    · rw [max_eq_left hy]
      rw [abs_le]; constructor <;> cases abs_cases (x - ρ) <;> linarith
  · rw [min_eq_left hx, max_eq_right (by norm_num)]
    rw [abs_le]; constructor <;> cases abs_cases (x - ρ) <;> linarith

end DiscoProofs
