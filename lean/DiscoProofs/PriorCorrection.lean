/-
Proposition 4: prior correction is Bayes-optimal within a concept cluster.

Classes form a finite type `Y`. All clients of the cluster share the class-conditional
likelihoods `f y = P(x | y) > 0` at a fixed input `x`; client `i` has prior `πi`, the
cluster has pooled prior `πk` (both positive).
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Field

namespace DiscoProofs

open Finset

variable {Y : Type*} [Fintype Y]

/-- Bayes posterior with prior `π` and likelihood `f`. -/
noncomputable def posterior (π f : Y → ℝ) (y : Y) : ℝ :=
  π y * f y / ∑ y', π y' * f y'

/-- The posterior of client `i` is proportional to the cluster posterior re-weighted by
the prior ratio `πi / πk`, with a constant that does not depend on the class. -/
theorem prior_correction_proportional (πi πk f : Y → ℝ)
    (hi : ∀ y, 0 < πi y) (hk : ∀ y, 0 < πk y) (hf : ∀ y, 0 < f y) [Nonempty Y] :
    ∃ c : ℝ, 0 < c ∧ ∀ y, posterior πi f y = c * (posterior πk f y * (πi y / πk y)) := by
  have hSi : 0 < ∑ y', πi y' * f y' :=
    Finset.sum_pos (fun y _ => mul_pos (hi y) (hf y)) Finset.univ_nonempty
  have hSk : 0 < ∑ y', πk y' * f y' :=
    Finset.sum_pos (fun y _ => mul_pos (hk y) (hf y)) Finset.univ_nonempty
  refine ⟨(∑ y', πk y' * f y') / (∑ y', πi y' * f y'), div_pos hSk hSi, ?_⟩
  intro y
  have := hk y
  unfold posterior
  field_simp

/-- Consequently the corrected score `log q_k(y) + log πi(y) - log πk(y)` orders the
classes exactly as the Bayes posterior of client `i`: the prior-corrected cluster model
makes the Bayes-optimal decision for every member. -/
theorem prior_correction_argmax (πi πk f : Y → ℝ)
    (hi : ∀ y, 0 < πi y) (hk : ∀ y, 0 < πk y) (hf : ∀ y, 0 < f y) [Nonempty Y] (y y' : Y) :
    posterior πi f y ≤ posterior πi f y' ↔
      Real.log (posterior πk f y) + Real.log (πi y) - Real.log (πk y) ≤
        Real.log (posterior πk f y') + Real.log (πi y') - Real.log (πk y') := by
  obtain ⟨c, hc, hprop⟩ := prior_correction_proportional πi πk f hi hk hf
  have hSk : 0 < ∑ y', πk y' * f y' :=
    Finset.sum_pos (fun y _ => mul_pos (hk y) (hf y)) Finset.univ_nonempty
  have hq : ∀ z, 0 < posterior πk f z := fun z =>
    div_pos (mul_pos (hk z) (hf z)) hSk
  have hg : ∀ z, 0 < posterior πk f z * (πi z / πk z) := fun z =>
    mul_pos (hq z) (div_pos (hi z) (hk z))
  have hlog : ∀ z, Real.log (posterior πk f z) + Real.log (πi z) - Real.log (πk z) =
      Real.log (posterior πk f z * (πi z / πk z)) := by
    intro z
    rw [Real.log_mul (hq z).ne' (div_pos (hi z) (hk z)).ne',
      Real.log_div (hi z).ne' (hk z).ne']
    ring
  rw [hlog y, hlog y', hprop y, hprop y', mul_le_mul_iff_of_pos_left hc,
    Real.log_le_log_iff (hg y) (hg y')]

end DiscoProofs
