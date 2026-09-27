/-
Proposition 2(i): the sensitivity of a clipped half-mean.

Signatures live in a real normed space `E`. Per-sample clipping to norm `C` is
`clip_C(x) = min(1, C/‖x‖) • x`. Replacing one record changes one clipped summand, so the
mean of `h` clipped vectors moves by at most `2C/h`; this is the sensitivity used to
calibrate the Gaussian mechanism.
-/
import Mathlib.Analysis.Normed.Module.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.Normed.Group.Basic

namespace DiscoProofs

open Finset

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Per-sample clipping. -/
noncomputable def clip (C : ℝ) (x : E) : E := (min 1 (C / ‖x‖)) • x

/-- A clipped vector has norm at most `C`. -/
theorem norm_clip_le (C : ℝ) (hC : 0 ≤ C) (x : E) : ‖clip C x‖ ≤ C := by
  unfold clip
  rcases eq_or_ne x 0 with hx | hx
  · simp [hx, hC]
  · have hnx : 0 < ‖x‖ := norm_pos_iff.mpr hx
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (le_min zero_le_one (by positivity))]
    calc min 1 (C / ‖x‖) * ‖x‖ ≤ (C / ‖x‖) * ‖x‖ :=
          mul_le_mul_of_nonneg_right (min_le_right _ _) hnx.le
      _ = C := by field_simp

/-- Replace-one sensitivity of the mean of `h` vectors of norm at most `C`: changing
the vector at a single index `s₀` moves the mean by at most `2C/h`. -/
theorem clipped_mean_sensitivity {n : ℕ} (hn : 0 < n) (C : ℝ) (g : Fin n → E)
    (hg : ∀ s, ‖g s‖ ≤ C) (s₀ : Fin n) (v : E) (hv : ‖v‖ ≤ C) :
    ‖(1 / (n : ℝ)) • ∑ s, g s - (1 / (n : ℝ)) • ∑ s, Function.update g s₀ v s‖
      ≤ 2 * C / n := by
  have hsum : ∑ s, g s - ∑ s, Function.update g s₀ v s = g s₀ - v := by
    rw [← Finset.sum_sub_distrib]
    rw [Finset.sum_eq_single s₀]
    · simp
    · intro b _ hb
      simp [Function.update_of_ne hb]
    · intro h; exact absurd (Finset.mem_univ s₀) h
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn
  rw [← smul_sub, hsum, norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity)]
  have h2 : ‖g s₀ - v‖ ≤ 2 * C := by
    calc ‖g s₀ - v‖ ≤ ‖g s₀‖ + ‖v‖ := norm_sub_le _ _
      _ ≤ C + C := add_le_add (hg s₀) hv
      _ = 2 * C := by ring
  calc 1 / (n : ℝ) * ‖g s₀ - v‖ ≤ 1 / (n : ℝ) * (2 * C) :=
        mul_le_mul_of_nonneg_left h2 (by positivity)
    _ = 2 * C / n := by ring

/-- The same bound for clipped samples: the mean of clipped vectors has replace-one
sensitivity `2C/h`, whatever the raw per-sample gradients are. -/
theorem clipped_signature_sensitivity {n : ℕ} (hn : 0 < n) (C : ℝ) (hC : 0 ≤ C)
    (x : Fin n → E) (s₀ : Fin n) (x' : E) :
    ‖(1 / (n : ℝ)) • ∑ s, clip C (x s) -
        (1 / (n : ℝ)) • ∑ s, Function.update (fun s => clip C (x s)) s₀ (clip C x') s‖
      ≤ 2 * C / n :=
  clipped_mean_sensitivity hn C (fun s => clip C (x s)) (fun s => norm_clip_le C hC (x s))
    s₀ (clip C x') (norm_clip_le C hC x')

end DiscoProofs
