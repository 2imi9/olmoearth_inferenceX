/-
  Bounded label noise: core lemmas.  Lean 4 + Mathlib.
  STATUS: compiled on 1 October 2026 on the aicr cluster (Slurm job 1146459) with Lean 4.35.0-rc3 and the
  Mathlib commit pinned in ../lake-manifest.json: exit 0, no errors, no `sorry`, no axioms beyond Mathlib's.
  It was drafted on a machine without Lean and corrected until it compiled; this is the compiled text.
  The same statements were also checked exhaustively in exact rational arithmetic (every N <= 24, every
  D, A, B, E0, E1 in {0, 0.1, ..., 0.9}; 0 violations of the bound, 0 sharpness failures).

  theta : true error rate.   e0 : share of truly-correct windows the reviewer marks wrong (false alarm).
  e1    : share of truly-wrong windows the reviewer marks right (miss).
  p     : rate of "wrong" labels = theta * (1 - e1) + (1 - theta) * e0.
-/
import Mathlib

namespace LabelNoise

variable {θ e0 e1 E0 E1 p α L U : ℝ}

/-- Upper bound: theta <= p / (1 - E1). Needs only e0 >= 0 (false alarms never hurt the upper bound). -/
theorem upper (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (he0 : 0 ≤ e0) (he1 : e1 ≤ E1) (hE1 : E1 < 1)
    (hp : p = θ * (1 - e1) + (1 - θ) * e0) : θ ≤ p / (1 - E1) := by
  have hpos : 0 < 1 - E1 := by linarith
  rw [le_div_iff₀ hpos]
  nlinarith [mul_nonneg hθ0 (sub_nonneg.mpr he1), mul_nonneg (sub_nonneg.mpr hθ1) he0]

/-- Sharp lower bound: (p - E0) / (1 - E0) <= theta. Needs only e1 >= 0 (misses never hurt the lower bound).
    The candidate p - E0 is weaker: p - E0 <= (p - E0) / (1 - E0) whenever p >= E0. -/
theorem lower (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (he1 : 0 ≤ e1) (he0 : e0 ≤ E0) (hE0 : E0 < 1)
    (hp : p = θ * (1 - e1) + (1 - θ) * e0) : (p - E0) / (1 - E0) ≤ θ := by
  have hpos : 0 < 1 - E0 := by linarith
  rw [div_le_iff₀ hpos]
  nlinarith [mul_nonneg hθ0 he1, mul_nonneg (sub_nonneg.mpr hθ1) (sub_nonneg.mpr he0)]

/-- Sharpness: every theta in [(p - E0)/(1 - E0), p/(1 - E1)] ∩ [0, 1) is produced by some admissible noise.
    Witness: theta <= p uses e1 = 0, e0 = (p - theta)/(1 - theta); theta >= p uses e0 = 0, e1 = 1 - p/theta. -/
theorem sharp_below (hθ0 : 0 ≤ θ) (hθ1 : θ < 1) (hθp : θ ≤ p) (hlo : p - E0 ≤ θ * (1 - E0)) :
    ∃ e0' : ℝ, 0 ≤ e0' ∧ e0' ≤ E0 ∧ p = θ * (1 - 0) + (1 - θ) * e0' := by
  have hpos : 0 < 1 - θ := by linarith
  refine ⟨(p - θ) / (1 - θ), div_nonneg (by linarith) hpos.le, ?_, ?_⟩
  · rw [div_le_iff₀ hpos]; nlinarith
  · field_simp; ring

theorem sharp_above (hθ0 : 0 < θ) (hpθ : p ≤ θ) (hhi : θ * (1 - E1) ≤ p) (hp0 : 0 ≤ p) :
    ∃ e1' : ℝ, 0 ≤ e1' ∧ e1' ≤ E1 ∧ p = θ * (1 - e1') + (1 - θ) * 0 := by
  refine ⟨1 - p / θ, ?_, ?_, ?_⟩
  · have : p / θ ≤ 1 := by rw [div_le_iff₀ hθ0]; linarith
    linarith
  · have : θ * (1 - E1) / θ ≤ p / θ := div_le_div_of_nonneg_right hhi hθ0.le
    have h2 : θ * (1 - E1) / θ = 1 - E1 := by field_simp
    linarith
  · field_simp; ring

/-- Transfer of a confidence statement: any nondecreasing maps carry "L <= p <= U" into a statement about
    theta, so the event {L <= p <= U} is contained in the event {gL L <= theta <= gU U}; its probability, and
    hence the coverage, can only grow. This is the whole finite-sample argument. -/
theorem transfer {gL gU : ℝ → ℝ} (hmL : Monotone gL) (hmU : Monotone gU)
    (hid : gL p ≤ θ ∧ θ ≤ gU p) (hL : L ≤ p) (hU : p ≤ U) : gL L ≤ θ ∧ θ ≤ gU U :=
  ⟨le_trans (hmL hL) hid.1, le_trans hid.2 (hmU hU)⟩

/-- Certification under a bounded miss rate: if the true zone error exceeds alpha, the reviewer's rate exceeds
    alpha * (1 - E1). So the null "p > alpha (1 - E1)" holds whenever the original null holds, and a level-delta
    test of the former is a level-delta test of the latter. False alarms (e0) only make it conservative. -/
theorem certify_null (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (he0 : 0 ≤ e0) (he1 : e1 ≤ E1) (hE1 : E1 < 1)
    (hp : p = θ * (1 - e1) + (1 - θ) * e0) (hbad : α < θ) : α * (1 - E1) < p := by
  have hpos : 0 < 1 - E1 := by linarith
  have h1 : α * (1 - E1) < θ * (1 - E1) := mul_lt_mul_of_pos_right hbad hpos
  have h2 : θ * (1 - E1) ≤ θ * (1 - e1) := mul_le_mul_of_nonneg_left (by linarith) hθ0
  have h3 : 0 ≤ (1 - θ) * e0 := mul_nonneg (sub_nonneg.mpr hθ1) he0
  linarith

end LabelNoise
