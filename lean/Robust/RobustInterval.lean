/-
  RobustInterval.lean : the lemmas behind estimate's and certify's bounds for windows that could not be
  judged (`?`) and for a reviewer who errs (oe_inferencex/estimate.py, estimate_error_rate; cli.py, certify).

    T1  deterministic bounds on the true error rate θ from the reviewer's rate p and noise bounds E0, E1,
        in the sharp form  max 0 ((p - E0)/(1 - E0)) ≤ θ ≤ min 1 (p/(1 - E1)),  and sharpness (every θ in the
        interval is produced by some admissible noise).
    T2  coverage transfer: the event "[L, U] covers p" is contained in the event "[gL L, gU U] covers θ",
        so the coverage can only grow (measure_mono).
    T3  bounds replaced by random upper bounds F0, F1: miscoverage ≤ miscoverage for p + P(a bound fails)
        (measure_mono + measure_union_le); no independence assumed.
    T4  unjudged windows: monotone interval ends and k ≤ k* ≤ k + u give [L k, U (k+u)] ⊇ [L k*, U k*];
        the noisy variant by a union bound.
    T5  certification core: θ > α forces p > α (1 - E1); p ≤ α' forces θ ≤ α'/(1 - E1).

  STATUS: compiled on 1 October 2026 on the aicr cluster (Slurm job 1146459) with Lean 4.35.0-rc3 and the
  Mathlib commit pinned in ../lake-manifest.json: exit 0, no errors, no `sorry`, no axioms beyond Mathlib's.
  It was drafted on a machine without Lean and corrected until it compiled; this is the compiled text.
-/
import Mathlib

open MeasureTheory
open scoped ENNReal

noncomputable section

namespace RobustInterval

/-! ## T1 in counts (finite population)

`D` truly wrong windows of `N`, `A` false alarms (truly correct, marked wrong), `B` misses (truly wrong, marked
right); the reviewer's count is `M = D - B + A`. Stated over ℝ (counts cast to ℝ). -/

theorem counts_upper {D A B M E1 : ℝ} (hA : 0 ≤ A) (hB : B ≤ E1 * D) (hM : M = D - B + A) :
    D * (1 - E1) ≤ M := by
  nlinarith

theorem counts_lower {N D A B M E0 : ℝ} (hB : 0 ≤ B) (hA : A ≤ E0 * (N - D)) (hM : M = D - B + A) :
    M - E0 * N ≤ D * (1 - E0) := by
  nlinarith

/-! ## T1 in rates

`θ` true error rate, `e0` false-alarm share among truly correct windows, `e1` miss share among truly wrong
windows, `p = θ (1 - e1) + (1 - θ) e0` the reviewer's rate. -/

/-- Upper end, multiplicative form. Uses `0 ≤ e0` (false alarms) and `e1 ≤ E1` only. -/
theorem upper_mul {θ e0 e1 E1 p : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (he0 : 0 ≤ e0) (he1 : e1 ≤ E1)
    (hp : p = θ * (1 - e1) + (1 - θ) * e0) : θ * (1 - E1) ≤ p := by
  have h1 : 0 ≤ θ * (E1 - e1) := mul_nonneg hθ0 (sub_nonneg.mpr he1)
  have h2 : 0 ≤ (1 - θ) * e0 := mul_nonneg (sub_nonneg.mpr hθ1) he0
  nlinarith [h1, h2]

/-- Lower end, multiplicative form. Uses `0 ≤ e1` (misses) and `e0 ≤ E0` only. -/
theorem lower_mul {θ e0 e1 E0 p : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (he1 : 0 ≤ e1) (he0 : e0 ≤ E0)
    (hp : p = θ * (1 - e1) + (1 - θ) * e0) : p - E0 ≤ θ * (1 - E0) := by
  have h1 : 0 ≤ θ * e1 := mul_nonneg hθ0 he1
  have h2 : 0 ≤ (1 - θ) * (E0 - e0) := mul_nonneg (sub_nonneg.mpr hθ1) (sub_nonneg.mpr he0)
  nlinarith [h1, h2]

/-- Upper end: `θ ≤ p / (1 - E1)`. -/
theorem upper_div {θ e0 e1 E1 p : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (he0 : 0 ≤ e0) (he1 : e1 ≤ E1)
    (hE1 : E1 < 1) (hp : p = θ * (1 - e1) + (1 - θ) * e0) : θ ≤ p / (1 - E1) := by
  have hc : 0 < 1 - E1 := by linarith
  have h := upper_mul hθ0 hθ1 he0 he1 hp
  have hinv : 0 ≤ (1 - E1)⁻¹ := inv_nonneg.mpr hc.le
  have hcc : (1 - E1) * (1 - E1)⁻¹ = 1 := mul_inv_cancel₀ hc.ne'
  have h2 : θ * (1 - E1) * (1 - E1)⁻¹ ≤ p * (1 - E1)⁻¹ := mul_le_mul_of_nonneg_right h hinv
  have h3 : θ * (1 - E1) * (1 - E1)⁻¹ = θ := by linear_combination θ * hcc
  rw [div_eq_mul_inv]
  linarith

/-- Lower end: `(p - E0) / (1 - E0) ≤ θ` (sharper than the candidate `p - E0`). -/
theorem lower_div {θ e0 e1 E0 p : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (he1 : 0 ≤ e1) (he0 : e0 ≤ E0)
    (hE0 : E0 < 1) (hp : p = θ * (1 - e1) + (1 - θ) * e0) : (p - E0) / (1 - E0) ≤ θ := by
  have hc : 0 < 1 - E0 := by linarith
  have h := lower_mul hθ0 hθ1 he1 he0 hp
  have hinv : 0 ≤ (1 - E0)⁻¹ := inv_nonneg.mpr hc.le
  have hcc : (1 - E0) * (1 - E0)⁻¹ = 1 := mul_inv_cancel₀ hc.ne'
  have h2 : (p - E0) * (1 - E0)⁻¹ ≤ θ * (1 - E0) * (1 - E0)⁻¹ := mul_le_mul_of_nonneg_right h hinv
  have h3 : θ * (1 - E0) * (1 - E0)⁻¹ = θ := by linear_combination θ * hcc
  rw [div_eq_mul_inv]
  linarith

/-- The candidate lower end `p - E0` is never above the sharp one when `E0 ≤ p` (so it is valid but looser). -/
theorem candidate_le_sharp {p E0 : ℝ} (hE00 : 0 ≤ E0) (hE0 : E0 < 1) (hp : E0 ≤ p) :
    p - E0 ≤ (p - E0) / (1 - E0) := by
  have hc : 0 < 1 - E0 := by linarith
  have hinv : 0 ≤ (1 - E0)⁻¹ := inv_nonneg.mpr hc.le
  have hcc : (1 - E0) * (1 - E0)⁻¹ = 1 := mul_inv_cancel₀ hc.ne'
  -- (p - E0) * (1 - E0)⁻¹ - (p - E0) = (p - E0) * E0 * (1 - E0)⁻¹ ≥ 0
  have h1 : 0 ≤ (p - E0) * E0 * (1 - E0)⁻¹ :=
    mul_nonneg (mul_nonneg (sub_nonneg.mpr hp) hE00) hinv
  have h2 : (p - E0) * (1 - E0)⁻¹ - (p - E0) = (p - E0) * E0 * (1 - E0)⁻¹ := by
    linear_combination (p - E0) * hcc
  rw [div_eq_mul_inv]
  linarith

/-- The two transforms. -/
def gL (E0 x : ℝ) : ℝ := max 0 ((x - E0) / (1 - E0))
def gU (E1 x : ℝ) : ℝ := min 1 (x / (1 - E1))

theorem gL_mono {E0 : ℝ} (hE0 : E0 < 1) : Monotone (gL E0) := by
  intro a b hab
  have hc : 0 ≤ (1 - E0)⁻¹ := inv_nonneg.mpr (by linarith)
  have h : (a - E0) / (1 - E0) ≤ (b - E0) / (1 - E0) := by
    rw [div_eq_mul_inv, div_eq_mul_inv]
    exact mul_le_mul_of_nonneg_right (by linarith) hc
  show max 0 ((a - E0) / (1 - E0)) ≤ max 0 ((b - E0) / (1 - E0))
  exact max_le (le_max_left _ _) (le_trans h (le_max_right _ _))

theorem gU_mono {E1 : ℝ} (hE1 : E1 < 1) : Monotone (gU E1) := by
  intro a b hab
  have hc : 0 ≤ (1 - E1)⁻¹ := inv_nonneg.mpr (by linarith)
  have h : a / (1 - E1) ≤ b / (1 - E1) := by
    rw [div_eq_mul_inv, div_eq_mul_inv]
    exact mul_le_mul_of_nonneg_right hab hc
  show min 1 (a / (1 - E1)) ≤ min 1 (b / (1 - E1))
  exact le_min (min_le_left _ _) (le_trans (min_le_right _ _) h)

/-- T1 (validity): θ lies in `[gL E0 p, gU E1 p]` whenever the noise is inside the box.
`E0 + E1 < 1` is not needed. -/
theorem identified {θ e0 e1 E0 E1 p : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (he0 : 0 ≤ e0) (he1 : 0 ≤ e1)
    (hle0 : e0 ≤ E0) (hle1 : e1 ≤ E1) (hE0 : E0 < 1) (hE1 : E1 < 1)
    (hp : p = θ * (1 - e1) + (1 - θ) * e0) : gL E0 p ≤ θ ∧ θ ≤ gU E1 p := by
  refine ⟨?_, ?_⟩
  · show max 0 ((p - E0) / (1 - E0)) ≤ θ
    exact max_le hθ0 (lower_div hθ0 hθ1 he1 hle0 hE0 hp)
  · show θ ≤ min 1 (p / (1 - E1))
    exact le_min hθ1 (upper_div hθ0 hθ1 he0 hle1 hE1 hp)

/-- Sharpness, below `p`: no misses and a false-alarm share `(p - θ)/(1 - θ) ≤ E0`. -/
theorem sharp_below {θ E0 p : ℝ} (hθ1 : θ < 1) (hθp : θ ≤ p) (hlo : p - E0 ≤ θ * (1 - E0)) :
    ∃ e0 : ℝ, 0 ≤ e0 ∧ e0 ≤ E0 ∧ p = θ * (1 - 0) + (1 - θ) * e0 := by
  have hc : 0 < 1 - θ := by linarith
  have hinv : 0 ≤ (1 - θ)⁻¹ := inv_nonneg.mpr hc.le
  have hcc : (1 - θ) * (1 - θ)⁻¹ = 1 := mul_inv_cancel₀ hc.ne'
  refine ⟨(p - θ) * (1 - θ)⁻¹, mul_nonneg (by linarith) hinv, ?_, ?_⟩
  · have h1 : p - θ ≤ E0 * (1 - θ) := by linarith
    have h2 : (p - θ) * (1 - θ)⁻¹ ≤ E0 * (1 - θ) * (1 - θ)⁻¹ := mul_le_mul_of_nonneg_right h1 hinv
    have h3 : E0 * (1 - θ) * (1 - θ)⁻¹ = E0 := by linear_combination E0 * hcc
    linarith
  · linear_combination (θ - p) * hcc

/-- Sharpness, above `p`: no false alarms and a miss share `1 - p/θ ≤ E1`. -/
theorem sharp_above {θ E1 p : ℝ} (hθ0 : 0 < θ) (hpθ : p ≤ θ) (hhi : θ * (1 - E1) ≤ p) :
    ∃ e1 : ℝ, 0 ≤ e1 ∧ e1 ≤ E1 ∧ p = θ * (1 - e1) + (1 - θ) * 0 := by
  have hinv : 0 ≤ θ⁻¹ := inv_nonneg.mpr hθ0.le
  have hcc : θ * θ⁻¹ = 1 := mul_inv_cancel₀ hθ0.ne'
  refine ⟨1 - p * θ⁻¹, ?_, ?_, ?_⟩
  · have h2 : p * θ⁻¹ ≤ θ * θ⁻¹ := mul_le_mul_of_nonneg_right hpθ hinv
    linarith
  · have h2 : θ * (1 - E1) * θ⁻¹ ≤ p * θ⁻¹ := mul_le_mul_of_nonneg_right hhi hinv
    have h3 : θ * (1 - E1) * θ⁻¹ = 1 - E1 := by linear_combination (1 - E1) * hcc
    linarith
  · linear_combination (-p) * hcc

/-- T1 (sharpness): every θ ∈ [0, 1] inside `[gL E0 p, gU E1 p]` is produced by some noise in the box,
so no narrower set is valid for every admissible reviewer. -/
theorem identified_sharp {θ E0 E1 p : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
    (hE00 : 0 ≤ E0) (hE10 : 0 ≤ E1) (hE0 : E0 < 1) (hE1 : E1 < 1)
    (hlo : gL E0 p ≤ θ) (hhi : θ ≤ gU E1 p) :
    ∃ e0 e1 : ℝ, 0 ≤ e0 ∧ e0 ≤ E0 ∧ 0 ≤ e1 ∧ e1 ≤ E1 ∧ p = θ * (1 - e1) + (1 - θ) * e0 := by
  have hlo' : (p - E0) / (1 - E0) ≤ θ := le_trans (le_max_right _ _) hlo
  have hhi' : θ ≤ p / (1 - E1) := le_trans hhi (min_le_right _ _)
  have c0 : 0 < 1 - E0 := by linarith
  have c1 : 0 < 1 - E1 := by linarith
  have k0 : (1 - E0) * (1 - E0)⁻¹ = 1 := mul_inv_cancel₀ c0.ne'
  have k1 : (1 - E1) * (1 - E1)⁻¹ = 1 := mul_inv_cancel₀ c1.ne'
  have mlo : p - E0 ≤ θ * (1 - E0) := by
    have h := mul_le_mul_of_nonneg_right hlo' c0.le
    rw [div_eq_mul_inv] at h
    have e : (p - E0) * (1 - E0)⁻¹ * (1 - E0) = p - E0 := by linear_combination (p - E0) * k0
    linarith
  have mhi : θ * (1 - E1) ≤ p := by
    have h := mul_le_mul_of_nonneg_right hhi' c1.le
    rw [div_eq_mul_inv] at h
    have e : p * (1 - E1)⁻¹ * (1 - E1) = p := by linear_combination p * k1
    linarith
  by_cases hθp : θ ≤ p
  · by_cases hlt : θ < 1
    · obtain ⟨e0, h0, h1, h2⟩ := sharp_below hlt hθp mlo
      exact ⟨e0, 0, h0, h1, le_rfl, hE10, h2⟩
    · have heq : θ = 1 := le_antisymm hθ1 (not_lt.mp hlt)
      have hpe : p = 1 := by linarith
      exact ⟨0, 0, le_rfl, hE00, le_rfl, hE10, by linear_combination hpe - heq⟩
  · have hpθ : p < θ := not_le.mp hθp
    have hθpos : 0 < θ := lt_of_le_of_lt hp0 hpθ
    obtain ⟨e1, h0, h1, h2⟩ := sharp_above hθpos hpθ.le mhi
    exact ⟨0, e1, le_rfl, hE00, h0, h1, h2⟩

/-! ## T2: coverage transfer -/

/-- Event inclusion, pointwise. -/
theorem transfer_event {gl gu : ℝ → ℝ} (hgl : Monotone gl) (hgu : Monotone gu) {θ p L U : ℝ}
    (hid : gl p ≤ θ ∧ θ ≤ gu p) (hL : L ≤ p) (hU : p ≤ U) : gl L ≤ θ ∧ θ ≤ gu U :=
  ⟨le_trans (hgl hL) hid.1, le_trans hid.2 (hgu hU)⟩

/-- T2: for every (θ, e0, e1) inside the box, any lower bound `c` on the probability that `[L, U]` covers `p`
is a lower bound on the probability that `[gL E0 L, gU E1 U]` covers θ. `μ` is the sampling design's law. -/
theorem coverage {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) (L U : Ω → ℝ)
    {θ e0 e1 E0 E1 p : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (he0 : 0 ≤ e0) (he1 : 0 ≤ e1)
    (hle0 : e0 ≤ E0) (hle1 : e1 ≤ E1) (hE0 : E0 < 1) (hE1 : E1 < 1)
    (hp : p = θ * (1 - e1) + (1 - θ) * e0) (c : ℝ≥0∞)
    (hcov : c ≤ μ {ω | L ω ≤ p ∧ p ≤ U ω}) :
    c ≤ μ {ω | gL E0 (L ω) ≤ θ ∧ θ ≤ gU E1 (U ω)} := by
  have hid := identified hθ0 hθ1 he0 he1 hle0 hle1 hE0 hE1 hp
  refine le_trans hcov (measure_mono ?_)
  intro ω hω
  simp only [Set.mem_setOf_eq] at hω ⊢
  exact transfer_event (gL_mono hE0) (gU_mono hE1) hid hω.1 hω.2

/-- T2, failure form: the miscoverage for θ is at most the miscoverage for p. -/
theorem miscoverage {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) (L U : Ω → ℝ)
    {θ e0 e1 E0 E1 p : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (he0 : 0 ≤ e0) (he1 : 0 ≤ e1)
    (hle0 : e0 ≤ E0) (hle1 : e1 ≤ E1) (hE0 : E0 < 1) (hE1 : E1 < 1)
    (hp : p = θ * (1 - e1) + (1 - θ) * e0) :
    μ {ω | ¬ (gL E0 (L ω) ≤ θ ∧ θ ≤ gU E1 (U ω))} ≤ μ {ω | ¬ (L ω ≤ p ∧ p ≤ U ω)} := by
  have hid := identified hθ0 hθ1 he0 he1 hle0 hle1 hE0 hE1 hp
  apply measure_mono
  intro ω hω
  simp only [Set.mem_setOf_eq] at hω ⊢
  intro h
  exact hω (transfer_event (gL_mono hE0) (gU_mono hE1) hid h.1 h.2)

/-! ## T3: estimated bounds (union bound, no independence) -/

theorem coverage_estimated {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) (L U F0 F1 : Ω → ℝ)
    {θ e0 e1 p : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (he0 : 0 ≤ e0) (he1 : 0 ≤ e1)
    (hF0 : ∀ ω, F0 ω < 1) (hF1 : ∀ ω, F1 ω < 1)
    (hp : p = θ * (1 - e1) + (1 - θ) * e0) :
    μ {ω | ¬ (gL (F0 ω) (L ω) ≤ θ ∧ θ ≤ gU (F1 ω) (U ω))} ≤
      μ {ω | ¬ (L ω ≤ p ∧ p ≤ U ω)} + μ {ω | ¬ (e0 ≤ F0 ω ∧ e1 ≤ F1 ω)} := by
  have hsub : {ω | ¬ (gL (F0 ω) (L ω) ≤ θ ∧ θ ≤ gU (F1 ω) (U ω))} ⊆
      {ω | ¬ (L ω ≤ p ∧ p ≤ U ω)} ∪ {ω | ¬ (e0 ≤ F0 ω ∧ e1 ≤ F1 ω)} := by
    intro ω hω
    simp only [Set.mem_setOf_eq, Set.mem_union] at hω ⊢
    by_contra hcon
    simp only [not_or, not_not] at hcon
    obtain ⟨⟨hL, hU⟩, ⟨h0, h1⟩⟩ := hcon
    have hid := identified hθ0 hθ1 he0 he1 h0 h1 (hF0 ω) (hF1 ω) hp
    exact hω (transfer_event (gL_mono (hF0 ω)) (gU_mono (hF1 ω)) hid hL hU)
  exact le_trans (measure_mono hsub) (measure_union_le _ _)

/-! ## T4: unjudged windows -/

/-- Pointwise: monotone ends and `k ≤ k* ≤ k + u`. -/
theorem unjudged_event {Lk Uk : ℕ → ℝ} (hL : Monotone Lk) (hU : Monotone Uk) {θ : ℝ} {k u ks : ℕ}
    (h1 : k ≤ ks) (h2 : ks ≤ k + u) (hcov : Lk ks ≤ θ ∧ θ ≤ Uk ks) : Lk k ≤ θ ∧ θ ≤ Uk (k + u) :=
  ⟨le_trans (hL h1) hcov.1, le_trans hcov.2 (hU h2)⟩

/-- T4(b): `ks` is the count of truly wrong windows in the sample, `k` the windows judged wrong, `u` the
unjudged ones; which windows go unjudged may depend arbitrarily on the sample and the truth. -/
theorem unjudged_coverage {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {Lk Uk : ℕ → ℝ}
    (hL : Monotone Lk) (hU : Monotone Uk) (θ : ℝ) (k u ks : Ω → ℕ)
    (h1 : ∀ ω, k ω ≤ ks ω) (h2 : ∀ ω, ks ω ≤ k ω + u ω) (c : ℝ≥0∞)
    (hcov : c ≤ μ {ω | Lk (ks ω) ≤ θ ∧ θ ≤ Uk (ks ω)}) :
    c ≤ μ {ω | Lk (k ω) ≤ θ ∧ θ ≤ Uk (k ω + u ω)} := by
  refine le_trans hcov (measure_mono ?_)
  intro ω hω
  simp only [Set.mem_setOf_eq] at hω ⊢
  exact unjudged_event hL hU (h1 ω) (h2 ω) hω

/-- T4(c): noisy judged labels plus unjudged windows. `pm` is the population rate with '?' read as right,
`pp` with '?' read as wrong; T1 applied to these two completed label sets gives `gL E0 pm ≤ θ` and
`θ ≤ gU E1 pp`. The miscoverage is at most the two one-sided miscoverages. -/
theorem unjudged_noisy {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {Lk Uk : ℕ → ℝ}
    {E0 E1 θ pm pp : ℝ} (hE0 : E0 < 1) (hE1 : E1 < 1) (hlo : gL E0 pm ≤ θ) (hhi : θ ≤ gU E1 pp)
    (k u : Ω → ℕ) :
    μ {ω | ¬ (gL E0 (Lk (k ω)) ≤ θ ∧ θ ≤ gU E1 (Uk (k ω + u ω)))} ≤
      μ {ω | ¬ (Lk (k ω) ≤ pm)} + μ {ω | ¬ (pp ≤ Uk (k ω + u ω))} := by
  have hsub : {ω | ¬ (gL E0 (Lk (k ω)) ≤ θ ∧ θ ≤ gU E1 (Uk (k ω + u ω)))} ⊆
      {ω | ¬ (Lk (k ω) ≤ pm)} ∪ {ω | ¬ (pp ≤ Uk (k ω + u ω))} := by
    intro ω hω
    simp only [Set.mem_setOf_eq, Set.mem_union] at hω ⊢
    by_contra hcon
    simp only [not_or, not_not] at hcon
    obtain ⟨hL, hU⟩ := hcon
    exact hω ⟨le_trans (gL_mono hE0 hL) hlo, le_trans hhi (gU_mono hE1 hU)⟩
  exact le_trans (measure_mono hsub) (measure_union_le _ _)

/-! ## T5: certification core -/

/-- The null shift: a true zone rate above α forces the reviewer's rate above α (1 - E1). -/
theorem certify_null {θ e0 e1 E1 p α : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (he0 : 0 ≤ e0) (he1 : e1 ≤ E1)
    (hE1 : E1 < 1) (hp : p = θ * (1 - e1) + (1 - θ) * e0) (hbad : α < θ) : α * (1 - E1) < p := by
  have hc : 0 < 1 - E1 := by linarith
  have h1 : α * (1 - E1) < θ * (1 - E1) := mul_lt_mul_of_pos_right hbad hc
  have h2 := upper_mul hθ0 hθ1 he0 he1 hp
  linarith

/-- A reviewer's zone rate at most α' bounds the true rate by α' / (1 - E1). -/
theorem certify_transfer {θ e0 e1 E1 p α' : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (he0 : 0 ≤ e0)
    (he1 : e1 ≤ E1) (hE1 : E1 < 1) (hp : p = θ * (1 - e1) + (1 - θ) * e0) (hcert : p ≤ α') :
    θ ≤ α' / (1 - E1) := by
  have h := upper_div hθ0 hθ1 he0 he1 hE1 hp
  have hc : 0 ≤ (1 - E1)⁻¹ := inv_nonneg.mpr (by linarith)
  have h2 : p / (1 - E1) ≤ α' / (1 - E1) := by
    rw [div_eq_mul_inv, div_eq_mul_inv]
    exact mul_le_mul_of_nonneg_right hcert hc
  linarith

/-- T5 as an event inclusion: "certified and the true rate exceeds α'/(1 - E1)" is contained in "certified
and the reviewer's rate exceeds α'", whose probability the package already bounds by δ. -/
theorem certify_event {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) (cert : Set Ω)
    {θ e0 e1 E1 p α' : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (he0 : 0 ≤ e0) (he1 : e1 ≤ E1)
    (hE1 : E1 < 1) (hp : p = θ * (1 - e1) + (1 - θ) * e0) :
    μ {ω | ω ∈ cert ∧ α' / (1 - E1) < θ} ≤ μ {ω | ω ∈ cert ∧ α' < p} := by
  apply measure_mono
  intro ω hω
  simp only [Set.mem_setOf_eq] at hω ⊢
  refine ⟨hω.1, ?_⟩
  by_contra hn
  have h := certify_transfer hθ0 hθ1 he0 he1 hE1 hp (not_lt.mp hn)
  linarith [hω.2]

end RobustInterval

end
