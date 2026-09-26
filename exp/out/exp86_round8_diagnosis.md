# exp86 round 8: diagnosis, and a blind audit of rounds 7 and 8 (read-only)

Written 25 September 2026, after round 8 was run and scored (commit 73bc9b3).

- **Round 8** is a development round on the exp87 build. It ran after the trial had passed at round 6 and before
  exp87's preregistration is frozen. It uses exp86's briefs and fixtures only; exp87's sealed held-out set was not
  opened. The trial's verdict does not depend on it.
- **Agent:** 6d25307, the signed head of PR 156. It adds 33 commits to round 7's agent (43ca1f8):
  - tool outputs with facts, required statements and forbidden claims;
  - statistical rules in code;
  - answer checks with one rewrite;
  - the middleware chain.
- **Package:** the agent's tools used olmoearth-inferencex 1.3.1 as installed.

## The graded verdict

Under the instrument in force (A8–A21), round 8 holds P1 and P3–P7, and parity passes 6 of 6. P2 fails on B1/studio
run 3 only.

**The P2 failure is a scorer miss (c).** The answer wrote the tool's date 2025-12-31 as "12-31-2025", and the reader
counted "12" as a number no tool returned. The date is right. No amendment is made, as for round 7.

The answer checks fired three times. Each time the model rewrote the answer once, and no sentence was left marked:

- a forbidden claim on B3/studio run 3;
- numbers on B7/files run 1 and B8/cluster run 1.

## A blind audit of rounds 7 and 8

The protocol is the one used for rounds 6 and 7:

- The answers of both rounds were copied under random labels P and Q, and paths that named the round were rewritten.
  Round 8's outputs carry the new fields, so the blinding is not complete.
- Four auditors each took a group of configurations in both labels. One adversarial verifier per auditor tried to
  refute every finding.
- This time the verifier also judged each confirmed finding under the owner's materiality rule
  (`docs/plan/agent_trial_v3.md`).

The full result, with the key (P is round 8, Q is round 7), is in `exp86_audit_rounds_7_8.json`.

| | Round 7 (Q) | Round 8 (P) |
|---|---|---|
| Confirmed findings | 63 | 35 |
| Answers with at least one | 26 of 30 | 24 of 30 |
| Material findings | 32 | 17 |
| Answers with a material finding | 16 of 30 | 11 of 30 |
| Configurations with a material finding | 7 of 10 | 5 of 10 |
| E1 listing generalised | 4 | 0 |
| E2 direction read backwards | 1 | 0 |
| E3 action or artifact claimed | 3 | 3 |
| E4 statistical advice against the tool | 8 | 1 |
| E5 evidence or magnitude overclaimed | 11 | 8 |
| E6 derived number | 10 | 7 |
| Other false or unsupported | 26 | 16 |

The verifiers refuted 13 findings. They also listed 8 findings the auditors had missed, which are not verified.

This audit counts round 7 at 16 of 30 answers with a material finding. The owner's sort of the rounds 6–7 audit
put it at 14. The difference is between auditors, and it is why both rounds were audited together again.

### What the build fixed

- **Patterns read off a listing (E1) and reversed directions (E2) are gone.** The comparisons now give where their
  differences sit, and their direction, as fields. Their listings are in a file.
- **Statistical advice against the tool (E4) fell from 8 to 1.** No confirmed finding of round 8 offers a looser
  alpha, another certification rule or a non-random design for certification. The one left is a labelling offer
  (below).

### What remains: the 17 material findings of round 8

| Cause | Findings | Configurations |
|---|---|---|
| One pooled correlation (r = −0.02 over 25 cells) read as "do not agree at all", "not anywhere", "high where the other is low" | 4 | B3/studio |
| Review sets and error rates offered for a regression band with no threshold; a sample of low-confidence windows promised "defensible" rates | 3 | B3/studio |
| Labels for one date, or one reference plus another model run, offered to settle which of two dated maps is right | 3 | B3/cluster, B7/files |
| A list claimed saved in `review_set_evidence.json`, which holds evidence text only | 2 | B8/cluster |
| Evidence the tool says does not cover the case applied to it ("comparable cases"); the pair called "Sen1Floods11 flood maps", a name taken from that evidence | 2 | B3/cluster, B7/files |
| The margin ratio of the 50 listed windows given for the 10 shown | 1 | B8/cluster |
| The weakest classes named wrongly: class 5 has 0 of 10 correct | 1 | B5/files |
| "Guaranteed-certifiable region" | 1 | B5/files |

Two findings the verifiers added trace to the tool itself. The files grid (F4: 400 chips of 14 × 14 windows stacked
in dataset order) carries no georeference, yet the spatial breakdown calls its first row band "north".

### The fixes that follow

1. The contract's fixed ids gain six claims:
   - `spatial_pattern_from_one_correlation`
   - `agreement_from_uncertain_correlation`
   - `review_set_for_unthresholded_regression`
   - `one_reference_settles_two_dates`
   - `evidence_outside_its_scope`
   - `certification_guaranteed`
2. The comparisons give the correlation with its n and a 95% interval, as a fact. They drop the numbers and names of
   evidence that does not cover the case, and they say "north" only for a georeferenced grid.
3. A review set that lists fewer windows than it ranks writes the full list to a file and names it.
4. The error estimate gives the weakest classes as a fact computed by code.
5. The harness gains a detector for each new id, a stricter check of "subset labelling is enough", and a check that a
   file named as holding a list does hold it.

Round 9 runs on the result, before exp87 freezes.
