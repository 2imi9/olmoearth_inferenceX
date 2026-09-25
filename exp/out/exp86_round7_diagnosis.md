# exp86 round 7: diagnosis, and a blind two-round audit (read-only)

Written 25 September 2026, after round 7 was run and scored (commit d60ce5f).

- **Round 7** ran after the trial had passed at round 6, so the trial's verdict does not depend on it.
- **Agent:** 43ca1f8, the signed head of PR 156. It holds round 6's agent (eb8a206, the same tree as 7bacc4d) plus
  one fix from the round 6 audit: `olmoearth_compare_review` returns `class_changes`, the share of differing windows
  on which A is the more confident side, and a note that its listing is not a sample.
- **Package:** the agent's tools used olmoearth-inferencex 1.3.0 as installed, as in rounds 1 to 6.

## The graded verdict

Under the instrument in force (A8–A21), round 7 **fails** on P5, on B3/cluster run 1. P1, P2, P3, P4, P6 and P7
hold, and parity passes 6 of 6.

**The P5 failure is a scorer miss (c).** A18's winner rule matched "upstream evidence shows the more confident side is
right on only about half or less of such windows".

- The sentence reports a rate. A18 excludes a rate only when a digit follows the predicate, so the number here, written
  in words, passed as a winner claim.
- The same answer declines in the sentence before: "Confidence doesn't pick a winner".
- No amendment is made. exp86 has passed; a fifteenth change after seeing results would repeat the pattern A8–A21
  already stand for. The reading is carried into exp87's rules before any exp87 run (`docs/plan/agent_trial_v3.md`).
- The sentence is still wrong on its own terms: the tool says the more confident side is right on 51 to 70 percent.
  The blind audit below counts it.

The number check fired twice, and both rewrites were accepted:

- B3/cluster run 3 flagged "57.8%", the complement of the tool's 42.2%, which is derived.
- B7/files run 1 flagged "1142".

## A blind audit of rounds 6 and 7

Round 6's first audit was one agent reading 30 answers. This one used a stronger protocol:

- The answers of rounds 6 and 7 were copied under random labels, X and Y. Paths that named the round were rewritten.
  One residual clue remained: round 7's compare outputs carry the new fields.
- Four auditors each took a group of configurations in both labels, and one adversarial verifier per auditor tried to
  refute every finding.
- A finding counts when the verifier confirmed it or only changed its class.
- The full result, and the key (X was round 6, Y round 7), are in `exp86_audit_rounds_6_7.json`.

| | Round 6 (X) | Round 7 (Y) |
|---|---|---|
| Confirmed findings | 54 | 48 |
| Answers with at least one | 23 of 30 | 23 of 30 |
| Refuted by the verifier | 2 | 6 |
| Further findings the verifier added (unverified) | 10 | 6 |
| E1 listing generalised | 5 | 5 |
| E2 direction read backwards | 2 | 1 |
| E3 action or artifact claimed | 2 | 1 |
| E4 statistical advice against the tool | 3 | 2 |
| E5 evidence or magnitude overclaimed | 12 | 8 |
| E6 derived number | 7 | 9 |
| Other false or unsupported | 23 | 22 |

The first audit's "at least 8 of 30" was a floor. Under the stronger protocol, 23 of 30 answers in each round state
something false or unsupported. Many of the "other" findings are minor, such as a result id called a prediction id,
but others are not (below).

### What the round-7 fix did

- **B7's direction errors are gone.**
  - Round 6: "B (post) is usually the more confident side", read off the first listed row; and "most differing windows
    flip class 1 -> 0", which is inverted.
  - Round 7 states neither. Its B7 answers take the direction from `class_changes` and the confidence share.
- **E1 persists where no aggregate exists.**
  - B3/cluster still reads place off the listing: "the top-50 differing windows sit mostly in row 0 (north edge)";
    "The most confidence-split zone runs along row 0".
  - B7 now invents spatial blocks from the ten listed windows.
  - The fix gave the direction but no spatial breakdown, and the listing is still inline.
- **E5 and E6 are unchanged in kind:**
  - evidence from the suite applied to cases the tool says no experiment grades;
  - "far more confident" across a margin gap of 0.04;
  - derived complements and ratios.

### The "other" findings, by kind

- **Claims about the world that no tool looked up.**
  - "There are no ground-truth labels for this AOI in Studio", though no tool searched.
  - WorldCover proposed as a dated reference for 2017 and 2018. It exists for 2020 and 2021 only.
- **Offers of methods that do not apply.**
  - An error rate or classification metrics for a regression score with no threshold.
  - A Bonferroni re-run after the prefix rule certified nothing. The tool already gives the per-level delta: every
    p-value is at least 0.554.
  - "Run a third dated map" to settle which is right.
- **Fields read as something else.**
  - A raster's sha256 prefix given as the model's revision (`a7c40be9` for `a347b15`).
  - `boundary_neighbours` read as "likely true class-boundary pixels" and as "edge cells rather than whole regions".
    In fact 512 of C1's 555 montane windows differ.
- **Counts and examples misread from listings,** such as "indices 4819–4834", of which the listing holds seven.

## What follows for the fixes

The audit confirms the build decided after the harness survey and the PPE reading, and it adds to it:

1. **Comparison outputs:**
   - a spatial breakdown (row and column bands, and where the differences concentrate);
   - the listing moved to a file, with a count and an order label;
   - class changes also given as undirected pairs, since "A ↔ B 37%" was read as both directions.
2. **Scoped evidence:** each tool's evidence says whether any experiment covers this case, in one sentence the
   answer can repeat. It replaces the multi-paragraph blocks the answers widened.
3. **Next steps and "not applicable" stated by the tools:**
   - A regression score has no error rate without a threshold.
   - Certification after a failed rule is not re-run under another rule.
   - A comparison across dates is not settled by another date.
4. **Fields labelled for reading:**
   - "model revision" beside "raster sha256";
   - what `boundary_neighbours` counts, and what it does not show.
5. **Answer checks in code:** numbers against their source fields, direction words against the fields, action and
   artifact claims against the run, and required warnings; then one rewrite.
6. **Derived quantities the answers keep computing,** returned as fields: complements, ratios, unused labels.

## For exp87

The strict P8 bar, no confirmed false claim in any of three runs, would fail today on all ten of exp86's
configurations: every one has at least one confirmed finding in round 7. That is the baseline. exp87 needs a stated materiality rule, decided at freezing. Otherwise one
mislabelled id weighs as much as an inverted direction. Every finding is reported whatever the rule.
