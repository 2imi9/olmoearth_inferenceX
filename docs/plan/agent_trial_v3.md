# Agent trial v3: claims, required content and held-out briefs (exp87 preregistration, DRAFT)

**Status: draft, not frozen.** This page is written on 25 September 2026, before the fixes it tests are built. It is
frozen, with the held-out set revealed, after the fixes are committed and before any counted run. Until then it may
change. After freezing, it changes only as stated under [Changes after freezing](#changes-after-freezing).

## Why a third trial

The [second trial](agent_trial_v2.md) (exp86) passed at round 6. Its seven criteria held on all ten configurations,
three runs each. But the pass came under the instrument amended fourteen times after results were seen (A8–A21).
Under the preregistered instrument, the same answers fail.

A blind audit of round 6 (`exp/out/exp86_round6_audit.md`) then found at least 8 of 30 answers with false or
contradicted prose that no criterion graded, and 3 derived numbers the number reader matched by chance. Its error
classes:

- **E1** a pattern generalised from a truncated or ordered listing;
- **E2** a direction the tool states, read backwards;
- **E3** an action or artifact claimed that does not exist;
- **E4** statistical advice against the tool's own notes (an alpha chosen after the bounds, certification from a
  non-random design);
- **E5** overclaimed evidence or magnitude;
- **E6** a number the model derived itself that passes by coincidence.

exp86's criteria cannot see E1–E5, and its P2 reader is blind to E6. This trial adds criteria for them. It also
avoids exp86's two faults:

- **Fitting the fixes to the fixtures.** Eight held-out briefs were sealed before any fix was built. Their SHA-256
  digest was committed in 42e8a24 (`exp/out/exp87_sealed_commitment.md`). Nobody building the fixes reads them.
- **Changing the rules after seeing results.** The instrument is frozen before the first counted run, and changes
  after that are limited to the ones under [Changes after freezing](#changes-after-freezing).

## What is tested

The OlmoEarth Agent after the fixes decided on 25 September 2026. They came from two reviews: a survey of current
harness designs, and a close reading of Google's Planetary Prediction Engine paper, both in the working notes. The
fixes are listed here as built, at the commit named in `round.json`:

1. **Tool outputs state what the model had been guessing:**
   - aggregates before listings, with a label on every listing (order, and that it is not a sample);
   - a spatial breakdown of the differences in a comparison;
   - directions stated as fields and as one sentence;
   - ratio fields;
   - derived counts a user needs;
   - warnings carried from one tool to the next;
   - evidence blocks that state their scope.
2. **Statistical rules enforced in code:** the design records alpha, certification refuses a different one, and
   next-step suggestions come from the design type.
3. **Checks of the answer in code, with one rewrite:**
   - each number is matched to its source field;
   - direction words must agree with the tool's fields;
   - action and artifact claims are matched against what the run did;
   - required warnings must be present.
4. **Small harness fixes:** the number pool no longer holds earlier answers or the memory block; the model's reasoning
   is recorded; the agent requires olmoearth-inferencex ≥ 1.3.1.

An LLM claim check (a verifier that sees the tool outputs and not the draft) is built only if the control round leaves
confirmed false claims that items 1–4 cannot catch. If built, it is a second arm run beside the control, never a
replacement.

## Briefs and fixtures

- **B1–B8**: exp86's ten configurations, with exp86's fixtures and wording unchanged, as a regression check.
- **The held-out set**: eight configurations, with fixtures built and facts computed before any fix, sealed. They
  are revealed at freezing. Their files are then copied into
  `exp/out/exp87_trial/`, and the recomputed manifest must reproduce the committed digest.

Each configuration is run three times, like exp86's. A round is 18 configurations, 54 runs.

## Criteria

**P1–P7** are exp86's criteria under its final instrument (A8–A21), unchanged, with one decision made before any
exp87 run. D1's rate exclusion also counts a number written in words, such as "about half" or "a third" ("most" is not a
number).
exp86 round 7 showed the need: "the more confident side is right on only about half or less of such windows" was
graded as a winner claim.

**P8, claims.** No answer states a material false claim (below) that the owner confirms. Every confirmed finding is
reported with its class (E1–E6 or other), material or not.

**Materiality** was decided by the owner on 25 September 2026, before freezing. A confirmed false claim is material
when it would change what a user believes about the map, the method or the evidence, or what they do next:

- a direction, a place or a magnitude stated wrongly;
- a number that is wrong, whether a tool returned it or the model derived it (E6);
- a file, a list or an action claimed that does not exist;
- advice on a method that does not apply, or that the tool's notes rule out;
- a fact about the world, stated as fact when no tool looked it up, that is false.

A slip that changes none of these is immaterial, such as a result id called a prediction id. It is reported beside
the verdict and does not fail the run. A derived number that is correct is reported as E6 and does not fail the run.
At freezing, every confirmed finding of exp86 rounds 6 and 7 (`exp/out/exp86_audit_rounds_6_7.json`) is sorted under
this rule, and the sorted list is committed as the anchor set. The owner decides new findings against it.

How a claim is decided:

- **Reference facts in code.** For each configuration, the facts a correct answer may state are computed from the
  fixtures by the package or the agent's tool functions, not by a model:
  - the dominant changes and their shares;
  - which side is more confident;
  - where differences concentrate;
  - estimates and intervals;
  - what certifies at which alpha;
  - the magnitudes of listed margins.
- **A blinded LLM screen.** It uses no Qwen model, and Claude is at most one of its members.
  - It reads every answer and its run's tool outputs and lists candidate false claims with evidence.
  - A second pass tries to refute each one.
  - The screen is blind to round, commit and arm.
- **Adjudication by the owner.** The repository's owner decides every flagged claim, and also reads a random sample of
  one answer in five from those with no flag. The sample is drawn with the package's sampler, and its seed is fixed
  at freezing.
  - A claim counts as false only when the owner confirms it.
  - A false claim found in the random sample is reported, and the screen's miss rate is estimated from it.
- **Drafts too.** When the harness's check rewrote an answer, the draft is screened as well. Its false claims are
  reported beside the final answer's. They do not grade P8, which grades what the user sees.

**P9, required content.** Every answer states what its brief requires. Examples:

- a comparison names the largest class change and the direction of change;
- an estimate states the design's own interval;
- a certification brief states whether anything certifies at the alpha asked.

The required items are listed per configuration, B1–B8 and the held-out set, at freezing. The owner adjudicates, and
the screen flags missing items. P9 keeps P8 from being passed by saying less.

**The pass rule** is exp86's: a configuration passes a criterion when all three runs pass it. A round passes when P1
to P9 hold on all 18 configurations.

**What a pass means.**

- **Answers with no material false claim.** 0 in 54 bounds the per-answer rate of material false claims at about
  5.4%, one-sided at 95% (1 − 0.05^(1/54)). It does not show that the rate is zero, and it says nothing of the
  immaterial slips, which are reported beside the verdict.
- **Rules reused.** P1–P7 are exp86's, whose rules were amended after seeing results. The held-out configurations are
  where those rules meet answers the fixes were not built on.

## Calibration before the first counted run

- **The screen's recall is measured on seeded errors before round 1.**
  - Real round 6 and 7 answers are edited to carry one known error each, in the observed forms: an inverted flip
    direction, a swapped confidence side, "the list was saved", a post-hoc alpha offer, an inflated magnitude, a
    derived percentage.
  - The seeded set also holds as many clean controls: true paraphrases, correct derived numbers the tools stated, and
    true hedges.
  - The seeded answers are confirmed by the owner.
- **The bar is at least 90% recall on each class.** Where the screen misses more, the owner reads every answer of
  that class's configurations.
- **The screen's false-flag rate** on the controls is reported.

## Changes after freezing

- The criteria, the reference facts, the required items, the screen's prompts and models, and the pass rule do not
  change after the first counted run.
- A bug in a reader of this page's rules may be fixed if both of these hold:
  - the fix is shown on the seeded set;
  - every round is regraded with it and the change is listed beside the verdict.
- A change of rule starts a new trial. It is never an amendment.

## What this trial cannot check

- Whether the "why" of an explanation is right, beyond the listed classes.
- Claims about the world that no tool output bears on. They are flagged when they present something as fact, but the
  owner decides them from outside knowledge.
- Whether the fixes help on briefs unlike B1–B8 and the held-out set. Eight held-out briefs are a check on fitting,
  not a test of generality.
