# Claim ledger

Every claim in these docs traces to a file under `exp/out/`. `docs/claims.yaml`
makes that a check rather than a discipline: one entry per claim, pinned to the
artifact fields that carry it, with a marker in every document that states it;
`tests/test_claims.py` fails when an artifact stops saying what a sentence says.

## A claim entry

```yaml
- id: w1-accuracy-gain                      # stable slug, the marker's name
  statement: Averaging the four tilings' decisions raises pixel accuracy ...
  status: supported                         # supported | mixed | rejected | measured | superseded
  experiments: [exp42]
  artifacts: [exp/out/exp42_summary.json]   # committed summary JSON or CSV
  check: |-                                 # Python over A: path -> parsed JSON, or CSV rows
    (lambda s: s['prereg']['supported'] is True and ...)(A['exp/out/exp42_summary.json'])
  cited_in: [{file: docs/Findings.md, marker: w1-accuracy-gain}]
```

Optional: `note` (a caveat) and `superseded_by: expNN` (status `superseded`).
`status` follows the ledger's verdict (`checked`, `built` and measurements are
`measured`; `not supported` is `rejected`); no artifact means `artifacts: []`,
`check: "True"` and a note saying so.

**The marker.** The sentence or table row that carries a claim ends with the
HTML comment `<!-- claim:<id> -->`, after the final pipe for a table row; it is
invisible on the rendered site and on GitHub. A line may carry several markers.

**The tests.** Every `check` is `True` on the artifacts (a failure names the
claim); every `cited_in` has its marker and every marker names a registered
claim; a superseded claim's citing paragraph or row names the superseding run;
every bold-verdict row of the technique ledger carries a marker; ids are unique.

**After a run lands.** Commit the artifact, then `python scripts/claims.py
stale`: it re-evaluates every check and prints the failing or superseded claims
with each citation's file and line. Rewrite those sentences, update the entries
(status, check, `superseded_by`), add entries and markers for the new claims, and
run `pytest tests/test_claims.py`. `claims.py list --exp exp50` shows what rests
on a run; `claims.py show <id>` prints an entry.
