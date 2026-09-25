# exp87: commitment to a sealed held-out set

Written 25 September 2026, before any fix after exp86 round 6's audit was built, and before round 7 was scored.

- The set is eight held-out briefs, with their fixtures and reference facts. Each fixture is either:
  - new model runs from the cluster scores provider; or
  - a new design or a transform of an existing fixture that changes the facts a correct answer must state.
- The reference facts were computed by olmoearth-inferencex and by the agent's own tool functions, never by a model.
- The files are kept outside both repositories. They are not read by anyone building the fixes, and they are revealed
  when exp87's preregistration is frozen.
- The set holds 32 files. The digest below is the SHA-256 of the sorted SHA-256 manifest of those files
  (`SEALED_MANIFEST.sha256`).

```
f1c45f433f83c66af0892c1f8babf0ef44ee6d288db55053bc337568e4c61033  SEALED_MANIFEST.sha256
```

At the reveal, the files are copied into `exp/out/exp87_trial/`. The manifest is recomputed there and must reproduce this
digest.
