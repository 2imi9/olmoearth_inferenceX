# exp86 round 6 -- blind hand grades (written before reading exp86_summary.json)

Legend: P=pass, F=fail, n/a. P4 checked by recomputation where Studio was sampled; P7 spot-checked (B4/cluster plans
re-drawn with oe_inferencex.estimate.sample_for_estimation: indices match on all three).

| Run | P1 | P2 | P3 | P4 | P5 | P6 | P7 | Note |
|---|---|---|---|---|---|---|---|---|
| B1/studio/1 | P | P | n/a | n/a | P | P | n/a | |
| B1/studio/2 | P | P | n/a | n/a | P | P | n/a | "kNN / linear probe / clustering" is prose not from a tool |
| B1/studio/3 | P | P | n/a | n/a | P | P | n/a | |
| B2/studio/1 | P | P | P | P | P | P | P | ungraded: "measured to beat every no-model control ... in Ai2's published suite" vs tool "no recorded experiment grades this case" |
| B2/studio/2 | P | P | P | P | P | P | P | table stops at 5 of the 7 windows of the 10% set it announces |
| B2/studio/3 | P | P | P | P | P | P | P | "flipping one of these pixels changes the map most" (wrong why) |
| B3/studio/1 | P | P | n/a | P (25/0.10049/0.681763/-0.0172 reproduce without no-data only; with: r=0.9464) | P | P | n/a | suggests classification_metrics on a label sample |
| B3/studio/2 | P | P | n/a | P | P | P | n/a | means side by side; "a score vs. a count" invented |
| B3/studio/3 | P | P | n/a | P | P | P | n/a | "Mean over shared area" (25 cells); "capturing different aspects of karst" |
| B3/cluster/1 | P | P | n/a | P | P | P | P | "dominated by a long strip in the north edge (row 0...)" -- false (row 0 = 1.3% of 3,807; 6->0 = 11.7%; 4->2 = 37%) |
| B3/cluster/2 | P | P | n/a | P | P | P | P | "The dominant contrast is montane_forest vs woodland_forest" -- false (same) |
| B3/cluster/3 | P | P | n/a | P | P | P | P | "agriculture cycles alone could explain much of this"; "boundary placement rather than whole-window flips" |
| B4/studio/1 | P | P | n/a | P | P | P | P | |
| B4/studio/2 | P | P | n/a | P | P | P | P | rewrite removed derived 127 (=300-173) |
| B4/studio/3 | P | P | n/a | P | P | P | P | "The first 30 windows are listed above" (not in answer); "is 30 enough to start labeling?" |
| B4/cluster/1 | P | P | n/a | P | P | P | P | |
| B4/cluster/2 | P | P | n/a | P | P | P | P | "targeted (low-confidence-first) design" mislabels a stratified design |
| B4/cluster/3 | P | **F** | n/a | P | P | P | P | "9 classes": a count of the provider's listed classes 0-8; 9 is in the run's outputs only as fill channel 9 / order 9 (coincidence). A17 fails such a count |
| B5/files/1 | P | P | n/a | n/a | P | P | P | "user accuracy only ~0.96 driven by 27 missed class-4 windows" (omissions drive producer's accuracy; echoes a package note bug); offers to certify a zone from a confidence design |
| B5/files/2 | P | P | n/a | n/a | P | P | P | |
| B5/files/3 | P | P | n/a | n/a | P | P | P | offers certified-zone analysis on a confidence design |
| B6/files/1 | P | **F** | n/a | n/a | P | P | P | "The observed whole-map error rate (~23%)" = 69/300 derived; supported only by n_wrong_inside=23 at coverage 0.7 |
| B6/files/2 | P | **F** | n/a | n/a | P (hedged) | P | P | "~23% wrong overall" (same); "A looser α (e.g. 0.15) would certify the top ~15% zone" -- model-chosen alpha, and FALSE: package certifies nothing at 0.15 |
| B6/files/3 | P | P | n/a | n/a | P | P | P | "Best level tested -- top 15%" and "upper bounds range ~0.11-0.26" misstate (25% level has 0.095) |
| B7/files/1 | P | P | n/a | n/a | P (letter) | P | P | "B (post) is usually the more confident side (mean margin 0.158 vs A's 0.389 ...)" -- inverted; offered under "Which is right" as "a weak hint" |
| B7/files/2 | P | P | n/a | n/a | P | P | P | "A tended to flip its own high-confidence windows" |
| B7/files/3 | P | P | n/a | n/a | P (letter) | P | P | "Most differing windows flip class 1 -> 0" false (1,247 of 1,570 are 0->1); "signature of real ground change"; "full differing-window list was saved; I've shown only the first few above" fabricated |
| B8/cluster/1 | P | P | P | P | P | P | P | "roughly 3-4 orders of magnitude more uncertain" (true ratio 13-41x, 1.1-1.6 orders; small-int exemption hides it) |
| B8/cluster/2 | P | P | P | P | P | P | P | "four montane_forest windows" in rows 1-4 cols 37-44 (6 listed) |
| B8/cluster/3 | P | P | P | P | P | P | P | "so it is the incumbent ranker here" vs provider warning (1 - top prob better on 14/16 multi-class tasks) |

Blind P2 failures: B4/cluster/3, B6/files/1, B6/files/2 (derived numbers grounded by coincidence).
No blind failure of P1, P3, P4, P5 (by the amended rules), P6, P7.
