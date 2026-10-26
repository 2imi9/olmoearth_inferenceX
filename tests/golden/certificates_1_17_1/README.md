Certificates issued by olmoearth-inferencex 1.17.1 (commit 1624ae1) on the quick-start map
(`examples/quickstart_map.py`, which writes the same map every time), kept as they were written:

- `random`: `sample --design random --budget 315 --target 0.5 --seed-from-map --alpha 0.05`, certified by the target rule;
- `class`: `sample --design class --budget 1200 --seed-from-map --alpha 0.2`, a zone per map class;
- `sequential`: `sample --design sequential --budget 300 --seed-from-map --alpha 0.1`;
- `class_named`: the class design as `class`, with `--condition-names 0=input 1=condition "2=good condition" 3=x`,
  names that hold words the class design's texts reword (1.17.2 first reworded them otherwise, and verify failed).

Each was labelled with `quickstart_map.label`, certified, and packed with `certificate --date 2024-06-01`.
`tests/test_bughunt_1_17_2.py` verifies them with the current release: a certificate stays verifiable when the
package changes (1.17.2 added `edge` to certify's result, which failed every one of them until verify learned to
compare without keys a recorded zone predates, and to run certify again with the rules of the release that wrote the
zone).
