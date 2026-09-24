"""The command line and the Python API give the same numbers on the same data (review of 2026-09-23).

The second review found `compare_inferences` grading unlabelled windows as a class where the command line skipped
them: the same arrays gave two different answers depending on the door a user came in by. Each test here runs one
input through both doors and requires equality, so a fix made on one side and not the other fails here."""
import csv
import json

import numpy as np
import pytest

from oe_inferencex import estimate as est
from oe_inferencex.assess import assess_prediction
from oe_inferencex.cli import main
from oe_inferencex.compare import compare_inferences


def _json(path):
    with open(path) as f:
        return json.load(f)


def test_compare_cli_and_api_agree_with_unlabelled_windows(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, (32, 32))
    b = np.where(rng.random((32, 32)) < 0.2, rng.integers(0, 3, (32, 32)), a)
    lab = np.where(rng.random((32, 32)) < 0.3, -1, np.where(rng.random((32, 32)) < 0.8, a, b))
    for name, x in (("a", a), ("b", b), ("lab", lab)):
        np.save(tmp_path / f"{name}.npy", x)
    out = tmp_path / "c"
    assert main(["compare", str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), "--patch", "1", "--labels",
                 str(tmp_path / "lab.npy"), "--out", str(out)]) == 0
    cli = _json(out / "comparison.json")
    api = compare_inferences(a, b, np.ones_like(a, bool), labels=lab)
    assert cli["n_disagree"] == api["n_disagree"] and cli["n_windows"] == api["n_windows"]
    for k in ("n", "errors_a", "errors_b", "corrected", "broken", "both"):
        assert cli["graded"]["crosstab"][k] == api["graded"]["crosstab"][k], k
    for k in ("n_disagree", "a_right", "b_right", "neither"):
        assert cli["graded"]["which_side"][k] == api["graded"]["which_side"][k], k


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    d = tmp_path_factory.mktemp("cons")
    rng = np.random.default_rng(4)
    logits = rng.normal(0, 1.5, (3, 64, 64))
    probs = np.exp(logits - logits.max(0)); probs /= probs.sum(0)
    truth = np.where(rng.random((16, 16)) < 0.8, assess_prediction(probs, is_logit=False)["arrays"]["pooled_argmax"],
                     rng.integers(0, 3, (16, 16)))
    np.save(d / "p.npy", probs.astype(np.float32))
    return d, probs.astype(np.float32), truth


def _sampled(scored, design, budget=120):
    d, probs, truth = scored
    csv_path = str(d / f"{design}.csv")
    assert main(["sample", str(d / "p.npy"), "--budget", str(budget), "--design", design, "--out", csv_path, "--seed", "1"]) == 0
    arr = assess_prediction(probs, is_logit=False)["arrays"]
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        i, j = int(r["window_row"]), int(r["window_col"])
        r["reference_class"] = str(int(truth[i, j]))
        r["wrong"] = str(int(arr["pooled_argmax"][i, j] != truth[i, j]))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    side = _json(csv_path[:-4] + ".json")
    sample = {k: (np.asarray(v) if k in ("indices", "strata", "strata_of_population", "tiles") else v) for k, v in side.items()}
    idx = np.array([int(r["index"]) for r in rows])
    wrong = np.array([int(r["wrong"]) for r in rows])
    ref = np.array([int(r["reference_class"]) for r in rows])
    return csv_path, sample, idx, wrong, ref, arr


@pytest.mark.parametrize("design", ["random", "confidence"])
def test_estimate_and_per_class_cli_and_api_agree(scored, tmp_path, design):
    csv_path, sample, idx, wrong, ref, arr = _sampled(scored, design)
    out = tmp_path / "e.json"
    assert main(["estimate", csv_path, "--per-class", "--out", str(out)]) == 0
    cli = _json(out)
    api = est.estimate_error_rate(sample, wrong)
    assert (cli["estimate"], cli["low"], cli["high"]) == (api["estimate"], api["low"], api["high"])
    map_class = np.where(arr["valid"], arr["pooled_argmax"], -1).ravel()
    pc = est.estimate_per_class(sample, ref, map_class, n_classes=3)
    assert cli["overall_accuracy"] == pc["overall_accuracy"]
    for c, row in pc["per_class"].items():
        for q in ("user_accuracy", "producer_accuracy", "reference_share"):
            assert cli["per_class"][str(c)][q] == row[q], (c, q)


@pytest.mark.parametrize("rule", ["prefix", "bonferroni"])
def test_certify_cli_and_api_agree(scored, tmp_path, rule):
    csv_path, sample, idx, wrong, ref, arr = _sampled(scored, "random", budget=200)
    out = tmp_path / "z.json"
    assert main(["certify", csv_path, "--alpha", "0.4", "--rule", rule, "--out", str(out)]) == 0
    cli = _json(out)
    api = est.certify_zone(arr["confidence"].ravel(), idx, wrong, 0.4, rule=rule, valid=arr["valid"].ravel())
    assert cli["coverage"] == api["coverage"] and cli["n_zone"] == api["n_zone"]
    assert [lv["accepted"] for lv in cli["levels"]] == [lv["accepted"] for lv in api["levels"]]
