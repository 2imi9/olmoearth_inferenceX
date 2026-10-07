"""The last two bugs of the 2026-10-06 hunt: a sample drawn with --nodata nan wrote a sidecar that strict JSON parsers
refuse, and the difference interval of two maps computed its ends in floats, one float inside a truth that sits on
an end."""
import json
import math
import shlex
from fractions import Fraction

import numpy as np

from oe_inferencex import cli
from oe_inferencex import estimate as est


def test_a_nan_nodata_sidecar_is_strict_json_and_reads_back(tmp_path, monkeypatch):
    """The sidecar held the bare token NaN; it now holds "nan", and certify reads the value back."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, (32, 32))
    p[:4, :4] = np.nan
    np.save(tmp_path / "p.npy", p)
    monkeypatch.chdir(tmp_path)
    cli.main(shlex.split("sample p.npy --budget 40 --design random --nodata nan --out s.csv"))
    text = open("s.json").read()
    side = json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))   # strict: no NaN token
    assert side["nodata"] == "nan"
    import csv
    rows = list(csv.DictReader(open("s.csv")))
    for r in rows:
        r["wrong"] = "0"
    with open("s.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert cli.main(shlex.split("certify s.csv --alpha 0.3 --nodata nan")) == 0


def test_the_difference_interval_holds_a_truth_that_sits_on_an_end():
    """A census of the differing windows: the interval is the point (KA - KB) / N, equal to the truth in floats."""
    N, D, KA, KB = 45, 10, 9, 0
    sample = {"indices": np.arange(D), "n_population": N, "n_disagree": D}
    ca, cb = np.ones(D, int), np.zeros(D, int)
    ref = np.where(np.arange(D) < KA, 1, 2)
    r = est.compare_from_disagreement(sample, ca, cb, ref)
    truth = (KA - KB) / N
    assert r["difference"]["low"] <= truth <= r["difference"]["high"]
    assert Fraction(r["difference"]["low"]) <= Fraction(KA - KB, N) <= Fraction(r["difference"]["high"]) or \
        math.isclose(r["difference"]["low"], truth)
