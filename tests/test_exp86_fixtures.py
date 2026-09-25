"""exp86's fixtures (exp/exp86_fixtures.py; docs/plan/agent_trial_v2.md, "Fixtures" and amendment A5).

Two kinds of test. The builder's pieces on small synthetic inputs: the acquisition periods read from file names, F4's
stacking of chips and its rows, F1's rows and facts, the filled sheet. And the built trial, exp/out/exp86_trial/, when
it is present: every fixture hashed in trial.json as the driver checks it, the provider's run as the driver reads it,
each configuration's workspace as the scorer's table gives it, the facts the plan states recomputed from the files,
the sheets filled from the expert labels, F4's dates, and no coordinate in a file the agent's tools read."""
import csv
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))

import exp86_agent_trial_v2 as e86  # noqa: E402
import exp86_driver as drv  # noqa: E402
import exp86_fixtures as fx  # noqa: E402

TRIAL = fx.TRIAL
built = pytest.mark.skipif(not os.path.exists(os.path.join(TRIAL, "trial.json")),
                           reason="exp/out/exp86_trial is not built (uv run python exp/exp86_fixtures.py)")


# --------------------------------------------------------------------------------------------- the builder's pieces
def test_acquisition_periods_come_from_the_file_names_of_the_event_only():
    names = ["EMSR279-11-0_s1grd_pre_20170914T060856.tif", "EMSR279-11-1_s1grd_pre_20170915T060109.tif",
             "EMSR279-11-0_s1grd_post_20180419T060132.tif", "EMSR279-11-1_s1grd_post_20180419T060107.tif",
             "EMSR279-1-0_s1grd_pre_20160101T000000.tif", "EMSR279-11-0_s1rtc_pre_20170101T000000.tif"]
    p = fx.acquisition_periods(names, "EMSR279-11")
    assert p["pre"]["period"] == "2017-09-14/2017-09-15" and p["post"]["period"] == "2018-04-19"
    assert p["pre"]["n_tiles"] == p["post"]["n_tiles"] == 2
    with pytest.raises(SystemExit, match="no date is invented"):
        fx.acquisition_periods(names[:2], "EMSR279-11")


def test_f4_stacks_the_events_chips_and_keeps_each_windows_class():
    rng = np.random.default_rng(0)
    n, g = 3, 4
    ok = rng.random((n + 2, g, g)) < 0.8
    masks = {"event": np.array(["E1", "E2", "E1", "E1", "E2"]), "ok": ok,
             "A_s1pre": rng.random((n + 2, g, g)) < 0.5, "B_s1post": rng.random((n + 2, g, g)) < 0.5,
             "margin_A_s1pre": rng.random((n + 2, g, g)).astype(np.float32) * 0.5 + 0.01,
             "margin_B_s1post": rng.random((n + 2, g, g)).astype(np.float32) * 0.5 + 0.01}
    out, facts = fx.f4_payloads(masks, "E1")
    sel = masks["event"] == "E1"
    assert out["pre"]["grid"] == [n * g, g] and facts["n_chips"] == n
    w = out["pre"]["windows"]
    assert w == np.flatnonzero(ok[sel].reshape(-1)).tolist() == out["post"]["windows"]
    chip, rest = divmod(w[5], g * g)                       # window (r, c) is row r % g of chip r // g
    r, c = divmod(w[5], g)
    assert (r // g, (r % g) * g + c) == (chip, rest)
    for tag, key in (("pre", "A_s1pre"), ("post", "B_s1post")):
        dec = masks[key][sel].reshape(-1)[w]
        m = masks[f"margin_{key}"][sel].reshape(-1)[w]
        rows = np.array(out[tag]["scores"])
        assert (rows.argmax(1) == dec).all() and np.allclose(rows.max(1), m, rtol=1e-6)
    masks["margin_A_s1pre"][0, 0, 0] = 0.0
    masks["ok"][0, 0, 0] = True
    with pytest.raises(SystemExit, match="finite and positive"):
        fx.f4_payloads(masks, "E1")


def test_f1_rows_are_the_labelled_windows_and_its_facts_recompute():
    probs = np.full((3, 4, 5), 1 / 3)
    probs[:, 0, 0] = [0.6, 0.3, 0.1]
    probs[:, 1, 1] = np.nan
    expert = np.zeros((4, 5), int)
    expert[2, 2] = -1
    payload, reference = fx.f1_payload(probs, expert, ["a", "b", "c"])
    assert payload["grid"] == [4, 5] and len(payload["windows"]) == 18
    assert 6 not in payload["windows"] and 12 not in payload["windows"]
    assert payload["scores"][0] == [0.6, 0.3, 0.1] and (reference == 0).all()
    facts = fx.f1_facts(payload, reference)
    assert facts["n_windows"] == 18 and facts["n_left_out"] == 2 and facts["review_cut"] == 1
    assert facts["error_rate"] == 0.0 and facts["tie_at_cut"] == (1, 16)     # 17 windows tie at margin 0


def test_the_sheet_is_filled_from_the_reference(tmp_path):
    p = tmp_path / "sheet.csv"
    p.write_text("order,window_index,row,col,map_class,stratum,wrong,reference_class\n0,7,0,7,2,0,,\n1,9,0,9,1,1,,\n")
    assert fx.fill_sheet(str(p), {7: 2, 9: 4}) == (2, 1)
    rows = list(csv.DictReader(open(p)))
    assert [(r["wrong"], r["reference_class"]) for r in rows] == [("0", "2"), ("1", "4")]


# --------------------------------------------------------------------------------------------- the built trial
@built
def test_every_fixture_is_hashed_as_the_driver_checks_it():
    have = drv.verified_fixtures(TRIAL)                     # raises on any unlisted, missing or changed file
    assert set(have) <= fx.expected_files() and fx.check(TRIAL) == []
    meta = json.load(open(os.path.join(TRIAL, "trial.json")))
    assert meta["model"] == e86.MODEL == fx.MODEL
    tops = {rel.split("/")[0] for rel in have}
    assert tops == {"F1", "F2", "F3", "F4", "C1"} | ({"C2"} if "C2" not in meta["pending"] else set())


@built
def test_the_providers_run_is_its_manifest_and_raster_only():
    runs = drv.provider_fixtures(TRIAL, drv.verified_fixtures(TRIAL))
    c1 = runs[f"C1/{fx.C1_RUN}"]
    assert c1["sha256"] == fx.C1_SHA and c1["fixture"] == "C1"
    meta = json.load(open(os.path.join(TRIAL, "trial.json")))
    if "C2" in meta["pending"]:
        assert f"C2/{fx.C2_RUN}" not in runs and meta["pending"]["C2"]["cluster_job"] == fx.C2_JOB
    else:
        assert runs[f"C2/{fx.C2_RUN}"]["fixture"] == "C2"


@built
def test_each_configuration_receives_only_its_own_fixtures():
    fixtures = drv.verified_fixtures(TRIAL)
    providers = drv.provider_fixtures(TRIAL, fixtures)
    consts = drv.scorer_constants()
    for config, tops in e86.WORKSPACE_FIXTURES.items():
        got = {rel.split("/")[0] for rel in drv.workspace_fixtures(fixtures, providers, config, consts)}
        assert got <= set(tops), config
        assert not (config.endswith("/studio") and got), config


@built
def test_the_brief_values_fill_the_preregistered_briefs_with_fixtures():
    rnd = json.load(open(os.path.join(TRIAL, "rounds", "1", "round.json")))
    meta = json.load(open(os.path.join(TRIAL, "trial.json")))
    assert rnd["brief_values"] == meta["brief_values"] and "B8/studio" in rnd["not_run"]
    fixtures = drv.verified_fixtures(TRIAL)
    for config, values in rnd["brief_values"].items():
        text = drv.fill_brief(e86.BRIEF_CONFIGS[config]["brief"], values)
        assert e86.brief_matches(config, text), config
        for v in drv.brief_files(e86.BRIEF_CONFIGS[config]["brief"], values).values():
            assert v in fixtures, (config, v)
        for v in drv.brief_run_dirs(e86.BRIEF_CONFIGS[config]["brief"], values).values():
            assert v.split("/")[0] in e86.WORKSPACE_FIXTURES[config], (config, v)


@built
def test_f1_is_the_fixture_the_plan_describes():
    payload = json.load(open(os.path.join(TRIAL, "fixtures", "F1", "dw_scores.json")))
    z = np.load(fx.dw_tile())
    reference = z["expert"].astype(int).ravel()[payload["windows"]]
    facts = fx.f1_facts(payload, reference)
    assert {k: (tuple(v) if k == "tie_at_cut" else v) for k, v in facts.items() if k in fx.PLAN_F1} == fx.PLAN_F1
    probs = z["probs"].astype(np.float32).astype(np.float64).reshape(9, -1).T[payload["windows"]]
    assert np.array_equal(np.asarray(payload["scores"]), probs)             # the float16 values, exactly


@built
def test_the_sheets_are_the_designs_windows_labelled_by_the_expert():
    z = np.load(fx.dw_tile())
    expert = z["expert"].astype(int).ravel()
    meta = json.load(open(os.path.join(TRIAL, "trial.json")))
    for tag, design in (("F2", "confidence"), ("F3", "random")):
        d = json.load(open(os.path.join(TRIAL, "fixtures", meta["fixture_notes"][tag]["design"])))
        rows = list(csv.DictReader(open(os.path.join(TRIAL, "fixtures", meta["fixture_notes"][tag]["labels"]))))
        assert d["design"] == design and d["budget"] == fx.BUDGET and d["seed"] == fx.SEED
        assert d["population"]["source"]["scores_path"] == "F1/dw_scores.json"
        assert [int(r["window_index"]) for r in rows] == [int(i) for i in d["sample"]["indices"]]
        for r in rows:
            w = int(r["window_index"])
            assert int(r["reference_class"]) == expert[w] >= 0
            assert int(r["wrong"]) == int(int(r["map_class"]) != expert[w])
        assert meta["fixture_notes"][tag]["n_wrong"] == sum(int(r["wrong"]) for r in rows)
    assert meta["fixture_notes"]["F3"]["certified"] == fx.PLAN_F3


@built
def test_f4s_dates_have_a_recorded_source_and_read_as_different_times():
    meta = json.load(open(os.path.join(TRIAL, "trial.json")))
    f4 = meta["fixture_notes"]["F4"]
    src = f4["date_source"]
    assert src["repo"] == fx.GEOID_REPO and src["revision"] == fx.GEOID_REVISION and src["revision"] in src["url"]
    assert meta["brief_values"]["B7/files"]["date_a"] == f4["date_a"] == f4["acquisitions"]["pre"]["period"]
    assert meta["brief_values"]["B7/files"]["date_b"] == f4["date_b"] == f4["acquisitions"]["post"]["period"]
    reading = e86.oe_compare.dates_reading(f4["date_a"], f4["date_b"])
    assert reading["status"] == "different_time" and reading["days_apart"] > 0
    pre = json.load(open(os.path.join(TRIAL, "fixtures", "F4", "emsr279-11_s1_pre.json")))
    post = json.load(open(os.path.join(TRIAL, "fixtures", "F4", "emsr279-11_s1_post.json")))
    assert pre["windows"] == post["windows"] and len(pre["scores"]) == len(post["scores"]) == f4["n_windows"]


@built
def test_no_file_the_agents_tools_read_holds_a_coordinate():
    """The JSON fixtures and the sheets a tool reads and may echo; the provider's manifest keeps its area, which is
    the provider's input and never in its result."""
    base = os.path.join(TRIAL, "fixtures")
    for rel in drv.verified_fixtures(TRIAL):
        path = os.path.join(base, rel)
        if rel.startswith(("C1/", "C2/")):
            continue
        found = []
        if rel.endswith(".json"):
            e86._walk_coordinates(json.load(open(path)), rel, found, "")
        else:
            header = open(path).readline().strip().split(",")
            found = [h for h in header if e86._coord_key(h)]
        assert not found, (rel, found[:3])
