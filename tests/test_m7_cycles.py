"""Exercise complete red/blue control flow with deterministic offline scorers."""
from types import SimpleNamespace

import pytest

from cbi.m4.context_prior import Fusion
from cbi.m4.detector import Detector, active_detector_dir, detector_fingerprint
from cbi.m7.blue_llm import run_llm_cycle
from cbi.util import append_jsonl, read_json
from test_m4_batch import _claim
from test_p4_offline import local_corpus


@pytest.fixture
def blue_setup(local_corpus, monkeypatch):
    cfg, _ = local_corpus
    baseline = cfg.paths.models / "detector_llm_v2"
    detector = Detector(cfg, None, None, Fusion(evidence_threshold=.6),
                        version="det-llm-2.0", backend="llm", evidence_variant="holistic")
    detector.save(baseline)
    cfg.paths.artifacts.mkdir(parents=True, exist_ok=True)
    (cfg.paths.artifacts / "round_20").mkdir()
    texts = {"optimal_cutpoint": "maximize separation", "subgroup_selection": "focused on early-stage",
             "evasive_reporting": "most promising"}

    def add_round(version, number):
        for i, family in enumerate(texts):
            append_jsonl(cfg.paths.experiment_log, {"round": number, "submission_id": f"r{number}_{i}",
                        "strategy_family": family, "detector_version": version,
                        "score": .2, "flag_rate": 0, "rejected_by": None})

    add_round(detector.version, 1)
    monkeypatch.setattr("cbi.m7.blue._submission_reports", lambda cfg, entry:
                        [(_claim(entry["submission_id"]), texts[entry["strategy_family"]])])

    def score_many(det, items, workers):
        results = []
        for claim, text, cid in items:
            score = .9 if cid.endswith("1") else .1
            if cid.startswith("test") and det.version.endswith("-2"):
                score = .9  # Cycle two violates the test FPR tolerance.
            results.append(SimpleNamespace(evidence_score=score, flag=True))
        return results

    monkeypatch.setattr("cbi.m7.blue_llm._score_many", score_many)
    red_calls = []

    def red_round(cfg, **kwargs):
        red_calls.append(kwargs)
        active = Detector.load(cfg, kwargs["detector_dir"])
        add_round(active.version, kwargs["round_no"])
        return [SimpleNamespace(score=.2)]

    monkeypatch.setattr("cbi.m7.blue_llm.run_round", red_round)
    return cfg, baseline, red_calls, texts


def test_three_cycles_preserve_baseline_and_reject_without_promotion(blue_setup):
    cfg, baseline, calls, texts = blue_setup
    original = detector_fingerprint(baseline)
    report = run_llm_cycle(cfg, cycles=3, workers=2, sandbox=False)
    assert report["status"] == "complete" and report["n_cycles"] == 3
    assert [r["accepted"] for r in report["cycles"]] == [True, False, True]
    assert [c["round_no"] for c in calls] == [21, 22, 23]
    assert all(c["workers"] == 2 and c["sandbox"] is False for c in calls)
    assert calls[1]["detector_dir"] == calls[0]["detector_dir"]
    assert report["cycles"][2]["parent_version"] == report["cycles"][0]["version"]
    assert detector_fingerprint(baseline) == original
    assert Detector.load(cfg, baseline).fusion.evidence_threshold == .6
    assert active_detector_dir(cfg.paths.models) == calls[-1]["detector_dir"]
    for row in report["cycles"]:
        assert not set(row["training_families"]) & set(report["holdout_families"])
        saved = Detector.load(cfg, row["detector_directory"])
        assert saved.evidence_variant == "holistic"
        for held in report["holdout_families"]:
            assert all(texts[held] not in example for example in saved.learned_examples)
        assert row["holdout_recall"]["n_reports"] == 1
    assert read_json(cfg.paths.reports / "P7_metrics.json")["n_cycles"] == 3


def test_failed_red_round_preserves_partial_report_and_unique_run(blue_setup, monkeypatch):
    cfg, baseline, _, _ = blue_setup
    first = run_llm_cycle(cfg, cycles=1, workers=1)
    first_dir = active_detector_dir(cfg.paths.models)
    first_hash = detector_fingerprint(first_dir)
    monkeypatch.setattr("cbi.m7.blue_llm.run_round", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="no graded submissions"):
        run_llm_cycle(cfg, cycles=3, workers=1)
    partial = read_json(cfg.paths.reports / "P7_metrics.json")
    assert partial["run_id"] != first["run_id"]
    assert partial["status"] == "failed" and partial["n_cycles"] == 0
    assert len(partial["cycles"]) == 1
    assert detector_fingerprint(first_dir) == first_hash


def test_retraining_p4_invalidates_blue_active_pointer(blue_setup):
    cfg, baseline, _, _ = blue_setup
    run_llm_cycle(cfg, cycles=1, workers=1)
    assert active_detector_dir(cfg.paths.models) != baseline
    updated = Detector.load(cfg, baseline)
    updated.fusion.evidence_threshold = .7
    updated.save(baseline)
    assert active_detector_dir(cfg.paths.models) == baseline


def test_cycles_must_be_positive(local_corpus):
    cfg, _ = local_corpus
    with pytest.raises(ValueError, match="positive"):
        run_llm_cycle(cfg, cycles=0)


def test_no_p4_v2_fails_before_scoring(local_corpus):
    cfg, _ = local_corpus
    with pytest.raises(RuntimeError, match="completed P4 v2"):
        run_llm_cycle(cfg)
