"""P4 preparation and calibration must work without an LLM connection."""
import json
from pathlib import Path

import pytest

from cbi.cli import main
from cbi.config import Paths, load_config
from cbi.m4.detector import Detector
from cbi.m4.cache import input_fingerprints, prepare_documents
from cbi.m4.llm_judge import _parse_judgement
from cbi.m4.train_llm import train_llm_detector
from cbi.util import write_json
from cbi.schemas import Claim
from test_m4_batch import _claim


@pytest.fixture
def local_corpus(tmp_path, monkeypatch):
    cfg = load_config(Path(__file__).resolve().parents[1])
    cfg.paths = Paths(root=tmp_path, snapshot=cfg.snapshot)
    claims, split, gt, scores = [], {}, {}, {}
    for name in ("train", "val", "test"):
        for label in (0, 1):
            cid = f"{name}{label}"
            claims.append(_claim(cid).model_dump(mode="json"))
            split[cid] = {"split": name}
            gt[cid] = {
                "claim_id": cid, "data_snapshot": cfg.snapshot,
                "contrast_type": "dichotomous", "gene": "TP53", "cancer_type": "LIHC",
                "endpoint": "OS", "reported_log_effect": 0.7, "reported_se": 0.2,
                "original_cohort_repro": {"status": "no_data"},
                "independent_cohorts": [{"cohort_id": "test-only", "log_effect": 0.1,
                                         "se": 0.2, "n": 100, "events": 30}],
                "meta": None, "multiverse": {"n_specs_reference": 1, "n_specs_used": 1,
                                            "frac_same_direction_sig": 0, "median_log_effect": 0},
                "reconciliation_flags": [], "robustness_score": 0.5,
                "inflation": 0.9 * label, "inflation_z": 3.0 * label,
                "gt_confidence": "medium", "weights_used": {},
            }
            scores[cid] = {"score": 0.8 - 0.6 * label, "holistic": 0.2 + 0.6 * label,
                           "flags": {}, "rationale": []}
            if name == "test":
                scores[cid].update(score=0.2 + 0.6 * label, holistic=0.9 - 0.8 * label)
            doc = cfg.paths.processed / "corpus" / "monitor_docs_v2" / f"{cid}.md"
            doc.parent.mkdir(parents=True, exist_ok=True)
            doc.write_text("Claim and methods for offline regression.")
    write_json(cfg.paths.processed / "corpus" / "claims.json", claims)
    write_json(cfg.paths.processed / "corpus" / "natural_split.json", {"by_claim": split})
    write_json(cfg.paths.processed / "ground_truth.json", gt)
    parsed = [Claim.model_validate(c) for c in claims]
    texts, sources, _ = prepare_documents(cfg, parsed)
    fingerprints = input_fingerprints(parsed, texts, sources, batch_size=1)
    for cid, row in scores.items():
        row["input_fingerprint"] = fingerprints[cid]

    def no_network(*args, **kwargs):
        pytest.fail("offline P4 attempted an API call")

    monkeypatch.setattr("cbi.m4.train_llm.llm_ping", no_network)
    monkeypatch.setattr("cbi.m4.train_llm.llm_checklist_score_many", no_network)
    monkeypatch.setattr("cbi.m4.train_llm.llm_checklist_score", no_network)
    for key in ("CBI_EVIDENCE_MODEL", "CBI_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    return cfg, {"model": "cached-model", "version": "monitor_docs_v2+flags_rubric_v1", "claims": scores}


def test_prepare_without_model_reports_missing_and_writes_no_detector(local_corpus, monkeypatch, capsys):
    cfg, _ = local_corpus
    monkeypatch.setattr("cbi.cli.load_config", lambda root: cfg)
    assert main(["p4", "--prepare-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["missing_scores"] == report["documents"] == 6
    assert report["api_requests"] == 0
    assert report["splits"]["val"] == {"n": 2, "positive": 1, "negative": 1}
    assert not report["ready_for_offline_training"]
    assert (cfg.paths.reports / "P4_preflight.json").is_file()
    assert not (cfg.paths.models / "detector_llm_v2" / "meta.json").exists()


def test_offline_missing_scores_fails_before_network(local_corpus):
    cfg, _ = local_corpus
    with pytest.raises(RuntimeError, match="0/6 cached"):
        train_llm_detector(cfg, offline=True)


def test_complete_cache_trains_offline_and_preserves_variant(local_corpus, monkeypatch):
    cfg, cached = local_corpus
    path = cfg.paths.processed / "p4_scores_v2.json"
    write_json(path, cached)
    original = path.read_bytes()
    detector, report = train_llm_detector(cfg, offline=True)
    assert path.read_bytes() == original
    assert detector.evidence_variant == report["evidence_variant"] == "holistic"
    assert report["evidence_rubric_test"]["auc"] == 1
    assert report["evidence_holistic_test"]["auc"] == 0
    operating = report["test_at_validation_threshold"]["evidence_5pct"]
    assert operating["threshold"] == detector.fusion.evidence_threshold
    assert operating["fpr"] == 1 and operating["recall"] == 0
    loaded = Detector.load(cfg, cfg.paths.models / "detector_llm_v2")
    judgement = {"score": 0.1, "holistic": 0.9, "rationale": []}
    monkeypatch.setattr("cbi.m4.detector.llm_checklist_score", lambda *a, **k: judgement)
    monkeypatch.setattr("cbi.m4.detector.llm_checklist_score_many", lambda *a, **k: [judgement, None])
    assert loaded.score(_claim("c0"), "text").evidence_score == 0.9
    assert loaded.flags_many([_claim("c0"), _claim("c1")], ["a", "b"]) == [True, True]
    from cbi.m7.blue_llm import _score_many
    monkeypatch.setattr("cbi.m7.blue_llm.llm_checklist_score_many", lambda *a, **k: [judgement, None])
    results = _score_many(loaded, [(_claim("c0"), "a", "c0"), (_claim("c1"), "b", "c1")], workers=1)
    assert [r.evidence_score for r in results] == [0.9, 0.9]


def test_invalid_cache_rows_are_not_counted_as_finished(local_corpus):
    cfg, cached = local_corpus
    cached["claims"]["val0"]["score"] = float("nan")
    del cached["claims"]["test0"]["holistic"]
    path = cfg.paths.processed / "p4_scores_v2.json"
    write_json(path, cached)
    original = path.read_bytes()
    _, report = train_llm_detector(cfg, prepare_only=True)
    assert report["cached_scores"] == 4
    assert set(report["invalid_cached_ids"]) == {"val0", "test0"}
    with pytest.raises(RuntimeError, match="4/6 cached"):
        train_llm_detector(cfg, offline=True)
    assert path.read_bytes() == original


def test_effective_evidence_model_controls_cache_matching(local_corpus, monkeypatch):
    cfg, cached = local_corpus
    write_json(cfg.paths.processed / "p4_scores_v2.json", cached)
    monkeypatch.setenv("CBI_LLM_MODEL", "different-general-model")
    monkeypatch.setenv("CBI_EVIDENCE_MODEL", "cached-model")
    _, report = train_llm_detector(cfg, prepare_only=True)
    assert report["cached_scores"] == 6
    monkeypatch.setenv("CBI_EVIDENCE_MODEL", "different-evidence-model")
    with pytest.raises(RuntimeError, match="model/version mismatch"):
        train_llm_detector(cfg)


def test_corrupt_cache_is_preserved(local_corpus):
    cfg, _ = local_corpus
    path = cfg.paths.processed / "p4_scores_v2.json"
    path.write_text('{"claims":')
    with pytest.raises(RuntimeError, match="Cannot read P4 cache"):
        train_llm_detector(cfg, offline=True)
    assert path.read_text() == '{"claims":'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_judgements_are_rejected(value):
    with pytest.raises(RuntimeError, match="finite"):
        _parse_judgement({"score": value})
