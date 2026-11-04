"""Paid scores may be reused only for the same source, claim and scoring contract."""
import pytest

from cbi.m4.cache import input_fingerprints, prepare_documents
from cbi.m4.train_llm import train_llm_detector
from cbi.schemas import Claim
from cbi.util import read_json, write_json
from test_m4_batch import _claim
from test_monitor_doc import _article
from test_p4_offline import local_corpus


@pytest.mark.parametrize("change", ["claim", "document", "prompt", "batch", "legacy"])
def test_changed_inputs_are_stale_without_overwriting_scores(local_corpus, monkeypatch, change):
    cfg, cache = local_corpus
    path = cfg.paths.processed / "p4_scores_v2.json"
    if change == "legacy":
        for row in cache["claims"].values():
            del row["input_fingerprint"]
    write_json(path, cache)
    original = path.read_bytes()
    if change == "claim":
        claims_path = cfg.paths.processed / "corpus" / "claims.json"
        claims = read_json(claims_path)
        claims[0]["extraction_confidence"] = .4
        write_json(claims_path, claims)
    elif change == "document":
        (cfg.paths.processed / "corpus" / "monitor_docs_v2" / "train0.md").write_text("Changed evidence.")
    elif change == "prompt":
        monkeypatch.setattr("cbi.m4.llm_judge.FLAG_RUBRIC", "A changed rubric.")
    elif change == "batch":
        monkeypatch.setenv("CBI_EVIDENCE_BATCH_SIZE", "2")
    _, report = train_llm_detector(cfg, prepare_only=True)
    assert len(report["stale_cached_ids"]) == (1 if change in {"claim", "document"} else 6)
    with pytest.raises(RuntimeError, match="Offline P4 requires complete valid scores"):
        train_llm_detector(cfg, offline=True)
    assert path.read_bytes() == original


def test_xml_and_claim_changes_rebuild_monitor_document(local_corpus, monkeypatch):
    cfg, _ = local_corpus
    claim = _claim("source_test")
    claim.source.pmcid = "PMC1"
    source = cfg.paths.raw / "pmc_real" / "PMC1.xml"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("first source")
    calls = []

    def parse(path):
        calls.append(path.read_text())
        article = _article()
        article.abstract = path.read_text()
        return article

    monkeypatch.setattr("cbi.m4.cache.pmc.parse_pmc_xml", parse)
    texts, sources, counts = prepare_documents(cfg, [claim])
    first = input_fingerprints([claim], texts, sources, batch_size=1)
    assert counts["rebuilt"] == 1 and "first source" in texts[claim.claim_id]
    _, _, counts = prepare_documents(cfg, [claim])
    assert counts["reused"] == 1 and len(calls) == 1
    source.write_text("revised source")
    texts, sources, counts = prepare_documents(cfg, [claim])
    assert counts["rebuilt"] == 1 and "revised source" in texts[claim.claim_id]
    assert input_fingerprints([claim], texts, sources, batch_size=1) != first
    claim.reported_cohort.n_reported = 777
    texts, _, counts = prepare_documents(cfg, [claim])
    assert counts["rebuilt"] == 1 and "n_reported=777" in texts[claim.claim_id]


def test_stale_scores_are_archived_before_replacement(local_corpus, monkeypatch):
    cfg, cached = local_corpus
    for row in cached["claims"].values():
        row.pop("input_fingerprint")
    path = cfg.paths.processed / "p4_scores_v2.json"
    write_json(path, cached)
    original = path.read_bytes()
    monkeypatch.setenv("CBI_LLM_MODEL", cached["model"])
    monkeypatch.setattr("cbi.m4.train_llm.llm_ping", lambda: True)
    monkeypatch.setattr("cbi.m4.train_llm.llm_checklist_score_many",
                        lambda items: [cached["claims"][c.claim_id] for c, _ in items])
    train_llm_detector(cfg, workers=1)
    archives = list(path.parent.glob("p4_scores_v2.archive-*.json"))
    assert len(archives) == 1 and archives[0].read_bytes() == original
    assert all("input_fingerprint" in row for row in read_json(path)["claims"].values())
