"""Content provenance for monitor documents and paid evidence judgements."""
from __future__ import annotations

import os
from pathlib import Path

from cbi.config import Config
from cbi.m1 import pmc
from cbi.m4 import llm_judge, monitor_doc
from cbi.schemas import Claim
from cbi.util import atomic_write_text, read_json, sha256_file, sha256_obj, sha256_text, write_json


def scoring_contract(batch_size: int) -> dict:
    """Hash implementation as well as mutable rubric and request settings.

    No credentials or endpoint URLs are persisted. Source hashing deliberately
    invalidates conservatively, including prompt/parser edits.
    """
    return {
        "version": 1, "judge_source": sha256_file(llm_judge.__file__),
        "checklist": llm_judge.CHECKLIST, "rubric": llm_judge.FLAG_RUBRIC,
        "flag_keys": llm_judge.FLAG_KEYS, "flag_weights": llm_judge.FLAG_WEIGHTS,
        "batch_size": batch_size,
        "batch_chars": int(os.environ.get("CBI_EVIDENCE_BATCH_CHARS", "14000")),
        "single_max_tokens": int(os.environ.get("CBI_EVIDENCE_MAX_TOKENS", "8192")),
        "batch_max_tokens": int(os.environ.get("CBI_EVIDENCE_BATCH_MAX_TOKENS", "16384")),
    }


def prepare_documents(cfg: Config, claims: list[Claim]) -> tuple[dict[str, str], dict[str, str], dict]:
    """Refresh source-backed documents when claim/XML/builder/content changes.

    Existing document-only inputs remain usable when no raw source exists, but
    that limitation is counted explicitly. Untracked source-backed documents
    are rebuilt once; a manifest is never used to bless old scores retroactively.
    """
    directory = cfg.paths.processed / "corpus" / "monitor_docs_v2"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    old = read_json(manifest_path) if manifest_path.exists() else {}
    if not isinstance(old, dict) or not isinstance(old.get("claims", {}), dict):
        raise RuntimeError(f"Invalid monitor document manifest: {manifest_path}")
    entries = dict(old.get("claims", {}))
    builder = {"builder": sha256_file(monitor_doc.__file__), "parser": sha256_file(pmc.__file__),
               "max_chars": int(os.environ.get("CBI_MONITOR_MAX_CHARS", "14000"))}
    texts, sources = {}, {}
    counts = {"rebuilt": 0, "reused": 0, "document_only": 0, "manuscript_fallback": 0}
    # Several claims can share one article; avoid repeated XML hashing/parsing.
    hashes, articles = {}, {}
    for claim in claims:
        cid = claim.claim_id
        path = directory / f"{cid}.md"
        xml = cfg.paths.raw / "pmc_real" / f"{claim.source.pmcid}.xml"
        manuscript = cfg.paths.processed / "corpus" / "manuscripts" / f"{cid}.md"
        source = xml if claim.source.pmcid and xml.is_file() else manuscript
        kind = "xml" if source == xml else "manuscript"
        if not source.is_file():
            source, kind = path, "document_only"
        if not source.is_file():
            raise RuntimeError(f"No monitor document or local manuscript source for {cid}")
        if source not in hashes:
            hashes[source] = sha256_file(source)
        source_hash = sha256_obj({"claim": claim.model_dump(mode="json"), "source": hashes[source],
                                  "kind": kind, "builder": builder})
        previous = entries.get(cid, {})
        reusable = (path.is_file() and previous.get("source_fingerprint") == source_hash
                    and previous.get("document_hash") == sha256_file(path))
        if reusable or kind == "document_only":
            text = path.read_text(encoding="utf-8")
            counts["reused"] += 1
        else:
            if kind == "xml":
                if source not in articles:
                    articles[source] = pmc.parse_pmc_xml(source)
                text = monitor_doc.build_monitor_document(articles[source], claim,
                                                          max_chars=builder["max_chars"])
            else:
                text = manuscript.read_text(encoding="utf-8")
            if not text.strip():
                raise RuntimeError(f"Empty monitor document for {cid}")
            atomic_write_text(path, text)
            counts["rebuilt"] += 1
        if not text.strip():
            raise RuntimeError(f"Empty monitor document for {cid}")
        counts["document_only"] += int(kind == "document_only")
        counts["manuscript_fallback"] += int(kind == "manuscript")
        texts[cid], sources[cid] = text, source_hash
        entries[cid] = {"source_fingerprint": source_hash, "document_hash": sha256_text(text), "kind": kind}
    write_json(manifest_path, {"version": 1, "claims": entries})
    return texts, sources, counts


def input_fingerprints(claims: list[Claim], texts: dict[str, str], sources: dict[str, str],
                       *, batch_size: int) -> dict[str, str]:
    contract = scoring_contract(batch_size)
    return {c.claim_id: sha256_obj({"claim": c.model_dump(mode="json"),
                                   "document": sha256_text(texts[c.claim_id]),
                                   "source": sources[c.claim_id], "contract": contract})
            for c in claims}
