"""Score P4 claims with a small, resumable, low-concurrency worker pool."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbi.config import load_config
from cbi.m4.llm_judge import llm_checklist_score_many
from cbi.m4.train import load_natural_corpus
from cbi.schemas import GroundTruthResult
from cbi.util import read_json, write_json


def main() -> int:
    cfg = load_config(ROOT)
    claims, split, _ = load_natural_corpus(cfg)
    gt = {cid: GroundTruthResult.model_validate(x)
          for cid, x in read_json(cfg.paths.processed / "ground_truth.json").items()}
    eligible = {cid for cid, r in gt.items() if r.label_eligible and r.independent_cohorts}
    ids = [c.claim_id for c in claims if split[c.claim_id] in {"train", "val", "test"}
           and c.claim_id in eligible]
    by_id = {c.claim_id: c for c in claims}
    texts = {cid: (cfg.paths.processed / "corpus" / "manuscripts" / f"{cid}.md").read_text(encoding="utf-8")
             for cid in ids}
    cache_path = cfg.paths.processed / "p4_scores.json"
    model = os.environ.get("CBI_LLM_MODEL", "")
    cache = {}
    if cache_path.is_file():
        try:
            old = read_json(cache_path)
            if old.get("model") == model:
                cache = dict(old.get("claims") or {})
        except Exception:
            cache = {}
    batch_size = max(1, int(os.environ.get("CBI_EVIDENCE_BATCH_SIZE", "4")))
    workers = max(1, int(os.environ.get("CBI_P4_WORKERS", "2")))
    print(f"ids={len(ids)} cached={len(cache)} workers={workers} batch={batch_size}", flush=True)

    def score_chunk(chunk: list[str]) -> dict[str, dict]:
        try:
            judgements = llm_checklist_score_many([(by_id[c], texts[c]) for c in chunk])
        except Exception as exc:
            print(f"chunk error {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            return {}
        out = {}
        for cid, judgement in zip(chunk, judgements):
            if judgement is None:
                continue
            out[cid] = {"score": float(judgement["score"]),
                        "rationale": list(judgement["rationale"])}
        return out

    for attempt in range(8):
        missing = [cid for cid in ids if cid not in cache]
        if not missing:
            break
        chunks = [missing[i:i + batch_size] for i in range(0, len(missing), batch_size)]
        print(f"attempt {attempt + 1}: missing={len(missing)} chunks={len(chunks)}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(score_chunk, chunk): chunk for chunk in chunks}
            for fut in as_completed(futures):
                gained = fut.result()
                if not gained:
                    continue
                cache.update(gained)
                write_json(cache_path, {"model": model, "claims": cache})
                print(f"cached={len(cache)}/{len(ids)}", flush=True)
        if len(cache) == len(ids):
            break
        time.sleep(min(60, 10 * (attempt + 1)))
    missing = [cid for cid in ids if cid not in cache]
    print(json.dumps({"ids": len(ids), "cached": len(cache), "missing": len(missing)}, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
