"""Build M3 context features from a local OpenAlex/Crossref snapshot.

Fails with a clear message when data/raw/context/context.json is absent; it does
not fabricate social features.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbi.config import load_config
from cbi.m3.features import ContextFeatureBuilder
from cbi.m3.sources import load_context_snapshot
from cbi.schemas import Claim, GroundTruthResult
from cbi.util import read_json, write_json


def main() -> int:
    cfg = load_config(ROOT)
    corpus = cfg.paths.processed / "corpus"
    claims = [Claim.model_validate(x) for x in read_json(corpus / "claims.json")]
    gt = {cid: GroundTruthResult.model_validate(v) for cid, v in
          read_json(cfg.paths.processed / "ground_truth.json").items() if "error" not in v}
    snapshot = load_context_snapshot(cfg)
    if not snapshot.papers:
        raise RuntimeError(
            f"no context snapshot at {cfg.paths.raw / 'context' / 'context.json'}; "
            "build one from OpenAlex/Crossref before P3"
        )
    variant = str(cfg.mvp.get("asof_variant", "pub_plus_3y"))
    builder = ContextFeatureBuilder(cfg, claims, gt, snapshot)
    features = builder.build_all(variant=variant)
    write_json(cfg.paths.processed / "context_features.json",
               {cid: f.model_dump(mode="json") for cid, f in features.items()})
    coverage = sum(1 for f in features.values() if f.context_coverage > 0) / len(features) if features else 0.0
    print(json.dumps({"claims": len(features), "variant": variant,
                      "mean_coverage": sum(f.context_coverage for f in features.values()) / len(features) if features else 0.0,
                      "nonzero_coverage": coverage}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
