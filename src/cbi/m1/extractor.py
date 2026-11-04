"""M1 claim extraction: model primary, rules validation only.

The model reads the article (including JATS tables rendered by
`cbi.m1.pmc.parse_pmc_xml`) and returns a schema-valid `Claim`.  The
deterministic consistency audit in `cbi.m1.audit` validates the result.  There
is no rule-based production fallback: without a configured model, M1 blocks.
"""
from __future__ import annotations

import os
from typing import Iterable

from cbi.config import Config
from cbi.m1.audit import audit_claim
from cbi.m1.hgnc import HGNCResolver
from cbi.m1.llm import (LLMServiceError, extract_batch_with_llm, extract_with_llm,
                        llm_available)
from cbi.m1.pmc import PMCArticle
from cbi.schemas import Claim


class LLMRequiredError(RuntimeError):
    """M1's production path requires a configured LLM; rules are validation only."""


class LLMExtractionError(RuntimeError):
    """A configured model failed or its output failed rule validation."""


class ClaimExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.resolver = HGNCResolver(cfg.paths.raw / "hgnc" / "hgnc_complete.tsv")
        self.allowed_cancers = set(cfg.mvp.get("cancers", []))
        self.pool_accessions = {
            str(c.get("accession") or c.get("cohort_id")).upper()
            for c in cfg.cohort_pool.get("cohorts", [])
        }

    def _finalize(self, claim: Claim | None, article: PMCArticle) -> Claim | None:
        if claim is None:
            return None
        symbol, hgnc_id = self.resolver.normalize(claim.entities.gene)
        if symbol:
            claim = claim.model_copy(update={"entities": claim.entities.model_copy(
                update={"gene": symbol, "hgnc_id": hgnc_id or claim.entities.hgnc_id}
            )})
        accessions = [a.upper() for a in claim.reported_cohort.accessions]
        if accessions:
            claim.reported_cohort.in_standard_pool = all(a in self.pool_accessions for a in accessions)
        issues = audit_claim(claim, allowed_cancers=self.allowed_cancers, resolver=self.resolver)
        if issues:
            if "unmapped_cancer_type" in issues:
                # Not one of the MVP cancers: outside scope, not an extraction error.
                return None
            raise LLMExtractionError(f"LLM claim failed rule validation: {issues}")
        return claim

    def extract_article(self, article: PMCArticle) -> Claim | None:
        if not llm_available():
            raise LLMRequiredError(
                "M1 requires a configured LLM endpoint/model "
                "(CBI_LLM_ENDPOINT + CBI_LLM_MODEL, or OPENAI_BASE_URL + OPENAI_MODEL); "
                "rules are validation-only, not a production fallback"
            )
        try:
            claim = extract_with_llm(article)
        except Exception as exc:
            raise LLMExtractionError(f"LLM extraction failed: {type(exc).__name__}: {exc}") from exc
        return self._finalize(claim, article)

    def extract_articles(self, articles: Iterable[PMCArticle], *,
                         batch_size: int | None = None) -> list[tuple[PMCArticle, Claim | None, Exception | None]]:
        """Extract articles in bounded batches, falling back per article when a
        batch answer is missing or invalid.

        The return value preserves input order and reports per-article errors
        instead of aborting a whole 1000-file audit on one bad model response.
        """
        items = list(articles)
        if not items:
            return []
        if not llm_available():
            raise LLMRequiredError(
                "M1 requires a configured LLM endpoint/model "
                "(CBI_LLM_ENDPOINT + CBI_LLM_MODEL, or OPENAI_BASE_URL + OPENAI_MODEL); "
                "rules are validation-only, not a production fallback"
            )
        size = max(1, int(batch_size if batch_size is not None
                          else os.environ.get("CBI_M1_BATCH_SIZE", "4")))
        results: list[tuple[PMCArticle, Claim | None, Exception | None]] = []
        if size == 1:
            for article in items:
                try:
                    results.append((article, self.extract_article(article), None))
                except LLMRequiredError:
                    raise
                except Exception as exc:
                    results.append((article, None, exc))
            return results
        for start in range(0, len(items), size):
            chunk = items[start:start + size]
            batch: dict[str, Claim | None] = {}
            try:
                batch = extract_batch_with_llm(chunk)
            except LLMServiceError as exc:
                # Do not turn one 5xx batch into four more requests.  Mark the
                # chunk as a service error and let the caller record it.
                results.extend((article, None, exc) for article in chunk)
                continue
            except Exception:
                # Fall back to the single-article path below.  The fallback is
                # intentionally explicit: a failed batch is not a negative claim.
                batch = {}
            for index, article in enumerate(chunk):
                key = article.pmcid or f"UNKNOWN{index}"
                if key in batch and batch[key] is not None:
                    try:
                        results.append((article, self._finalize(batch[key], article), None))
                        continue
                    except Exception:
                        # Invalid batch entry: retry singly so one bad object
                        # cannot drop an otherwise extractable article.
                        pass
                elif key in batch:
                    results.append((article, None, None))
                    continue
                try:
                    results.append((article, self.extract_article(article), None))
                except LLMRequiredError:
                    raise
                except Exception as exc:
                    results.append((article, None, exc))
        return results

    def extract(self, articles: Iterable[PMCArticle]) -> list[Claim]:
        out: list[Claim] = []
        for article in articles:
            claim = self.extract_article(article)
            if claim is not None:
                out.append(claim)
        return out
