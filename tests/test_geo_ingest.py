from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbi.cohort_pool.geo import collapse_probes, parse_platform_annotation, parse_series_matrix


def _write_gz(path: Path, text: str) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)
    return path


def test_series_matrix_and_annotation_mapping(tmp_path):
    series = _write_gz(tmp_path / "GSE1_series_matrix.txt.gz", '''!Series_title\t"test"
!Series_platform_id\t"GPL1"
!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"
!Sample_characteristics_ch1\t"event_os: 1"\t"event_os: 0"\t"event_os: 0"
!Sample_characteristics_ch1\t"duryears_os: 2.0"\t"duryears_os: 1.0"\t"duryears_os: 0"
!Sample_characteristics_ch1\t"age (years): 60"\t"age (years): 70"\t"age (years): 50"
!Sample_characteristics_ch1\t"gender (1=m, 2=f): 1"\t"gender (1=m, 2=f): 2"\t"gender (1=m, 2=f): 1"
!Sample_characteristics_ch1\t"tnm_staging_clinical: IIIA"\t"tnm_staging_clinical: I"\t"tnm_staging_clinical: II"
!Sample_characteristics_ch1\t"tissue: primary hepatocellular carcinoma tumor"\t"tissue: primary hepatocellular carcinoma tumor"\t"tissue: primary hepatocellular carcinoma tumor"
!series_matrix_table_begin
"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"
"P1"\t10\t20\t30
"P2"\t5\t7\t9
"P3"\t1\t2\t3
!series_matrix_table_end
''')
    clinical, expr, meta = parse_series_matrix(series)
    assert len(clinical) == 2  # GSM3 has zero follow-up
    assert clinical["os_event"].tolist() == [1, 0]
    assert clinical["stage_ajcc"].tolist() == ["III", "I"]
    assert set(clinical["sample_type"]) == {"primary"}
    assert clinical["os_time_days"].iloc[0] == pytest.approx(2.0 * 365.25)
    assert expr.shape == (3, 3)
    assert meta["platform"] == "GPL1"

    annot = _write_gz(tmp_path / "GPL1.annot.gz", '''^Annotation
!platform_table_begin
ID\tGene title\tGene symbol
P1\tgene one\tAAA
P2\tgene two\tBBB
P3\tgene three\tAAA
''')
    annotation = parse_platform_annotation(annot)
    genes = collapse_probes(expr, annotation)
    assert genes.shape == (3, 2)
    # P1 has the larger mean for AAA and must be the retained probe.
    assert genes.loc["GSM1", "AAA"] == 10
    assert genes.loc["GSM1", "BBB"] == 5


def test_real_cohort_evaluator(tmp_path):
    import pandas as pd
    import numpy as np
    from cbi.m2.real_cohort import evaluate_genes

    rng = np.random.default_rng(3)
    n = 180
    x = rng.normal(size=n)
    t = rng.exponential(800, size=n) * np.exp(-0.6 * x)
    c = rng.exponential(2000, size=n)
    clinical = pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(n)],
        "patient_id": [f"S{i}" for i in range(n)],
        "os_time_days": np.minimum(t, c),
        "os_event": (t <= c).astype(int),
        "stage_ajcc": "II",
        "age_years": 60,
        "sex": "female",
        "neoadjuvant": False,
        "sample_type": "primary",
        "platform": "test",
        "expr_unit": "log2",
    })
    expression = pd.DataFrame({"GENE1": x}, index=clinical["sample_id"])
    clinical.to_parquet(tmp_path / "clinical.parquet", index=False)
    expression.to_parquet(tmp_path / "expression.parquet")
    results = evaluate_genes(tmp_path, ["GENE1", "MISSING"])
    assert results[0]["gene"] == "GENE1" and results[0]["hr"] > 1.0
    assert results[1]["error"] == "gene not in expression matrix"
