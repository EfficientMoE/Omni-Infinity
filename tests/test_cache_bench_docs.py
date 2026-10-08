# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""The benchmark doc must exist, cover every suite, and be linked."""

from pathlib import Path

from benchmarks.caches.contract import FIELDS

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "cache_benchmarks.md"


def test_doc_exists_and_covers_every_suite():
    text = DOC.read_text()
    for token in (
        "WenhaoWang/VidProM",
        "VBench",
        "benchmarks.caches.micro",
        "benchmarks.caches.ablation",
        "benchmarks.caches.serve_trace",
        "benchmarks.caches.denoise_probe",
        "benchmarks.caches.denoise_fit",
        "OMNI_CONDITION_CACHE_DIR",
        "encoder-cache",
        "c5-teacache",
        "c5-taylor1",
        "c5-fbcache",
        "multi-prompt demo",
        "c5-uncalibrated",
    ):
        assert token in text, f"docs/cache_benchmarks.md missing {token}"


def test_doc_lists_every_contract_field():
    text = DOC.read_text()
    for field in FIELDS:
        assert field in text, f"contract field {field} undocumented"


def test_readme_links_the_doc():
    readme = (REPO / "README.md").read_text()
    assert "docs/cache_benchmarks.md" in readme
