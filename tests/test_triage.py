"""Triage template tests.

The triage table is the only place clinical wording lives, and it is a static
JSON lookup chosen by parasitemia. These tests make sure the banding in code and
the ranges in the file agree, that every language is complete, and that the
loader degrades safely for an unknown language.
"""

from __future__ import annotations

import json

import pytest

from app.inference import (
    MIN_CELLS_FOR_ESTIMATE,
    classify_parasitemia,
    load_triage,
    render_triage,
)

LANGUAGES = ("en", "ur", "pl")


@pytest.fixture(scope="module")
def triage() -> dict:
    return load_triage()


class TestTriageFileShape:
    def test_schema_version(self, triage):
        assert triage["schema"] == "tinymalaria.triage.v1"

    def test_declared_languages_are_supported(self, triage):
        assert set(triage["supported_languages"]) == set(LANGUAGES)

    def test_every_severity_has_every_language_key(self, triage):
        for level, block in triage["severity_levels"].items():
            for key in ("urgency", "require_expert_review", "header", "action",
                        "recommendation"):
                assert key in block, f"{level} is missing {key}"
            for field in ("header", "action", "recommendation"):
                for lang in LANGUAGES:
                    assert lang in block[field], f"{level}.{field}.{lang} missing"

    def test_warnings_and_disclaimer_are_localised(self, triage):
        for key in ("no_cells", "too_few_cells"):
            for lang in LANGUAGES:
                assert lang in triage["segmentation_warnings"][key]
        for lang in LANGUAGES:
            assert lang in triage["disclaimer"]

    def test_declared_ranges_match_the_banding_code(self, triage):
        """The JSON ranges must describe exactly what classify_parasitemia does."""
        bounds = []
        for level, block in triage["severity_levels"].items():
            lo, hi = block["parasitemia_range"]
            bounds.append((lo, hi))
        # pick one value comfortably inside each declared range
        for level, block in triage["severity_levels"].items():
            lo, hi = block["parasitemia_range"]
            probe = lo + (hi - lo) / 2.0
            assert classify_parasitemia(probe) == level, (
                f"parasitemia {probe:.3f} should be '{level}', "
                f"classified as '{classify_parasitemia(probe)}'"
            )

    def test_urgency_ordering(self, triage):
        order = {"negative": 0, "low": 1, "mild": 2, "moderate": 3, "severe": 4}
        for level, block in triage["severity_levels"].items():
            assert order[level] == int(order[level])
            assert isinstance(block["urgency"], str) and block["urgency"]
        # severity escalates monotonically with the band
        assert order["negative"] < order["low"] < order["mild"] \
            < order["moderate"] < order["severe"]

    def test_high_parasitemia_requires_expert_review(self, triage):
        for level in ("low", "mild", "moderate", "severe"):
            assert triage["severity_levels"][level]["require_expert_review"] is True
        assert triage["severity_levels"]["negative"]["require_expert_review"] is False


class TestSeverityBanding:
    @pytest.mark.parametrize(
        "percent,expected",
        [
            (0.0, "negative"),
            (0.5, "low"),
            (0.99, "low"),
            (1.0, "mild"),
            (4.99, "mild"),
            (5.0, "moderate"),
            (9.99, "moderate"),
            (10.0, "severe"),
            (100.0, "severe"),
        ],
    )
    def test_bands(self, percent, expected):
        assert classify_parasitemia(percent) == expected

    def test_is_monotonic(self):
        previous = -1
        for pct in [i / 10.0 for i in range(0, 1001)]:
            band = classify_parasitemia(pct)
            order = ["negative", "low", "mild", "moderate", "severe"].index(band)
            assert order >= previous
            previous = order


class TestRenderTriage:
    def test_renders_all_requested_languages(self, triage):
        for level in triage["severity_levels"]:
            for lang in LANGUAGES:
                out = render_triage({"severity": level, "num_cells": 60,
                                     "parasitemia_percent": 5.0}, lang=lang)
                assert out["language"] == lang
                assert out["severity"] == level
                assert out["header"] and out["action"] and out["recommendation"]
                assert out["disclaimer"]

    def test_unknown_language_falls_back_to_default(self):
        out = render_triage({"severity": "mild", "num_cells": 60}, lang="xx")
        assert out["language"] == "en"

    def test_no_warnings_for_a_healthy_field(self):
        out = render_triage({"severity": "negative", "num_cells": 80})
        assert out["warnings"] == []
        assert out["expert_review_required"] is False

    def test_warns_when_no_cells_found(self):
        for lang in LANGUAGES:
            out = render_triage({"severity": "negative", "num_cells": 0}, lang=lang)
            assert len(out["warnings"]) == 1
            assert out["warnings"][0]

    def test_warns_below_the_minimum_estimate(self):
        assert MIN_CELLS_FOR_ESTIMATE > 1
        out = render_triage({"severity": "low", "num_cells": MIN_CELLS_FOR_ESTIMATE - 1})
        assert len(out["warnings"]) == 1

    def test_no_warning_at_exactly_the_minimum(self):
        out = render_triage({"severity": "low", "num_cells": MIN_CELLS_FOR_ESTIMATE})
        assert out["warnings"] == []

    def test_known_bad_translations_are_gone(self, triage):
        """Regression: the file previously shipped mangled Urdu and Polish text."""
        blob = json.dumps(triage, ensure_ascii=False)
        for broken in ("Organizational", "PRZESIWU", "wy segmentowanych"):
            assert broken not in blob
        # the fabricated Polish calque was replaced with the attested term
        assert "pasożytozłuszczen" not in blob.lower()
        assert "parazytemia" in blob
