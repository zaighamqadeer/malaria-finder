"""Data-splitting and dataset-hygiene tests.

The design doc's "crucial rule" is that splits happen at slide/patient level,
because random image splits leak staining and illumination artifacts and
inflate reported sensitivity. These tests pin that behaviour down.
"""

from __future__ import annotations

import pytest

from src.dataset import patient_id_from_stem, patient_level_split


def _records(n_patients: int, per: int) -> list[dict]:
    recs = []
    for p in range(n_patients):
        for i in range(per):
            recs.append(
                {
                    "path": f"/tmp/p{p}_c{i}.png",
                    "label": (p + i) % 2,
                    "patient_id": f"slide_{p:03d}",
                    "domain": "nih",
                }
            )
    return recs


class TestPatientIdDerivation:
    @pytest.mark.parametrize(
        "stem,expected",
        [
            ("slide_000_cell_0000", "slide_000"),
            ("C100P61ThinF_IMG_20150618_171318_cell_2",
             "C100P61ThinF_IMG_20150618_171318"),
            ("cell_0123", "cell_0123"),
            ("IMG_20150618_171320", "IMG_20150618_171320"),
            ("field7_tile12", "field7"),
            ("field7-crop3", "field7"),
        ],
    )
    def test_strips_cell_batch_index(self, stem, expected):
        assert patient_id_from_stem(stem) == expected

    def test_distinct_slides_are_distinct(self):
        # Regression: the previous stem-prefix rule collapsed every sample
        # crop into the single patient id "slide", which emptied the train split.
        ids = {patient_id_from_stem(f"slide_{i:03d}_cell_0000") for i in range(8)}
        assert ids == {f"slide_{i:03d}" for i in range(8)}


class TestPatientLevelSplit:
    def test_no_patient_leakage(self):
        recs = _records(20, 25)
        train, val = patient_level_split(recs, val_fraction=0.2)
        assert {r["patient_id"] for r in train} & {r["patient_id"] for r in val} == set()

    def test_every_image_accounted_for(self):
        recs = _records(12, 10)
        train, val = patient_level_split(recs, val_fraction=0.25)
        assert len(train) + len(val) == len(recs)
        assert len(train) > 0 and len(val) > 0

    def test_exact_patient_separation(self):
        recs = _records(10, 5)
        train, val = patient_level_split(recs, val_fraction=0.3)
        train_p = {r["patient_id"] for r in train}
        val_p = {r["patient_id"] for r in val}
        assert train_p.isdisjoint(val_p)
        assert train_p | val_p == {f"slide_{i:03d}" for i in range(10)}

    def test_deterministic_for_a_fixed_seed(self):
        a = patient_level_split(_records(15, 7), val_fraction=0.2, seed=7)
        b = patient_level_split(_records(15, 7), val_fraction=0.2, seed=7)
        assert a == b

    def test_different_seeds_give_different_splits(self):
        recs = _records(30, 4)
        _, val_a = patient_level_split(recs, val_fraction=0.2, seed=1)
        _, val_b = patient_level_split(recs, val_fraction=0.2, seed=2)
        assert val_a != val_b

    def test_rejects_out_of_range_fraction(self):
        with pytest.raises(ValueError):
            patient_level_split(_records(4, 2), val_fraction=0.0)
        with pytest.raises(ValueError):
            patient_level_split(_records(4, 2), val_fraction=1.0)

    def test_stratified_split_keeps_label_balance(self):
        # 10 positive slides and 10 negative slides -> both classes in val.
        recs = []
        for p in range(20):
            label = 1 if p % 2 == 0 else 0
            for i in range(10):
                recs.append({"path": f"p{p}c{i}", "label": label,
                             "patient_id": f"slide_{p:03d}", "domain": "nih"})
        train, val = patient_level_split(recs, val_fraction=0.2)
        val_labels = {r["label"] for r in val}
        assert val_labels == {0, 1}

    def test_scan_image_folder_groups_by_slide(self, repo_root):
        from src.dataset import scan_image_folder

        recs = scan_image_folder(repo_root / "data" / "sample", patient_from="auto")
        patients = {r["patient_id"] for r in recs}
        assert len(patients) == 8
        assert all(r["patient_id"].startswith("slide_") for r in recs)
        # labels come from the class folder
        assert {r["label"] for r in recs} == {0, 1}

        train, val = patient_level_split(recs, val_fraction=0.3)
        assert len(train) > 0 and len(val) > 0
