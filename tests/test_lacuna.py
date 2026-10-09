"""Lacuna malaria dataset ingestion tests.

Every assertion here is built on facts verified against the published
Harvard Dataverse release (doi:10.7910/DVN/VEADSE):

* The authoritative annotation is ``Labels-CSV.csv`` with columns
  ``Image_name, xmin, ymin, width, height, Class`` in absolute pixels.
* Class names in the release: ``Parasitized cell``, ``Trophozoite``,
  ``Gametocyte``, ``Artifact``, ``WBC``.
* The CSVs contain a large number of *exact duplicate rows*, which must be
  de-duplicated or every class is silently doubled.
* A parallel ``Labels-YOLO`` tree exists but the release ships no
  ``classes.txt``, and its box geometry does not match the CSV, so the YOLO
  tree must require an explicit class mapping.

The fixtures below synthesise a miniature archive with those exact properties so
the tests never touch the network or the 7 GB download.
"""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest

from src.parse_lacuna import (
    Box,
    ParseConfig,
    background_tiles,
    build_class_map,
    discover_archives,
    parse_dataset,
    read_labels_csv,
    read_labels_yolo,
    square_crop,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fake_lacuna(tmp_path: Path) -> Path:
    """A tiny archive shaped exactly like the real release."""
    root = tmp_path / "lacuna"
    (root / "images").mkdir(parents=True)

    rows = [
        # (image, xmin, ymin, width, height, Class)
        ("f1.jpg", 100, 100, 60, 60, "Parasitized cell"),
        ("f1.jpg", 400, 200, 50, 50, "Parasitized cell"),
        ("f1.jpg", 800, 600, 70, 70, "Artifact"),
        ("f1.jpg", 850, 610, 30, 30, "WBC"),
        ("f2.jpg", 300, 300, 90, 90, "Trophozoite"),
        ("f2.jpg", 120, 480, 40, 40, "Gametocyte"),
        ("f2.jpg", 700, 80, 200, 200, "Artifact"),
        # an out-of-frame box (as the real annotations contain)
        ("f2.jpg", 0, 0, 4, 4, "Parasitized cell"),
    ]
    # exact duplicates, exactly like the published CSV
    rows += rows + rows

    with (root / "Labels-CSV.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Image_name", "xmin", "ymin", "width", "height", "Class"])
        for r in rows:
            writer.writerow(r)

    rng = np.random.default_rng(7)
    for name in ("f1.jpg", "f2.jpg"):
        img = (rng.random((1000, 1000, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(root / "images" / name), img)

    # a YOLO tree with no classes.txt, as shipped
    (root / "Labels-YOLO").mkdir()
    (root / "Labels-YOLO" / "f1.txt").write_text("0 0.1 0.1 0.05 0.05\n3 0.4 0.4 0.06 0.06\n")
    return root


# --------------------------------------------------------------------------- #
# CSV reading / de-duplication
# --------------------------------------------------------------------------- #
class TestLabelsCsv:
    def test_drops_duplicate_rows(self, fake_lacuna):
        rows = read_labels_csv(fake_lacuna / "Labels-CSV.csv")
        assert len(rows) == 8, "exact duplicate rows must be removed"

    def test_parses_absolute_pixel_coords(self, fake_lacuna):
        rows = read_labels_csv(fake_lacuna / "Labels-CSV.csv")
        first = rows[0]
        assert first == {"image": "f1.jpg", "x": 100, "y": 100, "w": 60, "h": 60,
                         "class": "Parasitized cell"}

    def test_class_names_preserved_verbatim(self, fake_lacuna):
        rows = read_labels_csv(fake_lacuna / "Labels-CSV.csv")
        assert {r["class"] for r in rows} == {
            "Parasitized cell", "Artifact", "WBC", "Trophozoite", "Gametocyte"
        }

    def test_read_labels_yolo_needs_class_names(self, fake_lacuna):
        rows = read_labels_yolo(fake_lacuna / "Labels-YOLO" / "f1.txt",
                                ["a", "b", "c", "d"])
        assert len(rows) == 2
        assert rows[0]["class"] == "a"
        assert rows[1]["class"] == "d"
        # normalised coords are preserved for the caller to scale
        assert 0.0 < rows[0]["cx"] < 1.0


# --------------------------------------------------------------------------- #
# Label mapping
# --------------------------------------------------------------------------- #
class TestClassMapping:
    def test_parasite_stages_are_positive(self):
        m = build_class_map(("Parasitized cell", "Trophozoite", "Gametocyte"),
                            ("Artifact", "WBC"))
        assert m["parasitizedcell"] == 1
        assert m["trophozoite"] == 1
        assert m["gametocyte"] == 1

    def test_artifacts_and_wbc_are_hard_negatives(self):
        m = build_class_map(("Parasitized cell",), ("Artifact", "WBC"))
        assert m["artifact"] == 0
        assert m["wbc"] == 0

    def test_whitespace_and_case_are_normalised(self):
        m = build_class_map(("Parasitized Cell",), (" White  Blood Cell ",))
        assert "parasitizedcell" in m
        assert "whitebloodcell" in m


# --------------------------------------------------------------------------- #
# Crop geometry
# --------------------------------------------------------------------------- #
class TestSquareCrop:
    def image(self):
        rng = np.random.default_rng(0)
        return (rng.random((400, 400, 3)) * 255).astype(np.uint8)

    def test_target_shape(self):
        out = square_crop(self.image(), Box(100, 100, 40, 30), 128, 0.18)
        assert out is not None and out.shape == (128, 128, 3)

    def test_box_hanging_off_the_topleft_corner(self):
        # Regression: a negative window index used to be used as a NumPy
        # end-relative index and read the wrong part of the frame.
        out = square_crop(self.image(), Box(0, 0, 30, 30), 128, 0.18)
        assert out is not None and out.shape == (128, 128, 3)

    def test_box_at_the_bottom_right_corner(self):
        out = square_crop(self.image(), Box(370, 370, 30, 30), 128, 0.18)
        assert out is not None and out.shape == (128, 128, 3)

    def test_fully_out_of_frame_returns_none(self):
        assert square_crop(self.image(), Box(500, 500, 30, 30), 128) is None

    def test_degenerate_box_returns_none(self):
        assert square_crop(self.image(), Box(10, 10, 0, 0), 128) is None

    def test_huge_box_is_clamped_to_the_frame(self):
        out = square_crop(self.image(), Box(0, 0, 400, 400), 128, 0.5)
        assert out is not None and out.shape == (128, 128, 3)


# --------------------------------------------------------------------------- #
# Background negatives
# --------------------------------------------------------------------------- #
class TestBackgroundTiles:
    def test_tiles_avoid_annotated_boxes(self):
        rng = np.random.default_rng(0)
        img = (rng.random((500, 500, 3)) * 255).astype(np.uint8)
        occupied = [Box(200, 200, 100, 100, "Parasitized cell")]
        tiles = background_tiles(img, 500, 500, 64, occupied, 5, __import__("random").Random(3))
        assert len(tiles) == 5
        for t in tiles:
            for o in occupied:
                inter_w = min(t.x + 64, o.x + o.w) - max(t.x, o.x)
                inter_h = min(t.y + 64, o.y + o.h) - max(t.y, o.y)
                assert max(0, inter_w) * max(0, inter_h) <= 0.5 * 64 * 64

    def test_tiny_image_yields_nothing(self):
        img = (np.ones((10, 10, 3)) * 200).astype(np.uint8)
        import random
        assert background_tiles(img, 10, 10, 64, [], 3, random.Random(1)) == []


# --------------------------------------------------------------------------- #
# End-to-end parse
# --------------------------------------------------------------------------- #
class TestParseDataset:
    def _cfg(self, fake_lacuna, out, **kw):
        base = dict(src=fake_lacuna, out=out, crop_size=64,
                    negatives_from_background=0.0, dry_run=False)
        base.update(kw)
        return ParseConfig(**base)

    def test_writes_imagefolder_layout(self, fake_lacuna, tmp_path):
        out = tmp_path / "crops"
        manifest = parse_dataset(self._cfg(fake_lacuna, out))
        assert (out / "Parasitized").is_dir()
        assert (out / "Uninfected").is_dir()
        assert manifest["num_crops"] > 0
        assert manifest["num_fields"] == 2

    def test_label_assignment(self, fake_lacuna, tmp_path):
        manifest = parse_dataset(self._cfg(fake_lacuna, tmp_path / "c"))
        by_src = {}
        for r in manifest["images"]:
            by_src.setdefault(r["source_class"], set()).add(r["label"])
        assert by_src["Parasitized cell"] == {1}
        assert by_src["Trophozoite"] == {1}
        assert by_src["Gametocyte"] == {1}
        assert by_src["Artifact"] == {0}
        assert by_src["WBC"] == {0}

    def test_crops_are_the_configured_size(self, fake_lacuna, tmp_path):
        out = tmp_path / "c"
        parse_dataset(self._cfg(fake_lacuna, out, crop_size=64))
        paths = sorted(out.rglob("*.png"))
        assert paths
        for p in paths[:4]:
            assert cv2.imread(str(p)).shape[:2] == (64, 64)

    def test_patient_id_is_the_field_of_view(self, fake_lacuna, tmp_path):
        manifest = parse_dataset(self._cfg(fake_lacuna, tmp_path / "c"))
        ids = {r["patient_id"] for r in manifest["images"]}
        assert ids == {"f1", "f2"}

    def test_no_patient_leakage_is_possible(self, fake_lacuna, tmp_path):
        """Splitting by field must isolate the two captures."""
        from src.dataset import patient_level_split

        manifest = parse_dataset(self._cfg(fake_lacuna, tmp_path / "c"))
        records = manifest["images"]
        train, val = patient_level_split(records, val_fraction=0.5)
        assert {r["patient_id"] for r in train} & {r["patient_id"] for r in val} == set()

    def test_background_negatives_added(self, fake_lacuna, tmp_path):
        manifest = parse_dataset(self._cfg(fake_lacuna, tmp_path / "c",
                                           negatives_from_background=0.5))
        assert manifest["source_class_counts"]["background"] > 0
        assert manifest["label_counts"]["0"] > 0

    def test_manifest_records_provenance(self, fake_lacuna, tmp_path):
        out = tmp_path / "c"
        parse_dataset(self._cfg(fake_lacuna, out))
        m = json.loads((out / "manifest.json").read_text())
        assert m["license"] == "CC BY 4.0"
        assert "Lacuna" in m["source"]
        assert m["dataset"] == "lacuna_phone"
        assert m["positive_classes"] == ["Parasitized cell", "Trophozoite",
                                         "Gametocyte"]
        assert m["negative_classes"] == ["Artifact", "WBC"]

    def test_custom_positive_set(self, fake_lacuna, tmp_path):
        manifest = parse_dataset(self._cfg(fake_lacuna, tmp_path / "c",
                                           positive=("Artifact",),
                                           negative=("Parasitized cell",)))
        for r in manifest["images"]:
            expected = 1 if r["source_class"] == "Artifact" else 0
            assert r["label"] == expected

    def test_dry_run_writes_nothing(self, fake_lacuna, tmp_path):
        out = tmp_path / "c"
        parse_dataset(self._cfg(fake_lacuna, out, dry_run=True))
        assert not out.exists()

    def test_missing_source_is_actionable(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            parse_dataset(ParseConfig(src=tmp_path / "nope", out=tmp_path / "o"))
        assert "download_lacuna.py" in str(exc.value)

    def test_discovers_images_and_annotations(self, fake_lacuna):
        images, annotations = discover_archives(fake_lacuna)
        assert len(images) == 2
        assert set(annotations) == {"f1.jpg", "f2.jpg"}
        assert len(annotations["f1.jpg"]) == 4


# --------------------------------------------------------------------------- #
# Downloader (network calls mocked out)
# --------------------------------------------------------------------------- #
class TestDuplicateTreeDetection:
    """The NIH Kaggle archive unzipped twice reports 2x the true image count.

    Two copies of the tree are *distinct files*, so path identity does not catch
    them; a crop is identified by its filename. Regression test for a manifest
    that silently reported 55,116 images instead of 27,558.
    """

    def _tree(self, root: Path, copies: int = 1) -> None:
        import cv2

        rng = np.random.default_rng(0)
        for cls in ("Parasitized", "Uninfected"):
            for slide in range(3):
                for cell in range(4):
                    d = root / cls
                    d.mkdir(parents=True, exist_ok=True)
                    # real NIH filenames embed a unique field id, so they never
                    # collide across the two class folders
                    name = f"F{slide:03d}_{cls[:4]}_IMG_1_cell_{cell:02d}.png"
                    img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
                    cv2.imwrite(str(d / name), img)
        # plant `copies - 1` nested duplicates of the whole tree
        for _ in range(copies - 1):
            nested = root / "cell_images"
            nested.mkdir(parents=True, exist_ok=True)
            for p in root.rglob("*.png"):
                if nested in p.parents:
                    continue
                dst = nested / p.relative_to(root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(p.read_bytes())

    def test_clean_tree_is_unchanged(self, tmp_path):
        from src.prepare_data import _dedupe_records

        src = tmp_path / "cell_images"
        self._tree(src, copies=1)
        recs = [{"path": str(p)} for p in src.rglob("*.png")]
        unique, dropped = _dedupe_records(recs)
        assert dropped == 0
        assert len(unique) == len(recs)

    def test_nested_duplicate_is_collapsed(self, tmp_path):
        from src.prepare_data import _dedupe_records

        src = tmp_path / "cell_images"
        self._tree(src, copies=2)
        recs = [{"path": str(p)} for p in src.rglob("*.png")]
        assert len(recs) == 48  # 24 crops, two copies each
        unique, dropped = _dedupe_records(recs)
        assert dropped == 24
        assert len(unique) == 24

    def test_collect_records_warns_on_a_doubled_tree(self, tmp_path):
        """End-to-end: the manifest must not double the dataset."""
        from src.prepare_data import SOURCES, collect_records, _dedupe_records

        src = tmp_path / "cell_images"
        self._tree(src, copies=2)
        info = SOURCES["nih_full"]
        recs = collect_records(src, info.domain, info.source_name, info.license)
        unique, dropped = _dedupe_records(recs)
        assert dropped == 24, "the nested tree was not collapsed"
        assert len(unique) == 24


class TestPrepareDataPaths:
    def test_registry_path_is_the_write_target(self, monkeypatch):
        """--dataset nih_full must write where src/train.py looks."""
        from src.prepare_data import _registry_manifest_path

        path = _registry_manifest_path("nih_full")
        assert path is not None
        assert str(path).endswith("data/nih/manifest.json"), (
            "prepare_data wrote data/nih_full/ but train.py reads data/nih/ - "
            "exactly the mismatch that broke Stage 1"
        )

    def test_lacuna_registry_path(self):
        from src.prepare_data import _registry_manifest_path

        path = _registry_manifest_path("lacuna_phone")
        assert path is not None
        assert str(path).endswith("lacuna_crops/manifest.json")

    def test_unknown_key_returns_none(self):
        from src.prepare_data import _registry_manifest_path

        assert _registry_manifest_path("not_a_dataset") is None


class TestDownloaderUnits:
    def test_part_index_parsing(self):
        from src.download_lacuna import DataFile

        assert DataFile(1, "Thick_Ghana.part1.rar", 1).part_index == 1
        assert DataFile(2, "Thin_Uganda.rar", 1).part_index is None

    def test_part_index_zero_padded(self):
        from src.download_lacuna import DataFile

        assert DataFile(1, "x.part07.rar", 1).part_index == 7

    def test_resolve_by_name(self):
        from src.download_lacuna import DataFile, resolve_files

        files = [DataFile(10428455, "Thin_Uganda.rar", 10),
                 DataFile(7225281, "DS.pdf", 5)]
        assert resolve_files(files, ["Thin_Uganda.rar"])[0].name == "Thin_Uganda.rar"

    def test_resolve_by_id(self):
        from src.download_lacuna import DataFile, resolve_files

        files = [DataFile(10428455, "Thin_Uganda.rar", 10)]
        assert resolve_files(files, ["10428455"])[0].id == 10428455

    def test_resolve_by_glob(self):
        from src.download_lacuna import DataFile, resolve_files

        files = [DataFile(i, f"Thick_Ghana.part{i}.rar", 1) for i in (1, 2, 3)]
        picked = resolve_files(files, ["Thick_Ghana.part?.rar"])
        assert [f.name for f in picked] == [
            "Thick_Ghana.part1.rar", "Thick_Ghana.part2.rar",
            "Thick_Ghana.part3.rar"]

    def test_unknown_name_raises_with_available_list(self):
        from src.download_lacuna import DataFile, resolve_files

        with pytest.raises(KeyError) as exc:
            resolve_files([DataFile(1, "a.rar", 1)], ["nope.rar"])
        assert "a.rar" in str(exc.value)

    def test_deduplicates(self):
        from src.download_lacuna import DataFile, resolve_files

        files = [DataFile(1, "a.rar", 1)]
        assert len(resolve_files(files, ["a.rar", "a.rar"])) == 1

    def test_find_unrar_reports_none_when_missing(self, monkeypatch):
        import src.download_lacuna as dl

        monkeypatch.setattr(dl.shutil, "which", lambda _name: None)
        assert dl.find_unrar() is None

    def test_extract_without_backend_gives_instructions(self, tmp_path, monkeypatch):
        """The 'no extractor' message must not depend on what is installed.

        This used to only pass on machines with no 7z/bsdtar/unrar on PATH:
        where one exists, find_unrar() returns it and the code fell through to
        rarfile, raising a bare FileNotFoundError instead of the instructions.
        Force the missing-backend branch so the test is hermetic.
        """
        from src.download_lacuna import extract_rar

        monkeypatch.setattr(
            "src.download_lacuna.find_unrar", lambda explicit=None: None
        )
        archive = tmp_path / "present.rar"
        archive.write_bytes(b"Rar!\x1a\x07\x00" + b"\x00" * 64)  # valid RAR4 magic
        with pytest.raises(RuntimeError) as exc:
            extract_rar(archive, tmp_path, tool="/nonexistent/unrar")
        msg = str(exc.value)
        assert "apt-get install" in msg
        assert "unrar-free" in msg

    def test_extract_reports_a_missing_archive_clearly(self, tmp_path, monkeypatch):
        """A missing archive must not surface as a raw rarfile traceback."""
        from src.download_lacuna import extract_rar

        # pretend a backend exists so we reach the archive check
        monkeypatch.setattr(
            "src.download_lacuna.find_unrar", lambda explicit=None: "/usr/bin/true"
        )
        with pytest.raises(FileNotFoundError) as exc:
            extract_rar(tmp_path / "nope.rar", tmp_path, tool="/usr/bin/true")
        assert "--files" in str(exc.value)


class TestDatasetRegistry:
    def test_lacuna_registered(self):
        from src.train import DATASETS

        spec = DATASETS["lacuna_phone"]
        assert spec.license == "CC BY 4.0"
        assert spec.domain == "phone"
        assert spec.stage == 2
        assert "lacuna" in spec.manifest.lower()

    def test_manifest_path_points_at_the_parser_output(self):
        from src.train import DATASETS

        spec = DATASETS["lacuna_phone"]
        assert spec.manifest.endswith("lacuna_crops/manifest.json")
        assert spec.folder.endswith("lacuna_crops")
