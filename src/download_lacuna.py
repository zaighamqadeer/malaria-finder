"""Harvard Dataverse downloader + RAR extractor for the Lacuna Malaria Datasets.

Dataset (verified against the Dataverse API while writing this module):

    Persistent ID : doi:10.7910/DVN/VEADSE
    Title         : Lacuna Malaria Datasets
    Author        : Makerere AI Lab, Makerere University (Uganda)
    License       : CC BY 4.0
    Citation      : Makerere AI Lab. "Lacuna Malaria Datasets", Harvard
                    Dataverse.

Files on the server (as published):

    DATASHEET_FOR_MALARIA_DATASETS-GHANA.pdf      126 KB
    DATASHEET_FOR_MALARIA_DATASETS-UGANDA.pdf     125 KB
    Thick_Ghana.part1.rar                        2.15 GB
    Thick_Ghana.part2.rar                        2.15 GB
    Thick_Ghana.part3.rar                         695 MB
    Thin_Images_Ghana.rar                        1.51 GB
    Thin_Uganda.rar                              730 MB
                                               --------
                                        total  ~7.2 GB

Each RAR contains ``images/<epoch_ms>.jpg`` smartphone captures, a
``Labels-YOLO/<stem>.txt`` file per image, and — the authoritative
annotation — ``Labels-CSV.csv``. See ``src/parse_lacuna.py`` for the format
and for why the CSV is preferred over the YOLO files.

Usage
-----
    python src/download_lacuna.py --list
    python src/download_lacuna.py --files Thin_Uganda.rar
    python src/download_lacuna.py --files Thick_Ghana.part1.rar,Thick_Ghana.part2.rar,Thick_Ghana.part3.rar
    python src/download_lacuna.py --all --out data/raw/lacuna
"""

from __future__ import annotations

# Support BOTH documented entry points: `python src/download_lacuna.py ...`
# and `python -m src.download_lacuna`.
if __package__ in (None, ""):  # pragma: no cover - exercised via the shell
    import os
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"  # noqa: A001

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

PERSISTENT_ID = "doi:10.7910/DVN/VEADSE"
API_BASE = "https://dataverse.harvard.edu/api"
ACCESS_URL = API_BASE + "/access/datafile/{file_id}"
USER_AGENT = "TinyMalariaNet-data-prep/1.0 (+%s)" % PERSISTENT_ID


@dataclass(frozen=True)
class DataFile:
    id: int
    name: str
    size: int
    md5: str | None = None

    @property
    def size_mb(self) -> float:
        return self.size / 1e6

    @property
    def is_rar(self) -> bool:
        return self.name.lower().endswith((".rar", ".r00", ".part1.rar"))

    @property
    def part_index(self) -> int | None:
        """1 for ``Thick_Ghana.part1.rar``, None for a single-part archive."""
        stem = self.name.lower()
        if ".part" in stem and stem.endswith(".rar"):
            digits = "".join(ch for ch in stem.split(".part")[-1] if ch.isdigit())
            return int(digits) if digits else None
        return None


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def _request(url: str, timeout: int = 180, stream: bool = False) -> Any:
    from urllib.request import Request, urlopen

    req = Request(url, headers={"User-Agent": USER_AGENT,
                                "Accept": "application/json"})
    return urlopen(req, timeout=timeout)


def fetch_dataset_metadata(persistent_id: str = PERSISTENT_ID) -> dict[str, Any]:
    """Return the parsed latest-version metadata block from the Dataverse API."""
    url = f"{API_BASE}/datasets/:persistentId/versions/?persistentId={quote(persistent_id, safe='')}"
    with _request(url) as resp:  # noqa: S310 - fixed public API endpoint
        payload = json.loads(resp.read().decode())
    versions = payload.get("data") or []
    if not versions:
        raise RuntimeError(f"No versions returned for {persistent_id}")
    return versions[0]["metadataBlocks"]["citation"]


def list_files(persistent_id: str = PERSISTENT_ID) -> list[DataFile]:
    """Fetch the live file manifest from Dataverse (never a hardcoded list)."""
    url = f"{API_BASE}/datasets/:persistentId/?persistentId={quote(persistent_id, safe='')}"
    with _request(url) as resp:  # noqa: S310
        payload = json.loads(resp.read().decode())
    files: list[DataFile] = []
    for entry in payload["data"]["latestVersion"].get("files", []):
        df = entry["dataFile"]
        files.append(
            DataFile(
                id=int(df["id"]),
                name=str(df.get("filename")),
                size=int(df.get("filesize") or 0),
                md5=df.get("md5"),
            )
        )
    if not files:
        raise RuntimeError("The Dataverse API returned no files.")
    return files


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_file(df: DataFile, dest: Path, chunk: int = 1 << 20,
                  resume: bool = True) -> Path:
    """Stream one Dataverse file to ``dest``; resumable and size-verified."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    mode, have = "wb", 0
    if resume and dest.exists():
        have = dest.stat().st_size
        if have == df.size and df.size:
            print(f"[lacuna] {df.name} already complete ({df.size_mb:,.1f} MB)")
            return dest
        mode = "ab"  # append / resume

    url = ACCESS_URL.format(file_id=df.id)
    with _request(url) as resp, dest.open(mode) as fh:
        total = int(resp.headers.get("content-length") or df.size)
        if mode == "ab" and resp.status == 200:
            # A 200 (not 206) means the server ignored the range request, so the
            # partial file must be discarded or the archive will be corrupt.
            fh.truncate(0)
            have = 0
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            have += len(block)
            if total:
                done = 100.0 * have / total
                sys.stdout.write(f"\r[lacuna] {df.name}: {have/1e6:,.1f}/"
                                 f"{total/1e6:,.1f} MB ({done:5.1f}%)")
                sys.stdout.flush()
    sys.stdout.write("\n")

    got = dest.stat().st_size
    if df.size and got != df.size:
        raise IOError(f"size mismatch for {df.name}: {got} != {df.size} "
                      "delete the partial file and retry")
    return dest


def resolve_files(available: list[DataFile], patterns: Iterable[str]) -> list[DataFile]:
    """Map comma-separated names/exact ids/globs onto manifest entries."""
    wanted = [p.strip() for p in ",".join(patterns).split(",") if p.strip()]
    by_name = {f.name.lower(): f for f in available}
    picked: list[DataFile] = []
    for pat in wanted:
        if pat.isdigit() and int(pat) in {f.id for f in available}:
            picked.append(next(f for f in available if f.id == int(pat)))
            continue
        f = by_name.get(pat.lower())
        if f is None:
            # glob on the name
            import fnmatch

            matches = [f for f in available if fnmatch.fnmatch(f.name.lower(), pat.lower())]
            if not matches:
                raise KeyError(f"no file matching '{pat}'. "
                               f"Available: {sorted(by_name)}")
            picked.extend(matches)
        else:
            picked.append(f)
    # de-duplicate, keep manifest order
    seen: set[int] = set()
    out: list[DataFile] = []
    for f in picked:
        if f.id not in seen:
            seen.add(f.id)
            out.append(f)
    return out


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def find_unrar(explicit: str | None = None) -> str | None:
    """Locate a usable RAR backend.

    ``rarfile`` (the Python library) can *read* RAR archives but delegates
    extraction to an external binary, so one of these must exist. We check the
    usual suspects plus a few bundled locations.
    """
    candidates = [explicit] if explicit else []
    candidates += [
        os.environ.get("UNRAR_TOOL", ""),
        "unrar",
        "unrar-free",
        "unar",
        "7z", "7zz", "7za",
        "bsdtar",
        "/usr/local/bin/unrar",
        "/opt/unrar/unrar",
    ]
    for cand in candidates:
        if cand and shutil.which(cand):
            return shutil.which(cand)
    return None


def extract_rar(archive: Path, dest: Path, tool: str | None = None) -> Path:
    """Extract a single or multi-part RAR archive into ``dest``.

    ``archive`` must be the FIRST part for multi-part archives
    (``Thick_Ghana.part1.rar``); the remaining parts are picked up automatically
    because they share the same directory.
    """
    archive = Path(archive)
    if not archive.exists():
        raise FileNotFoundError(
            f"Archive not found: {archive}\n"
            f"Pass --files <name> to download it first, or --local-dir / "
            f"--extract-only if it lives elsewhere."
        )

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    backend = find_unrar(tool)
    if backend is None:
        raise RuntimeError(
            "No RAR extractor found. `rarfile` needs an external backend:\n"
            "  Ubuntu/Debian : sudo apt-get install -y unrar-free   # or: unrar\n"
            "  macOS         : brew install unrar\n"
            "  generic       : download UnRAR from "
            "https://www.rarlab.com/rar/unrarsrc-7.1.10.tar.gz\n"
            "Then re-run with --unrar /path/to/unrar"
        )

    import rarfile

    rarfile.UNRAR_TOOL = backend
    try:
        with rarfile.RarFile(str(archive)) as rf:
            names = rf.namelist()
            print(f"[lacuna] {archive.name}: {len(names)} entries, "
                  f"{sum(1 for n in names if n.startswith('images/'))} images")
            rf.extractall(str(dest))
    except rarfile.Error as exc:
        raise RuntimeError(
            f"Failed to extract {archive} with {backend}. If this is a "
            f"multi-part archive, make sure every part is present in "
            f"{archive.parent} and that you passed .part1.rar. ({exc})"
        ) from exc
    print(f"[lacuna] extracted -> {dest}")
    return dest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download & unpack the Lacuna Malaria Datasets "
                    "(Harvard Dataverse, CC BY 4.0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--list", action="store_true",
                   help="show the live file manifest and exit")
    p.add_argument("--files", default=None,
                   help="comma-separated file names / ids / globs (e.g. "
                        "'Thin_Uganda.rar' or 'Thick_Ghana.part?.rar')")
    p.add_argument("--all", action="store_true",
                   help="download every archive in the dataset (~7.2 GB)")
    p.add_argument("--datasheets", action="store_true",
                   help="also fetch the Ghana/Uganda datasheet PDFs")
    p.add_argument("--out", default="data/raw/lacuna",
                   help="download directory")
    p.add_argument("--extract-to", default="data/lacuna",
                   help="extraction target (relative to the repo root)")
    p.add_argument("--extract", action="store_true",
                   help="extract archives after downloading")
    p.add_argument("--extract-only", action="store_true",
                   help="skip downloading, extract what is already present")
    p.add_argument("--unrar", default=None,
                   help="path to an unrar/unrar-free/7z/bsdtar binary")
    p.add_argument("--refresh", action="store_true",
                   help="re-download files that already exist locally")
    return p


def run(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parent.parent

    print(f"[lacuna] querying {API_BASE} for {PERSISTENT_ID}")
    files = list_files()
    total = sum(f.size for f in files)
    print(f"[lacuna] {len(files)} files, {total/1e9:.2f} GB total")

    if args.list:
        for f in files:
            print(f"  {f.id:>10d}  {f.name:48s} {f.size:>15,}")
        return 0

    if not (args.files or args.all or args.extract_only):
        print("[lacuna] nothing to do. Use --files '<name>' , --all or --extract-only.")
        return 1

    if args.files or args.all:
        wanted = resolve_files(files, args.files.split(",") if args.files
                               else [f.name for f in files]
                               if args.all else [])
        if not args.all and not args.datasheets:
            wanted = [f for f in wanted if not f.name.endswith(".pdf")]
        if not wanted:
            print("[lacuna] no matching files")
            return 1
        total_mb = sum(f.size for f in wanted) / 1e6
        print(f"[lacuna] will fetch {len(wanted)} file(s) ({total_mb:,.1f} MB): "
              + ", ".join(f.name for f in wanted))

        download_dir = (repo_root / args.out) if not Path(args.out).is_absolute() \
            else Path(args.out)
        for f in wanted:
            dest = download_dir / f.name
            if dest.exists() and not args.refresh:
                if f.size and dest.stat().st_size == f.size:
                    print(f"[lacuna] {f.name} present, skipping (use --refresh)")
                    continue
            download_file(f, dest, resume=not args.refresh)

    if not (args.extract or args.extract_only):
        return 0

    # ---- extraction ---------------------------------------------------- #
    source_dir = (repo_root / args.out) if not Path(args.out).is_absolute() \
        else Path(args.out)
    extract_dir = (repo_root / args.extract_to) if not Path(args.extract_to).is_absolute() \
        else Path(args.extract_to)

    archives = {p.name.lower(): p for p in source_dir.glob("*.rar")}
    if not archives:
        print(f"[lacuna] no .rar files under {source_dir}; nothing to extract")
        return 1

    # one archive per distinct volume: prefer .part1.rar over later parts
    groups: dict[str, list[Path]] = {}
    for name, path in archives.items():
        base = name.split(".part")[0] if ".part" in name else name
        groups.setdefault(base, []).append(path)

    for base, parts in sorted(groups.items()):
        parts.sort(key=lambda p: (p.name.lower().split(".part")[-1] if ".part"
                                  in p.name.lower() else ""))
        first = parts[0]
        missing = ""
        if first.name.lower().endswith(".part1.rar") is False and len(parts) > 1:
            missing = f" (expected .part1.rar, found {len(parts)} parts)"
        try:
            extract_rar(first, extract_dir, tool=args.unrar)
        except RuntimeError as exc:
            print(f"[lacuna] {exc}{missing}")
            return 3
    print(f"[lacuna] done. Next step:")
    print(f"  python src/parse_lacuna.py --src {extract_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
