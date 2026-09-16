#!/usr/bin/env python3
"""Losslessly archive bulky/redundant files from data/generated.

Canonical datasets, merged pass@k rollouts, summaries, audit reports, exported
tasks, and the newest trajectory viewer are deliberately retained in place.
Sources are removed only after the compressed archive has been listed and its
members exactly match the selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


PART_RE = re.compile(r"\.part\d+\.(?:jsonl|log)$")
SHARD_RE = re.compile(r"(?:_fixed|_shard)\d+\.")
SHARD_DIR_RE = re.compile(r"^(?:shard|quarter)\d+$")
LATEST_VIEWER = "refuse_parallel_multiturn10_steps7_15_grok45_20_20260728_v2.html"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_sources(root: Path) -> list[Path]:
    selected: set[Path] = set()

    for path in root.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if PART_RE.search(name) or SHARD_RE.search(name) or "_smoke1_" in name:
            selected.add(path)
        if (
            name.startswith("trajectory_review_")
            and name.endswith(".html")
            and name != LATEST_VIEWER
        ):
            selected.add(path)

    for path in root.glob("pass*/**/*"):
        if not path.is_file():
            continue
        if path.name == "events.jsonl":
            selected.add(path)
        elif path.name == "rollouts.jsonl" and SHARD_DIR_RE.match(path.parent.name):
            selected.add(path)
        elif path.name.startswith("vllm_") and path.suffix == ".log":
            selected.add(path)

    return sorted(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/generated"))
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/generated/archives/generated_intermediates_20260728.tar.zst"),
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    archive = args.archive.resolve()
    sources = select_sources(root)
    relative = [path.relative_to(root).as_posix() for path in sources]
    source_bytes = sum(path.stat().st_size for path in sources)

    preview = {
        "root": str(root),
        "archive": str(archive),
        "selected_files": len(sources),
        "source_bytes": source_bytes,
        "source_mib": round(source_bytes / 1024 / 1024, 2),
        "apply": args.apply,
        "sample": relative[:20],
    }
    print(json.dumps(preview, indent=2))
    if not args.apply:
        return 0
    if not sources:
        raise ValueError("No intermediate files selected")
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {archive}")

    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    completed = subprocess.run(
        [
            "tar",
            "--zstd",
            "-cf",
            str(temporary),
            "--files-from=-",
        ],
        cwd=root,
        input="\n".join(relative) + "\n",
        text=True,
        check=True,
        capture_output=True,
    )
    if completed.stderr:
        print(completed.stderr)

    listing = subprocess.run(
        ["tar", "--zstd", "-tf", str(temporary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if sorted(listing) != sorted(relative):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Archive member verification failed")

    temporary.replace(archive)
    archive_digest = sha256(archive)
    source_records = [
        {
            "path": rel,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path, rel in zip(sources, relative)
    ]
    manifest = {
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_digest,
        "source_root": str(root),
        "source_bytes": source_bytes,
        "source_files": source_records,
        "restore_command": f"tar --zstd -xf {archive} -C {root}",
    }
    manifest_path = archive.with_suffix(archive.suffix + ".manifest.json")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_tmp.replace(manifest_path)

    # The archive membership and manifest hashes are now durable. Removing these
    # exact sources is safe and reversible using restore_command above.
    for path in sources:
        path.unlink()

    print(
        json.dumps(
            {
                "archived_files": len(sources),
                "source_mib": round(source_bytes / 1024 / 1024, 2),
                "archive_mib": round(archive.stat().st_size / 1024 / 1024, 2),
                "manifest": str(manifest_path),
                "restore_command": manifest["restore_command"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
