from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

def pipeline_event_files() -> list[str]:
    # Keep in sync with PipelineConfig event candidates in ttc_bunching_pipeline/config.py.
    files = [
        "df2025_all.csv",      # primary
        "506-2024-1.csv",      # extras
        "df_2024-2",
        "df_2026",
        "all506_df-new-1.csv", # fallback pair
        "all506_df-new-2.csv",
    ]

    # Deduplicate while keeping order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def split_csv_file(src: Path, chunk_dir: Path, max_bytes: int) -> list[Path]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out_files: list[Path] = []

    with src.open("rb") as rf:
        header = rf.readline()
        if not header:
            raise ValueError(f"File is empty: {src}")

        part_idx = 1
        part_path = chunk_dir / f"{src.name}.part{part_idx:04d}.csv"
        wf = part_path.open("wb")
        out_files.append(part_path)
        wf.write(header)
        part_size = len(header)

        try:
            for line in rf:
                if part_size + len(line) > max_bytes and part_size > len(header):
                    wf.close()
                    part_idx += 1
                    part_path = chunk_dir / f"{src.name}.part{part_idx:04d}.csv"
                    wf = part_path.open("wb")
                    out_files.append(part_path)
                    wf.write(header)
                    part_size = len(header)

                wf.write(line)
                part_size += len(line)
        finally:
            wf.close()

    return out_files


def assemble_csv(parts: Iterable[Path], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    parts = list(parts)
    if len(parts) == 0:
        raise ValueError(f"No parts provided for {output_file.name}")

    with output_file.open("wb") as wf:
        for i, part in enumerate(parts):
            with part.open("rb") as rf:
                if i > 0:
                    # Skip repeated header line from subsequent parts.
                    rf.readline()
                while True:
                    b = rf.read(1024 * 1024)
                    if not b:
                        break
                    wf.write(b)


def run_split(args: argparse.Namespace) -> int:
    source_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    chunk_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(float(args.max_mb) * 1024 * 1024)

    files = args.files or pipeline_event_files()
    manifest: dict[str, object] = {
        "source_dir": str(source_dir),
        "max_chunk_mb": float(args.max_mb),
        "files": [],
    }

    for name in files:
        src = source_dir / name
        if not src.exists():
            print(f"[skip] missing source file: {src}")
            continue

        print(f"[split] {src.name}")
        parts = split_csv_file(src=src, chunk_dir=chunk_dir, max_bytes=max_bytes)
        part_rels = [str(p.relative_to(out_dir)) for p in parts]
        part_sizes_mb = [round(p.stat().st_size / (1024 * 1024), 3) for p in parts]
        manifest["files"].append(
            {
                "name": src.name,
                "source_size_bytes": int(src.stat().st_size),
                "source_sha256": sha256_file(src),
                "parts": part_rels,
                "part_sizes_mb": part_sizes_mb,
            }
        )
        print(
            f"  -> {len(parts)} parts, largest={max(part_sizes_mb):.3f} MB"
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] wrote manifest: {manifest_path}")
    return 0


def run_assemble(args: argparse.Namespace) -> int:
    in_dir = Path(args.in_dir).resolve()
    manifest_path = in_dir / "manifest.json"
    out_dir = Path(args.out_dir).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    if not isinstance(files, list):
        raise ValueError("Invalid manifest format: 'files' must be a list")

    for fmeta in files:
        name = str(fmeta["name"])
        parts = [in_dir / str(p) for p in fmeta["parts"]]
        missing = [str(p) for p in parts if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing parts for {name}: {missing}")

        dst = out_dir / name
        if dst.exists() and not args.force:
            print(f"[skip] exists (use --force to overwrite): {dst}")
            continue

        print(f"[assemble] {name}")
        assemble_csv(parts=parts, output_file=dst)

        source_hash = str(fmeta.get("source_sha256", "")).strip()
        if source_hash:
            out_hash = sha256_file(dst)
            if out_hash == source_hash:
                print("  -> sha256 OK")
            else:
                print("  -> sha256 MISMATCH (content may still be CSV-equivalent)")

    print("[ok] assembly complete")
    print("Primary + extras used by pipeline:")
    files = pipeline_event_files()
    print(f"  primary: {files[0] if files else '(none)'}")
    print(f"  extras: {files[1:4] if len(files) >= 4 else files[1:]}")
    print(f"  fallback: {files[-2:] if len(files) >= 2 else files}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Split large pipeline CSVs into GitHub-friendly chunks and "
            "reassemble them later."
        )
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("split", help="Split source files into chunked CSV parts")
    ps.add_argument("--source-dir", required=True, help="Directory containing source CSVs")
    ps.add_argument("--out-dir", default="data_files", help="Output directory for chunks + manifest")
    ps.add_argument("--max-mb", type=float, default=45.0, help="Max size per chunk in MB")
    ps.add_argument(
        "--files",
        nargs="*",
        default=None,
        help=(
            "Optional explicit file list (relative to source-dir). "
            "Default uses pipeline primary/extras/fallback files."
        ),
    )
    ps.set_defaults(func=run_split)

    pa = sub.add_parser("assemble", help="Reassemble chunked CSV parts back to original files")
    pa.add_argument("--in-dir", default="data_files", help="Directory containing manifest + chunk files")
    pa.add_argument("--out-dir", default=".", help="Output directory for reconstructed files")
    pa.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    pa.set_defaults(func=run_assemble)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
