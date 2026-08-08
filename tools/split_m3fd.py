"""Create a deterministic 3,780/420 M3FD train/test split."""

import argparse
import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def index_files(directory: Path, extensions):
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    indexed = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in indexed:
            raise ValueError(
                f"Duplicate stem {path.stem!r} in {directory}: "
                f"{indexed[path.stem].name}, {path.name}"
            )
        indexed[path.stem] = path
    return indexed


def preview(values, limit=5):
    values = sorted(values)
    return ", ".join(values[:limit]) + (" ..." if len(values) > limit else "")


def validate_matching_stems(infrared, visible, annotations):
    all_stems = set(infrared) | set(visible) | set(annotations)
    common = set(infrared) & set(visible) & set(annotations)
    if all_stems != common:
        problems = []
        for label, indexed in (
            ("Ir", infrared),
            ("Vis", visible),
            ("Annotation", annotations),
        ):
            missing = all_stems - set(indexed)
            if missing:
                problems.append(f"missing from {label}: {preview(missing)}")
        raise ValueError("M3FD stems do not match; " + "; ".join(problems))
    return common


def validate_xml_files(annotations, strict_filename=False):
    filename_mismatches = []
    for stem, path in sorted(annotations.items()):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"Invalid Pascal VOC XML: {path}: {exc}") from exc
        if root.tag != "annotation":
            raise ValueError(f"Unexpected XML root in {path}: {root.tag!r}")
        xml_filename = root.findtext("filename")
        if xml_filename and Path(xml_filename).stem != stem:
            mismatch = (path, xml_filename, stem)
            if strict_filename:
                raise ValueError(
                    f"Annotation filename mismatch in {path}: "
                    f"XML names {xml_filename!r}, expected stem {stem!r}"
                )
            filename_mismatches.append(mismatch)
    if filename_mismatches:
        examples = "; ".join(
            f"{path.name}: {xml_filename!r} -> {stem!r}"
            for path, xml_filename, stem in filename_mismatches[:3]
        )
        print(
            "warning: ignored legacy <filename> mismatches in "
            f"{len(filename_mismatches)} XML files; external Ir/Vis/Annotation stems "
            f"remain authoritative. Examples: {examples}"
        )
    return len(filename_mismatches)


def split_key(stem: str, seed: int):
    return hashlib.sha256(f"{seed}:{stem}".encode("utf-8")).digest(), stem


def validate_existing_destination(output_root: Path, assignments, sources):
    reused = 0
    for split, split_stems in assignments.items():
        for folder, indexed in sources.items():
            directory = output_root / split / folder
            if not directory.exists():
                continue
            if not directory.is_dir():
                raise FileExistsError(f"Expected a destination directory: {directory}")
            expected_names = {indexed[stem].name for stem in split_stems}
            existing = {path.name: path for path in directory.iterdir()}
            unexpected = set(existing) - expected_names
            if unexpected:
                raise FileExistsError(
                    f"Destination contains files outside the requested fixed split: "
                    f"{directory}: {preview(unexpected)}"
                )
            for stem in split_stems:
                source = indexed[stem]
                destination = directory / source.name
                if not destination.exists():
                    continue
                if not destination.is_file():
                    raise FileExistsError(f"Expected a destination file: {destination}")
                if destination.stat().st_size != source.stat().st_size:
                    raise FileExistsError(
                        f"Existing destination size differs from source: {destination}"
                    )
                reused += 1
    return reused


def transfer(source: Path, destination: Path, mode: str):
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        os.link(source, destination)


def atomic_write_text(path: Path, content: str):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    os.replace(temporary, path)


def write_lines(path: Path, values):
    atomic_write_text(path, "".join(f"{value}\n" for value in values))


def split_fingerprint(train_stems, test_stems, seed):
    payload = (
        f"seed={seed}\n[train]\n"
        + "\n".join(train_stems)
        + "\n[test]\n"
        + "\n".join(test_stems)
        + "\n"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split M3FD/{Ir,Vis,Annotation} into deterministic train/test folders. "
            "The original folders are preserved."
        )
    )
    parser.add_argument("--root", required=True, help="M3FD root containing Ir/, Vis/, Annotation/.")
    parser.add_argument(
        "--output-root",
        help="Destination root. Default: --root (creates M3FD/train and M3FD/test).",
    )
    parser.add_argument("--train-count", type=int, default=3780)
    parser.add_argument("--test-count", type=int, default=420)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="copy is safest; hardlink saves disk space but requires the same filesystem.",
    )
    parser.add_argument(
        "--skip-xml-validation",
        action="store_true",
        help="Skip Pascal VOC XML parsing. Stem matching is still enforced.",
    )
    parser.add_argument(
        "--strict-xml-filename",
        action="store_true",
        help=(
            "Require the legacy XML <filename> field to match the external XML stem. "
            "Disabled by default because official M3FD annotations retain old names."
        ),
    )
    args = parser.parse_args()

    if args.train_count < 1 or args.test_count < 1:
        parser.error("--train-count and --test-count must both be positive")
    if args.skip_xml_validation and args.strict_xml_filename:
        parser.error("--skip-xml-validation and --strict-xml-filename cannot be combined")

    root = Path(args.root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else root
    infrared = index_files(root / "Ir", IMAGE_EXTENSIONS)
    visible = index_files(root / "Vis", IMAGE_EXTENSIONS)
    annotations = index_files(root / "Annotation", (".xml",))
    stems = validate_matching_stems(infrared, visible, annotations)
    expected = args.train_count + args.test_count
    if len(stems) != expected:
        raise ValueError(
            f"Expected exactly {expected} matched M3FD pairs for "
            f"{args.train_count}/{args.test_count}, found {len(stems)}."
        )
    if not args.skip_xml_validation:
        validate_xml_files(annotations, strict_filename=args.strict_xml_filename)
    ordered = sorted(stems, key=lambda value: split_key(value, args.seed))
    train_stems = ordered[: args.train_count]
    test_stems = ordered[args.train_count :]
    assignments = {"train": train_stems, "test": test_stems}

    for split in assignments:
        for folder in ("Ir", "Vis", "Annotation"):
            (output_root / split / folder).mkdir(parents=True, exist_ok=True)
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    sources = {"Ir": infrared, "Vis": visible, "Annotation": annotations}
    reused = validate_existing_destination(output_root, assignments, sources)
    transferred = 0
    for split, split_stems in assignments.items():
        for stem in split_stems:
            for folder, indexed in sources.items():
                source = indexed[stem]
                destination = output_root / split / folder / source.name
                if destination.exists():
                    continue
                transfer(source, destination, args.mode)
                transferred += 1

    write_lines(meta_dir / "train.txt", train_stems)
    write_lines(meta_dir / "test.txt", test_stems)
    fingerprint = split_fingerprint(train_stems, test_stems, args.seed)
    metadata = {
        "dataset": "M3FD",
        "source_root": str(root),
        "method": "sha256(seed:stem) ascending",
        "seed": args.seed,
        "mode": args.mode,
        "train_count": len(train_stems),
        "test_count": len(test_stems),
        "fingerprint_sha256": fingerprint,
    }
    atomic_write_text(
        meta_dir / "split.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )

    print(f"matched M3FD pairs: {len(stems)}")
    print(f"destination files: copied={transferred}, reused={reused}")
    print(f"train: {len(train_stems)} -> {output_root / 'train'}")
    print(f"test:  {len(test_stems)} -> {output_root / 'test'}")
    print(f"split lists: {meta_dir}")
    print(f"fingerprint_sha256: {fingerprint}")
    print("original Ir/Vis/Annotation folders were preserved")


if __name__ == "__main__":
    main()
