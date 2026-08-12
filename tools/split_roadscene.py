"""Create the fixed 171/50 RoadScene train/test split safely and reproducibly."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def index_files(directory: Path):
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    indexed = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
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


def matching_stems(infrared, visible):
    all_stems = set(infrared) | set(visible)
    common = set(infrared) & set(visible)
    if all_stems != common:
        problems = []
        missing_ir = all_stems - set(infrared)
        missing_vis = all_stems - set(visible)
        if missing_ir:
            problems.append(f"missing from infrared: {preview(missing_ir)}")
        if missing_vis:
            problems.append(f"missing from visible: {preview(missing_vis)}")
        raise ValueError("RoadScene stems do not match; " + "; ".join(problems))
    if not common:
        raise RuntimeError("No matching RoadScene infrared/visible pairs were found.")
    return common


def validate_images(stems, infrared, visible):
    for stem in sorted(stems):
        try:
            with Image.open(infrared[stem]) as image:
                ir_size = image.size
                image.verify()
            with Image.open(visible[stem]) as image:
                vis_size = image.size
                image.verify()
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"Unreadable RoadScene image for stem {stem!r}: {exc}") from exc
        if ir_size != vis_size:
            raise ValueError(
                f"IR/VIS size mismatch for {stem}: infrared={ir_size}, visible={vis_size}"
            )


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
                    "Destination contains files outside the requested fixed split: "
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
            "Split RoadScene/{infrared,visible} into the fixed 171/50 source-image split. "
            "The original folders are preserved."
        )
    )
    parser.add_argument(
        "--root", required=True, help="RoadScene root containing infrared/ and visible/."
    )
    parser.add_argument(
        "--output-root",
        help="Destination root. Default: --root (creates RoadScene/train and RoadScene/test).",
    )
    parser.add_argument("--train-count", type=int, default=171)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="copy is safest; hardlink saves space but requires the same filesystem.",
    )
    parser.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip decoding and aligned-size validation. Stem matching is still enforced.",
    )
    args = parser.parse_args()

    if args.train_count < 1 or args.test_count < 1:
        parser.error("--train-count and --test-count must both be positive")

    root = Path(args.root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else root
    infrared = index_files(root / "infrared")
    visible = index_files(root / "visible")
    stems = matching_stems(infrared, visible)
    expected = args.train_count + args.test_count
    if len(stems) != expected:
        raise ValueError(
            f"Expected exactly {expected} matched RoadScene pairs for "
            f"{args.train_count}/{args.test_count}, found {len(stems)}."
        )
    if not args.skip_image_validation:
        validate_images(stems, infrared, visible)

    ordered = sorted(stems, key=lambda value: split_key(value, args.seed))
    train_stems = ordered[: args.train_count]
    test_stems = ordered[args.train_count :]
    assignments = {"train": train_stems, "test": test_stems}
    sources = {"infrared": infrared, "visible": visible}

    for split in assignments:
        for folder in sources:
            (output_root / split / folder).mkdir(parents=True, exist_ok=True)
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

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
        "dataset": "RoadScene",
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

    print(f"matched RoadScene pairs: {len(stems)}")
    print(f"destination files: copied={transferred}, reused={reused}")
    print(f"train: {len(train_stems)} -> {output_root / 'train'}")
    print(f"test:  {len(test_stems)} -> {output_root / 'test'}")
    print(f"split lists: {meta_dir}")
    print(f"fingerprint_sha256: {fingerprint}")
    print("original infrared/visible folders were preserved")


if __name__ == "__main__":
    main()
