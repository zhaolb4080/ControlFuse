import argparse
import json
from pathlib import Path


EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def indexed(directory: Path):
    return {path.stem: path for path in directory.iterdir() if path.suffix.lower() in EXTENSIONS}


def main():
    parser = argparse.ArgumentParser(description="Build a ControlFuse JSONL manifest from aligned folders.")
    parser.add_argument("--visible-dir", required=True)
    parser.add_argument("--infrared-dir", required=True)
    parser.add_argument("--mask-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--instruction", default="Enhance the entire scene.")
    parser.add_argument("--negative-instruction", default="Suppress all relevant content.")
    parser.add_argument("--granularity", choices=("global", "semantic", "instance"), default="global")
    args = parser.parse_args()
    vis, ir = indexed(Path(args.visible_dir)), indexed(Path(args.infrared_dir))
    masks = indexed(Path(args.mask_dir)) if args.mask_dir else {}
    names = sorted(set(vis) & set(ir))
    if not names:
        raise RuntimeError("No matching visible/infrared stems found.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for name in names:
            row = {
                "name": name,
                "visible": str(vis[name].resolve()),
                "infrared": str(ir[name].resolve()),
                "instruction": args.instruction,
                "negative_instruction": args.negative_instruction,
                "granularity": args.granularity,
            }
            if name in masks:
                row["mask"] = str(masks[name].resolve())
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(names)} samples to {output}")


if __name__ == "__main__":
    main()
