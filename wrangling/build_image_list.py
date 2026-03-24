from __future__ import annotations

from pathlib import Path


def parse_extensions(raw: str) -> set[str]:
    exts: set[str] = set()
    for item in raw.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        exts.add(ext)
    if not exts:
        raise ValueError("At least one file extension must be provided")
    return exts


def build_image_list(
    split: str = "train",
    dataset_root: Path = Path("data/Tagging_work"),
    images_dir: Path | None = None,
    output_file: Path | None = None,
    prefix: str | None = None,
    exts_raw: str = ".jpg",
    recursive: bool = False,
) -> None:
    images_dir = images_dir if images_dir else dataset_root / "images" / split
    output_file = output_file if output_file else dataset_root / f"{split}.txt"
    prefix = prefix if prefix is not None else f"images/{split}"
    exts = parse_extensions(exts_raw)

    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    paths = images_dir.rglob("*") if recursive else images_dir.iterdir()
    files = sorted(
        p.relative_to(images_dir).as_posix()
        for p in paths
        if p.is_file() and p.suffix.lower() in exts
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(f"{prefix}/{name}" for name in files) + ("\n" if files else ""),
        encoding="utf-8",
    )

    print(
        f"Wrote {len(files)} lines to {output_file} "
        f"(input={images_dir}, exts={sorted(exts)}, recursive={recursive})"
    )


def main() -> None:
    # Edit these values directly instead of passing CLI arguments.
    split = "train"
    dataset_root = Path("data/extracted_frames/processed/classifition")
    images_dir = None
    output_file = None
    prefix = None
    exts_raw = ".jpg"
    recursive = False

    build_image_list(
        split=split,
        dataset_root=dataset_root,
        images_dir=images_dir,
        output_file=output_file,
        prefix=prefix,
        exts_raw=exts_raw,
        recursive=recursive,
    )


if __name__ == "__main__":
    main()