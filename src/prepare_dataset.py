"""Build an uncontaminated 150-image multi-modality medical benchmark.

Sources:
1. TsinghuaC3I/MedXpertQA (MM config, test split - 2025 clinical multimodal benchmark)
2. OctoMed/NEJM-Image-Challenge (recent clinical image challenges + human crowd votes)
"""

from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

MONTH_ORDER = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_nejm_date(date_str: str) -> tuple[int, int, int]:
    """Convert NEJM date string like 'oct-05-2023' into sortable (year, month, day)."""
    try:
        mon_s, day_s, year_s = date_str.strip().lower().split("-")
        return (int(year_s), MONTH_ORDER.get(mon_s, 0), int(day_s))
    except Exception:
        return (0, 0, 0)


def _strip_medxpert_choices(question: str) -> str:
    """Remove inline 'Answer Choices: ...' suffix so prompts stay uniform across sources."""
    marker = "\nAnswer Choices:"
    if marker in question:
        return question.split(marker, 1)[0].strip()
    return question.strip()


def _parse_pct(val: Any) -> float:
    """Parse percentage strings like '62%' into floats."""
    if val is None:
        return 0.0
    s = str(val).strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _coerce_pil_image(img_obj: Any, zip_handle: zipfile.ZipFile | None = None, zip_index: dict[str, str] | None = None) -> Image.Image:
    """Resolve PIL Image from an Image object, bytes/dict, or filename inside images.zip."""
    if isinstance(img_obj, Image.Image):
        return img_obj
    if isinstance(img_obj, dict) and img_obj.get("bytes"):
        return Image.open(io.BytesIO(img_obj["bytes"]))
    if isinstance(img_obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(img_obj))
    if isinstance(img_obj, str):
        if zip_handle is not None and zip_index is not None:
            member = zip_index.get(img_obj) or zip_index.get(Path(img_obj).name)
            if member:
                with zip_handle.open(member) as fh:
                    return Image.open(io.BytesIO(fh.read()))
        path = Path(img_obj)
        if path.exists():
            return Image.open(path)
    raise TypeError(f"Unsupported image payload type: {type(img_obj)!r}")


def _save_pil_image(
    img_obj: Any,
    dest: Path,
    max_side: int = 768,
    zip_handle: zipfile.ZipFile | None = None,
    zip_index: dict[str, str] | None = None,
) -> None:
    """Save RGB image resized to at most max_side on its longest edge."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = _coerce_pil_image(img_obj, zip_handle=zip_handle, zip_index=zip_index)
    rgb = img.convert("RGB")
    if max(rgb.size) > max_side:
        rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    rgb.save(dest, format="PNG")


def _sample_medxpert(n_samples: int, seed: int, img_dir: Path) -> list[dict[str, Any]]:
    """Sample single-image cases stratified across body systems from MedXpertQA-MM."""
    ds = load_dataset("TsinghuaC3I/MedXpertQA", "MM", split="test")
    images_zip_path = hf_hub_download(
        repo_id="TsinghuaC3I/MedXpertQA",
        filename="images.zip",
        repo_type="dataset",
    )
    single_rows = [row for row in ds if len(row.get("images") or []) == 1]

    by_system: dict[str, list[dict[str, Any]]] = {}
    for row in single_rows:
        sys_name = str(row.get("body_system") or "Other")
        by_system.setdefault(sys_name, []).append(row)

    rng = random.Random(seed)
    for bucket in by_system.values():
        rng.shuffle(bucket)

    # Round-robin stratified sampling across body systems for broad modality coverage
    selected: list[dict[str, Any]] = []
    systems = sorted(by_system.keys())
    idx = 0
    while len(selected) < n_samples and any(by_system.values()):
        sys_name = systems[idx % len(systems)]
        bucket = by_system[sys_name]
        if bucket:
            selected.append(bucket.pop())
        idx += 1

    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(images_zip_path, "r") as zf:
        zip_index = {Path(name).name: name for name in zf.namelist()}
        for name in zf.namelist():
            zip_index[name] = name

        for i, row in enumerate(selected):
            case_id = f"medxpert_{row.get('id', i)}"
            img_path = img_dir / f"{case_id}.png"
            _save_pil_image(row["images"][0], img_path, zip_handle=zf, zip_index=zip_index)

            raw_opts = row["options"]
            options = {str(k).upper(): str(v).strip() for k, v in raw_opts.items()}
            records.append(
                {
                    "case_id": case_id,
                    "source": "MedXpertQA-MM (2025)",
                    "domain": str(row.get("body_system") or "Clinical"),
                    "medical_task": str(row.get("medical_task") or "Diagnosis"),
                    "case_date": "2025",
                    "image_path": str(img_path),
                    "question": _strip_medxpert_choices(str(row["question"])),
                    "options_json": json.dumps(options, ensure_ascii=False),
                    "correct_letter": str(row["label"]).strip().upper(),
                    "human_vote_pct": None,
                    "human_plurality_correct": None,
                }
            )
    return records


def _sample_nejm(n_samples: int, min_year: int, seed: int, img_dir: Path) -> list[dict[str, Any]]:
    """Sample recent cases from NEJM-Image-Challenge with human clinician vote stats."""
    ds = load_dataset("OctoMed/NEJM-Image-Challenge", split="test")

    enriched: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for row in ds:
        parsed_date = _parse_nejm_date(str(row.get("date", "")))
        if parsed_date[0] >= min_year and row.get("image") is not None:
            enriched.append((parsed_date, row))

    # Fallback: if fewer than n_samples in min_year+, sort newest-first and take top
    if len(enriched) < n_samples:
        enriched = [
            (_parse_nejm_date(str(row.get("date", ""))), row)
            for row in ds
            if row.get("image") is not None
        ]
        enriched.sort(key=lambda pair: pair[0], reverse=True)
        pool = [row for _, row in enriched[: max(n_samples * 2, n_samples)]]
    else:
        enriched.sort(key=lambda pair: pair[0], reverse=True)
        pool = [row for _, row in enriched]

    rng = random.Random(seed)
    rng.shuffle(pool)
    selected = pool[:n_samples]

    records: list[dict[str, Any]] = []
    for i, row in enumerate(selected):
        case_id = f"nejm_{row.get('image_id', i)}"
        img_path = img_dir / f"{case_id}.png"
        _save_pil_image(row["image"], img_path)

        options: dict[str, str] = {}
        for item in row["options"]:
            letter, text = str(item).split(":", 1)
            options[letter.strip().upper()] = text.strip()

        correct_letter = str(row["answer"]).split(":", 1)[0].strip().upper()
        votes = {
            letter: _parse_pct(row.get(f"option_{letter}_votes"))
            for letter in options.keys()
        }
        human_vote_pct = votes.get(correct_letter, 0.0)
        top_letter = max(votes.items(), key=lambda kv: kv[1])[0] if votes else ""
        human_plurality = 100.0 if top_letter == correct_letter else 0.0

        records.append(
            {
                "case_id": case_id,
                "source": "NEJM-Image-Challenge",
                "domain": "NEJM Clinical Case",
                "medical_task": "Diagnosis",
                "case_date": str(row.get("date", "")),
                "image_path": str(img_path),
                "question": str(row["question"]).strip(),
                "options_json": json.dumps(options, ensure_ascii=False),
                "correct_letter": correct_letter,
                "human_vote_pct": human_vote_pct,
                "human_plurality_correct": human_plurality,
            }
        )
    return records


def build_benchmark_dataset(
    output_dir: Path,
    medxpert_count: int = 75,
    nejm_count: int = 75,
    nejm_min_year: int = 2021,
    seed: int = 42,
) -> Path:
    """Create the combined 150-image benchmark folder and manifest.csv."""
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    records = _sample_medxpert(medxpert_count, seed=seed, img_dir=img_dir) + _sample_nejm(
        nejm_count, min_year=nejm_min_year, seed=seed, img_dir=img_dir
    )

    df = pd.DataFrame(records)
    df.to_csv(manifest_path, index=False)
    print(f"Saved {len(df)} benchmark cases to {manifest_path}")
    print(df["source"].value_counts().to_string())
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 150-image clinical VLM benchmark.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmark_150"))
    parser.add_argument("--medxpert-count", type=int, default=75)
    parser.add_argument("--nejm-count", type=int, default=75)
    parser.add_argument("--nejm-min-year", type=int, default=2021)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_benchmark_dataset(
        output_dir=args.output_dir,
        medxpert_count=args.medxpert_count,
        nejm_count=args.nejm_count,
        nejm_min_year=args.nejm_min_year,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
