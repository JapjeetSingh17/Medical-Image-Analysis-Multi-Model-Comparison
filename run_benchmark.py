"""End-to-end CLI runner for dataset prep, sequential 4-bit benchmarking, ranking, and charts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.demo_app import generate_static_demo_cards
from src.evaluate import evaluate_all_models
from src.models import MODEL_REGISTRY, MedicalVLMRunner
from src.parser import parse_model_response
from src.plot_results import generate_all_charts
from src.prepare_dataset import build_benchmark_dataset


def _t4_without_bf16() -> bool:
    """Return True when running on a CUDA GPU that lacks native bfloat16 (e.g., Turing T4)."""
    try:
        import torch

        return torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    except Exception:
        return False


def _load_existing_case_ids(jsonl_path: Path) -> set[str]:
    """Read already-evaluated case_ids so interrupted runs resume seamlessly."""
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                done.add(str(json.loads(line)["case_id"]))
    return done


def _select_smoke_subset(df: pd.DataFrame, count: int = 6) -> pd.DataFrame:
    """Pick a balanced tiny subset (half MedXpertQA, half NEJM) for fast verification."""
    half = max(1, count // 2)
    p1 = df[df["source"] == "MedXpertQA-MM (2025)"].head(half)
    p2 = df[df["source"] == "NEJM-Image-Challenge"].head(count - len(p1))
    return pd.concat([p1, p2], ignore_index=True)


def run_benchmark(
    data_dir: Path,
    results_dir: Path,
    model_keys: list[str],
    smoke_test: bool = False,
    hf_token: str | None = None,
    explicit_models: bool = False,
) -> None:
    """Execute dataset setup, sequential 4-bit inference, evaluation, and plotting."""
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        build_benchmark_dataset(output_dir=data_dir)

    df = pd.read_csv(manifest_path)
    if smoke_test:
        df = _select_smoke_subset(df, count=6)
        print(f"Running SMOKE TEST on {len(df)} cases across {len(model_keys)} models.")
    else:
        print(f"Running FULL benchmark on {len(df)} cases across {len(model_keys)} models.")

    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    token = hf_token or os.environ.get("HF_TOKEN") or None
    no_bf16_gpu = _t4_without_bf16()

    for key in model_keys:
        spec = MODEL_REGISTRY[key]
        if spec.family == "gemma3" and no_bf16_gpu and not explicit_models:
            print(
                f"[SKIP] {spec.display_name} auto-skipped on T4 GPU (no native bfloat16; "
                "fp32 fallback takes ~12 min/image). Pass --models medgemma-1.5-4b to force it."
            )
            continue
        if spec.gated and not token:
            print(
                f"[SKIP] {spec.display_name} requires HF_TOKEN (gated repo {spec.hf_id}). "
                "Set HF_TOKEN or pass --hf-token to include it."
            )
            continue

        out_jsonl = raw_dir / f"{key}.jsonl"
        done_ids = _load_existing_case_ids(out_jsonl)
        pending = df[~df["case_id"].isin(done_ids)]
        if pending.empty:
            print(f"[RESUME] {spec.display_name}: all {len(df)} cases already checkpointed.")
            continue

        print(f"\n=== Loading {spec.display_name} ({spec.hf_id}) ===")
        runner = MedicalVLMRunner(spec, hf_token=token)
        try:
            runner.load()
        except Exception as exc:
            print(
                f"[SKIP] Could not load {spec.display_name} ({spec.hf_id}): {exc}\n"
                + (
                    f"       Make sure you accepted the license at https://huggingface.co/{spec.hf_id} "
                    "with the same HF account as your HF_TOKEN."
                    if spec.gated
                    else ""
                )
            )
            runner.unload()
            continue

        with out_jsonl.open("a", encoding="utf-8") as fh:
            for _, row in tqdm(pending.iterrows(), total=len(pending), desc=spec.key):
                options = json.loads(row["options_json"])
                img = Image.open(row["image_path"]).convert("RGB")
                raw_resp, latency = runner.predict_mc(
                    img, str(row["question"]), options
                )
                parsed = parse_model_response(raw_resp, options)

                record: dict[str, Any] = {
                    "case_id": row["case_id"],
                    "model_key": spec.key,
                    "model_name": spec.display_name,
                    "predicted_letter": parsed.predicted_letter,
                    "format_compliant": parsed.format_compliant,
                    "reasoning": parsed.reasoning,
                    "raw_response": raw_resp,
                    "latency_sec": round(latency, 3),
                    "peak_vram_gb": runner.peak_vram_gb(),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()

        print(f"Finished {spec.display_name} | Peak VRAM: {runner.peak_vram_gb():.2f} GB")
        runner.unload()

    summary_df = evaluate_all_models(manifest_path, raw_dir, results_dir)
    print("\n=== Ranked Leaderboard ===")
    print(summary_df.to_string(index=False))

    charts = generate_all_charts(results_dir)
    cards = generate_static_demo_cards(manifest_path, raw_dir, results_dir / "demos")
    print(f"\nGenerated {len(charts)} percentage charts and {len(cards)} demo cards.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Image Analyzer Multi-Model Benchmark")
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmark_150"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    explicit_models = args.models is not None
    model_keys = args.models if explicit_models else list(MODEL_REGISTRY.keys())

    run_benchmark(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        model_keys=model_keys,
        smoke_test=args.smoke_test and not args.full,
        hf_token=args.hf_token,
        explicit_models=explicit_models,
    )


if __name__ == "__main__":
    main()
