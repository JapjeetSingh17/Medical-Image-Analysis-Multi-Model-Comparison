"""Score, rank, and summarize medical VLM benchmark results in percentages (0-100%)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import f1_score


def load_predictions_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load checkpointed per-image predictions from a JSONL file."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_human_baselines(manifest_df: pd.DataFrame) -> dict[str, float]:
    """Compute NEJM human clinician crowd accuracy metrics in percentages."""
    nejm = manifest_df[manifest_df["source"] == "NEJM-Image-Challenge"]
    if nejm.empty:
        return {"nejm_human_vote_pct": 0.0, "nejm_human_plurality_pct": 0.0}
    vote_pct = float(pd.to_numeric(nejm["human_vote_pct"], errors="coerce").dropna().mean() or 0.0)
    plur_pct = float(
        pd.to_numeric(nejm["human_plurality_correct"], errors="coerce").dropna().mean() or 0.0
    )
    return {
        "nejm_human_vote_pct": round(vote_pct, 1),
        "nejm_human_plurality_pct": round(plur_pct, 1),
    }


def score_model_predictions(
    manifest_df: pd.DataFrame,
    preds: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute percentage accuracy, macro F1, per-source, and efficiency metrics for one model."""
    pred_df = pd.DataFrame(preds)
    merged = manifest_df.merge(pred_df, on="case_id", how="inner", suffixes=("", "_pred"))
    if merged.empty:
        raise ValueError("No matching case_ids between manifest and predictions.")

    y_true = merged["correct_letter"].astype(str).str.upper().tolist()
    y_pred = merged["predicted_letter"].fillna("INVALID").astype(str).str.upper().tolist()
    valid_labels = sorted(set(y_true))

    overall_acc = 100.0 * sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    macro_f1 = 100.0 * f1_score(y_true, y_pred, labels=valid_labels, average="macro", zero_division=0)
    format_pct = 100.0 * float(merged["format_compliant"].astype(bool).mean())

    def _subset_acc(source_name: str) -> float:
        sub = merged[merged["source"] == source_name]
        if sub.empty:
            return 0.0
        matches = (
            sub["correct_letter"].astype(str).str.upper()
            == sub["predicted_letter"].fillna("INVALID").astype(str).str.upper()
        )
        return round(100.0 * float(matches.mean()), 1)

    domain_acc: dict[str, float] = {}
    for domain, grp in merged.groupby("domain"):
        matches = (
            grp["correct_letter"].astype(str).str.upper()
            == grp["predicted_letter"].fillna("INVALID").astype(str).str.upper()
        )
        domain_acc[str(domain)] = round(100.0 * float(matches.mean()), 1)

    mean_latency = float(merged["latency_sec"].astype(float).mean())
    peak_vram = float(merged["peak_vram_gb"].astype(float).max())

    return {
        "model_key": str(merged["model_key"].iloc[0]),
        "model_name": str(merged["model_name"].iloc[0]),
        "num_cases": len(merged),
        "overall_accuracy_pct": round(overall_acc, 1),
        "macro_f1_pct": round(macro_f1, 1),
        "medxpert_accuracy_pct": _subset_acc("MedXpertQA-MM (2025)"),
        "nejm_accuracy_pct": _subset_acc("NEJM-Image-Challenge"),
        "format_compliance_pct": round(format_pct, 1),
        "mean_latency_sec": round(mean_latency, 2),
        "peak_vram_gb": round(peak_vram, 2),
        "domain_accuracy_pct": domain_acc,
    }


def evaluate_all_models(manifest_path: Path, raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Evaluate every checkpointed model JSONL, rank them, and write summary.csv + report.md."""
    manifest_df = pd.read_csv(manifest_path)
    human_stats = compute_human_baselines(manifest_df)

    jsonl_files = sorted(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No prediction JSONL files found in {raw_dir}")

    summaries: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    for jf in jsonl_files:
        preds = load_predictions_jsonl(jf)
        if not preds:
            continue
        metrics = score_model_predictions(manifest_df, preds)
        domain_map = metrics.pop("domain_accuracy_pct")
        summaries.append(metrics)
        for dom, val in domain_map.items():
            domain_rows.append(
                {
                    "model_key": metrics["model_key"],
                    "model_name": metrics["model_name"],
                    "domain": dom,
                    "accuracy_pct": val,
                }
            )

    summary_df = pd.DataFrame(summaries).sort_values(
        by=["overall_accuracy_pct", "macro_f1_pct", "mean_latency_sec"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    summary_df.insert(0, "rank", range(1, len(summary_df) + 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(domain_rows).to_csv(out_dir / "domain_breakdown.csv", index=False)
    (out_dir / "human_baseline.json").write_text(
        json.dumps(human_stats, indent=2), encoding="utf-8"
    )

    _write_markdown_report(summary_df, human_stats, out_dir / "report.md")
    return summary_df


def _write_markdown_report(
    summary_df: pd.DataFrame,
    human_stats: dict[str, float],
    report_path: Path,
) -> None:
    """Write a clean Markdown leaderboard report ready for course submission."""
    lines = [
        "# Medical Image Analyzer — Multi-Model Evaluation Report",
        "",
        "## Ranked Leaderboard (Percentages)",
        "",
        "| Rank | Model | Cases | Overall Acc (%) | Macro F1 (%) | MedXpertQA (%) | NEJM (%) | Format OK (%) | Latency (s/img) | Peak VRAM (GB) |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| #{int(row['rank'])} | **{row['model_name']}** | {int(row['num_cases'])} | "
            f"{row['overall_accuracy_pct']:.1f}% | {row['macro_f1_pct']:.1f}% | "
            f"{row['medxpert_accuracy_pct']:.1f}% | {row['nejm_accuracy_pct']:.1f}% | "
            f"{row['format_compliance_pct']:.1f}% | {row['mean_latency_sec']:.2f}s | "
            f"{row['peak_vram_gb']:.2f} GB |"
        )

    lines.extend(
        [
            "",
            "## NEJM Human Clinician Crowd Reference",
            f"- **Human Clinician Plurality Accuracy (NEJM Top Vote):** {human_stats['nejm_human_plurality_pct']:.1f}%",
            f"- **Human Clinician Mean Correct Vote Share:** {human_stats['nejm_human_vote_pct']:.1f}%",
            "",
            "## Generated Percentage Charts",
            "- `charts/01_model_ranking_overall.png`",
            "- `charts/02_source_and_human_comparison.png`",
            "- `charts/03_domain_breakdown_heatmap.png`",
            "- `charts/04_accuracy_vs_vram_speed.png`",
            "",
            "> **Disclaimer:** For educational and research benchmarking only; not intended for clinical diagnosis.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score and rank medical VLM predictions.")
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark_150/manifest.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    df = evaluate_all_models(args.manifest, args.raw_dir, args.out_dir)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
