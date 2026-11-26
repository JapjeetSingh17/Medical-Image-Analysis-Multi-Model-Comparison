"""Generate percentage-annotated comparison charts for the ranked medical VLMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _configure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"


def plot_overall_ranking(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Plot horizontal ranked bar chart comparing Overall Accuracy (%) and Macro F1 (%)."""
    df = summary_df.sort_values("rank", ascending=False).copy()
    labels = [f"#{int(r)}  {n}" for r, n in zip(df["rank"], df["model_name"])]

    y = range(len(df))
    height = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars_acc = ax.barh(
        [i + height / 2 for i in y],
        df["overall_accuracy_pct"],
        height=height,
        label="Overall Accuracy (%)",
        color="#1f77b4",
    )
    bars_f1 = ax.barh(
        [i - height / 2 for i in y],
        df["macro_f1_pct"],
        height=height,
        label="Macro F1 Score (%)",
        color="#2ca02c",
        alpha=0.88,
    )

    for bar in list(bars_acc) + list(bars_f1):
        val = bar.get_width()
        ax.text(
            val + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 108)
    ax.set_xlabel("Score (%)")
    ax.set_title("Medical Image Analyzer — Model Ranking (Accuracy & Macro F1 %)", pad=14)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_source_and_human_comparison(
    summary_df: pd.DataFrame,
    human_stats: dict[str, float],
    out_path: Path,
) -> None:
    """Plot grouped bars for MedXpertQA (%) vs NEJM (%) with Human Clinician Crowd line."""
    df = summary_df.sort_values("rank", ascending=True).copy()
    x = range(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(
        [i - width / 2 for i in x],
        df["medxpert_accuracy_pct"],
        width=width,
        label="MedXpertQA-MM 2025 (%)",
        color="#3b82f6",
    )
    b2 = ax.bar(
        [i + width / 2 for i in x],
        df["nejm_accuracy_pct"],
        width=width,
        label="NEJM Recent Cases (%)",
        color="#f59e0b",
    )

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.3,
            f"{h:.1f}%",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )

    human_plur = human_stats.get("nejm_human_plurality_pct", 0.0)
    if human_plur > 0:
        ax.axhline(
            human_plur,
            color="#dc2626",
            linestyle="--",
            linewidth=2,
            label=f"NEJM Human Clinician Crowd ({human_plur:.1f}%)",
        )

    ax.set_xticks(list(x))
    short_labels = [
        f"#{int(r)}\n{n.replace(' (', '\n(')}" for r, n in zip(df["rank"], df["model_name"])
    ]
    ax.set_xticklabels(short_labels, fontsize=9.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Benchmark Breakdown by Dataset Source vs. Human Clinician Baseline", pad=14)
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_domain_heatmap(
    summary_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot a percentage heatmap across body systems / clinical domains."""
    if domain_df.empty:
        return

    order = summary_df.sort_values("rank")["model_name"].tolist()
    pivot = domain_df.pivot(index="model_name", columns="domain", values="accuracy_pct")
    pivot = pivot.reindex([m for m in order if m in pivot.index])

    fig_w = max(9.5, 1.35 * len(pivot.columns))
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Accuracy (%)"},
        linewidths=0.6,
        ax=ax,
    )
    for txt in ax.texts:
        txt.set_text(f"{txt.get_text()}%")

    ax.set_title("Per-Domain / Modality Accuracy Breakdown (%)", pad=14)
    ax.set_xlabel("Clinical Domain / Body System")
    ax.set_ylabel("")
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_efficiency(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Plot Accuracy (%) vs Inference Latency (s/img), scaled by Peak Memory (GB)."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    sizes = [max(v, 0.8) * 160 for v in summary_df["peak_vram_gb"]]
    scatter = ax.scatter(
        summary_df["mean_latency_sec"],
        summary_df["overall_accuracy_pct"],
        s=sizes,
        c=summary_df["peak_vram_gb"],
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=1.2,
    )

    max_x = max(float(summary_df["mean_latency_sec"].max()), 5.0)
    max_y = max(float(summary_df["overall_accuracy_pct"].max()), 45.0)
    offsets = [(14, 24), (16, 12), (16, -24), (16, -38), (14, -22)]
    sorted_df = summary_df.sort_values("rank").reset_index(drop=True)

    for idx, row in sorted_df.iterrows():
        dx, dy = offsets[idx % len(offsets)]
        ha = "left"
        if float(row["mean_latency_sec"]) > max_x * 0.72:
            dx, ha = -14, "right"
        label = (
            f"#{int(row['rank'])} {row['model_name']}\n"
            f"({row['overall_accuracy_pct']:.1f}%, {row['peak_vram_gb']:.2f} GB)"
        )
        ax.annotate(
            label,
            (row["mean_latency_sec"], row["overall_accuracy_pct"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            fontsize=8.8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BBBBBB", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="#666666", lw=0.9),
        )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Peak Memory / VRAM (GB)")
    ax.set_xlim(-max_x * 0.08, max_x * 1.15)
    ax.set_ylim(0, min(108, max(60, max_y + 22)))
    ax.set_xlabel("Mean Inference Time (seconds / image)")
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("Accuracy (%) vs. Speed & Memory Trade-off", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_all_charts(results_dir: Path) -> list[Path]:
    """Generate all 4 percentage charts from results_dir/summary.csv."""
    _configure_style()
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(results_dir / "summary.csv")
    domain_df = pd.read_csv(results_dir / "domain_breakdown.csv")
    human_path = results_dir / "human_baseline.json"
    human_stats = json.loads(human_path.read_text(encoding="utf-8")) if human_path.exists() else {}

    paths = [
        charts_dir / "01_model_ranking_overall.png",
        charts_dir / "02_source_and_human_comparison.png",
        charts_dir / "03_domain_breakdown_heatmap.png",
        charts_dir / "04_accuracy_vs_vram_speed.png",
    ]
    plot_overall_ranking(summary_df, paths[0])
    plot_source_and_human_comparison(summary_df, human_stats, paths[1])
    plot_domain_heatmap(summary_df, domain_df, paths[2])
    plot_accuracy_vs_efficiency(summary_df, paths[3])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate percentage ranking charts.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    generated = generate_all_charts(args.results_dir)
    for p in generated:
        print(f"Generated chart: {p}")


if __name__ == "__main__":
    main()
