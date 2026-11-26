"""Interactive Gradio demo and static presentation showcase generator (3 clinical demos)."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.models import MODEL_REGISTRY, MedicalVLMRunner
from src.parser import parse_model_response


def select_demo_cases(manifest_path: Path, count: int = 3) -> pd.DataFrame:
    """Pick 3 diverse clinical modality cases from the manifest for presentation demos."""
    df = pd.read_csv(manifest_path)
    chosen_indices: list[int] = []
    seen_domains: set[str] = set()

    # Prefer at least 1 NEJM case + 2 distinct MedXpertQA body systems
    nejm_rows = df[df["source"] == "NEJM-Image-Challenge"]
    if not nejm_rows.empty:
        idx = int(nejm_rows.index[0])
        chosen_indices.append(idx)
        seen_domains.add(str(df.loc[idx, "domain"]))

    for idx, row in df.iterrows():
        if len(chosen_indices) >= count:
            break
        dom = str(row["domain"])
        if idx not in chosen_indices and dom not in seen_domains:
            chosen_indices.append(int(idx))
            seen_domains.add(dom)

    for idx in df.index:
        if len(chosen_indices) >= count:
            break
        if int(idx) not in chosen_indices:
            chosen_indices.append(int(idx))

    return df.loc[chosen_indices].reset_index(drop=True)


def generate_static_demo_cards(
    manifest_path: Path,
    raw_dir: Path,
    out_dir: Path,
    count: int = 3,
) -> list[Path]:
    """Render 3 side-by-side backup PNG demo cards from saved benchmark predictions."""
    demo_df = select_demo_cases(manifest_path, count=count)
    preds_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for jf in sorted(raw_dir.glob("*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            preds_by_model.setdefault(rec["model_name"], {})[rec["case_id"]] = rec

    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for idx, row in demo_df.iterrows():
        case_id = row["case_id"]
        options = json.loads(row["options_json"])
        correct = str(row["correct_letter"]).upper()
        img = Image.open(row["image_path"]).convert("RGB")

        fig, (ax_img, ax_txt) = plt.subplots(
            1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1, 1.45]}
        )
        ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title(f"Demo Case #{idx + 1}: {row['domain']}\n({row['source']})", fontsize=11)

        lines = [
            textwrap.fill(f"Question: {row['question']}", width=76)[:280],
            "",
            f"Ground Truth: {correct} — {options.get(correct, '')}",
            "-" * 74,
        ]
        for model_name, by_case in preds_by_model.items():
            rec = by_case.get(case_id)
            if not rec:
                continue
            pred_l = rec.get("predicted_letter") or "N/A"
            status = "CORRECT" if pred_l == correct else "WRONG"
            lines.append(
                f"[{status}] {model_name}: {pred_l} ({rec.get('latency_sec', 0):.2f}s)"
            )
            snippet = (rec.get("reasoning") or rec.get("raw_response") or "").replace("\n", " ")
            wrapped = textwrap.fill(f"Notes: {snippet[:140]}", width=72, initial_indent="   ", subsequent_indent="          ")
            lines.append(wrapped)

        ax_txt.axis("off")
        ax_txt.text(
            0.0,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=9.5,
            family="monospace",
            wrap=True,
        )
        fig.tight_layout()
        card_path = out_dir / f"demo_case_{idx + 1}.png"
        fig.savefig(card_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        generated.append(card_path)

    return generated


def launch_gradio_demo(
    manifest_path: Path,
    model_keys: list[str],
    hf_token: str | None = None,
    share: bool = True,
) -> None:
    """Launch an interactive Gradio side-by-side live demo for presentation."""
    import gradio as gr

    demo_df = select_demo_cases(manifest_path, count=3)
    presets: dict[str, dict[str, Any]] = {}
    for idx, row in demo_df.iterrows():
        label = f"Demo {idx + 1}: {row['domain']} ({row['source']})"
        presets[label] = {
            "image_path": row["image_path"],
            "question": row["question"],
            "options_json": row["options_json"],
            "correct": row["correct_letter"],
        }

    def load_preset(preset_label: str) -> tuple[Image.Image, str, str, str]:
        item = presets[preset_label]
        opts = json.loads(item["options_json"])
        opts_lines = "\n".join(f"{k}: {v}" for k, v in sorted(opts.items()))
        gt = f"Ground Truth: {item['correct']} ({opts.get(item['correct'], '')})"
        return Image.open(item["image_path"]).convert("RGB"), item["question"], opts_lines, gt

    def analyze_image(
        image: Image.Image | None,
        question: str,
        options_text: str,
        selected_models: list[str],
    ) -> str:
        if image is None or not question.strip():
            return "Please upload or select a medical image and enter a clinical question."
        options: dict[str, str] = {}
        for line in options_text.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                options[k.strip().upper()] = v.strip()
        if not options:
            options = {"A": "Normal / Negative", "B": "Abnormal / Positive"}

        md_sections: list[str] = []
        for key in selected_models:
            spec = MODEL_REGISTRY[key]
            runner = MedicalVLMRunner(spec, hf_token=hf_token)
            runner.load()
            raw_resp, latency = runner.predict_mc(image, question, options)
            vram = runner.peak_vram_gb()
            runner.unload()

            parsed = parse_model_response(raw_resp, options)
            pred_disp = (
                f"**{parsed.predicted_letter}** ({options.get(parsed.predicted_letter, '')})"
                if parsed.predicted_letter
                else "**Unparsed**"
            )
            section = [
                f"### {spec.display_name}",
                f"- **Prediction:** {pred_disp}",
                f"- **Latency:** `{latency:.2f}s` | **Peak VRAM:** `{vram:.2f} GB`",
            ]
            if parsed.reasoning:
                section.append(f"- **Reasoning (`<think>`):** {parsed.reasoning}")
            clean_raw = raw_resp.replace("<", "&lt;").replace(">", "&gt;")
            section.append(f"- **Raw Output:** {clean_raw}")
            md_sections.append("\n".join(section))

        return "\n\n---\n\n".join(md_sections)

    def on_custom_upload(img: Image.Image | None) -> str:
        return "Custom Image (No Preset Ground Truth)" if img is not None else ""

    first_label = next(iter(presets))
    init_img, init_q, init_opts, init_gt = load_preset(first_label)

    with gr.Blocks(title="Medical Image Analyzer — Multi-VLM Demo") as demo:
        gr.Markdown("# Medical Image Analyzer — Live Multi-Model Comparison")
        gr.Markdown(
            "Select one of the **3 pre-loaded multi-modality clinical cases** or upload your own medical image."
        )
        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=list(presets.keys()), value=first_label, label="Presentation Demo Preset"
            )
            model_checks = gr.CheckboxGroup(
                choices=model_keys,
                value=model_keys[:2],
                label="Models to Compare (Run Sequentially)",
            )
        with gr.Row():
            img_in = gr.Image(value=init_img, type="pil", label="Medical Image")
            with gr.Column():
                q_in = gr.Textbox(value=init_q, lines=4, label="Clinical Question")
                opts_in = gr.Textbox(value=init_opts, lines=5, label="Options (Letter: Text)")
                gt_out = gr.Textbox(value=init_gt, interactive=False, label="Reference Answer")
        run_btn = gr.Button("Analyze Image Across Selected Models", variant="primary")
        out_md = gr.Markdown(label="Model Comparison Output")

        preset_dd.change(load_preset, inputs=[preset_dd], outputs=[img_in, q_in, opts_in, gt_out])
        img_in.upload(on_custom_upload, inputs=[img_in], outputs=[gt_out])
        run_btn.click(
            analyze_image,
            inputs=[img_in, q_in, opts_in, model_checks],
            outputs=[out_md],
        )

    demo.launch(share=share, debug=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo cards or launch Gradio demo.")
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark_150/manifest.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--demos-dir", type=Path, default=Path("results/demos"))
    parser.add_argument("--launch-gradio", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
    )
    args = parser.parse_args()

    if args.launch_gradio:
        launch_gradio_demo(args.manifest, model_keys=args.models, share=True)
    else:
        cards = generate_static_demo_cards(args.manifest, args.raw_dir, args.demos_dir)
        for c in cards:
            print(f"Generated static demo card: {c}")


if __name__ == "__main__":
    main()
