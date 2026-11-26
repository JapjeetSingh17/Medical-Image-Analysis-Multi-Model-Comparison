# Medical Image Analysis: Multi-Model VLM Comparison

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JapjeetSingh17/Medical-Image-Analysis-Multi-Model-Comparison/blob/main/notebooks/medical_vlm_benchmark_colab.ipynb)

An **AI in Healthcare** course mini-project that benchmarks **small/medium medical-image-trained Vision-Language Models (VLMs)** on an **uncontaminated 150-image multi-modality clinical dataset**, ranks them by accuracy and Macro F1 (%), compares them against a **Human Clinician Crowd baseline**, generates percentage-based visualizations, and provides a **3-case Gradio Live Presentation Demo**.

---

## Why This Benchmark Design?

1. **Avoids Dataset Leakage:** Instead of classic over-trained datasets (`MedMNIST`, `VQA-RAD`, `HAM10000`) that appear in many VLM pretraining corpora, we curate **150 clinical cases** from:
   - **75 single-image cases from [`TsinghuaC3I/MedXpertQA`](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) (`MM`, 2025)** — stratified across body systems (Cardiovascular, Skeletal, Nervous, etc.).
   - **75 recent cases from [`OctoMed/NEJM-Image-Challenge`](https://huggingface.co/datasets/OctoMed/NEJM-Image-Challenge)** — real *New England Journal of Medicine* clinical image cases spanning Dermatology, Radiology, Ophthalmology, Pathology, and Cardiology, complete with **physician crowd voting stats**.
2. **Runs Free on Google Colab T4 (15 GB VRAM):** Loads one 4-bit NF4 model at a time (`torch.float16`, `sdpa`/`eager` attention fallback) with explicit VRAM cleanup (`gc.collect()` + `torch.cuda.empty_cache()`) and per-image `.jsonl` checkpoints saved to Google Drive.
3. **Presentation-Ready:** Includes an interactive **Gradio Live Demo** (`share=True`) preloaded with 3 multi-modality cases + custom image upload, plus 3 static backup demo PNG cards.

---

## Models Compared

| Model | Hugging Face Repo | Size | Access | Medical Specialization |
| :--- | :--- | :---: | :--- | :--- |
| **MedGemma 1.5 (4B)** | `google/medgemma-1.5-4b-it` | 4B | Gated (`HF_TOKEN`) | Gemma-3 + MedSigLIP clinical multimodal SFT |
| **LLaVA-Med v1.5 (7B)** | `chaoyinshe/llava-med-v1.5-mistral-7b-hf` | 7B | Ungated | Mistral-7B + PubMed Central biomedical alignment |
| **MedVLM-R1 (2B)** | `JZPeterPan/MedVLM-R1` | 2B | Ungated | Qwen2-VL-2B + GRPO RL (`<think>...</think>`) |
| **Qwen2.5-VL-MedSFT (3B)** | `CAIR-HKISI/Qwen2.5-VL-3B-MedSFT` | 3B | Ungated | Qwen2.5-VL-3B medical supervised fine-tuning |

---

## Project Structure

```text
.
├── .agents/plugins/colab/                # Antigravity colab-mcp customization plugin
├── notebooks/
│   └── medical_vlm_benchmark_colab.ipynb # Step-by-step Google Colab T4 notebook
├── src/
│   ├── prepare_dataset.py                # Builds 150-image MedXpertQA + NEJM manifest
│   ├── models.py                         # 4-bit T4 sequential model loaders & adapters
│   ├── parser.py                         # Extracts <think>, <answer>, and option letters
│   ├── evaluate.py                       # Ranks models & exports summary.csv / report.md
│   ├── plot_results.py                   # Generates 4 percentage comparison PNG charts
│   └── demo_app.py                       # Gradio 3-case live demo + static backup cards
├── tests/
│   └── test_parser_and_metrics.py        # Local pytest suite (parser, ranking, charts)
├── run_benchmark.py                      # Unified CLI (--smoke-test / --full)
├── requirements-colab.txt                # Colab T4 dependencies
└── TEAM_EXPLANATION.md                   # 2-part member breakdown for professor evaluation
```

---

## Quick Start (Google Colab T4)

1. Click the **Open in Colab** badge above (or upload `notebooks/medical_vlm_benchmark_colab.ipynb` to Colab).
2. Switch runtime to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
3. *(Optional for MedGemma 1.5)* Accept Google's license on [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it) and add your read token to Colab Secrets as `HF_TOKEN`.
4. Run all cells:
   - Step 4 builds the 150-image dataset on Google Drive.
   - Step 5A runs a 6-image `--smoke-test`; Step 5B runs the `--full` 150-image benchmark with resume checkpoints.
   - Step 6 displays the ranked leaderboard & **4 percentage graphs**:
     - `01_model_ranking_overall.png` — Overall Accuracy (%) & Macro F1 (%)
     - `02_source_and_human_comparison.png` — MedXpertQA vs NEJM (%) + **Human Clinician Crowd (%)**
     - `03_domain_breakdown_heatmap.png` — Accuracy (%) across clinical domains/body systems
     - `04_accuracy_vs_vram_speed.png` — Accuracy (%) vs. Latency (s/img) & Peak VRAM (GB)
   - Step 7 launches the **Gradio Live Demo** for class presentation.

---

## Antigravity `colab-mcp` Plugin Setup

This repository includes an Antigravity customization plugin at `.agents/plugins/colab/` configuring Google's official [`googlecolab/colab-mcp`](https://github.com/googlecolab/colab-mcp) bridge via `uvx` so local coding agents can interact with an open Colab browser runtime.

---

## Local Testing

```bash
uv venv --python 3.12
uv pip install -r requirements-colab.txt
uv run pytest -v
```

> **Clinical Disclaimer:** This project is strictly for academic and educational benchmarking in healthcare AI. Model predictions must not be used for real medical diagnosis or clinical decision-making.
