# Project Explanation & 2-Member Work Division Guide

> **Purpose:** This document explains the entire **Medical Image Analyzer — Multi-Model VLM Comparison** project and cleanly divides ownership into **Part 1 (Member 1)** and **Part 2 (Member 2)** so your professor can clearly evaluate each team member's technical contribution during your viva/presentation.

---

## High-Level System Flow (Shared Overview)

1. **Uncontaminated Dataset Curation (`150` images):** Combines `75` expert clinical multimodal cases from **MedXpertQA-MM (2025)** and `75` recent clinical cases from **NEJM-Image-Challenge** (including real physician crowd vote percentages).
2. **Sequential 4-bit Cloud Execution (Google Colab T4, 15 GB VRAM):** Runs **4 medical-trained Vision-Language Models** (`MedGemma 1.5 4B`, `LLaVA-Med v1.5 7B`, `MedVLM-R1 2B`, `Qwen2.5-VL-MedSFT 3B`) one at a time in 4-bit NF4 quantization with Drive checkpointing.
3. **Structured Parsing, Ranking & Percentage Visualization:** Extracts `<think>` reasoning and `<answer>` option letters, ranks models (`#1` to `#4`) by Overall Accuracy (%) and Macro F1 (%), compares against **Human Clinician Crowd Accuracy (%)**, and plots 4 percentage charts.
4. **Live Presentation Demo:** Launches a **Gradio** web interface preloaded with **3 diverse multi-modality cases** (plus custom image upload) and generates 3 static backup demo cards.

---

# Part 1 — Member 1: Data Engineering, Evaluation Metrics & Percentage Visualizations

**Recommended Presenter:** *Member 1 (e.g., Japjeet Singh)*  
**Core Files Owned:**
- [`src/prepare_dataset.py`](src/prepare_dataset.py) — Uncontaminated multi-modality dataset pipeline
- [`src/evaluate.py`](src/evaluate.py) — Scoring, human baseline calculation, and ranking engine
- [`src/plot_results.py`](src/plot_results.py) — 4 high-resolution percentage comparison charts
- [`tests/test_parser_and_metrics.py`](tests/test_parser_and_metrics.py) *(metrics & chart tests)*

### What Member 1 Built & How to Explain It to the Professor

1. **Why We Rejected Standard "Inbuilt" Datasets (Scientific Rigor):**
   - Datasets like `MedMNIST`, `VQA-RAD`, and `HAM10000` are widely memorized during VLM pretraining/SFT, inflating scores artificially.
   - Instead, I built `prepare_dataset.py` to sample **150 clinical cases** across two modern benchmarks:
     - **75 single-image cases from `TsinghuaC3I/MedXpertQA` (`MM`, 2025):** Stratified round-robin across `body_system` categories (`Cardiovascular`, `Skeletal`, `Nervous`, etc.) so no single organ system dominates.
     - **75 recent cases from `OctoMed/NEJM-Image-Challenge`:** Sorted newest-first (`_parse_nejm_date()`) to minimize data leakage, spanning Dermatology, Radiology, Ophthalmology, and Pathology, while preserving `option_A_votes`–`option_E_votes` to compute **Human Clinician Crowd Accuracy (%)**.
   - Every image is normalized to RGB PNG (`<= 768px` longest edge) and indexed in `manifest.csv`.

2. **Ranking & Quantitative Evaluation Engine (`evaluate.py`):**
   - Merges checkpointed `.jsonl` predictions with ground-truth labels in `manifest.csv`.
   - Computes standardized percentage metrics (`0–100%`):
     - **Overall Accuracy (%)** and **Macro F1 Score (%)** (prevents class imbalance bias).
     - **Source Split Accuracy (%):** `MedXpertQA-MM (%)` vs `NEJM-Image-Challenge (%)`.
     - **Per-Domain Accuracy (%)** across body systems.
     - **Format Compliance (%)** (how reliably the model follows diagnostic instructions).
     - **Human Clinician Baseline:** Both **Human Plurality Accuracy (%)** (did the majority of doctors pick the right answer?) and **Mean Correct Vote Share (%)**.
   - Ranks models `#1–#4` by `(Overall Accuracy %, Macro F1 %, -Mean Latency)` and exports `summary.csv` and `report.md`.

3. **Percentage Visualizations (`plot_results.py`):**
   - `01_model_ranking_overall.png`: Ranked horizontal bar chart of Overall Accuracy (%) & Macro F1 (%).
   - `02_source_and_human_comparison.png`: Grouped dataset comparison with a dashed red **NEJM Human Clinician Crowd (%)** benchmark line.
   - `03_domain_breakdown_heatmap.png`: Domain-by-model accuracy (%) heatmap showing where each model excels or struggles.
   - `04_accuracy_vs_vram_speed.png`: Accuracy (%) vs. Latency (s/img) scatter plot scaled by Peak VRAM (GB).

### Likely Professor Questions for Member 1
- *Q: Why use Macro F1 alongside Overall Accuracy?*  
  **A:** Accuracy alone can hide weak performance on rarer option letters or harder body systems; Macro F1 weights every class equally.
- *Q: How did you calculate the Human Clinician baseline?*  
  **A:** Using NEJM's real physician poll distributions (`option_A_votes`..`option_E_votes`), we measure both the % of cases where the top-voted option matched ground truth (Plurality Accuracy) and the average % of doctors voting for the right answer.

---

# Part 2 — Member 2: Model Engineering, T4 Optimization, Answer Parsing & Live Demo

**Recommended Presenter:** *Member 2 (Teammate)*  
**Core Files Owned:**
- [`src/models.py`](src/models.py) — 4-bit quantized VLM registry, prompt templating & GPU memory lifecycle
- [`src/parser.py`](src/parser.py) — Reasoning (`<think>`) extraction & multi-stage answer parser
- [`run_benchmark.py`](run_benchmark.py) — Resumable CLI orchestrator (`--smoke-test` / `--full`)
- [`src/demo_app.py`](src/demo_app.py) — Interactive Gradio 3-case live demo & static backup cards
- [`notebooks/medical_vlm_benchmark_colab.ipynb`](notebooks/medical_vlm_benchmark_colab.ipynb) & `.agents/plugins/colab/`

### What Member 2 Built & How to Explain It to the Professor

1. **Selecting 4 Distinct Medical VLM Paradigms (`models.py`):**
   - **MedGemma 1.5 (4B):** Google's Gemma-3 + specialized MedSigLIP encoder.
   - **LLaVA-Med v1.5 (7B):** Mistral-7B aligned on PubMed Central biomedical figures (`LlavaForConditionalGeneration`).
   - **MedVLM-R1 (2B):** Qwen2-VL-2B fine-tuned via GRPO Reinforcement Learning to emit clinical chain-of-thought inside `<think>...</think>` before answering.
   - **Qwen2.5-VL-MedSFT (3B):** Qwen2.5-VL-3B trained via medical supervised fine-tuning.

2. **Hardware Constraints & Colab T4 Engineering (`models.py` + `run_benchmark.py`):**
   - **Problem 1 (VRAM & CPU RAM Limits):** Running 4 models simultaneously would exceed Colab T4's 15 GB VRAM and 12.7 GB CPU RAM.  
     **Solution:** Configured 4-bit NF4 `BitsAndBytesConfig` (`low_cpu_mem_usage=True`, `device_map="auto"`), bringing each model down to ~2.0–4.8 GB VRAM, and built `unload()` (`del`, `gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`) to free memory sequentially between models.
   - **Problem 2 (T4 Turing GPU Architecture):** T4 (`sm_75`) lacks native `bfloat16` and `flash_attention_2`.  
     **Solution:** Forced `torch.float16` compute dtype and implemented `_load_model_with_attn_fallback()` (`sdpa` → `eager`).
   - **Problem 3 (Session Timeouts):** Added Drive-backed `.jsonl` checkpointing in `run_benchmark.py` so interrupted runs resume without losing completed cases.

3. **Zero-Shot Prompt Standardization & Robust Answer Parsing (`parser.py`):**
   - Built `build_clinical_prompt()` to give all models identical multiple-choice options, adapting only the `<think>` instruction for RL reasoning model `MedVLM-R1`.
   - Built a 3-tier parser: (1) `<answer>X</answer>` extraction, (2) regex option-letter extraction (`"Answer: B"`, `"(C)"`), and (3) verbatim option-text fallback—while logging `format_compliant` rate (%).

4. **2–3 Live Presentation Demos (`demo_app.py`):**
   - Built a **Gradio web app** (`launch_gradio_demo()`) preloaded with **3 distinct clinical cases** across modalities plus custom image upload, allowing live side-by-side comparison of predictions, `<think>` reasoning traces, latency, and VRAM.
   - Also implemented `generate_static_demo_cards()` to save 3 backup PNG cards (`results/demos/demo_case_1..3.png`) in case classroom internet drops.

### Likely Professor Questions for Member 2
- *Q: Does 4-bit quantization hurt medical accuracy?*  
  **A:** NF4 (NormalFloat4) with double quantization and `float16` compute preserves nearly all reasoning accuracy while cutting VRAM ~3–4×, making 7B models runnable on a free 15 GB T4 GPU.
- *Q: How do you ensure a fair comparison between a reasoning model (`MedVLM-R1`) and direct SFT models (`MedGemma`, `LLaVA-Med`)?*  
  **A:** Both receive the exact same question and options; `parser.py` isolates `<think>` blocks cleanly so reasoning length never penalizes answer extraction, and we measure both Accuracy (%) and Format Compliance (%).

---

## Live Presentation Script (5-Minute Flow)

1. **Member 1 (1.5 min):** Explain the problem, why we rejected `MedMNIST` to avoid leakage, the 150-image `MedXpertQA` + `NEJM` dataset, and show the 4 percentage ranking charts (`01`–`04`).
2. **Member 2 (2 min):** Explain the 4 model architectures, 4-bit T4 memory engineering, and `<think>`/`<answer>` parsing.
3. **Joint Live Demo (1.5 min):** Open the Gradio app (or backup `demo_case_1..3.png` cards) and walk through **3 multi-modality cases** (e.g., an NEJM clinical diagnosis case + 2 MedXpertQA radiology/organ cases), showing how `MedVLM-R1`'s reasoning compares against `MedGemma 1.5` and `LLaVA-Med`.
