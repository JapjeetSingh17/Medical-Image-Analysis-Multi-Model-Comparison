# Project Explanation & 2-Member Work Division Guide

> **Purpose:** This document explains the entire **Medical Image Analyzer — Multi-Model VLM Comparison** project and cleanly divides ownership into **Part 1 (Member 1)** and **Part 2 (Member 2)** so your professor can clearly evaluate each team member's technical contribution during your viva/presentation.

---

## Final Observed Results Summary (150 Clinical Cases)

| Rank | Model | Overall Acc (%) | Macro F1 (%) | MedXpertQA (%) | NEJM (%) | Format OK (%) | Latency (s/img) | Peak Memory |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **MedVLM-R1 (2B, MLX 4-bit)** | **34.7%** | 29.6% | **22.7%** | 46.7% | **100.0%** | 2.33s | 2.95 GB |
| **#2** | **MedGemma 1.5 (4B, MLX 4-bit)** | **34.0%** | **33.1%** | 16.0% | **52.0%** | 94.7% | 11.81s | 3.77 GB |
| **#3** | **Qwen2.5-VL-MedSFT (3B)** | **30.7%** | 28.7% | 20.0% | 41.3% | **100.0%** | **0.67s** | 3.13 GB |
| **#4** | **LLaVA-Med v1.5 (7B)** | **28.7%** | 25.1% | 16.0% | 41.3% | 16.7% | 2.22s | 4.45 GB |
| **#5** | **BiomedCLIP (200M)** | **20.0%** | 19.6% | 20.0% | 20.0% | **100.0%** | **0.25s** | **0.74 GB** |

- **NEJM Human Clinician Plurality Accuracy:** `100.0%` (`51.8%` mean correct vote share)
- **Key Takeaway:** Reinforcement-learning reasoning (`MedVLM-R1 2B`) matched/edged `MedGemma 1.5 4B` overall (**34.7% vs 34.0%**) and led on the hardest MedXpertQA-MM split (**22.7%**) while running **5× faster** in **2.95 GB** unified memory; meanwhile `BiomedCLIP` (**20.0%**) demonstrates why pure contrastive encoders struggle on complex clinical vignettes compared to generative medical VLMs.

---

# Part 1 — Member 1: Data Engineering, Evaluation Metrics & Percentage Visualizations

**Recommended Presenter:** *Member 1 (e.g., Japjeet Singh)*  
**Core Files Owned:**
- [`src/prepare_dataset.py`](src/prepare_dataset.py) — Uncontaminated multi-modality dataset pipeline
- [`src/evaluate.py`](src/evaluate.py) — Scoring, human baseline calculation, and ranking engine
- [`src/plot_results.py`](src/plot_results.py) — 4 high-resolution percentage comparison charts
- [`tests/test_parser_and_metrics.py`](tests/test_parser_and_metrics.py) *(metrics, zip extraction & chart tests)*

### What Member 1 Built & How to Explain It to the Professor

1. **Why We Rejected Standard "Inbuilt" Datasets (Scientific Rigor):**
   - Classic datasets like `MedMNIST`, `VQA-RAD`, and `HAM10000` are widely memorized during VLM pretraining/SFT, inflating accuracy artificially.
   - Instead, I built `prepare_dataset.py` to sample **150 clinical cases** across two modern benchmarks:
     - **75 single-image cases from `TsinghuaC3I/MedXpertQA` (`MM`, 2025):** Extracts only the sampled images selectively from `images.zip` and stratifies round-robin across `body_system` categories (`Cardiovascular`, `Skeletal`, `Nervous`, `Respiratory`, etc.).
     - **75 recent cases from `OctoMed/NEJM-Image-Challenge`:** Sorted newest-first (`_parse_nejm_date()`) to minimize data leakage, spanning Dermatology, Radiology, Ophthalmology, and Pathology, while preserving `option_A_votes`–`option_E_votes` to compute **Human Clinician Crowd Accuracy (%)**.
   - Every image is normalized to RGB PNG (`<= 768px` longest edge) and indexed in `manifest.csv`.

2. **Ranking & Quantitative Evaluation Engine (`evaluate.py`):**
   - Merges checkpointed `.jsonl` predictions with ground-truth labels in `manifest.csv`.
   - Computes standardized percentage metrics (`0–100%`):
     - **Overall Accuracy (%)** and **Macro F1 Score (%)**.
     - **Source Split Accuracy (%):** `MedXpertQA-MM (%)` vs `NEJM-Image-Challenge (%)`.
     - **Per-Domain Accuracy (%)** across 13 clinical body systems/domains.
     - **Format Compliance (%)** and **Human Clinician Baseline** (`100.0%` plurality, `51.8%` mean correct vote share).
   - Ranks models `#1–#5` by `(Overall Accuracy %, Macro F1 %, -Mean Latency)` and exports `results/summary.csv` and `results/report.md`.

3. **Percentage Visualizations (`plot_results.py`):**
   - `results/charts/01_model_ranking_overall.png`: Ranked horizontal bar chart of Overall Accuracy (%) & Macro F1 (%).
   - `results/charts/02_source_and_human_comparison.png`: Grouped dataset comparison with the **NEJM Human Clinician Crowd (`100.0%`)** reference line.
   - `results/charts/03_domain_breakdown_heatmap.png`: Domain-by-model accuracy (%) heatmap across body systems.
   - `results/charts/04_accuracy_vs_vram_speed.png`: Accuracy (%) vs. Latency (s/img) scatter plot scaled by Peak Memory (GB).

### Likely Professor Questions for Member 1
- *Q: Why use Macro F1 alongside Overall Accuracy?*  
  **A:** Macro F1 weights every answer class equally so a model that collapses onto one option letter (or struggles on harder classes) is penalized—for example, `MedGemma 1.5` leads in Macro F1 (`33.1%`) even though `MedVLM-R1` edges it slightly on Overall Accuracy (`34.7%` vs `34.0%`).
- *Q: How did you calculate the Human Clinician baseline?*  
  **A:** Using NEJM's real physician poll distributions (`option_A_votes`..`option_E_votes`), we measure both Plurality Accuracy (`100.0%`) and the average % of doctors voting for the right diagnosis (`51.8%`).

---

# Part 2 — Member 2: Model Engineering, Hardware Optimization, Parsing & Live Demo

**Recommended Presenter:** *Member 2 (Teammate)*  
**Core Files Owned:**
- [`src/models.py`](src/models.py) — Dual-backend (Apple MLX 4-bit + CUDA 4-bit NF4 + BiomedCLIP) adapters & memory lifecycle
- [`src/parser.py`](src/parser.py) — Reasoning (`<think>` / Gemma `<unused94>thought`) extraction & 4-tier answer parser
- [`run_benchmark.py`](run_benchmark.py) — Resumable CLI orchestrator (`--smoke-test` / `--full`, auto-detecting Apple Silicon MLX vs CUDA)
- [`src/demo_app.py`](src/demo_app.py) — Interactive Gradio 3-case live demo & static backup cards (`results/demos/`)
- [`notebooks/medical_vlm_benchmark_colab.ipynb`](notebooks/medical_vlm_benchmark_colab.ipynb) & `.agents/plugins/colab/`

### What Member 2 Built & How to Explain It to the Professor

1. **Comparing 5 Distinct Medical VLM Paradigms (`models.py`):**
   - **MedVLM-R1 (2B):** Qwen2-VL-2B fine-tuned via GRPO Reinforcement Learning (converted to 4-bit MLX locally).
   - **MedGemma 1.5 (4B):** Google's Gemma-3 + MedSigLIP encoder (`mlx-community/medgemma-1.5-4b-it-4bit`).
   - **Qwen2.5-VL-MedSFT (3B):** Qwen2.5-VL-3B trained via medical supervised fine-tuning.
   - **LLaVA-Med v1.5 (7B):** Mistral-7B aligned on PubMed Central biomedical figures (`LlavaForConditionalGeneration`).
   - **BiomedCLIP (200M):** Contrastive ViT-B/16 + PubMedBERT baseline trained on 15M biomedical image-text pairs.

2. **Real-World Hardware Debugging & Dual-Backend Engineering (`models.py` + `run_benchmark.py`):**
   - **Challenge 1 (8 GB M1 RAM & Colab T4 Limits):** Loaded models **sequentially** in 4-bit quantization (`0.74–4.45 GB` peak memory) with explicit cleanup (`gc.collect()`, `mx.clear_cache()`, `torch.cuda.empty_cache()`), plus `.jsonl` checkpointing so interrupted runs resume without losing completed cases.
   - **Challenge 2 (T4 `float16` Overflow on Gemma 3):** Observed empirically that `MedGemma 1.5` overflows to `NaN` in `float16` on Turing T4 GPUs (producing blank outputs) and runs slowly in `fp32`, so we implemented Apple Silicon **4-bit MLX** execution where `MedGemma 1.5` runs cleanly in `3.77 GB` memory at `11.81s/img`.
   - **Challenge 3 (`MedVLM-R1` Config Validation):** Built `_prepare_model_source()` to create a zero-copy symlinked snapshot patching `use_cache: null` and `flash_attention_2` for `transformers 5.x` compatibility and 4-bit MLX conversion.

3. **Zero-Shot Prompt Standardization & 4-Tier Answer Parser (`parser.py`):**
   - Built a 4-stage parser: (1) `<answer>X</answer>` extraction, (2) explicit prefix matching (`Answer: C`), (3) verbatim option-text matching (prioritized ahead of loose letters so English `"a"` never hijacks option `A`), and (4) uppercase-only standalone letter fallback.

4. **2–3 Live Presentation Demos (`demo_app.py`):**
   - Built an interactive **Gradio web app** (`launch_gradio_demo()`) preloaded with **3 distinct clinical cases** across modalities plus custom image upload.
   - Generated 3 static backup PNG cards (`results/demos/demo_case_1.png`–`3.png`) ready for slides even without live GPU access.

### Likely Professor Questions for Member 2
- *Q: How did you run 2B–4B medical VLMs on an 8 GB MacBook Air M1?*  
  **A:** Using Apple's native **MLX (`mlx-vlm`)** with **4-bit quantization** (cutting weights ~3–4×), resizing images to `512×512`, and loading one model at a time with `mx.clear_cache()` between runs—keeping peak memory between `0.74 GB` and `3.77 GB`.
- *Q: Why did `LLaVA-Med (7B)` score lower (`28.7%`) than smaller `2B–4B` models?*  
  **A:** Despite having the most parameters (`7B`), `LLaVA-Med v1.5` uses an older Mistral-7B + CLIP alignment pipeline and followed the structured `<answer>` format only `16.7%` of the time, whereas `MedVLM-R1 (2B)` and `MedGemma 1.5 (4B)` use stronger modern vision encoders and instruction/RL tuning.

---

## Live Presentation Script (5-Minute Flow)

1. **Member 1 (1.5 min):** Explain the problem, why we rejected `MedMNIST` to avoid leakage, the 150-image `MedXpertQA` + `NEJM` dataset, and walk through the 4 percentage ranking charts (`results/charts/01`–`04`).
2. **Member 2 (2 min):** Explain the 5 model architectures, the real hardware trade-offs (Colab T4 vs Apple M1 MLX 4-bit), and the 4-tier answer parser.
3. **Joint Live Demo (1.5 min):** Open the Gradio app (`.venv/bin/python -m src.demo_app --manifest data/benchmark_150/manifest.csv --launch-gradio`) or show `results/demos/demo_case_1..3.png` to compare model predictions across **3 multi-modality clinical cases**.
