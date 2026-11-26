# Medical Image Analyzer — Multi-Model Evaluation Report

## Ranked Leaderboard (Percentages)

| Rank | Model | Cases | Overall Acc (%) | Macro F1 (%) | MedXpertQA (%) | NEJM (%) | Format OK (%) | Latency (s/img) | Peak VRAM (GB) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| #1 | **MedVLM-R1 (2B, MLX 4-bit)** | 150 | 34.7% | 29.6% | 22.7% | 46.7% | 100.0% | 2.33s | 2.95 GB |
| #2 | **MedGemma 1.5 (4B, MLX 4-bit)** | 150 | 34.0% | 33.1% | 16.0% | 52.0% | 94.7% | 11.81s | 3.77 GB |
| #3 | **Qwen2.5-VL-MedSFT (3B)** | 150 | 30.7% | 28.7% | 20.0% | 41.3% | 100.0% | 0.67s | 3.13 GB |
| #4 | **LLaVA-Med v1.5 (7B)** | 150 | 28.7% | 25.1% | 16.0% | 41.3% | 16.7% | 2.22s | 4.45 GB |
| #5 | **BiomedCLIP (200M)** | 150 | 20.0% | 19.6% | 20.0% | 20.0% | 100.0% | 0.25s | 0.74 GB |

## NEJM Human Clinician Crowd Reference
- **Human Clinician Plurality Accuracy (NEJM Top Vote):** 100.0%
- **Human Clinician Mean Correct Vote Share:** 51.8%

## Generated Percentage Charts
- `charts/01_model_ranking_overall.png`
- `charts/02_source_and_human_comparison.png`
- `charts/03_domain_breakdown_heatmap.png`
- `charts/04_accuracy_vs_vram_speed.png`

> **Disclaimer:** For educational and research benchmarking only; not intended for clinical diagnosis.
