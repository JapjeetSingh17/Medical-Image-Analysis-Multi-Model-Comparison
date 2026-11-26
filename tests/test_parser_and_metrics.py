"""Unit and integration tests for parsing, ranking, percentage metrics, charts, and demo cards."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from src.demo_app import generate_static_demo_cards
from src.evaluate import evaluate_all_models
from src.parser import parse_model_response
from src.plot_results import generate_all_charts


def test_parse_medvlm_think_and_answer_tags() -> None:
    opts = {"A": "Normal", "B": "Pneumothorax", "C": "Pleural effusion"}
    raw = "<think>Costophrenic angle blunting is absent; pleural line visible.</think>\n<answer>B</answer>"
    parsed = parse_model_response(raw, opts)
    assert parsed.predicted_letter == "B"
    assert parsed.format_compliant is True
    assert "pleural line" in parsed.reasoning


def test_parse_plain_answer_prefix_and_fallback() -> None:
    opts = {"A": "Melanoma", "B": "Basal cell carcinoma", "C": "Seborrheic keratosis"}
    p1 = parse_model_response("Answer: C. Characteristic stuck-on appearance.", opts)
    assert p1.predicted_letter == "C"
    assert p1.format_compliant is True

    p2 = parse_model_response("The lesion is most consistent with Basal cell carcinoma.", opts)
    assert p2.predicted_letter == "B"
    assert p2.format_compliant is False

    p3 = parse_model_response("Unable to determine.", opts)
    assert p3.predicted_letter is None
    assert p3.format_compliant is False


def test_evaluate_ranking_and_generate_charts_and_demos(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (64, 64), color=(i * 50, 80, 120)).save(img_dir / f"img_{i}.png")

    opts = json.dumps({"A": "Alpha", "B": "Beta", "C": "Gamma"})
    manifest_rows = [
        {
            "case_id": "m1",
            "source": "MedXpertQA-MM (2025)",
            "domain": "Cardiovascular",
            "medical_task": "Diagnosis",
            "case_date": "2025",
            "image_path": str(img_dir / "img_0.png"),
            "question": "Q1?",
            "options_json": opts,
            "correct_letter": "A",
            "human_vote_pct": None,
            "human_plurality_correct": None,
        },
        {
            "case_id": "m2",
            "source": "MedXpertQA-MM (2025)",
            "domain": "Skeletal",
            "medical_task": "Treatment",
            "case_date": "2025",
            "image_path": str(img_dir / "img_1.png"),
            "question": "Q2?",
            "options_json": opts,
            "correct_letter": "B",
            "human_vote_pct": None,
            "human_plurality_correct": None,
        },
        {
            "case_id": "n1",
            "source": "NEJM-Image-Challenge",
            "domain": "NEJM Clinical Case",
            "medical_task": "Diagnosis",
            "case_date": "oct-05-2023",
            "image_path": str(img_dir / "img_2.png"),
            "question": "Q3?",
            "options_json": opts,
            "correct_letter": "C",
            "human_vote_pct": 64.0,
            "human_plurality_correct": 100.0,
        },
        {
            "case_id": "n2",
            "source": "NEJM-Image-Challenge",
            "domain": "NEJM Clinical Case",
            "medical_task": "Diagnosis",
            "case_date": "nov-02-2023",
            "image_path": str(img_dir / "img_3.png"),
            "question": "Q4?",
            "options_json": opts,
            "correct_letter": "A",
            "human_vote_pct": 30.0,
            "human_plurality_correct": 0.0,
        },
    ]
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    raw_dir = tmp_path / "results" / "raw"
    raw_dir.mkdir(parents=True)

    # Model A gets 4/4 (100%), Model B gets 2/4 (50%)
    model_a_preds = [
        ("m1", "A", 1.2),
        ("m2", "B", 1.1),
        ("n1", "C", 1.3),
        ("n2", "A", 1.0),
    ]
    model_b_preds = [
        ("m1", "A", 2.0),
        ("m2", "C", 2.1),
        ("n1", "B", 2.2),
        ("n2", "A", 1.9),
    ]

    for key, name, rows in [
        ("model-a", "Model Alpha (4B)", model_a_preds),
        ("model-b", "Model Beta (2B)", model_b_preds),
    ]:
        lines = [
            json.dumps(
                {
                    "case_id": cid,
                    "model_key": key,
                    "model_name": name,
                    "predicted_letter": pred,
                    "format_compliant": True,
                    "reasoning": "Sample clinical reasoning.",
                    "raw_response": f"<answer>{pred}</answer>",
                    "latency_sec": lat,
                    "peak_vram_gb": 3.4 if key == "model-a" else 2.0,
                }
            )
            for cid, pred, lat in rows
        ]
        (raw_dir / f"{key}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    results_dir = tmp_path / "results"
    summary_df = evaluate_all_models(manifest_path, raw_dir, results_dir)

    assert list(summary_df["model_key"]) == ["model-a", "model-b"]
    assert list(summary_df["rank"]) == [1, 2]
    assert summary_df.iloc[0]["overall_accuracy_pct"] == 100.0
    assert summary_df.iloc[1]["overall_accuracy_pct"] == 50.0
    assert (results_dir / "report.md").exists()

    charts = generate_all_charts(results_dir)
    assert len(charts) == 4
    for chart in charts:
        assert chart.exists() and chart.stat().st_size > 1000

    cards = generate_static_demo_cards(manifest_path, raw_dir, results_dir / "demos", count=3)
    assert len(cards) == 3
    for card in cards:
        assert card.exists() and card.stat().st_size > 1000
