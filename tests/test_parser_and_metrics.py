"""Unit and integration tests for parsing, ranking, percentage metrics, charts, and demo cards."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from src.demo_app import generate_static_demo_cards
from src.evaluate import evaluate_all_models
from src.parser import parse_model_response
from src.plot_results import generate_all_charts
from src.prepare_dataset import _save_pil_image


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

    # Ensure lowercase English word 'a' never beats verbatim option text
    p4 = parse_model_response(
        "The diagnosis is likely Pearly penile papules (PPP). Consult a healthcare professional.",
        {"A": "Condyloma", "D": "Pearly penile papules"},
    )
    assert p4.predicted_letter == "D"
    assert p4.format_compliant is False


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


def test_save_pil_image_from_zip_filename(tmp_path: Path) -> None:
    zip_path = tmp_path / "images.zip"
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(10, 120, 200)).save(buf, format="JPEG")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("images/MM-0-a.jpeg", buf.getvalue())

    out_png = tmp_path / "out.png"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_index = {Path(name).name: name for name in zf.namelist()}
        _save_pil_image("MM-0-a.jpeg", out_png, zip_handle=zf, zip_index=zip_index)

    assert out_png.exists()
    with Image.open(out_png) as saved:
        assert saved.mode == "RGB"
        assert saved.size == (32, 32)


def test_run_benchmark_skips_failing_gated_model(tmp_path: Path, monkeypatch) -> None:
    import run_benchmark as rb

    img_dir = tmp_path / "data" / "images"
    img_dir.mkdir(parents=True)
    img_path = img_dir / "c1.png"
    Image.new("RGB", (32, 32), color=(20, 40, 60)).save(img_path)

    manifest_path = tmp_path / "data" / "manifest.csv"
    pd.DataFrame(
        [
            {
                "case_id": "c1",
                "source": "MedXpertQA-MM (2025)",
                "domain": "Skeletal",
                "medical_task": "Diagnosis",
                "case_date": "2025",
                "image_path": str(img_path),
                "question": "Q?",
                "options_json": json.dumps({"A": "Opt1", "B": "Opt2"}),
                "correct_letter": "A",
                "human_vote_pct": None,
                "human_plurality_correct": None,
            }
        ]
    ).to_csv(manifest_path, index=False)

    class FakeRunner:
        def __init__(self, spec, hf_token=None):
            self.spec = spec

        def load(self):
            if self.spec.key == "medgemma-1.5-4b":
                raise OSError("403 Client Error: gated repo")

        def predict_mc(self, image, question, options, image_path=None):
            return "<answer>A</answer>", 0.5

        def peak_vram_gb(self):
            return 2.0

        def unload(self):
            pass

    monkeypatch.setattr(rb, "MedicalVLMRunner", FakeRunner)

    results_dir = tmp_path / "results"
    rb.run_benchmark(
        data_dir=tmp_path / "data",
        results_dir=results_dir,
        model_keys=["medgemma-1.5-4b", "medvlm-r1-2b"],
        smoke_test=True,
        hf_token="dummy-token",
    )

    summary = pd.read_csv(results_dir / "summary.csv")
    assert list(summary["model_key"]) == ["medvlm-r1-2b"]
    assert summary.iloc[0]["overall_accuracy_pct"] == 100.0


def test_sanitize_config_and_gemma3_dtype() -> None:
    from types import SimpleNamespace
    from src.models import _sanitize_config

    cfg = SimpleNamespace(use_cache=None, text_config=SimpleNamespace(use_cache=None))
    fixed = _sanitize_config(cfg)
    assert fixed.use_cache is True
    assert fixed.text_config.use_cache is True

    gemma_raw = "<unused94>thought\nPneumonia infiltrate visible.<unused95><answer>B</answer>"
    parsed = parse_model_response(gemma_raw, {"A": "Normal", "B": "Pneumonia"})
    assert parsed.predicted_letter == "B"
    assert "Pneumonia infiltrate" in parsed.reasoning


def test_patch_json_file_fixes_null_use_cache(tmp_path: Path) -> None:
    from src.models import _patch_json_file

    src = tmp_path / "config.json"
    dest = tmp_path / "patched_config.json"
    src.write_text(
        json.dumps(
            {
                "model_type": "qwen2_vl",
                "use_cache": None,
                "attn_implementation": "flash_attention_2",
                "text_config": {"use_cache": None},
            }
        ),
        encoding="utf-8",
    )
    _patch_json_file(src, dest)
    out = json.loads(dest.read_text(encoding="utf-8"))
    assert out["use_cache"] is True
    assert out["text_config"]["use_cache"] is True
    assert "attn_implementation" not in out


def test_format_english_summary_strips_xml_tags() -> None:
    from src.demo_app import format_english_summary

    raw = "<think>Normal lung fields without consolidation.</think><answer>The chest X-ray appears normal with clear lung fields.</answer>"
    assert format_english_summary(raw) == "The chest X-ray appears normal with clear lung fields."

    raw_letter_only = "<think>Left lower lobe infiltrate visible on chest radiograph.</think><answer>B</answer>"
    assert format_english_summary(raw_letter_only) == "Left lower lobe infiltrate visible on chest radiograph."
