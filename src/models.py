"""Unified 4-bit Colab T4 model registry and sequential inference adapters."""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class ModelSpec:
    """Metadata and loader family for a medical Vision-Language Model."""

    key: str
    display_name: str
    hf_id: str
    family: str  # "gemma3" | "llava" | "qwen2_vl" | "qwen2_5_vl"
    params_label: str
    gated: bool
    uses_think_tags: bool = False
    max_new_tokens: int = 96


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "medgemma-1.5-4b": ModelSpec(
        key="medgemma-1.5-4b",
        display_name="MedGemma 1.5 (4B)",
        hf_id="google/medgemma-1.5-4b-it",
        family="gemma3",
        params_label="4B",
        gated=True,
        max_new_tokens=48,
    ),
    "llava-med-7b": ModelSpec(
        key="llava-med-7b",
        display_name="LLaVA-Med v1.5 (7B)",
        hf_id="chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        family="llava",
        params_label="7B",
        gated=False,
        max_new_tokens=96,
    ),
    "medvlm-r1-2b": ModelSpec(
        key="medvlm-r1-2b",
        display_name="MedVLM-R1 (2B)",
        hf_id="JZPeterPan/MedVLM-R1",
        family="qwen2_vl",
        params_label="2B",
        gated=False,
        uses_think_tags=True,
        max_new_tokens=220,
    ),
    "qwen2.5-vl-3b-medsft": ModelSpec(
        key="qwen2.5-vl-3b-medsft",
        display_name="Qwen2.5-VL-MedSFT (3B)",
        hf_id="CAIR-HKISI/Qwen2.5-VL-3B-MedSFT",
        family="qwen2_5_vl",
        params_label="3B",
        gated=False,
        max_new_tokens=96,
    ),
    "biomedclip": ModelSpec(
        key="biomedclip",
        display_name="BiomedCLIP (200M)",
        hf_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        family="biomedclip",
        params_label="200M",
        gated=False,
        max_new_tokens=32,
    ),
    "medgemma-1.5-4b-mlx": ModelSpec(
        key="medgemma-1.5-4b-mlx",
        display_name="MedGemma 1.5 (4B, MLX 4-bit)",
        hf_id="mlx-community/medgemma-1.5-4b-it-4bit",
        family="mlx_vlm",
        params_label="4B-4bit",
        gated=False,
        max_new_tokens=64,
    ),
    "medvlm-r1-2b-mlx": ModelSpec(
        key="medvlm-r1-2b-mlx",
        display_name="MedVLM-R1 (2B, MLX 4-bit)",
        hf_id="JZPeterPan/MedVLM-R1",
        family="mlx_vlm",
        params_label="2B-4bit",
        gated=False,
        uses_think_tags=True,
        max_new_tokens=200,
    ),
    "medgemma-1.0-4b-mlx": ModelSpec(
        key="medgemma-1.0-4b-mlx",
        display_name="MedGemma 1.0 (4B, MLX 4-bit)",
        hf_id="mlx-community/medgemma-4b-it-4bit",
        family="mlx_vlm",
        params_label="4B-4bit",
        gated=False,
        max_new_tokens=64,
    ),
}

CUDA_DEFAULT_MODELS = [
    "medgemma-1.5-4b",
    "llava-med-7b",
    "medvlm-r1-2b",
    "qwen2.5-vl-3b-medsft",
    "biomedclip",
]

MLX_DEFAULT_MODELS = [
    "medgemma-1.5-4b-mlx",
    "medvlm-r1-2b-mlx",
    "biomedclip",
]


def build_clinical_prompt(
    question: str,
    options: dict[str, str],
    uses_think_tags: bool = False,
) -> str:
    """Create a standardized zero-shot multiple-choice prompt for fair comparison."""
    opts_text = "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))
    if uses_think_tags:
        instruction = (
            "Analyze the medical image and clinical question carefully. "
            "Reason step-by-step inside <think>...</think> tags, then provide ONLY the "
            "single best option letter inside <answer>...</answer> tags (e.g., <answer>B</answer>)."
        )
    else:
        instruction = (
            "Analyze the medical image and clinical question carefully. "
            "Output ONLY the single best option letter inside <answer>...</answer> tags first "
            "(e.g., <answer>B</answer>), followed by a concise 1-sentence justification."
        )
    return f"{instruction}\n\nQuestion:\n{question}\n\nOptions:\n{opts_text}"


def _select_compute_dtype(family: str) -> Any:
    """Pick a numerically safe compute dtype for the current GPU and model family.

    Gemma 3 / MedGemma overflows to NaN in float16 on Turing (T4) GPUs; use
    bfloat16 when natively supported or float32 otherwise.
    """
    import torch

    if family == "gemma3":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    return torch.float16


def _build_bnb_config(compute_dtype: Any) -> Any:
    """Create 4-bit NF4 BitsAndBytesConfig using the family-safe compute dtype."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _sanitize_config(config: Any) -> Any:
    """Ensure config fields like use_cache are valid booleans instead of None."""
    if getattr(config, "use_cache", None) is None:
        config.use_cache = True
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "use_cache", None) is None:
        text_cfg.use_cache = True
    return config


def _patch_json_file(src: Path, dest: Path) -> None:
    """Copy a Hugging Face JSON config and fix null use_cache / flash_attention_2 fields."""
    data = json.loads(src.read_text(encoding="utf-8"))
    if data.get("use_cache") is None:
        data["use_cache"] = True
    if isinstance(data.get("text_config"), dict) and data["text_config"].get("use_cache") is None:
        data["text_config"]["use_cache"] = True
    if data.get("attn_implementation") == "flash_attention_2":
        data.pop("attn_implementation", None)
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_patched_snapshot_dir(hf_id: str, hf_token: str | None) -> str:
    """Create a symlinked local mirror of a HF snapshot with patched config JSONs."""
    from huggingface_hub import snapshot_download

    snap_dir = Path(snapshot_download(repo_id=hf_id, token=hf_token))
    patch_dir = Path(tempfile.mkdtemp(prefix="hf_patched_"))
    for entry in snap_dir.iterdir():
        target = patch_dir / entry.name
        if entry.name in ("config.json", "generation_config.json") and entry.is_file():
            _patch_json_file(entry, target)
        else:
            target.symlink_to(entry)
    return str(patch_dir)


def _prepare_model_source(hf_id: str, hf_token: str | None) -> tuple[str, str | None]:
    """Return (load_path_or_id, temp_dir_to_cleanup) handling invalid HF config JSONs."""
    from transformers import AutoConfig

    proc_kwargs = {"token": hf_token} if hf_token else {}
    try:
        AutoConfig.from_pretrained(hf_id, **proc_kwargs)
        return hf_id, None
    except Exception as exc:
        if "use_cache" not in str(exc) and "StrictDataclassFieldValidationError" not in type(exc).__name__:
            raise
        patched_dir = _build_patched_snapshot_dir(hf_id, hf_token)
        return patched_dir, patched_dir


def _load_model_with_attn_fallback(model_cls: Any, hf_id: str, **kwargs: Any) -> Any:
    """Load HF model preferring sdpa and falling back to eager for T4 compatibility."""
    for attn_impl in ("sdpa", "eager"):
        try:
            return model_cls.from_pretrained(hf_id, attn_implementation=attn_impl, **kwargs)
        except (ValueError, NotImplementedError):
            continue
    return model_cls.from_pretrained(hf_id, **kwargs)


class MedicalVLMRunner:
    """Sequential 4-bit medical VLM loader and inference wrapper (CUDA or Apple MLX)."""

    def __init__(self, spec: ModelSpec, hf_token: str | None = None) -> None:
        self.spec = spec
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or None
        self.model: Any = None
        self.processor: Any = None
        self._mlx_config: Any = None
        self._mlx_peak_gb: float = 0.0
        self._temp_dir: str | None = None

    def _resolve_mlx_path(self) -> str:
        """Return an MLX-ready HF repo or convert a PyTorch HF repo to 4-bit MLX locally."""
        if self.spec.hf_id.startswith("mlx-community/"):
            return self.spec.hf_id

        local_mlx_dir = Path("models") / f"{self.spec.key}-4bit"
        if (local_mlx_dir / "config.json").exists():
            return str(local_mlx_dir)

        from mlx_vlm.convert import convert

        src_path, temp_dir = _prepare_model_source(self.spec.hf_id, self.hf_token)
        try:
            local_mlx_dir.parent.mkdir(parents=True, exist_ok=True)
            convert(
                hf_path=src_path,
                mlx_path=str(local_mlx_dir),
                quantize=True,
                q_bits=4,
                q_group_size=64,
            )
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
        return str(local_mlx_dir)

    def load(self) -> None:
        """Load 4-bit quantized weights and processor into GPU/Metal memory."""
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForImageTextToText,
            AutoProcessor,
            LlavaForConditionalGeneration,
            Qwen2VLForConditionalGeneration,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if self.spec.family == "mlx_vlm":
            import mlx.core as mx
            from mlx_vlm import load as mlx_load
            from mlx_vlm.utils import load_config as mlx_load_config

            mx.reset_peak_memory()
            mlx_path = self._resolve_mlx_path()
            self.model, self.processor = mlx_load(mlx_path)
            self._mlx_config = mlx_load_config(mlx_path)
            self._mlx_peak_gb = round(mx.get_peak_memory() / (1024**3), 2)
            return

        if self.spec.family == "biomedclip":
            from open_clip import create_model_from_pretrained, get_tokenizer

            hub_id = f"hf-hub:{self.spec.hf_id}"
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            model, preprocess = create_model_from_pretrained(hub_id)
            model.to(device)
            model.eval()
            self.model = model
            self.processor = (preprocess, get_tokenizer(hub_id), device)
            return

        compute_dtype = _select_compute_dtype(self.spec.family)
        bnb_cfg = _build_bnb_config(compute_dtype)
        model_source, self._temp_dir = _prepare_model_source(self.spec.hf_id, self.hf_token)

        use_token = self.hf_token if model_source == self.spec.hf_id else None
        proc_kwargs = {"token": use_token} if use_token else {}

        config = AutoConfig.from_pretrained(model_source, **proc_kwargs)
        config = _sanitize_config(config)

        common_kwargs: dict[str, Any] = {
            "config": config,
            "quantization_config": bnb_cfg,
            "torch_dtype": compute_dtype,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if use_token:
            common_kwargs["token"] = use_token

        self.processor = AutoProcessor.from_pretrained(model_source, **proc_kwargs)

        if self.spec.family == "gemma3":
            self.model = _load_model_with_attn_fallback(
                AutoModelForImageTextToText, model_source, **common_kwargs
            )
        elif self.spec.family == "llava":
            self.model = _load_model_with_attn_fallback(
                LlavaForConditionalGeneration, model_source, **common_kwargs
            )
        elif self.spec.family == "qwen2_vl":
            self.model = _load_model_with_attn_fallback(
                Qwen2VLForConditionalGeneration, model_source, **common_kwargs
            )
        elif self.spec.family == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration

            self.model = _load_model_with_attn_fallback(
                Qwen2_5_VLForConditionalGeneration, model_source, **common_kwargs
            )
        else:
            raise ValueError(f"Unsupported model family: {self.spec.family}")

        if getattr(self.model, "generation_config", None) is not None:
            if getattr(self.model.generation_config, "use_cache", None) is None:
                self.model.generation_config.use_cache = True

        self.model.eval()

    def predict(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 320,
        image_path: str | Path | None = None,
    ) -> tuple[str, float]:
        """Run single-image inference and return (raw_text, latency_seconds)."""
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError(f"Model {self.spec.key} is not loaded.")

        if self.spec.family == "mlx_vlm":
            import mlx.core as mx
            from mlx_vlm import generate as mlx_generate
            from mlx_vlm.prompt_utils import apply_chat_template as mlx_apply_chat_template

            temp_img_path: Path | None = None
            if image_path is not None and Path(image_path).exists():
                img_ref = str(image_path)
            else:
                fd, tmp_name = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                temp_img_path = Path(tmp_name)
                image.convert("RGB").save(temp_img_path, format="PNG")
                img_ref = str(temp_img_path)

            try:
                formatted_prompt = mlx_apply_chat_template(
                    self.processor, self._mlx_config, prompt, num_images=1
                )
                start = time.perf_counter()
                result = mlx_generate(
                    self.model,
                    self.processor,
                    formatted_prompt,
                    image=[img_ref],
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                    resize_shape=(512, 512),
                    verbose=False,
                )
                latency = time.perf_counter() - start
                peak = float(getattr(result, "peak_memory", 0.0) or 0.0)
                if peak == 0.0:
                    peak = mx.get_peak_memory() / (1024**3)
                self._mlx_peak_gb = max(self._mlx_peak_gb, round(peak, 2))
                return str(result.text).strip(), latency
            finally:
                if temp_img_path is not None:
                    temp_img_path.unlink(missing_ok=True)

        rgb_img = image.convert("RGB")
        start = time.perf_counter()

        with torch.inference_mode():
            inputs = self._prepare_inputs(rgb_img, prompt)
            inputs = {
                k: (v.to(self.model.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }
            prompt_len = inputs["input_ids"].shape[-1]
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            generated_ids = output_ids[0][prompt_len:]
            decoded = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        latency = time.perf_counter() - start
        return decoded, latency

    def predict_mc(
        self,
        image: Image.Image,
        question: str,
        options: dict[str, str],
        image_path: str | Path | None = None,
    ) -> tuple[str, float]:
        """Run unified multiple-choice inference for generative VLMs or BiomedCLIP."""
        import torch

        if self.spec.family != "biomedclip":
            prompt = build_clinical_prompt(
                question=question,
                options=options,
                uses_think_tags=self.spec.uses_think_tags,
            )
            return self.predict(
                image,
                prompt,
                max_new_tokens=self.spec.max_new_tokens,
                image_path=image_path,
            )

        if self.model is None or self.processor is None:
            raise RuntimeError(f"Model {self.spec.key} is not loaded.")

        preprocess, tokenizer, device = self.processor
        sorted_opts = sorted(options.items())
        labels = [k for k, _ in sorted_opts]
        texts = [f"this medical image shows {v}" for _, v in sorted_opts]

        start = time.perf_counter()
        img_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
        tok_tensor = tokenizer(texts, context_length=256).to(device)

        with torch.inference_mode():
            img_feat = self.model.encode_image(img_tensor)
            txt_feat = self.model.encode_text(tok_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            logits = (img_feat @ txt_feat.T) * self.model.logit_scale.exp()
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        latency = time.perf_counter() - start
        ranked = sorted(zip(labels, probs), key=lambda kv: kv[1], reverse=True)
        best_letter = ranked[0][0]
        scores_str = ", ".join(f"{lbl}: {p * 100:.1f}%" for lbl, p in ranked)
        return f"<answer>{best_letter}</answer> (BiomedCLIP similarity: {scores_str})", latency

    def predict_freeform(
        self,
        image: Image.Image,
        custom_question: str = "",
        image_path: str | Path | None = None,
    ) -> tuple[str, float]:
        """Analyze a medical image and return a plain-English clinical summary."""
        import torch

        if self.spec.family == "biomedclip":
            if self.model is None or self.processor is None:
                raise RuntimeError(f"Model {self.spec.key} is not loaded.")
            concepts = [
                "a normal chest X-ray without acute cardiopulmonary disease",
                "a chest X-ray showing pneumonia, consolidation, or lung nodule",
                "an electrocardiogram (ECG) tracing",
                "a brain or spine MRI scan",
                "a CT scan showing internal organ pathology",
                "a dermatology photograph of a skin lesion or rash",
                "an ultrasound scan",
                "a histopathology or microscopy tissue slide",
                "a bone X-ray showing a fracture or skeletal abnormality",
            ]
            preprocess, tokenizer, device = self.processor
            texts = [f"this medical image shows {c}" for c in concepts]
            start = time.perf_counter()
            img_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
            tok_tensor = tokenizer(texts, context_length=256).to(device)
            with torch.inference_mode():
                img_feat = self.model.encode_image(img_tensor)
                txt_feat = self.model.encode_text(tok_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                probs = ((img_feat @ txt_feat.T) * self.model.logit_scale.exp()).softmax(dim=-1).cpu().numpy()[0]
            latency = time.perf_counter() - start
            ranked = sorted(zip(concepts, probs), key=lambda kv: kv[1], reverse=True)
            top1, p1 = ranked[0]
            top2, p2 = ranked[1]
            sentence = (
                f"This medical image most closely matches {top1} ({p1 * 100:.1f}% confidence), "
                f"followed by {top2} ({p2 * 100:.1f}%)."
            )
            return sentence, latency

        extra = f" Focus specifically on: {custom_question.strip()}" if custom_question.strip() else ""
        prompt = (
            "You are a medical imaging AI assistant. Examine this medical image and write 2 to 3 "
            "clear, plain-English sentences describing: (1) the imaging modality and anatomical region, "
            "(2) the key visual findings, and (3) the most likely clinical impression."
            f"{extra} Do not output XML tags, bullet points, or multiple-choice letters."
        )
        return self.predict(
            image,
            prompt,
            max_new_tokens=max(self.spec.max_new_tokens, 140),
            image_path=image_path,
        )

    def _prepare_inputs(self, rgb_img: Image.Image, prompt: str) -> dict[str, Any]:
        """Format multimodal chat inputs according to model family."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if self.spec.family == "gemma3":
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

        if self.spec.family in ("qwen2_vl", "qwen2_5_vl"):
            from qwen_vl_utils import process_vision_info

            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            return self.processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        # LLaVA family (with safe fallback if tokenizer lacks chat_template)
        try:
            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            chat_text = f"USER: <image>\n{prompt}\nASSISTANT:"
        if "<image>" not in chat_text:
            chat_text = f"USER: <image>\n{prompt}\nASSISTANT:"
        return self.processor(text=chat_text, images=rgb_img, return_tensors="pt")

    def peak_vram_gb(self) -> float:
        """Return peak allocated GPU/Metal memory in GB since load()."""
        import torch

        if self.spec.family == "mlx_vlm":
            return self._mlx_peak_gb
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024**3), 2)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            try:
                return round(torch.mps.current_allocated_memory() / (1024**3), 2)
            except Exception:
                return 0.0
        return 0.0

    def unload(self) -> None:
        """Explicitly release GPU/Metal memory and system RAM before the next model loads."""
        import torch

        self.model = None
        self.processor = None
        self._mlx_config = None
        if self._temp_dir:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        gc.collect()
        if self.spec.family == "mlx_vlm":
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
