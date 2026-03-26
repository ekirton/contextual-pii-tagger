"""PIIDetector: primary public API for contextual PII detection.

Spec: specifications/detection-interface.md §1, §3
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from contextual_pii_tagger.entities import DetectionResult
from contextual_pii_tagger.output_parser import parse_output
from contextual_pii_tagger.prompt import assemble_prompt

# Lazy import — only needed when loading adapter models.
try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None  # type: ignore[assignment,misc]


class PIIDetector:
    """Load a fine-tuned model and run quasi-identifier detection."""

    def __init__(self, model: object, tokenizer: object) -> None:
        self._model = model
        self._tokenizer = tokenizer

    # ── §1.1: from_pretrained ────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, model_path: str) -> PIIDetector:
        """Load model and tokenizer from *model_path*.

        Supports local directories (merged or base+adapter) and
        HuggingFace model IDs.

        RAISES:
            FileNotFoundError – local path does not exist.
            ValueError – incompatible architecture.
        """
        path = Path(model_path)

        # A HuggingFace ID looks like "username/model-name" (exactly one slash,
        # no leading slash).  Anything else is treated as a local path.
        looks_like_hf_id = (
            "/" in model_path
            and not model_path.startswith("/")
            and model_path.count("/") == 1
        )
        is_local = not looks_like_hf_id

        if is_local and not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 4-bit quantization on CUDA; native dtype elsewhere (MPS/CPU)
        load_kwargs: dict = {"device_map": "auto", "torch_dtype": "auto"}
        if torch.cuda.is_available():
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        # Detect adapter vs. merged weights
        has_adapter = is_local and (path / "adapter_config.json").exists()

        if has_adapter:
            if PeftModel is None:
                raise ImportError(
                    "peft is required for adapter models: pip install peft"
                )
            # adapter_config.json contains the base model reference
            import json

            adapter_cfg = json.loads((path / "adapter_config.json").read_text())
            base_model_name = adapter_cfg.get(
                "base_model_name_or_path", "microsoft/Phi-3-mini-4k-instruct"
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, **load_kwargs
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, **load_kwargs
            )

        model.eval()
        device = next(model.parameters()).device
        print(f"Model loaded on device: {device}")
        return cls(model, tokenizer)

    # ── §1.2: detect ─────────────────────────────────────────────────────

    def detect(self, text: str) -> DetectionResult:
        """Classify quasi-identifier categories in *text*.

        RAISES:
            ValueError – if *text* is empty.
        """
        if not text:
            raise ValueError("text must be non-empty")

        prompt_tokens = assemble_prompt(text, self._tokenizer)
        raw_output = _generate(prompt_tokens, self._model, self._tokenizer)
        return parse_output(raw_output)

    def predict(self, text: str) -> DetectionResult:
        """Alias for :meth:`detect` satisfying the ``Predictor`` protocol."""
        return self.detect(text)



# ── §3: Model inference ─────────────────────────────────────────────────


def _generate(prompt_tokens: list[int], model: object, tokenizer: object) -> str:
    """Run greedy decoding on *prompt_tokens* and return decoded completion."""
    input_ids = torch.tensor([prompt_tokens], device=model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            temperature=1.0,  # greedy (do_sample=False overrides)
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Strip the prompt tokens from the output
    completion_ids = output_ids[0][len(prompt_tokens) :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)
