"""Merge LoRA adapter into base model for standalone inference.

Spec: specifications/training.md §4

Heavy ML dependencies imported lazily so the module can be imported
and tested without them installed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def merge_adapter(
    base_model_path: str, adapter_path: str, output_path: str
) -> None:
    """Merge LoRA adapter into the base model and save as standalone.

    REQUIRES:
        - *base_model_path* points to the Phi-3 Mini base model.
        - *adapter_path* points to trained LoRA adapter weights.
        - *output_path* is a writable directory.

    ENSURES:
        - Writes merged model to *output_path* loadable without peft.
        - Includes tokenizer files.
        - Inference results identical to base+adapter.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

    # Load and apply adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge adapter weights into base model
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_model.save_pretrained(output_path)

    # Save tokenizer alongside model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    logger.info("Merged model saved to %s", output_path)
