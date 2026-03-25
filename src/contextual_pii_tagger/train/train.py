"""QLoRA fine-tuning loop for contextual PII tagger.

Spec: specifications/training.md §3, §5

Heavy ML dependencies (torch, peft, trl, datasets) are
imported lazily inside functions so the module can be imported and tested
without those packages installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load and return training configuration from a YAML file.

    RAISES:
        FileNotFoundError – if *config_path* does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def train(config: dict, dataset: list[dict[str, str]]) -> None:
    """Run QLoRA fine-tuning.

    REQUIRES:
        - *config* contains all hyperparameters per training.md §3.
        - *dataset* is a list of dicts with ``"text"`` keys.

    ENSURES:
        - LoRA adapter weights saved to ``config["output_dir"]``.
        - Base model weights not modified.
        - Training uses bf16 precision.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from trl import SFTConfig, SFTTrainer

    output_dir = config["output_dir"]
    use_bnb = torch.cuda.is_available() and config.get("load_in_4bit", False)

    # ── Load tokenizer ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load base model ───────────────────────────────────────────────
    model_kwargs: dict = {}
    if use_bnb:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, config["bnb_4bit_compute_dtype"]),
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    elif torch.backends.mps.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "mps"
        logger.info("Using MPS backend (no 4-bit quantization)")
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        logger.info("Using CPU (no 4-bit quantization)")

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        **model_kwargs,
    )

    # ── Apply LoRA adapter ───────────────────────────────────────────
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training arguments ───────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        max_length=config["max_seq_length"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    # ── Convert dataset to HuggingFace Dataset ───────────────────────
    hf_dataset = Dataset.from_list(dataset)

    # ── Train ────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    # ── Save adapter ─────────────────────────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Adapter saved to %s", output_dir)
