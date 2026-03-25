"""Tests for the training loop.

All expectations derived from specifications/training.md §3, §5.

ML dependencies (peft, trl, datasets, bitsandbytes) are mocked via
sys.modules since they are imported lazily inside train().
"""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import yaml

from contextual_pii_tagger.train.train import load_config


# ── Helpers ──────────────────────────────────────────────────────────────


def _sample_config(tmp_path: Path) -> dict:
    """Return a config dict matching spec hyperparameters."""
    return {
        "base_model": "microsoft/Phi-3-mini-4k-instruct",
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "max_seq_length": 1024,
        "output_dir": str(tmp_path / "output"),
    }


def _write_config(tmp_path: Path) -> Path:
    cfg = _sample_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg))
    return config_path


class _MockLoraConfig:
    """Captures LoRA config arguments for assertions."""

    def __init__(self, *, r, lora_alpha, lora_dropout, target_modules, task_type):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.task_type = task_type


@contextmanager
def _ml_mocks():
    """Context manager that installs mock ML modules and yields mocks dict."""
    mocks = {}

    # Build mock modules ──────────────────────────────────────────────

    # peft
    peft_mod = ModuleType("peft")
    peft_mod.__spec__ = MagicMock()
    peft_mod.__version__ = "0.14.0"
    peft_mod.LoraConfig = _MockLoraConfig
    peft_mod.TaskType = MagicMock()
    peft_mod.TaskType.CAUSAL_LM = "CAUSAL_LM"
    peft_mod.PeftModel = MagicMock()
    peft_mod.PeftMixedModel = MagicMock()
    mock_get_peft = MagicMock()
    mock_peft_model = MagicMock()
    mock_get_peft.return_value = mock_peft_model
    peft_mod.get_peft_model = mock_get_peft
    mocks["get_peft_model"] = mock_get_peft
    mocks["peft_model"] = mock_peft_model

    # trl
    trl_mod = ModuleType("trl")
    trl_mod.__spec__ = MagicMock()
    mock_trainer = MagicMock()
    mock_sft_cls = MagicMock(return_value=mock_trainer)
    trl_mod.SFTTrainer = mock_sft_cls
    mocks["SFTTrainer"] = mock_sft_cls
    mocks["trainer_instance"] = mock_trainer

    # datasets
    datasets_mod = ModuleType("datasets")
    datasets_mod.__spec__ = MagicMock()
    datasets_mod.Dataset = MagicMock()
    datasets_mod.Dataset.from_list = MagicMock(return_value=MagicMock())

    # Mock torch (provide bfloat16 attr)
    mock_torch = MagicMock()
    mock_torch.bfloat16 = "bfloat16"

    # Mock transformers
    transformers_mod = ModuleType("transformers")
    transformers_mod.__spec__ = MagicMock()
    mock_model_cls = MagicMock()
    mock_model_cls.from_pretrained = MagicMock(return_value=MagicMock())
    transformers_mod.AutoModelForCausalLM = mock_model_cls
    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tok.eos_token = "<|end|>"
    mock_tok_cls = MagicMock()
    mock_tok_cls.from_pretrained = MagicMock(return_value=mock_tok)
    transformers_mod.AutoTokenizer = mock_tok_cls
    transformers_mod.BitsAndBytesConfig = MagicMock()
    transformers_mod.TrainingArguments = MagicMock()

    mocks["AutoModelForCausalLM"] = mock_model_cls
    mocks["AutoTokenizer"] = mock_tok_cls

    # Install ─────────────────────────────────────────────────────────
    fake_modules = {
        "peft": peft_mod,
        "trl": trl_mod,
        "datasets": datasets_mod,
        "torch": mock_torch,
        "transformers": transformers_mod,
    }
    saved = {}
    for name, mod in fake_modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    # Force reimport of the train module to pick up mocked imports
    mod_name = "contextual_pii_tagger.train.train"
    saved_train = sys.modules.pop(mod_name, None)

    try:
        train_mod = importlib.import_module(mod_name)
        mocks["train_fn"] = train_mod.train
        yield mocks
    finally:
        # Restore
        if saved_train is not None:
            sys.modules[mod_name] = saved_train
        else:
            sys.modules.pop(mod_name, None)
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


# ── §Config: load_config ────────────────────────────────────────────────


class TestLoadConfig:
    """ENSURES: config.yaml parsed into dict with correct types."""

    def test_loads_valid_config(self, tmp_path):
        config_path = _write_config(tmp_path)
        cfg = load_config(str(config_path))
        assert cfg["base_model"] == "microsoft/Phi-3-mini-4k-instruct"
        assert cfg["lora_r"] == 16
        assert cfg["lora_alpha"] == 32
        assert cfg["learning_rate"] == 2e-4

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")


# ── §3: train ───────────────────────────────────────────────────────────


class TestTrain:
    """ENSURES: QLoRA fine-tuning with correct configuration.

    Tests mock the entire ML stack to verify configuration wiring.
    """

    def test_trains_with_correct_lora_config(self, tmp_path):
        cfg = _sample_config(tmp_path)
        dataset = [{"text": "sample training text"}]
        with _ml_mocks() as mocks:
            mocks["train_fn"](cfg, dataset)
            mocks["get_peft_model"].assert_called_once()
            lora_cfg = mocks["get_peft_model"].call_args[0][1]
            assert lora_cfg.r == 16
            assert lora_cfg.lora_alpha == 32
            assert lora_cfg.lora_dropout == 0.05

    def test_trainer_called(self, tmp_path):
        cfg = _sample_config(tmp_path)
        dataset = [{"text": "sample training text"}]
        with _ml_mocks() as mocks:
            mocks["train_fn"](cfg, dataset)
            mocks["trainer_instance"].train.assert_called_once()

    def test_adapter_saved_after_training(self, tmp_path):
        cfg = _sample_config(tmp_path)
        dataset = [{"text": "sample training text"}]
        with _ml_mocks() as mocks:
            mocks["train_fn"](cfg, dataset)
            mocks["peft_model"].save_pretrained.assert_called_once()
