"""Tests for merge adapter.

All expectations derived from specifications/training.md §4.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock, call

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────


@contextmanager
def _ml_mocks():
    """Context manager that installs mock ML modules and yields mocks dict."""
    mocks = {}

    # peft
    peft_mod = ModuleType("peft")
    peft_mod.__spec__ = MagicMock()
    peft_mod.__version__ = "0.14.0"
    mock_peft_model_cls = MagicMock()
    mock_merged_model = MagicMock()
    mock_peft_instance = MagicMock()
    mock_peft_instance.merge_and_unload.return_value = mock_merged_model
    mock_peft_model_cls.from_pretrained.return_value = mock_peft_instance
    peft_mod.PeftModel = mock_peft_model_cls
    peft_mod.PeftMixedModel = MagicMock()
    mocks["PeftModel"] = mock_peft_model_cls
    mocks["peft_instance"] = mock_peft_instance
    mocks["merged_model"] = mock_merged_model

    # transformers
    transformers_mod = ModuleType("transformers")
    transformers_mod.__spec__ = MagicMock()
    mock_model_cls = MagicMock()
    mock_base_model = MagicMock()
    mock_model_cls.from_pretrained.return_value = mock_base_model
    transformers_mod.AutoModelForCausalLM = mock_model_cls
    mock_tok = MagicMock()
    mock_tok_cls = MagicMock()
    mock_tok_cls.from_pretrained.return_value = mock_tok
    transformers_mod.AutoTokenizer = mock_tok_cls
    mocks["AutoModelForCausalLM"] = mock_model_cls
    mocks["AutoTokenizer"] = mock_tok_cls
    mocks["base_model"] = mock_base_model
    mocks["tokenizer"] = mock_tok

    # torch (minimal mock)
    mock_torch = MagicMock()
    mock_torch.bfloat16 = "bfloat16"

    fake_modules = {
        "peft": peft_mod,
        "transformers": transformers_mod,
        "torch": mock_torch,
    }
    saved = {}
    for name, mod in fake_modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    mod_name = "contextual_pii_tagger.train.merge"
    saved_merge = sys.modules.pop(mod_name, None)

    try:
        merge_mod = importlib.import_module(mod_name)
        mocks["merge_adapter"] = merge_mod.merge_adapter
        yield mocks
    finally:
        if saved_merge is not None:
            sys.modules[mod_name] = saved_merge
        else:
            sys.modules.pop(mod_name, None)
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


# ── §4: merge_adapter ───────────────────────────────────────────────────


class TestMergeAdapter:
    """ENSURES: merged model loadable without peft; identical inference."""

    def test_loads_base_model_and_adapter(self, tmp_path):
        output = tmp_path / "merged"
        with _ml_mocks() as mocks:
            mocks["merge_adapter"]("/base", "/adapter", str(output))
            mocks["AutoModelForCausalLM"].from_pretrained.assert_called_once()
            mocks["PeftModel"].from_pretrained.assert_called_once()

    def test_merges_and_saves(self, tmp_path):
        output = tmp_path / "merged"
        with _ml_mocks() as mocks:
            mocks["merge_adapter"]("/base", "/adapter", str(output))
            mocks["peft_instance"].merge_and_unload.assert_called_once()
            mocks["merged_model"].save_pretrained.assert_called_once_with(str(output))

    def test_saves_tokenizer(self, tmp_path):
        output = tmp_path / "merged"
        with _ml_mocks() as mocks:
            mocks["merge_adapter"]("/base", "/adapter", str(output))
            mocks["tokenizer"].save_pretrained.assert_called_once_with(str(output))
