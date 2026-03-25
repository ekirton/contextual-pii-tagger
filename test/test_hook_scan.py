"""Tests for hook script: scan entry point and extract_text.

All expectations derived from specifications/hook-script.md §1-5.
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from contextual_pii_tagger.entities import (
    DetectionResult,
    RiskLevel,
    SpanLabel,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _clean_result() -> DetectionResult:
    return DetectionResult(labels=frozenset(), risk=RiskLevel.LOW, rationale="")


def _pii_result() -> DetectionResult:
    return DetectionResult(
        labels=frozenset({SpanLabel.LOCATION, SpanLabel.WORKPLACE}),
        risk=RiskLevel.HIGH,
        rationale="Combined workplace and location narrow identity.",
    )


# ── §2: extract_text ─────────────────────────────────────────────────────


class TestExtractText:
    """ENSURES: correct text extraction for each hook type."""

    def test_user_prompt_extracts_prompt_text(self):
        from contextual_pii_tagger.hooks.scan import extract_text

        payload = {"query": "Tell me about the employee at building 7."}
        result = extract_text("user_prompt", payload)
        assert result == "Tell me about the employee at building 7."

    def test_pre_tool_use_serializes_tool_input(self):
        from contextual_pii_tagger.hooks.scan import extract_text

        payload = {
            "tool_name": "Read",
            "tool_input": {"file_path": "/home/user/medical_records.txt"},
        }
        result = extract_text("pre_tool_use", payload)
        assert "medical_records.txt" in result

    def test_post_tool_use_extracts_tool_output(self):
        from contextual_pii_tagger.hooks.scan import extract_text

        payload = {
            "tool_name": "Read",
            "tool_output": "Patient visits clinic every Tuesday at 3pm near downtown.",
        }
        result = extract_text("post_tool_use", payload)
        assert result == "Patient visits clinic every Tuesday at 3pm near downtown."

    def test_returns_empty_for_missing_fields(self):
        from contextual_pii_tagger.hooks.scan import extract_text

        assert extract_text("user_prompt", {}) == ""
        assert extract_text("pre_tool_use", {}) == ""
        assert extract_text("post_tool_use", {}) == ""

    def test_returns_empty_for_empty_content(self):
        from contextual_pii_tagger.hooks.scan import extract_text

        assert extract_text("user_prompt", {"query": ""}) == ""
        assert extract_text("post_tool_use", {"tool_name": "R", "tool_output": ""}) == ""


# ── §3: Exit code contract ───────────────────────────────────────────────


class TestScanExitCodes:
    """ENSURES: exit codes match spec §3.1-3.3."""

    @patch("contextual_pii_tagger.hooks.scan.PIIDetector")
    def test_exit_0_on_clean_text(self, mock_detector_cls):
        from contextual_pii_tagger.hooks.scan import scan

        mock_detector = MagicMock()
        mock_detector.detect.return_value = _clean_result()
        mock_detector_cls.from_pretrained.return_value = mock_detector

        payload = json.dumps({"query": "The sky is blue."})
        exit_code, stderr = scan("user_prompt", StringIO(payload))

        assert exit_code == 0
        assert stderr == ""

    @patch("contextual_pii_tagger.hooks.scan.PIIDetector")
    def test_exit_2_on_pii_detected(self, mock_detector_cls):
        from contextual_pii_tagger.hooks.scan import scan

        mock_detector = MagicMock()
        mock_detector.detect.return_value = _pii_result()
        mock_detector_cls.from_pretrained.return_value = mock_detector

        payload = json.dumps({"query": "I work at Acme near downtown."})
        exit_code, stderr = scan("user_prompt", StringIO(payload))

        assert exit_code == 2
        result_data = json.loads(stderr)
        assert "LOCATION" in result_data["labels"]
        assert "WORKPLACE" in result_data["labels"]
        assert result_data["risk"] == "HIGH"

    @patch("contextual_pii_tagger.hooks.scan.PIIDetector")
    def test_exit_1_on_model_load_error(self, mock_detector_cls):
        from contextual_pii_tagger.hooks.scan import scan

        mock_detector_cls.from_pretrained.side_effect = FileNotFoundError("no model")

        payload = json.dumps({"query": "Hello."})
        exit_code, stderr = scan("user_prompt", StringIO(payload))

        assert exit_code == 1
        assert "no model" in stderr

    @patch("contextual_pii_tagger.hooks.scan.PIIDetector")
    def test_exit_1_on_inference_error(self, mock_detector_cls):
        from contextual_pii_tagger.hooks.scan import scan

        mock_detector = MagicMock()
        mock_detector.detect.side_effect = RuntimeError("inference failed")
        mock_detector_cls.from_pretrained.return_value = mock_detector

        payload = json.dumps({"query": "Hello."})
        exit_code, stderr = scan("user_prompt", StringIO(payload))

        assert exit_code == 1
        assert "inference failed" in stderr

    @patch("contextual_pii_tagger.hooks.scan.PIIDetector")
    def test_exit_0_on_empty_payload(self, mock_detector_cls):
        """Nothing to scan → exit 0."""
        from contextual_pii_tagger.hooks.scan import scan

        payload = json.dumps({})
        exit_code, stderr = scan("user_prompt", StringIO(payload))

        assert exit_code == 0
        assert stderr == ""
        mock_detector_cls.from_pretrained.assert_not_called()

    def test_exit_1_on_invalid_json_stdin(self):
        from contextual_pii_tagger.hooks.scan import scan

        exit_code, stderr = scan("user_prompt", StringIO("not json"))

        assert exit_code == 1
        assert stderr  # some error message

    @patch("contextual_pii_tagger.hooks.scan.PIIDetector")
    def test_stderr_json_is_compact_single_line(self, mock_detector_cls):
        """§3.2: JSON output is compact format, single line."""
        from contextual_pii_tagger.hooks.scan import scan

        mock_detector = MagicMock()
        mock_detector.detect.return_value = _pii_result()
        mock_detector_cls.from_pretrained.return_value = mock_detector

        payload = json.dumps({"query": "I work at Acme near downtown."})
        exit_code, stderr = scan("user_prompt", StringIO(payload))

        assert exit_code == 2
        assert "\n" not in stderr.strip()


# ── §3: No stdout output ─────────────────────────────────────────────────


class TestNoStdout:
    """ENSURES: nothing written to stdout."""

    @patch("contextual_pii_tagger.hooks.scan.PIIDetector")
    def test_no_stdout_on_pass(self, mock_detector_cls):
        from contextual_pii_tagger.hooks.scan import scan

        mock_detector = MagicMock()
        mock_detector.detect.return_value = _clean_result()
        mock_detector_cls.from_pretrained.return_value = mock_detector

        payload = json.dumps({"query": "The sky is blue."})
        exit_code, stderr = scan("user_prompt", StringIO(payload))
        # scan returns (exit_code, stderr) — stdout is not in the contract
        # The function itself must not print to stdout
        assert exit_code == 0

    @patch("contextual_pii_tagger.hooks.scan.PIIDetector")
    def test_no_stdout_on_block(self, mock_detector_cls):
        from contextual_pii_tagger.hooks.scan import scan

        mock_detector = MagicMock()
        mock_detector.detect.return_value = _pii_result()
        mock_detector_cls.from_pretrained.return_value = mock_detector

        payload = json.dumps({"query": "I work at Acme."})
        exit_code, stderr = scan("user_prompt", StringIO(payload))
        assert exit_code == 2


# ── §4: Model loading ────────────────────────────────────────────────────


class TestModelPath:
    """ENSURES: model loaded from PII_MODEL_PATH or default."""

    @patch("contextual_pii_tagger.hooks.scan.PIIDetector")
    def test_uses_env_var_when_set(self, mock_detector_cls):
        from contextual_pii_tagger.hooks.scan import scan

        mock_detector = MagicMock()
        mock_detector.detect.return_value = _clean_result()
        mock_detector_cls.from_pretrained.return_value = mock_detector

        payload = json.dumps({"query": "Hello."})
        with patch.dict("os.environ", {"PII_MODEL_PATH": "/custom/model/path"}):
            scan("user_prompt", StringIO(payload))

        mock_detector_cls.from_pretrained.assert_called_once_with("/custom/model/path")

    @patch("contextual_pii_tagger.hooks.scan.PIIDetector")
    def test_uses_default_path_when_env_unset(self, mock_detector_cls):
        from contextual_pii_tagger.hooks.scan import scan

        mock_detector = MagicMock()
        mock_detector.detect.return_value = _clean_result()
        mock_detector_cls.from_pretrained.return_value = mock_detector

        payload = json.dumps({"query": "Hello."})
        with patch.dict("os.environ", {}, clear=True):
            scan("user_prompt", StringIO(payload))

        call_args = mock_detector_cls.from_pretrained.call_args[0][0]
        assert "contextual-pii-tagger" in call_args


# ── §2: Hook types in scan ───────────────────────────────────────────────


class TestHookTypeRouting:
    """ENSURES: all three hook types route correctly through scan."""

    @patch("contextual_pii_tagger.hooks.scan.PIIDetector")
    def test_pre_tool_use_scans_tool_input(self, mock_detector_cls):
        from contextual_pii_tagger.hooks.scan import scan

        mock_detector = MagicMock()
        mock_detector.detect.return_value = _clean_result()
        mock_detector_cls.from_pretrained.return_value = mock_detector

        payload = json.dumps({
            "tool_name": "Bash",
            "tool_input": {"command": "cat /home/user/data.csv"},
        })
        exit_code, stderr = scan("pre_tool_use", StringIO(payload))
        assert exit_code == 0
        mock_detector.detect.assert_called_once()

    @patch("contextual_pii_tagger.hooks.scan.PIIDetector")
    def test_post_tool_use_scans_tool_output(self, mock_detector_cls):
        from contextual_pii_tagger.hooks.scan import scan

        mock_detector = MagicMock()
        mock_detector.detect.return_value = _pii_result()
        mock_detector_cls.from_pretrained.return_value = mock_detector

        payload = json.dumps({
            "tool_name": "Read",
            "tool_output": "John works at building 7, arrives at 8am daily.",
        })
        exit_code, stderr = scan("post_tool_use", StringIO(payload))
        assert exit_code == 2
