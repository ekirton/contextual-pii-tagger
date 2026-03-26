"""Tests for shared Ollama utilities.

Covers §3.2 (LLM Call Resilience) from specifications/data-generation.md.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from contextual_pii_tagger.data.cli_utils import call_ollama, strip_code_fences


class TestStripCodeFences:
    """Tests for markdown code fence stripping."""

    def test_strips_json_fence(self):
        text = '```json\n[{"a": 1}]\n```'
        assert strip_code_fences(text) == '[{"a": 1}]'

    def test_strips_bare_fence(self):
        text = '```\n[1, 2, 3]\n```'
        assert strip_code_fences(text) == '[1, 2, 3]'

    def test_passes_through_plain_json(self):
        text = '[{"a": 1}]'
        assert strip_code_fences(text) == '[{"a": 1}]'

    def test_passes_through_empty_string(self):
        assert strip_code_fences("") == ""


def _mock_ollama_response(content: list | dict) -> MagicMock:
    """Build a mock urllib response for Ollama /api/chat."""
    body = json.dumps({
        "message": {"role": "assistant", "content": json.dumps(content)},
    }).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestCallOllama:
    """Tests for the Ollama HTTP caller."""

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_parses_json_array(self, mock_urlopen):
        mock_urlopen.return_value = _mock_ollama_response([{"text": "hello"}])
        result = call_ollama("prompt", "qwen2.5:3b")
        assert result == [{"text": "hello"}]

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_extracts_array_from_dict(self, mock_urlopen):
        """Ollama with format:json may return {"examples": [...]}."""
        mock_urlopen.return_value = _mock_ollama_response(
            {"examples": [{"text": "hello"}]}
        )
        result = call_ollama("prompt", "qwen2.5:3b")
        assert result == [{"text": "hello"}]

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_strips_code_fences(self, mock_urlopen):
        body = json.dumps({
            "message": {
                "role": "assistant",
                "content": '```json\n[{"text": "hello"}]\n```',
            },
        }).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp
        result = call_ollama("prompt", "qwen2.5:3b")
        assert result == [{"text": "hello"}]

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_empty_response_raises(self, mock_urlopen):
        """Empty content must raise RuntimeError (§3.2)."""
        body = json.dumps({
            "message": {"role": "assistant", "content": ""},
        }).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp
        with pytest.raises(RuntimeError, match="[Ee]mpty"):
            call_ollama("prompt", "qwen2.5:3b")

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_connection_error_raises_runtime_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with pytest.raises(RuntimeError, match="Ollama"):
            call_ollama("prompt", "qwen2.5:3b")

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_sends_correct_model(self, mock_urlopen):
        mock_urlopen.return_value = _mock_ollama_response([{"text": "hi"}])
        call_ollama("test prompt", "qwen2.5:14b")
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        payload = json.loads(req.data)
        assert payload["model"] == "qwen2.5:14b"

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_sends_json_format(self, mock_urlopen):
        mock_urlopen.return_value = _mock_ollama_response([{"text": "hi"}])
        call_ollama("test prompt", "qwen2.5:3b")
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        payload = json.loads(req.data)
        assert payload["format"] == "json"

    @patch("contextual_pii_tagger.data.cli_utils.urllib.request.urlopen")
    def test_dict_without_array_raises(self, mock_urlopen):
        mock_urlopen.return_value = _mock_ollama_response({"status": "ok"})
        with pytest.raises(RuntimeError, match="no array"):
            call_ollama("prompt", "qwen2.5:3b")
