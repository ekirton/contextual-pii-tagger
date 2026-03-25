"""Shared utilities for calling Ollama and parsing responses."""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL,
)


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON output."""
    m = _CODE_FENCE_RE.match(text)
    return m.group(1) if m else text


def call_ollama(
    prompt: str,
    model: str,
    base_url: str = _DEFAULT_OLLAMA_BASE_URL,
    timeout: int = 300,
) -> list[Any]:
    """Call the Ollama chat API and return the parsed JSON array.

    Uses ``/api/chat`` with ``format: "json"`` to constrain the model
    to valid JSON output.
    """
    url = f"{base_url}/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    content = body.get("message", {}).get("content", "")

    if isinstance(content, str):
        content = strip_code_fences(content).strip()
        if not content:
            raise RuntimeError("Empty response from Ollama")
        content = json.loads(content)

    # Ollama with format: "json" may return a dict with an array inside
    if isinstance(content, dict):
        for value in content.values():
            if isinstance(value, list):
                return value
        raise RuntimeError(
            f"Ollama returned a JSON object with no array field: {list(content.keys())}"
        )

    if isinstance(content, list):
        return content

    raise RuntimeError(f"Unexpected Ollama response format: {type(content)}")
