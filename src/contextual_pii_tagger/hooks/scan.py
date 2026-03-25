"""Claude Code hook script for contextual PII scanning.

Spec: specifications/hook-script.md §1-5
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import IO

from contextual_pii_tagger.detector import PIIDetector

_DEFAULT_MODEL_PATH = str(Path.home() / ".cache" / "contextual-pii-tagger")


def extract_text(hook_type: str, payload: dict) -> str:
    """Extract scannable text from the hook event payload.

    Spec: §2.1
    """
    if hook_type == "user_prompt":
        text = payload.get("query", "")
    elif hook_type == "pre_tool_use":
        tool_input = payload.get("tool_input")
        if tool_input is None:
            return ""
        if isinstance(tool_input, str):
            text = tool_input
        else:
            text = json.dumps(tool_input)
    elif hook_type == "post_tool_use":
        text = payload.get("tool_output", "")
    else:
        return ""

    if not isinstance(text, str):
        return ""
    return text


def scan(hook_type: str, stdin: IO[str]) -> tuple[int, str]:
    """Run the hook scan pipeline.

    Returns (exit_code, stderr_content).
    Spec: §1, §3
    """
    try:
        raw = stdin.read()
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return 1, str(exc)

    text = extract_text(hook_type, payload)
    if not text:
        return 0, ""

    try:
        model_path = os.environ.get("PII_MODEL_PATH", _DEFAULT_MODEL_PATH)
        detector = PIIDetector.from_pretrained(model_path)
        result = detector.detect(text)
    except Exception as exc:
        return 1, str(exc)

    if not result.labels:
        return 0, ""

    return 2, json.dumps(result.to_dict(), separators=(",", ":"))
