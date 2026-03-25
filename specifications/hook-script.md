# Hook Script Specification

**Architecture:** [hook-integration.md](../doc/architecture/hook-integration.md)
**Features:** F-09

> **Note:** This specification covers the Python hook script. For the Rust binary alternative, see [rust-binary.md](rust-binary.md).

---

## 1. Entry Point

```
scan(hook_type: string, stdin: stream) -> (exit_code: integer, stderr: string)
```

The hook script is invoked as a subprocess by Claude Code. It reads the event payload from stdin, runs detection, and communicates the result via exit code and stderr.

**REQUIRES:**
- `hook_type` is one of: `user_prompt`, `pre_tool_use`, `post_tool_use`.
- `stdin` contains a JSON payload from Claude Code (format varies by hook type).
- The model is available at the path specified by `PII_MODEL_PATH` environment variable, or at `~/.cache/contextual-pii-tagger/` if unset.

**ENSURES:**
- Exactly one of the three exit codes is returned (see Section 3).
- On exit code 2, stderr contains a JSON-serialized DetectionResult.
- On exit code 0, stderr is empty.
- On exit code 1, stderr contains an error message string (not JSON).
- No output is written to stdout.

---

## 2. Payload Extraction

### 2.1 extract_text

```
extract_text(hook_type: string, payload: dict) -> string
```

Extracts the text content to scan from the hook event payload.

**REQUIRES:**
- `hook_type` is a valid hook type string.
- `payload` is a parsed JSON dictionary from stdin.

**ENSURES:**
- For `user_prompt`: returns the value of the prompt text field from the payload.
- For `pre_tool_use`: serializes tool arguments to a single string (JSON-encoded if structured).
- For `post_tool_use`: extracts the tool output text from the payload.
- Returns a non-empty string if the payload contains usable content.

**RAISES:**
- Returns an empty string if the payload is missing the expected fields or the content is empty. The caller treats an empty string as "nothing to scan" (exit 0).

---

## 3. Exit Code Contract

| Exit Code | Condition | stderr Content |
|-----------|-----------|----------------|
| 0 | No PII detected, or no text to scan | Empty |
| 2 | PII detected (labels non-empty) | JSON-serialized DetectionResult |
| 1 | Any error during processing | Error message string |

### 3.1 Exit 0 — Pass

**REQUIRES:**
- `extract_text` returned an empty string, OR
- `PIIDetector.detect(text)` returned a DetectionResult with empty labels.

**ENSURES:**
- Exit code is 0.
- stderr is empty.
- Claude Code proceeds with the action.

### 3.2 Exit 2 — Block

**REQUIRES:**
- `PIIDetector.detect(text)` returned a DetectionResult with non-empty labels.

**ENSURES:**
- Exit code is 2.
- stderr contains the DetectionResult serialized as JSON (compact format, single line).
- The JSON output satisfies all DetectionResult invariants from entities.md Section 3.
- Claude Code blocks the action and reads stderr.

### 3.3 Exit 1 — Error (Fail-Open)

**REQUIRES:**
- An exception occurred during payload parsing, model loading, or inference.

**ENSURES:**
- Exit code is 1.
- stderr contains a human-readable error message.
- Claude Code proceeds with the action (fail-open behavior).

---

## 4. Model Loading

**REQUIRES:**
- The environment variable `PII_MODEL_PATH` is set to a valid model directory, OR the default path `~/.cache/contextual-pii-tagger/` contains valid model weights.

**ENSURES:**
- The model is loaded from `PII_MODEL_PATH` if set, otherwise from the default path.
- If the model cannot be loaded, exit code 1 is returned.

**MAINTAINS:**
- The model is loaded fresh on each hook invocation (hooks are stateless processes).
- No model state is cached between invocations.

---

## 5. Timeout

**MAINTAINS:**
- The hook must complete within the timeout configured in Claude Code settings (default: 10,000 ms).
- If the hook exceeds the timeout, Claude Code terminates the process and proceeds with the action (same effect as exit code 1).
- The script does not implement its own timeout logic; timeout enforcement is Claude Code's responsibility.
