# Claude Code Hook Integration

**Features:** F-09 (Claude Code Privacy Hooks), F-10 (Tier 1 Detection), F-11 (Scanner Binary)
**Requirements:** R-DMO-01, R-DMO-02, R-T1-01, R-BIN-01, R-BIN-02

---

## 1. Overview

The hook integration demonstrates the detector as a privacy gate in Claude Code sessions. Three hooks intercept data at different points in the session, pass the content to the Detection Interface for local inference, and block the action if quasi-identifier PII is detected.

```
┌──────────────────────────────────────────────────────────────┐
│                      Claude Code Session                     │
│                                                              │
│  User Prompt ──> [UserPromptSubmit Hook] ──> Send to API     │
│                                                              │
│  Tool Call   ──> [PreToolUse Hook]       ──> Execute Tool    │
│                                                              │
│  Tool Output ──> [PostToolUse Hook]      ──> Return to Claude│
│                                                              │
└──────────────────────────────────────────────────────────────┘
                          │
                          v
              ┌───────────────────────┐
              │   Hook Script         │
              │                       │
              │  Extract text ──>     │
              │  PIIDetector.detect() │
              │  ──> Evaluate result  │
              │  ──> Exit 0 or 2     │
              └───────────────────────┘
```

## 2. Hook Points

| Hook | Event | Content Scanned | Risk Addressed |
|------|-------|----------------|----------------|
| `UserPromptSubmit` | User submits a prompt | The full prompt text | User pastes PII directly into a prompt |
| `PreToolUse` | Claude is about to invoke a tool | Tool arguments (e.g., file paths, search queries) | Claude is about to access a resource containing PII |
| `PostToolUse` | A tool has returned output | Tool output (e.g., file contents, command output) | Data returned from a tool contains PII |

## 3. Hook Script Flow

Each hook runs the same decision logic:

1. **Extract text.** Parse the hook event payload to extract the text content to scan. The payload format differs by hook type:
   - `UserPromptSubmit`: The prompt text is the payload.
   - `PreToolUse`: Tool arguments are serialized to a text string.
   - `PostToolUse`: Tool output is extracted from the payload.

2. **Detect.** Call `PIIDetector.detect(text)` to get a DetectionResult.

3. **Evaluate.** Check the DetectionResult:
   - If `labels` is empty (no PII detected): exit with code 0 (pass).
   - If `labels` is non-empty (PII detected): exit with code 2 (block) and write the DetectionResult as JSON to stderr.

4. **Claude receives findings.** When a hook exits with code 2, Claude Code reads the stderr output and treats it as context. Claude then presents the findings to the user conversationally.

## 4. Exit Code Contract

| Exit Code | Meaning | Effect |
|-----------|---------|--------|
| 0 | No PII detected | Action proceeds normally |
| 2 | PII detected | Action is blocked; stderr contains DetectionResult as JSON |
| 1 | Hook error | Action proceeds (fail-open); error is logged |

The fail-open behavior on exit code 1 ensures that hook errors (model loading failure, parse error, etc.) do not permanently block the user's workflow. This is appropriate for a proof of concept but should be reconsidered for production use.

## 5. Hook Configuration

The hooks are configured in the Claude Code settings file (`.claude/settings.json` or `.claude/settings.local.json`).

### 5.1 Python hooks (single tier)

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python -m contextual_pii_tagger.hooks.scan --hook user_prompt",
            "timeout": 10000
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python -m contextual_pii_tagger.hooks.scan --hook pre_tool_use",
            "timeout": 10000
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python -m contextual_pii_tagger.hooks.scan --hook post_tool_use",
            "timeout": 10000
          }
        ]
      }
    ]
  }
}
```

### 5.2 Rust binary hooks (both tiers)

Two hooks per event — Tier 1 runs first (fast, pattern-based), Tier 2 runs second (model inference). If Tier 1 blocks, Tier 2 is skipped.

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          { "type": "command", "command": "pii-scanner --tier1 --hook user_prompt" },
          { "type": "command", "command": "pii-scanner --tier2 --hook user_prompt", "timeout": 10000 }
        ]
      }
    ],
    "PreToolUse": [
      {
        "hooks": [
          { "type": "command", "command": "pii-scanner --tier1 --hook pre_tool_use" },
          { "type": "command", "command": "pii-scanner --tier2 --hook pre_tool_use", "timeout": 10000 }
        ]
      }
    ],
    "PostToolUse": [
      {
        "hooks": [
          { "type": "command", "command": "pii-scanner --tier1 --hook post_tool_use" },
          { "type": "command", "command": "pii-scanner --tier2 --hook post_tool_use", "timeout": 10000 }
        ]
      }
    ]
  }
}
```

The `timeout` is set to 10 seconds for Tier 2 to accommodate CPU inference latency. Tier 1 does not need an explicit timeout (completes in under 10ms). Both read the event payload from stdin and write findings to stderr.

## 6. Conversational Presentation

When a hook blocks an action, Claude receives the DetectionResult JSON on stderr. Claude is expected to:

1. Parse the findings and present them in natural language (e.g., "I noticed your prompt contains information that could identify someone — specifically, a workplace mention and a routine that together narrow down who you might be.")
2. Explain the risk level and rationale.
3. Offer the user three options:
   - **Redact:** Remove the detected quasi-identifiers before proceeding.
   - **Revise:** Let the user rewrite their prompt with the findings in mind.
   - **Continue:** Proceed despite the warning.

This conversational flow is handled by Claude's existing instruction-following capability — no special prompting or configuration is required beyond providing the findings as context.

## 7. Model Loading

The hook script must load the model on each invocation (since hooks are stateless processes). The model path is configured via an environment variable (`PII_MODEL_PATH`) defaulting to `~/.cache/contextual-pii-tagger/`.

### 7.1 Python hooks

- The merged model variant is used (no `peft` dependency, faster loading).
- Cold-start latency is several seconds (Python interpreter + torch import + model load).

### 7.2 Rust binary hooks

- The Tier 2 model is loaded in GGUF format via llama.cpp with memory-mapped I/O (~200ms load time).
- `PII_MODEL_PATH` points to a `.gguf` file (e.g., `~/.cache/contextual-pii-tagger/model.gguf`).
- Tier 1 requires no model loading (pattern-based detection via redact-core).
