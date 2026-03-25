# F-09: Claude Code Privacy Hooks

**Priority:** P1
**Requirements:** R-DMO-01, R-DMO-02

## What This Feature Does

Demonstrates the detector as a set of Claude Code hooks that intercept data at three points in a coding session. The hooks can be backed by either the Python hook script or the compiled Rust binary (see F-11).

- **Before a prompt is sent** — scans what the user is about to send for quasi-identifier PII.
- **Before a tool runs** — scans tool arguments (e.g., file paths) for PII exposure risk.
- **After a tool runs** — scans tool output (e.g., file contents returned to Claude) for PII.

When PII is detected, the hook blocks the action and surfaces the findings to Claude, which then presents them conversationally to the user. The user can choose to redact the sensitive content, revise their prompt, or continue as-is.

## Why It Exists

This is the proof-of-concept demonstration of the product's core value: a lightweight, fully offline privacy gate that prevents accidental PII disclosure to external LLM APIs. By integrating with Claude Code — a real developer tool — the demo shows the product working in a practical context rather than in isolation.

The conversational presentation is important: rather than forcing users to learn special commands or read raw JSON output, Claude explains what was found and offers options. This makes privacy protection feel like a natural part of the workflow.

## Design Tradeoffs

- The default posture is **block and inform** — actions are stopped when PII is detected, not just flagged. This is the safer default for a privacy tool, but it means false positives will interrupt the user's workflow. The tradeoff is acceptable because quasi-identifier PII is genuinely sensitive and the cost of disclosure is high.
- The hook covers three data flow points (prompt, tool input, tool output) to provide comprehensive coverage. This adds configuration complexity but ensures PII is not leaked through any channel.
- This feature targets Claude Code specifically. It does not provide hooks for other editors or tools.

## What This Feature Does Not Provide

- Integration with IDEs, editors, or tools other than Claude Code.
- A background scanning or monitoring mode.
- User preferences for sensitivity thresholds or per-category overrides.
- Tier 1 detection within the Tier 2 hook (Tier 1 runs as a separate hook — see F-10).

## Acceptance Criteria

### AC-01: Hook blocks on PII detection
**GIVEN** a Claude Code session with the privacy hooks installed
**WHEN** the user submits a prompt, a tool is about to run, or a tool has returned output that contains quasi-identifier PII
**THEN** the hook blocks the action and provides the detection findings
*(Traces to R-DMO-01)*

### AC-02: Conversational presentation
**GIVEN** a hook has blocked an action due to detected PII
**WHEN** Claude receives the findings
**THEN** Claude presents the findings to the user in natural language and offers options: redact, revise, or continue
*(Traces to R-DMO-02)*

### AC-03: Clean text passes through
**GIVEN** a Claude Code session with the privacy hooks installed
**WHEN** the user submits a prompt, a tool runs, or a tool returns output that contains no quasi-identifier PII
**THEN** the action proceeds without interruption

### AC-04: Fully offline
**GIVEN** the hooks are running
**WHEN** PII detection is performed
**THEN** no data leaves the local machine during inference
