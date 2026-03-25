//! Top-level scan dispatch.
//!
//! Spec: specifications/rust-binary.md §1

use crate::payload;
use crate::tier1;
use crate::tier2;
use serde_json::Value;
use std::path::Path;

/// Scan result: exit code + optional stderr content.
pub struct ScanOutput {
    pub exit_code: i32,
    pub stderr: String,
}

/// Run a Tier 1 scan on the hook payload.
pub fn run_tier1(hook_type: &str, payload: &Value) -> ScanOutput {
    let text = payload::extract_text(hook_type, payload);
    if text.is_empty() {
        return ScanOutput { exit_code: 0, stderr: String::new() };
    }

    match tier1::scan_tier1(&text) {
        Ok(output) => {
            if output.findings.is_empty() {
                ScanOutput { exit_code: 0, stderr: String::new() }
            } else {
                ScanOutput { exit_code: 2, stderr: output.to_json() }
            }
        }
        Err(e) => ScanOutput { exit_code: 1, stderr: e },
    }
}

/// Run a Tier 2 scan on the hook payload.
pub fn run_tier2(hook_type: &str, payload: &Value, model_path: &Path) -> ScanOutput {
    let text = payload::extract_text(hook_type, payload);
    if text.is_empty() {
        return ScanOutput { exit_code: 0, stderr: String::new() };
    }

    match tier2::scan_tier2(&text, model_path) {
        Ok(result) => {
            if result.labels.is_empty() {
                ScanOutput { exit_code: 0, stderr: String::new() }
            } else {
                ScanOutput { exit_code: 2, stderr: result.to_json() }
            }
        }
        Err(e) => ScanOutput { exit_code: 1, stderr: e },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tier1_empty_text_exits_0() {
        let payload = json!({});
        let output = run_tier1("user_prompt", &payload);
        assert_eq!(output.exit_code, 0);
        assert!(output.stderr.is_empty());
    }

    #[test]
    fn tier1_clean_text_exits_0() {
        let payload = json!({"query": "hello world"});
        let output = run_tier1("user_prompt", &payload);
        assert_eq!(output.exit_code, 0);
        assert!(output.stderr.is_empty());
    }

    #[test]
    fn tier2_empty_text_exits_0() {
        let payload = json!({});
        let output = run_tier2("user_prompt", &payload, Path::new("/nonexistent"));
        assert_eq!(output.exit_code, 0);
        assert!(output.stderr.is_empty());
    }

    #[test]
    fn tier2_with_text_and_missing_model_exits_1() {
        let payload = json!({"query": "some text to scan"});
        let output = run_tier2("user_prompt", &payload, Path::new("/nonexistent/model.gguf"));
        assert_eq!(output.exit_code, 1); // fail-open on model load error
        assert!(!output.stderr.is_empty());
    }
}
