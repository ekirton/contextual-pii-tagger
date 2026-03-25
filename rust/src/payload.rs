//! Payload extraction from Claude Code hook events.
//!
//! Spec: specifications/rust-binary.md §5

use serde_json::Value;

/// Extract scannable text from a hook event payload.
///
/// Returns an empty string if the expected field is missing or empty.
pub fn extract_text(hook_type: &str, payload: &Value) -> String {
    match hook_type {
        "user_prompt" => payload
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),

        "pre_tool_use" => match payload.get("tool_input") {
            Some(v) if v.is_string() => v.as_str().unwrap_or("").to_string(),
            Some(v) if !v.is_null() => serde_json::to_string(v).unwrap_or_default(),
            _ => String::new(),
        },

        "post_tool_use" => payload
            .get("tool_output")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),

        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn user_prompt_extracts_query() {
        let payload = json!({"query": "hello world"});
        assert_eq!(extract_text("user_prompt", &payload), "hello world");
    }

    #[test]
    fn user_prompt_missing_query() {
        let payload = json!({});
        assert_eq!(extract_text("user_prompt", &payload), "");
    }

    #[test]
    fn pre_tool_use_string_input() {
        let payload = json!({"tool_input": "some file path"});
        assert_eq!(extract_text("pre_tool_use", &payload), "some file path");
    }

    #[test]
    fn pre_tool_use_dict_input_serialized() {
        let payload = json!({"tool_input": {"file": "test.py", "line": 42}});
        let result = extract_text("pre_tool_use", &payload);
        assert!(result.contains("test.py"));
        assert!(result.contains("42"));
    }

    #[test]
    fn pre_tool_use_missing_input() {
        let payload = json!({});
        assert_eq!(extract_text("pre_tool_use", &payload), "");
    }

    #[test]
    fn post_tool_use_extracts_output() {
        let payload = json!({"tool_output": "file contents here"});
        assert_eq!(extract_text("post_tool_use", &payload), "file contents here");
    }

    #[test]
    fn post_tool_use_missing_output() {
        let payload = json!({});
        assert_eq!(extract_text("post_tool_use", &payload), "");
    }

    #[test]
    fn unknown_hook_type_returns_empty() {
        let payload = json!({"query": "test"});
        assert_eq!(extract_text("unknown", &payload), "");
    }

    #[test]
    fn empty_string_values_return_empty() {
        let payload = json!({"query": ""});
        assert_eq!(extract_text("user_prompt", &payload), "");
    }

    #[test]
    fn null_values_return_empty() {
        let payload = json!({"query": null});
        assert_eq!(extract_text("user_prompt", &payload), "");
    }
}
