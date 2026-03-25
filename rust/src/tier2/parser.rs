//! Output parser for Tier 2 model completions.
//!
//! Total function — never panics, always returns a valid DetectionResult.
//!
//! Spec: specifications/rust-binary.md §3.3, specifications/output-parser.md

use crate::entities::{DetectionResult, RiskLevel, SpanLabel};
use std::collections::BTreeSet;

/// Parse raw model output into a DetectionResult.
///
/// This is a total function: it never panics and always returns a valid result.
pub fn parse_output(raw: &str) -> DetectionResult {
    let json_value = extract_json(raw);

    match json_value {
        Some(obj) => build_result(&obj),
        None => DetectionResult::empty(),
    }
}

/// Try to extract a JSON object from raw text using multiple strategies.
fn extract_json(raw: &str) -> Option<serde_json::Value> {
    let trimmed = raw.trim();

    // Strategy 1: Direct parse
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if v.is_object() {
            return Some(v);
        }
    }

    // Strategy 2: Markdown code fences
    if let Some(content) = extract_from_code_fence(trimmed) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
            if v.is_object() {
                return Some(v);
            }
        }
    }

    // Strategy 3: Find embedded JSON object
    if let Some(content) = extract_embedded_json(trimmed) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
            if v.is_object() {
                return Some(v);
            }
        }
    }

    // Strategy 4: Repair and retry
    let repaired = repair_json(trimmed);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&repaired) {
        if v.is_object() {
            return Some(v);
        }
    }

    // Also try repair on extracted content
    if let Some(content) = extract_embedded_json(trimmed) {
        let repaired = repair_json(&content);
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&repaired) {
            if v.is_object() {
                return Some(v);
            }
        }
    }

    None
}

/// Extract content from markdown code fences (```json ... ``` or ``` ... ```).
fn extract_from_code_fence(s: &str) -> Option<String> {
    let start_markers = ["```json", "```"];
    for marker in start_markers {
        if let Some(start_idx) = s.find(marker) {
            let content_start = start_idx + marker.len();
            if let Some(end_idx) = s[content_start..].find("```") {
                return Some(s[content_start..content_start + end_idx].trim().to_string());
            }
        }
    }
    None
}

/// Find the first { ... } substring that might be JSON.
fn extract_embedded_json(s: &str) -> Option<String> {
    let start = s.find('{')?;
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in s[start..].char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..start + i + 1].to_string());
                }
            }
            _ => {}
        }
    }

    // If unclosed, return from { to end (repair will handle it)
    if depth > 0 {
        return Some(s[start..].to_string());
    }

    None
}

/// Attempt to repair common JSON malformations.
fn repair_json(s: &str) -> String {
    let mut result = s.to_string();

    // Single quotes → double quotes (simple heuristic)
    result = result.replace('\'', "\"");

    // Trailing commas before } or ]
    let re_trailing_obj = regex_lite::Regex::new(r",\s*\}").unwrap();
    result = re_trailing_obj.replace_all(&result, "}").to_string();
    let re_trailing_arr = regex_lite::Regex::new(r",\s*\]").unwrap();
    result = re_trailing_arr.replace_all(&result, "]").to_string();

    // Close unclosed braces/brackets
    let open_braces = result.chars().filter(|&c| c == '{').count();
    let close_braces = result.chars().filter(|&c| c == '}').count();
    for _ in 0..(open_braces.saturating_sub(close_braces)) {
        result.push('}');
    }
    let open_brackets = result.chars().filter(|&c| c == '[').count();
    let close_brackets = result.chars().filter(|&c| c == ']').count();
    for _ in 0..(open_brackets.saturating_sub(close_brackets)) {
        result.push(']');
    }

    result
}

/// Build a DetectionResult from a parsed JSON object.
fn build_result(obj: &serde_json::Value) -> DetectionResult {
    // Extract labels
    let labels: BTreeSet<SpanLabel> = obj
        .get("labels")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .filter_map(SpanLabel::from_str_opt)
                .collect()
        })
        .unwrap_or_default();

    // Extract risk
    let risk = obj
        .get("risk")
        .and_then(|v| v.as_str())
        .and_then(RiskLevel::from_str_opt)
        .unwrap_or(RiskLevel::Low);

    // Extract rationale
    let rationale = obj
        .get("rationale")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    DetectionResult::new_enforced(labels, risk, rationale)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Total function guarantee ---

    #[test]
    fn empty_input_returns_empty_result() {
        let r = parse_output("");
        assert!(r.labels.is_empty());
        assert_eq!(r.risk, RiskLevel::Low);
    }

    #[test]
    fn garbage_input_returns_empty_result() {
        let r = parse_output("not json at all }{][");
        assert!(r.labels.is_empty());
    }

    // --- Direct parse ---

    #[test]
    fn valid_json_parsed() {
        let input = r#"{"labels": ["WORKPLACE", "ROUTINE"], "risk": "MEDIUM", "rationale": "combo"}"#;
        let r = parse_output(input);
        assert!(r.labels.contains(&SpanLabel::Workplace));
        assert!(r.labels.contains(&SpanLabel::Routine));
        assert_eq!(r.risk, RiskLevel::Medium);
        assert_eq!(r.rationale, "combo");
    }

    #[test]
    fn valid_json_single_label() {
        let input = r#"{"labels": ["LOCATION"], "risk": "MEDIUM", "rationale": "location found"}"#;
        let r = parse_output(input);
        assert_eq!(r.labels.len(), 1);
        assert!(r.labels.contains(&SpanLabel::Location));
    }

    // --- Code fence extraction ---

    #[test]
    fn json_in_code_fence() {
        let input = "Here is the result:\n```json\n{\"labels\": [\"WORKPLACE\"], \"risk\": \"MEDIUM\", \"rationale\": \"work\"}\n```";
        let r = parse_output(input);
        assert!(r.labels.contains(&SpanLabel::Workplace));
    }

    #[test]
    fn json_in_plain_code_fence() {
        let input = "```\n{\"labels\": [], \"risk\": \"LOW\", \"rationale\": \"\"}\n```";
        let r = parse_output(input);
        assert!(r.labels.is_empty());
        assert_eq!(r.risk, RiskLevel::Low);
    }

    // --- Embedded JSON ---

    #[test]
    fn embedded_json_in_text() {
        let input = "The analysis shows {\"labels\": [\"DEMOGRAPHIC\"], \"risk\": \"MEDIUM\", \"rationale\": \"demographics\"} as the result.";
        let r = parse_output(input);
        assert!(r.labels.contains(&SpanLabel::Demographic));
    }

    // --- JSON repair ---

    #[test]
    fn trailing_comma_repaired() {
        let input = r#"{"labels": ["LOCATION",], "risk": "MEDIUM", "rationale": "loc",}"#;
        let r = parse_output(input);
        assert!(r.labels.contains(&SpanLabel::Location));
    }

    #[test]
    fn single_quotes_repaired() {
        let input = "{'labels': ['WORKPLACE'], 'risk': 'MEDIUM', 'rationale': 'work'}";
        let r = parse_output(input);
        assert!(r.labels.contains(&SpanLabel::Workplace));
    }

    #[test]
    fn unclosed_brace_repaired() {
        let input = r#"{"labels": ["ROUTINE"], "risk": "MEDIUM", "rationale": "routine""#;
        let r = parse_output(input);
        assert!(r.labels.contains(&SpanLabel::Routine));
    }

    #[test]
    fn unclosed_bracket_repaired() {
        let input = r#"{"labels": ["ROUTINE", "WORKPLACE"], "risk": "HIGH", "rationale": "combo"}"#;
        let r = parse_output(input);
        assert!(r.labels.contains(&SpanLabel::Routine));
        assert!(r.labels.contains(&SpanLabel::Workplace));
    }

    // --- Field extraction with defaults ---

    #[test]
    fn missing_labels_defaults_to_empty() {
        let input = r#"{"risk": "LOW", "rationale": ""}"#;
        let r = parse_output(input);
        assert!(r.labels.is_empty());
    }

    #[test]
    fn missing_risk_defaults_to_low() {
        let input = r#"{"labels": [], "rationale": ""}"#;
        let r = parse_output(input);
        assert_eq!(r.risk, RiskLevel::Low);
    }

    #[test]
    fn missing_rationale_defaults_to_empty() {
        let input = r#"{"labels": ["LOCATION"], "risk": "LOW"}"#;
        let r = parse_output(input);
        assert!(r.rationale.is_empty());
    }

    // --- Label validation ---

    #[test]
    fn invalid_labels_dropped() {
        let input = r#"{"labels": ["WORKPLACE", "INVALID", "BOGUS"], "risk": "MEDIUM", "rationale": "work"}"#;
        let r = parse_output(input);
        assert_eq!(r.labels.len(), 1);
        assert!(r.labels.contains(&SpanLabel::Workplace));
    }

    #[test]
    fn duplicate_labels_collapsed() {
        let input = r#"{"labels": ["WORKPLACE", "WORKPLACE", "ROUTINE"], "risk": "HIGH", "rationale": "combo"}"#;
        let r = parse_output(input);
        assert_eq!(r.labels.len(), 2);
    }

    // --- Consistency enforcement ---

    #[test]
    fn empty_labels_forced_to_low() {
        let input = r#"{"labels": [], "risk": "HIGH", "rationale": "wrong"}"#;
        let r = parse_output(input);
        assert_eq!(r.risk, RiskLevel::Low);
        assert!(r.rationale.is_empty());
    }

    #[test]
    fn low_risk_clears_rationale() {
        let input = r#"{"labels": ["LOCATION"], "risk": "LOW", "rationale": "should be cleared"}"#;
        let r = parse_output(input);
        assert!(r.rationale.is_empty());
    }

    #[test]
    fn multi_label_high_auto_generates_rationale() {
        let input = r#"{"labels": ["WORKPLACE", "ROUTINE"], "risk": "HIGH", "rationale": ""}"#;
        let r = parse_output(input);
        assert_eq!(r.rationale, "Multiple quasi-identifiers detected.");
    }

    // --- Invalid risk value ---

    #[test]
    fn invalid_risk_defaults_to_low() {
        let input = r#"{"labels": [], "risk": "CRITICAL", "rationale": ""}"#;
        let r = parse_output(input);
        assert_eq!(r.risk, RiskLevel::Low);
    }
}
