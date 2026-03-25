//! Prompt assembly for Tier 2 inference.
//!
//! Spec: specifications/rust-binary.md §3.1

/// Maximum token sequence length.
pub const MAX_SEQUENCE_LENGTH: usize = 1024;

/// The instruction template (must match Python prompt.py exactly).
pub const PROMPT_TEMPLATE: &str = "<|user|>\n\
Classify which quasi-identifier PII categories are present in the \
following text. Return the list of category labels from the taxonomy, \
an overall risk score (LOW/MEDIUM/HIGH), and a brief rationale.\n\
\n\
Text: {text}\n\
<|end|>\n\
<|assistant|>\n";

/// Format the prompt template with the given text.
pub fn get_template_text(text: &str) -> String {
    PROMPT_TEMPLATE.replace("{text}", text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_contains_markers() {
        let result = get_template_text("hello world");
        assert!(result.contains("<|user|>"));
        assert!(result.contains("<|end|>"));
        assert!(result.contains("<|assistant|>"));
        assert!(result.contains("Text: hello world"));
    }

    #[test]
    fn template_preserves_text() {
        let text = "I work at St. Mary's Hospital on Tuesdays.";
        let result = get_template_text(text);
        assert!(result.contains(text));
    }

    #[test]
    fn max_sequence_length_is_1024() {
        assert_eq!(MAX_SEQUENCE_LENGTH, 1024);
    }
}
