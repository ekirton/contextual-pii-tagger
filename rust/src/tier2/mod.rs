//! Tier 2 quasi-identifier detection pipeline.
//!
//! Spec: specifications/rust-binary.md §3

pub mod inference;
pub mod parser;
pub mod prompt;

use crate::entities::DetectionResult;
use std::path::Path;

/// Run the Tier 2 detection pipeline: prompt assembly → inference → parse.
pub fn scan_tier2(text: &str, model_path: &Path) -> Result<DetectionResult, String> {
    let prompt_text = prompt::get_template_text(text);

    let raw_output = inference::generate(model_path, &prompt_text)?;

    Ok(parser::parse_output(&raw_output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn scan_tier2_fails_without_model() {
        let result = scan_tier2("test text", &PathBuf::from("/nonexistent/model.gguf"));
        assert!(result.is_err());
    }
}
