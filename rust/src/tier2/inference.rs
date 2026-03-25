//! Model loading and inference via llama.cpp.
//!
//! When the `tier2` feature is enabled, uses `llama-cpp-2` for GGUF model
//! loading and greedy text generation. Without the feature, returns an error.
//!
//! Spec: specifications/rust-binary.md §3.2, §6

use std::path::Path;

/// Maximum number of new tokens to generate.
const MAX_NEW_TOKENS: usize = 256;

/// Load a GGUF model and run inference.
///
/// Returns the raw completion string (prompt tokens stripped).
pub fn generate(model_path: &Path, prompt: &str) -> Result<String, String> {
    if !model_path.exists() {
        return Err(format!("Model file not found: {}", model_path.display()));
    }

    #[cfg(feature = "tier2")]
    {
        generate_impl(model_path, prompt)
    }

    #[cfg(not(feature = "tier2"))]
    {
        let _ = (model_path, prompt);
        Err("Tier 2 inference requires --features tier2 (llama.cpp not compiled)".to_string())
    }
}

/// Real implementation using llama-cpp-2.
#[cfg(feature = "tier2")]
fn generate_impl(model_path: &Path, prompt: &str) -> Result<String, String> {
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::LlamaModel;
    use llama_cpp_2::sampling::LlamaSampler;
    use std::num::NonZeroU32;

    // Initialize backend
    let backend = LlamaBackend::init().map_err(|e| format!("Backend init failed: {e}"))?;

    // Load model (CPU only, no GPU layers)
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .map_err(|e| format!("Model load failed: {e}"))?;

    // Create context with 1280 token window (1024 prompt + 256 generation)
    let ctx_params =
        LlamaContextParams::default().with_n_ctx(NonZeroU32::new(1280));
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .map_err(|e| format!("Context creation failed: {e}"))?;

    // Tokenize the prompt
    let prompt_tokens = model
        .str_to_token(prompt, true)
        .map_err(|e| format!("Tokenization failed: {e}"))?;

    let n_prompt = prompt_tokens.len();

    // Create batch and add all prompt tokens
    let mut batch = LlamaBatch::new(n_prompt + MAX_NEW_TOKENS, 1);
    for (i, &token) in prompt_tokens.iter().enumerate() {
        let is_last = i == n_prompt - 1;
        batch
            .add(token, i as i32, &[0], is_last)
            .map_err(|e| format!("Batch add failed: {e}"))?;
    }

    // Decode prompt
    ctx.decode(&mut batch)
        .map_err(|e| format!("Prompt decode failed: {e}"))?;

    // Set up greedy sampler (temperature 0 = deterministic)
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

    // Generate tokens
    let mut output_pieces: Vec<String> = Vec::new();
    let mut n_decoded = 0;

    loop {
        if n_decoded >= MAX_NEW_TOKENS {
            break;
        }

        // Sample next token
        let token = sampler.sample(&ctx, -1);

        // Check for end of generation
        if model.is_eog_token(token) {
            break;
        }

        // Decode token to text
        let piece = model
            .token_to_str(token, None)
            .map_err(|e| format!("Token decode failed: {e}"))?;
        output_pieces.push(piece);

        n_decoded += 1;

        // Prepare next batch with the generated token
        batch.clear();
        batch
            .add(token, (n_prompt + n_decoded) as i32, &[0], true)
            .map_err(|e| format!("Batch add failed: {e}"))?;

        ctx.decode(&mut batch)
            .map_err(|e| format!("Decode failed: {e}"))?;
    }

    Ok(output_pieces.join(""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn generate_fails_with_missing_model() {
        let path = PathBuf::from("/nonexistent/model.gguf");
        let result = generate(&path, "test prompt");
        assert!(result.is_err());
    }

    #[cfg(feature = "tier2")]
    mod with_llama {
        use super::*;

        // Integration tests require a real GGUF model file.
        // Set PII_MODEL_PATH to run these.

        #[test]
        #[ignore = "requires GGUF model file"]
        fn generate_produces_output() {
            let model_path = std::env::var("PII_MODEL_PATH")
                .map(PathBuf::from)
                .expect("PII_MODEL_PATH must be set for integration tests");

            let prompt = "<|user|>\nClassify which quasi-identifier PII categories are present in the following text. Return the list of category labels from the taxonomy, an overall risk score (LOW/MEDIUM/HIGH), and a brief rationale.\n\nText: The weather is nice today.\n<|end|>\n<|assistant|>\n";

            let result = generate(&model_path, prompt);
            assert!(result.is_ok(), "generate should succeed: {:?}", result.err());

            let output = result.unwrap();
            assert!(!output.is_empty(), "output should not be empty");
        }
    }
}
