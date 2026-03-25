//! PII Scanner Binary — CLI entry point.
//!
//! Spec: specifications/rust-binary.md §1

mod entities;
mod payload;
mod scan;
mod tier1;
mod tier2;

use clap::Parser;
use std::io::Read;
use std::path::PathBuf;
use std::process;

#[derive(Parser)]
#[command(name = "pii-scanner", about = "PII detection for Claude Code hooks")]
struct Cli {
    /// Run Tier 1 direct identifier detection
    #[arg(long, group = "tier")]
    tier1: bool,

    /// Run Tier 2 quasi-identifier detection
    #[arg(long, group = "tier")]
    tier2: bool,

    /// Hook event type
    #[arg(long, value_parser = ["user_prompt", "pre_tool_use", "post_tool_use"])]
    hook: String,
}

fn main() {
    let cli = Cli::parse();

    // Read payload from stdin
    let mut input = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut input) {
        eprint!("Failed to read stdin: {e}");
        process::exit(1);
    }

    let payload: serde_json::Value = match serde_json::from_str(&input) {
        Ok(v) => v,
        Err(e) => {
            eprint!("Failed to parse JSON payload: {e}");
            process::exit(1);
        }
    };

    let output = if cli.tier1 {
        scan::run_tier1(&cli.hook, &payload)
    } else if cli.tier2 {
        let model_path = std::env::var("PII_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("~"))
                    .join(".cache/contextual-pii-tagger/model.gguf")
            });
        scan::run_tier2(&cli.hook, &payload, &model_path)
    } else {
        eprint!("Specify --tier1 or --tier2");
        process::exit(1);
    };

    if !output.stderr.is_empty() {
        eprint!("{}", output.stderr);
    }

    process::exit(output.exit_code);
}
