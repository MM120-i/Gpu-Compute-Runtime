use std::path::PathBuf;
use clap::Parser;

#[derive(Parser)]
#[command(name = "gcr", about = "GPU Compute Runtime — GLSL to SPIR-V pipeline")]
pub struct Cli {
    pub input: PathBuf,

    #[arg(short = 'o', long)]
    pub output: PathBuf,

    #[arg(long)]
    pub no_preprocess: bool,

    #[arg(long)]
    pub no_unroll: bool,

    #[arg(short = 'I', long)]
    pub include_dirs: Vec<String>,

    #[arg(short = 'D')]
    pub defines: Vec<String>,

    #[arg(long)]
    pub optimize: bool,

    #[arg(long)]
    pub validate: bool,

    #[arg(short, long)]
    pub verbose: bool,
}