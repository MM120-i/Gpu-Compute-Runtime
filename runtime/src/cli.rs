use std::path::PathBuf;
use clap::{Parser, Subcommand, Args};

#[derive(Parser)]
#[command(name = "gcr", about = "GPU Compute Runtime — GLSL to SPIR-V pipeline")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    #[command(about = "Compile a GLSL shader to SPIR-V")]
    Compile(CompileArgs),
    #[command(about = "Render a Mandelbrot set image on the GPU")]
    Mandelbrot(MandelbrotArgs),
}

#[derive(Args)]
pub struct CompileArgs {
    pub input: PathBuf,

    #[arg(short = 'o', long)]
    pub output: PathBuf,

    #[arg(long)]
    pub no_preprocess: bool,

    #[arg(long)]
    pub no_unroll: bool,

    #[arg(long)]
    pub no_ast_opt: bool,

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

#[derive(Args)]
pub struct MandelbrotArgs {
    #[arg(long, default_value = "1920")]
    pub width: u32,

    #[arg(long, default_value = "1080")]
    pub height: u32,

    #[arg(long, default_value = "1000")]
    pub iters: u32,

    #[arg(long)]
    pub cx: Option<f32>,

    #[arg(long)]
    pub cy: Option<f32>,

    #[arg(long)]
    pub scale: Option<f32>,

    #[arg(long, value_enum)]
    pub zoom: Option<ZoomPreset>,

    #[arg(long, default_value = "mandelbrot.ppm")]
    pub output: PathBuf,

    #[arg(long)]
    pub benchmark: bool,
}

#[derive(Clone, clap::ValueEnum)]
pub enum ZoomPreset {
    Full,
    Seahorse,
    Elephant,
    Spiral,
}
