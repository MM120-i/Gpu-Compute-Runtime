use std::ffi::OsStr;
use std::fs;
use std::process::Command;
use clap::Parser;

mod cli;

fn optimize_spirv(spirv: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir();
    let input_path = dir.join("gcr_opt_in.spv");
    let output_path = dir.join("gcr_opt_out.spv");

    fs::write(&input_path, spirv)?;

    let output = Command::new("spirv-opt")
        .args([
            OsStr::new("-O"),
            input_path.as_os_str(),
            OsStr::new("-o"),
            output_path.as_os_str(),
        ])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("spirv-opt failed: {}", stderr.trim()).into());
    }

    let result = fs::read(&output_path)?;
    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&output_path);

    Ok(result)
}

fn validate_spirv(spirv: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir();
    let path = dir.join("gcr_val.spv");

    fs::write(&path, spirv)?;

    let output = Command::new("spirv-val")
        .arg(&path)
        .output()?;

    let _ = fs::remove_file(&path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("spirv-val failed: {}", stderr.trim()).into());
    }

    Ok(())
}

fn run_compile(args: cli::CompileArgs) -> Result<(), Box<dyn std::error::Error>> {
    let source: String = fs::read_to_string(&args.input)?;

    let source: String = if !args.no_preprocess {
        let source_path: std::borrow::Cow<'_, str> = args.input.to_string_lossy();

        let include_dirs: Option<String> = if args.include_dirs.is_empty() {
            None
        } 
        else {
            Some(args.include_dirs.join(";"))
        };

        let defines: Option<String> = if args.defines.is_empty() {
            None
        } 
        else {
            Some(args.defines.join(";"))
        };

        let result: String = runtime::shaderc::preprocess(
            &source,
            Some(&source_path),
            include_dirs.as_deref(),
            defines.as_deref(),
        )?;

        if args.verbose {
            eprintln!("[gcr] preprocessing: OK");
        }

        result
    } 
    else {
        source
    };

    let source: String = if !args.no_unroll {
        let result: String = runtime::shaderc::unroll(&source)?;

        if args.verbose {
            eprintln!("[gcr] loop unrolling: OK");
        }

        result
    } 
    else {
        source
    };

    let pass_flags: i32 = if args.no_ast_opt {
        0
    } 
    else {
        runtime::shaderc::PASS_CONSTANT_PROPAGATION
    };

    let source: String = if pass_flags != 0 {
        let result: String = runtime::shaderc::pipeline(&source, pass_flags)?;

        if args.verbose {
            eprintln!("[gcr] AST Constant propagation: OK");
        }

        result
    } 
    else {
        source
    };

    let (spirv, warnings) = runtime::shaderc::compile_glsl_with_errors(&source)?;

    if args.verbose {
        eprintln!("[gcr] compilation: OK ({} bytes)", spirv.len());

        if !warnings.is_empty() {
            eprintln!("[gcr] warnings:\n{}", warnings);
        }
    }

    let spirv: Vec<u8> = if args.optimize {
        let result: Vec<u8> = optimize_spirv(&spirv)?;

        if args.verbose {
            eprintln!("[gcr] spirv-opt: OK");
        }

        result
    } 
    else {
        spirv
    };

    if args.validate {
        validate_spirv(&spirv)?;

        if args.verbose {
            eprintln!("[gcr] spirv-val: OK");
        }
    }

    fs::write(&args.output, &spirv)?;
    eprintln!("[gcr] output: {} ({} bytes)", args.output.display(), spirv.len());

    Ok(())
}

fn lookup_zoom(preset: cli::ZoomPreset) -> (f32, f32, f32) {
    match preset {
        cli::ZoomPreset::Full => (-0.5, 0.0, 3.5),
        cli::ZoomPreset::Seahorse => (-0.743643887, 0.131825904, 0.0000001),
        cli::ZoomPreset::Elephant => (0.275, 0.006, 0.00001),
        cli::ZoomPreset::Spiral => (-0.7269, 0.1889, 0.0002),
    }
}

fn run_mandelbrot(args: cli::MandelbrotArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (cx, cy, scale) = if let Some(zoom) = args.zoom {
        lookup_zoom(zoom)
    } else {
        (
            args.cx.unwrap_or(-0.5),
            args.cy.unwrap_or(0.0),
            args.scale.unwrap_or(3.5),
        )
    };

    eprintln!(
        "[gcr] rendering {}×{} @ {} iters (cx={}, cy={}, scale={})",
        args.width, args.height, args.iters, cx, cy, scale
    );

    if args.benchmark {
        let mut ctx: runtime::context::GpuContext = runtime::context::GpuContext::new()?;
        let profiler: runtime::gpu::GpuProfiler = runtime::gpu::profiler::GpuProfiler::new(&ctx)?;
        let (result, _report) = runtime::demos::mandelbrot::bench_mandelbrot(&mut ctx, &profiler)?;
        let mb: &serde_json::Value = &result["mandelbrot"];

        println!("{}", serde_json::to_string_pretty(&result)?);

        let gpu: f64 = mb["gpu_ms"].as_f64().unwrap_or(0.0);
        let cpu: f64 = mb["cpu_ms"].as_f64().unwrap_or(0.0);
        let sp: f64 = mb["speedup"].as_f64().unwrap_or(0.0);

        eprintln!("[gcr] GPU {:.2}ms | CPU {:.2}ms | {:.1}x speedup", gpu, cpu, sp);
    } 
    else {
        let mut ctx: runtime::context::GpuContext = runtime::context::GpuContext::new()?;
        runtime::demos::mandelbrot::render_to_file(
            &mut ctx,
            args.width,
            args.height,
            args.iters,
            cx,
            cy,
            scale,
            args.output.to_str().ok_or("invalid output path")?,
        )?;

        eprintln!("[gcr] saved to {}", args.output.display());
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli: cli::Cli = cli::Cli::parse();

    match cli.command {
        cli::Commands::Compile(args) => run_compile(args),
        cli::Commands::Mandelbrot(args) => run_mandelbrot(args),
    }
}
