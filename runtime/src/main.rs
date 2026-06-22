use clap::Parser;
use std::ffi::OsStr;
use std::fs;
use std::process::Command;

mod cli;

fn optimize_spirv(spirv: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let dir: std::path::PathBuf = std::env::temp_dir();
    let input_path: std::path::PathBuf = dir.join("gcr_opt_in.spv");
    let output_path: std::path::PathBuf = dir.join("gcr_opt_out.spv");

    fs::write(&input_path, spirv)?;

    let output: std::process::Output = Command::new("spirv-opt")
        .args([
            OsStr::new("-O"),
            input_path.as_os_str(),
            OsStr::new("-o"),
            output_path.as_os_str(),
        ])
        .output()?;

    if !output.status.success() {
        let stderr: std::borrow::Cow<'_, str> = String::from_utf8_lossy(&output.stderr);
        return Err(format!("spirv-opt failed: {}", stderr.trim()).into());
    }

    let result: Vec<u8> = fs::read(&output_path)?;
    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&output_path);

    Ok(result)
}

fn validate_spirv(spirv: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let dir: std::path::PathBuf = std::env::temp_dir();
    let path: std::path::PathBuf = dir.join("gcr_val.spv");

    fs::write(&path, spirv)?;

    let output: std::process::Output = Command::new("spirv-val")
        .arg(&path)
        .output()?;

    let _ = fs::remove_file(&path);

    if !output.status.success() {
        let stderr: std::borrow::Cow<'_, str> = String::from_utf8_lossy(&output.stderr);
        return Err(format!("spirv-val failed: {}", stderr.trim()).into());
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: cli::Cli = cli::Cli::parse();
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
    else{
        runtime::shaderc::PASS_CONSTANT_PROPAGATION
    };

    let source: String = if pass_flags != 0 {
        let result: String = runtime::shaderc::pipeline(&source, pass_flags)?;

        if args.verbose {
            eprintln!("[gcr] AST Constant propagation: OK");
        }

        result
    }
    else{
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
