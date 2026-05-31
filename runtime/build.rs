use std::path::PathBuf;

fn main() {
    let out_dir: PathBuf = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let spv_path: PathBuf = out_dir.join("double.spv");
    let status: std::process::ExitStatus = std::process::Command::new("glslc")
        .args(&[
            "shaders/double.comp",
            "-o",
            &spv_path.to_string_lossy(),
        ])
        .status()
        .expect("glslc not found — install Vulkan SDK and ensure bin/ is on PATH");

    assert!(status.success(), "glslc compilation failed");
    
    println!("cargo:rustc-env=DOUBLE_SPV={}", spv_path.display());
    println!("cargo:rerun-if-changed=shaders/double.comp");
}
