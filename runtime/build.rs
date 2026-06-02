use std::path::PathBuf;

fn main() {
    // c++ FFI bridge
    let vulkan_sdk: String = std::env::var("VULKAN_SDK").expect("VULKAN_SDK not set - install Vulkan SDK and reboot bro");
    let lib_dir: String = format!("{}/Lib", vulkan_sdk);
    let include_dir: String = format!("{}/Include", vulkan_sdk);

    cc::Build::new()
        .cpp(true)
        .file("cpp/shaderc_bridge.cpp")
        .include(&include_dir)
        .compile("shaderc_bridge");

    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=static=shaderc_combined");

    // pre-compiled test shader stuff
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
