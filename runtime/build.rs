use std::path::PathBuf;

fn main() {
    // c++ FFI bridge
    let vulkan_sdk: String = std::env::var("VULKAN_SDK").expect("VULKAN_SDK not set - install Vulkan SDK and reboot bro");
    let lib_dir: String = format!("{}/Lib", vulkan_sdk);
    let include_dir: String = format!("{}/Include", vulkan_sdk);

    cc::Build::new()
        .cpp(true)
        .flag("-std:c++17")
        .file("../compiler/bridge/shaderc_bridge.cpp")
        .file("../compiler/preprocessor/preprocessor.cpp")
        .file("../compiler/unroller/loop_unroller.cpp")
        .file("../compiler/ast/glsl_lexer.cpp")
        .file("../compiler/ast/glsl_parser.cpp")
        .file("../compiler/ast/emitter.cpp")
        .file("../compiler/ast/constant_propagation.cpp")
        .file("../compiler/compiler_pipeline.cpp")
        .include(&include_dir)
        .include("../compiler/ast")
        .compile("shaderc_bridge");

    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=static=shaderc_combined");

    // pre-compiled test shader stuff
    let out_dir: PathBuf = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let spv_path: PathBuf = out_dir.join("double.spv");
    let status: std::process::ExitStatus = std::process::Command::new("glslc")
        .args(&[
            "../kernels/double.comp",
            "-o",
            &spv_path.to_string_lossy(),
        ])
        .status()
        .expect("glslc not found — install Vulkan SDK and ensure bin/ is on PATH");

    assert!(status.success(), "glslc compilation failed");
    
    println!("cargo:rustc-env=DOUBLE_SPV={}", spv_path.display());
    println!("cargo:rerun-if-changed=../kernels/double.comp");
    println!("cargo:rerun-if-changed=../compiler/bridge/shaderc_bridge.cpp");
    println!("cargo:rerun-if-changed=../compiler/preprocessor/preprocessor.cpp");
    println!("cargo:rerun-if-changed=../compiler/preprocessor/preprocessor.h");
    println!("cargo:rerun-if-changed=../compiler/unroller/loop_unroller.cpp");
    println!("cargo:rerun-if-changed=../compiler/unroller/loop_unroller.h");
    println!("cargo:rerun-if-changed=../compiler/unroller/loop_unroller.h");
    println!("cargo:rerun-if-changed=../compiler/ast/ast.h");
    println!("cargo:rerun-if-changed=../compiler/ast/glsl_lexer.h");
    println!("cargo:rerun-if-changed=../compiler/ast/glsl_lexer.cpp");
    println!("cargo:rerun-if-changed=../compiler/ast/emitter.h");
    println!("cargo:rerun-if-changed=../compiler/ast/emitter.cpp");
    println!("cargo:rerun-if-changed=../compiler/ast/glsl_parser.h");
    println!("cargo:rerun-if-changed=../compiler/ast/glsl_parser.cpp");
    println!("cargo:rerun-if-changed=../compiler/ast/constant_propagation.h");
    println!("cargo:rerun-if-changed=../compiler/ast/constant_propagation.cpp");
    println!("cargo:rerun-if-changed=../compiler/compiler_pipeline.cpp");
    println!("cargo:rerun-if-changed=../compiler/compiler_pipeline.h");
}
