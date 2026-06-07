#[test]
fn unroll_noop_when_no_loops() {
    let source: &str = "#version 460\nvoid main() { float x = 1.0; }\n";
    let result: String = runtime::shaderc::unroll(source).unwrap();
    assert_eq!(result, source);
}

#[test]
fn unroll_simple_constant_loop() {
    let source: &str = r#"
#version 460
void main() {
    float sum = 0.0;
    for (int i = 0; i < 3; i++) {
        sum += float(i);
    }
}
"#;
    let result: String = runtime::shaderc::unroll(source).unwrap();

    assert!(result.contains("float(0)"), "missing float(0)");
    assert!(result.contains("float(1)"), "missing float(1)");
    assert!(result.contains("float(2)"), "missing float(2)");

    assert!(!result.contains("for"), "unrolled output still contains 'for'");
}

#[test]
fn unroll_skips_non_constant_bound() {
    let source: &str = r#"
#version 460
layout(local_size_x = 64) in;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    float sum = 0.0;
    for (int k = 0; k < idx; k++) {
        sum += 1.0;
    }
}
"#;
    let result: String = runtime::shaderc::unroll(source).unwrap();

    assert!(
        result.contains("for(int k = 0; k < idx; k++)"),
        "non-constant loop was modified:\n{}",
        result
    );
}

#[test]
fn unroll_skips_break_continue() {
    let source: &str = r#"
#version 460
void main() {
    float sum = 0.0;
    for (int i = 0; i < 8; i++) {
        if (i > 4) break;
        sum += float(i);
    }
}
"#;
    let result: String = runtime::shaderc::unroll(source).unwrap();

    assert!(result.contains("for("), "for-loop with break was unrolled");
    assert!(result.contains("break"), "break was removed");
}

#[test]
fn unroll_multiple_loops() {
    let source: &str = r#"
#version 460
void main() {
    float sum = 0.0;
    for (int i = 0; i < 2; i++) {
        sum += float(i);
    }
    float prod = 1.0;
    for (int j = 0; j < 3; j++) {
        prod *= float(j);
    }
}
"#;
    let result: String = runtime::shaderc::unroll(source).unwrap();

    assert!(result.contains("float(0)"));
    assert!(result.contains("float(1)"));
    assert!(result.contains("float(2)"));
    assert!(!result.contains("for"), "unrolled output still contains 'for'");
}

#[test]
fn unroll_bounded_by_less_equal() {
    let source: &str = r#"
#version 460
void main() {
    float prod = 1.0;
    for (int j = 0; j <= 3; j++) {
        prod *= float(j);
    }
}
"#;
    let result: String = runtime::shaderc::unroll(source).unwrap();

    assert!(result.contains("float(0)"));
    assert!(result.contains("float(1)"));
    assert!(result.contains("float(2)"));
    assert!(result.contains("float(3)"));
    assert!(!result.contains("for"), "<= bound loop not unrolled");
}

#[test]
fn unroll_output_compiles_to_valid_spirv() {
    let source: &str = r#"
#version 460
layout(local_size_x = 64) in;
layout(binding = 0) buffer Out { float data[]; } out_buf;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    float sum = 0.0;
    for (int i = 0; i < 4; i++) {
        sum += float(i);
    }
    out_buf.data[idx] = sum;
}
"#;
    let unrolled: String = runtime::shaderc::unroll(source).unwrap();
    let (spirv, _) = runtime::shaderc::compile_glsl_with_errors(&unrolled)
        .expect("unrolled GLSL should compile to valid SPIR-V");
    assert!(!spirv.is_empty());
}

#[test]
fn preprocess_noop_without_directives() {
    let source: &str = "#version 460\nvoid main() {}\n";
    let result: String = runtime::shaderc::preprocess(source, None, None, None)
        .expect("preprocess should succeed");

    assert!(result.contains("#version 460"));
    assert!(result.contains("void main()"));
}

#[test]
fn preprocess_expands_define_in_source() {
    let source: &str = r#"
#version 460
#define N 4
void main() {
    float x = N;
}
"#;
    let result: String = runtime::shaderc::preprocess(source, None, None, None).expect("preprocess should succeed");

    assert!(result.contains("x = 4"), "N was not expanded to 4:\n{}", result);
    assert!(!result.contains("N"), "N still appears in output:\n{}", result);
}

#[test]
fn preprocess_and_unroller_full_pipeline() {
    let source: &str = r#"
#version 460
#define ITERS 3
layout(local_size_x = 64) in;
layout(binding = 0) buffer Out { float data[]; } out_buf;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    float sum = 0.0;
    for (int i = 0; i < ITERS; i++) {
        sum += float(i);
    }
    out_buf.data[idx] = sum;
}
"#;

    let preprocessed: String = runtime::shaderc::preprocess(source, None, None, None).expect("preprocess should succeed");

    assert!(
        preprocessed.contains("i < 3"),
        "ITERS was not expanded in preprocessed output:\n{}",
        preprocessed
    );

    let unrolled: String = runtime::shaderc::unroll(&preprocessed).expect("unroll should succeed");

    assert!(unrolled.contains("float(0)"));
    assert!(unrolled.contains("float(1)"));
    assert!(unrolled.contains("float(2)"));
    assert!(!unrolled.contains("for"), "unrolled output still has 'for'");

    let (spirv, _) = runtime::shaderc::compile_glsl_with_errors(&unrolled).expect("final GLSL should compile to valid SPIR-V");
    
    assert!(!spirv.is_empty(), "SPIR-V output should not be empty");
}
