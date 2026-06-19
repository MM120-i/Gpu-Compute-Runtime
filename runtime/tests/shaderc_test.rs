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

#[test]
fn lexer_operators() {
    let tokens: Vec<String> = runtime::shaderc::tokenize("a + b").unwrap();
    assert_eq!(tokens, vec!["IDENTIFIER", "PLUS", "IDENTIFIER", "END_OF_FILE"]);
}

#[test]
fn lexer_keywords() {
    let tokens: Vec<String> = runtime::shaderc::tokenize("void main()").unwrap();
    assert_eq!(tokens, vec!["KW_VOID", "IDENTIFIER", "OPEN_PAREN", "CLOSE_PAREN", "END_OF_FILE"]);
}

#[test]
fn lexer_numbers() {
    let tokens: Vec<String> = runtime::shaderc::tokenize("42 3.14f").unwrap();
    assert_eq!(tokens, vec!["INT_LITERAL", "FLOAT_LITERAL", "END_OF_FILE"]);
}

#[test]
fn lexer_comparison() {
    let tokens: Vec<String> = runtime::shaderc::tokenize("i <= 8").unwrap();
    assert_eq!(tokens, vec!["IDENTIFIER", "LESS_EQUALS", "INT_LITERAL", "END_OF_FILE"]);
}

#[test]
fn emit_var_decl(){
    let result: String = runtime::shaderc::emit_test().unwrap();
    assert_eq!(result, "float x = 3.14;\n");
}


fn run_propagate(source: &str) -> String {
    runtime::shaderc::propagate_and_emit(source)
        .expect("constant propagation should succeed")
}

#[test]
fn fold_int_arithmetic() {
    let result: String = run_propagate(r#"
#version 460
void main() { int a = 5 * 2 + 3; }
"#);
    assert!(result.contains("int a = 13;") || result.contains("int a = 13;\n"),
        "5 * 2 + 3 should fold to 13:\n{}", result);
}

#[test]
fn fold_add_zero() {
    let result: String = run_propagate(r#"
#version 460
void main() { int a = 42; int b = a + 0; }
"#);
    assert!(!result.contains("a + 0"), "a + 0 should fold to just a:\n{}", result);
    assert!(result.contains("int b = a;\n") || result.contains("int b = a;\n"),
        "b should be directly assigned from a:\n{}", result);
}

#[test]
fn fold_mul_by_one() {
    let result: String = run_propagate(r#"
#version 460
void main() { int a = 42; int b = a * 1; }
"#);
    assert!(!result.contains("a * 1"), "a * 1 should fold to just a:\n{}", result);
    assert!(result.contains("int b = a;\n") || result.contains("int b = a;\n"),
        "b should be directly assigned from a:\n{}", result);
}

#[test]
fn fold_mul_by_zero() {
    let result: String = run_propagate(r#"
#version 460
void main() { int a = 42; int b = a * 0; }
"#);
    assert!(result.contains("b = 0;") || result.contains("b = 0;\n"),
        "a * 0 should fold to 0:\n{}", result);
}

#[test]
fn fold_sub_zero() {
    let result: String = run_propagate(r#"
#version 460
void main() { int a = 42; int b = a - 0; }
"#);
    assert!(!result.contains("a - 0"), "a - 0 should fold to a:\n{}", result);
    assert!(result.contains("int b = a;\n") || result.contains("int b = a;\n"),
        "b should be directly assigned from a:\n{}", result);
}

#[test]
fn fold_double_negate() {
    let result: String = run_propagate(r#"
#version 460
void main() { int a = 42; int b = -(-a); }
"#);
    assert!(!result.contains("-(-a)"), "-(-a) should fold to a:\n{}", result);
    assert!(result.contains("b = a;\n") || result.contains("b = a;\n"),
        "b should be directly assigned from a:\n{}", result);
}

#[test]
fn fold_double_not() {
    let result: String = run_propagate(r#"
#version 460
void main() { bool a = true; bool b = !(!a); }
"#);
    assert!(!result.contains("!(!a)"), "!(!a) should fold to a:\n{}", result);
    assert!(result.contains("b = a;\n") || result.contains("b = a;\n"),
        "b should be directly assigned from a:\n{}", result);
}

#[test]
fn fold_float_binary() {
    let result: String = run_propagate(r#"
#version 460
void main() { float a = 2.5; float b = a * 1.0; }
"#);
    assert!(!result.contains("a * 1.0"), "a * 1.0 should fold to a:\n{}", result);
}

#[test]
fn fold_int_comparison() {
    let result: String = run_propagate(r#"
#version 460
void main() { int a = 1; int b = 3 < 5; }
"#);
    assert!(result.contains("int b = 1;") || result.contains("b = 1;"),
        "3 < 5 should fold to 1 (true):\n{}", result);
}

#[test]
fn propagate_is_idempotent() {
    let source: &str = r#"
#version 460
layout(local_size_x = 1) in;
void main() {
    float x = 1.0;
    float y = x + 0.0;
}
"#;
    let first: String = run_propagate(source);
    let second: String = run_propagate(&first);
    assert_eq!(first, second, "second pass should not change anything");
}

#[test]
fn propagated_shader_compiles_to_valid_spirv() {
    let source: &str = r#"
#version 460
layout(local_size_x = 1) in;
layout(binding = 0) buffer Out { float d[]; } out_buf;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    float sum = 0.0;
    for (int i = 0; i < 4; i++) {
        sum += float(i);
    }
    out_buf.d[idx] = sum * 1.0 + 0.0;
}
"#;
    let propagated: String = run_propagate(source);
    let (spirv, _) = runtime::shaderc::compile_glsl_with_errors(&propagated).expect("propagated GLSL should compile to valid SPIR-V");
    assert!(!spirv.is_empty());
}

#[test]
fn propagate_with_preprocess_and_unroll_full_pipeline() {
    let source: &str = r#"
#version 460
#define ITERS 3
layout(local_size_x = 64) in;
layout(binding = 0) buffer Out { float d[]; } out_buf;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    float sum = 0.0;
    for (int i = 0; i < ITERS; i++) {
        sum += float(i);
    }
    float factor = 2.0 * 3.0;
    out_buf.d[idx] = sum * factor + 0.0;
}
"#;

    let preprocessed: String = runtime::shaderc::preprocess(source, None, None, None).expect("preprocess should succeed");
    assert!(preprocessed.contains("i < 3"), "ITERS should expand");

    let unrolled: String = runtime::shaderc::unroll(&preprocessed).expect("unroll should succeed");
    assert!(!unrolled.contains("for"), "loops should be unrolled");

    let propagated: String = runtime::shaderc::propagate_and_emit(&unrolled).expect("propagation should succeed");
    assert!(propagated.contains("6.0"), "2.0 * 3.0 should fold to 6.0:\n{}", propagated);

    let (spirv, _) = runtime::shaderc::compile_glsl_with_errors(&propagated).expect("final GLSL should compile to valid SPIR-V");
    assert!(!spirv.is_empty());
}

#[test]
fn no_ast_opt_flag_is_present() {
    let source: &str = r#"
#version 460
layout(local_size_x = 1) in;
void main() {
    int a = 5 * 2;
}
"#;
    let with_opt: String = run_propagate(source);
    assert!(with_opt.contains("int a = 10;") || with_opt.contains("a = 10;"), "with AST opt, 5*2 should fold to 10:\n{}", with_opt);

    let (spirv, _) = runtime::shaderc::compile_glsl_with_errors(source).expect("raw source with 5*2 should compile");
    assert!(!spirv.is_empty());
}
