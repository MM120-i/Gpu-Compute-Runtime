use std::fs;

#[test]
fn run_scan_benchmark() {
    let mut ctx: runtime::context::GpuContext = runtime::context::GpuContext::new().expect("create GpuContext for scan benchmark");
    let result: serde_json::Value = runtime::bench::scan::run_scan(&mut ctx).expect("scan benchmark failed");
    let json: String = serde_json::to_string_pretty(&result).expect("serialize benchmark result");

    println!("{}", json);

    fs::write("bench_results.json", &json).expect("write bench_results.json");

    let scan: &serde_json::Value = &result["scan"];
    assert!(scan["correct"].as_bool().unwrap(), "scan correctness check failed");
    
    eprintln!(
        "[bench] scan: {:.2} ms GPU, {:.2} ms CPU, {:.2} GB/s, {:.2}x speedup",
        scan["gpu_ms"].as_f64().unwrap(),
        scan["cpu_ms"].as_f64().unwrap(),
        scan["bandwidth_gbps"].as_f64().unwrap(),
        scan["speedup"].as_f64().unwrap(),
    );
}

#[test]
fn run_histogram_benchmark() {
    let mut ctx: runtime::context::GpuContext = runtime::context::GpuContext::new().expect("create GpuContext for histogram benchmark");
    let result: serde_json::Value = runtime::bench::histogram::run_histogram(&mut ctx).expect("histogram benchmark failed");
    let json: String = serde_json::to_string_pretty(&result).expect("serialize benchmark result");

    println!("{}", json);

    let hist: &serde_json::Value = &result["histogram"];
    assert!(hist["correct"].as_bool().unwrap(), "histogram correctness check failed");
}