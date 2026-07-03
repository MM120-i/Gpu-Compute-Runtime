use std::fs;
use runtime::gpu::profiler::{GpuProfiler, BenchmarkReport};

fn print_summary(key: &str, data: &serde_json::Value) {
    let gpu = data["gpu_ms"].as_f64().unwrap_or(0.0);
    let cpu = data["cpu_ms"].as_f64().unwrap_or(0.0);
    let bw  = data["bandwidth_gbps"].as_f64().unwrap_or(0.0);
    let sp  = data["speedup"].as_f64().unwrap_or(0.0);
    eprintln!("[bench] {}: {:.2} ms GPU, {:.2} ms CPU, {:.2} GB/s, {:.2}x speedup", key, gpu, cpu, bw, sp);
}

#[test]
fn run_all_benchmarks() {
    let mut combined: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let mut reports: Vec<BenchmarkReport> = Vec::new();

    // scan
    {
        let mut ctx = runtime::context::GpuContext::new().expect("create GpuContext");
        let profiler = GpuProfiler::new(&ctx);
        let (result, report) = runtime::bench::scan::run_scan(&mut ctx, &profiler).expect("scan benchmark failed");
        let scan = &result["scan"];

        assert!(scan["correct"].as_bool().unwrap(), "scan correctness check failed");

        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        print_summary("scan", scan);
        combined.insert("scan".into(), result["scan"].clone());
        reports.push(report);
    }

    // histogram
    {
        let mut ctx = runtime::context::GpuContext::new().expect("create GpuContext");
        let profiler = GpuProfiler::new(&ctx);
        let (result, report) = runtime::bench::histogram::run_histogram(&mut ctx, &profiler).expect("histogram benchmark failed");
        let hist = &result["histogram"];

        assert!(hist["correct"].as_bool().unwrap(), "histogram correctness check failed");

        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        print_summary("histogram", hist);
        combined.insert("histogram".into(), result["histogram"].clone());
        reports.push(report);
    }

    // spmv
    {
        let mut ctx = runtime::context::GpuContext::new().expect("create GpuContext");
        let profiler = GpuProfiler::new(&ctx);
        let (result, report) = runtime::bench::spmv::run_spmv(&mut ctx, &profiler).expect("spmv benchmark failed");
        let spmv = &result["spmv"];

        assert!(spmv["correct"].as_bool().unwrap(), "spmv correctness check failed");

        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        print_summary("spmv", spmv);
        combined.insert("spmv".into(), result["spmv"].clone());
        reports.push(report);
    }

    {
        let ctx = runtime::context::GpuContext::new().expect("create GpuContext");
        let display = GpuProfiler::new(&ctx);
        display.print_report(&ctx.device_name(), &reports);
    }

    let json = serde_json::to_string_pretty(&combined).expect("serialize");
    let path = "../docs/WebPage/chart.js/public/bench_results.json";

    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent).expect("create public dir");
    }

    fs::write("bench_results.json", &json).expect("write bench_results.json");
    fs::write(path, &json).expect("write bench_results.json (dashboard)");

    eprintln!("[bench] all results written to bench_results.json");
}