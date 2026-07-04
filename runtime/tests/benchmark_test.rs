use std::fs;
use serde_json::json;
use sysinfo::System;

fn collect_system_info() -> serde_json::Value {
    let mut sys: System = System::new_all();
    sys.refresh_all();

    let cpu: String = sys.cpus().first().map(|c: &sysinfo::Cpu| c.brand().to_string()).unwrap_or_else(|| "Unknown CPU".into());
    let cores: usize = sys.cpus().len();
    let ram_gb: f64 = sys.total_memory() as f64 / 1_073_741_824.0;
    let os: String = format!("{} {}", System::name().unwrap_or_else(|| "Unknown OS".into()), System::os_version().unwrap_or_else(|| String::new()));
    let os: String = os.trim().to_string();

    json!({
        "cpu": cpu,
        "cores": cores,
        "ram_gb": (ram_gb * 100.0).round() / 100.0,
        "os": os,
    })
}

fn print_summary(key: &str, data: &serde_json::Value) {
    let gpu: f64 = data["gpu_ms"].as_f64().unwrap_or(0.0);
    let cpu: f64 = data["cpu_ms"].as_f64().unwrap_or(0.0);
    let bw: f64  = data["bandwidth_gbps"].as_f64().unwrap_or(0.0);
    let sp: f64  = data["speedup"].as_f64().unwrap_or(0.0);

    eprintln!("[bench] {}: {:.2} ms GPU, {:.2} ms CPU, {:.2} GB/s, {:.2}x speedup", key, gpu, cpu, bw, sp);
}

#[test]
fn run_all_benchmarks() {
    let mut combined: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();

    let mut gpu_name: String = String::new();
    let mut vulkan_ver: String = String::new();
    let mut subgroup_sz: u32 = 0;

    // scan
    {
        let mut ctx: runtime::context::GpuContext = runtime::context::GpuContext::new().expect("create GpuContext");
        gpu_name = ctx.device_name();
        vulkan_ver = ctx.vulkan_version();
        subgroup_sz = ctx.subgroup_size();
        let result: serde_json::Value = runtime::bench::scan::run_scan(&mut ctx).expect("scan benchmark failed");
        let scan: &serde_json::Value = &result["scan"];

        assert!(scan["correct"].as_bool().unwrap(), "scan correctness check failed");

        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        print_summary("scan", scan);
        combined.insert("scan".into(), result["scan"].clone());
    }

    // histogram
    {
        let mut ctx: runtime::context::GpuContext = runtime::context::GpuContext::new().expect("create GpuContext");
        let result: serde_json::Value = runtime::bench::histogram::run_histogram(&mut ctx).expect("histogram benchmark failed");
        let hist: &serde_json::Value = &result["histogram"];

        assert!(hist["correct"].as_bool().unwrap(), "histogram correctness check failed");

        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        print_summary("histogram", hist);
        combined.insert("histogram".into(), result["histogram"].clone());
    }

    // spmv
    {
        let mut ctx: runtime::context::GpuContext = runtime::context::GpuContext::new().expect("create GpuContext");
        let result: serde_json::Value = runtime::bench::spmv::run_spmv(&mut ctx).expect("spmv benchmark failed");
        let spmv: &serde_json::Value = &result["spmv"];

        assert!(spmv["correct"].as_bool().unwrap(), "spmv correctness check failed");

        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        print_summary("spmv", spmv);
        combined.insert("spmv".into(), result["spmv"].clone());
    }

    let mut sys: serde_json::Value = collect_system_info();
    sys["gpu"] = serde_json::Value::String(gpu_name.clone());
    sys["vulkan"] = serde_json::Value::String(vulkan_ver);
    sys["subgroup_size"] = serde_json::Value::Number(subgroup_sz.into());

    let mut ordered: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    ordered.insert("system".into(), sys);

    for (k, v) in combined {
        ordered.insert(k, v);
    }

    let json: String = serde_json::to_string_pretty(&ordered).expect("serialize");
    let path: &str = "../docs/WebPage/chart.js/public/bench_results.json";

    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent).expect("create public dir");
    }

    fs::write("bench_results.json", &json).expect("write bench_results.json");
    fs::write(path, &json).expect("write bench_results.json (dashboard)");

    eprintln!("[bench] all results written to bench_results.json");
}
