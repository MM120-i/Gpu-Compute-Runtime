use std::time::Instant;
use serde_json::{json, Value};
use ash::vk::{DescriptorSet, BufferUsageFlags};
use rand::Rng;
use rand_chacha::ChaCha12Rng;
use rand::SeedableRng;
use gpu_allocator::MemoryLocation; 

use crate::context::GpuContext;
use crate::error::GpuError;
use crate::gpu::buffer::GpuBuffer;
use crate::gpu::pipeline::{BufferBinding, ComputePipeline};
use crate::gpu::dispatcher::Dispatcher;
use crate::gpu::profiler::{GpuProfiler, BenchmarkReport};

const WG_SIZE: u32 = 256;
const BUCKETS: usize = 256;
const BENCH_ELEMENTS: usize = 1_048_576;
const ITERATIONS: u32 = 10;
const RANGE: u32 = 1024;

const HISTOGRAM_GLSL: &str = include_str!("../../../kernels/benchmarks/histogram.comp");

pub struct HistogramState {
    pipeline: ComputePipeline,
    desc: DescriptorSet,
    dispatcher: Dispatcher,
    in_buf: GpuBuffer,
    out_buf: GpuBuffer,
}

pub fn init_histogram(ctx: &mut GpuContext) -> Result<HistogramState, GpuError> {
    let in_buf: GpuBuffer = ctx.create_buffer(
        BENCH_ELEMENTS as u64 * std::mem::size_of::<u32>() as u64,
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu,
    )?;

    let out_buf: GpuBuffer = ctx.create_buffer(
        BUCKETS as u64 * std::mem::size_of::<u32>() as u64,
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::GpuToCpu,
    )?;

    let bindings: [BufferBinding; 2] = [
        BufferBinding { slot: 0 },
        BufferBinding { slot: 1 },
    ];
    let pipeline: ComputePipeline = ComputePipeline::from_glsl_no_opt(ctx, HISTOGRAM_GLSL, "main", &bindings)?;
    let desc: DescriptorSet = pipeline.create_descriptor_set(ctx, &[&in_buf, &out_buf])?;
    let dispatcher: Dispatcher = Dispatcher::new(ctx)?;

    Ok(HistogramState { 
        pipeline, 
        desc, 
        dispatcher, 
        in_buf, 
        out_buf 
    })
}

pub fn destroy_histogram(ctx: &mut GpuContext, state: HistogramState) {
    ctx.destroy_dispatcher(state.dispatcher);
    ctx.destroy_pipeline(state.pipeline);
    ctx.destroy_buffer(state.in_buf);
    ctx.destroy_buffer(state.out_buf);
}

fn histogram_cpu(data: &[u32]) -> Vec<u32> {
    let mut hist: Vec<u32> = vec![0u32; BUCKETS];

    for &val in data {
        let bucket: usize = (val as usize) % BUCKETS;
        hist[bucket] += 1;
    }

    hist
}

fn dispatch_histogram(ctx: &mut GpuContext, state: &mut HistogramState, n: u32) -> Result<(), GpuError> {
    let wg: crate::gpu::WorkgroupCount = Dispatcher::workgroup_count_1d(n, WG_SIZE);
    state.dispatcher.dispatch(ctx, &state.pipeline, state.desc, wg)?;
    Ok(())
}

pub fn run_histogram(ctx: &mut GpuContext, profiler: &GpuProfiler) -> Result<(Value, BenchmarkReport), GpuError> {
    let device_name: String = ctx.device_name();
    let mut state: HistogramState = init_histogram(ctx)?;

    // ── Correctness tests (all data padded to 256 so no stale reads) ──
    {
        // Random data
        let mut rng: ChaCha12Rng = ChaCha12Rng::seed_from_u64(1);
        let d: Vec<u32> = (0..256).map(|_| rng.next_u32() % 256).collect();
        let exp = histogram_cpu(&d);
        state.in_buf.upload(&d)?;
        state.out_buf.upload(&vec![0u32; BUCKETS])?;
        dispatch_histogram(ctx, &mut state, d.len() as u32)?;
        let result: Vec<u32> = state.out_buf.download()?;
        assert_eq!(result, exp, "histogram: random 256");
    }
    {
        // All same bucket
        let d = vec![42u32; 256];
        let exp = histogram_cpu(&d);
        state.in_buf.upload(&d)?;
        state.out_buf.upload(&vec![0u32; BUCKETS])?;
        dispatch_histogram(ctx, &mut state, d.len() as u32)?;
        let result: Vec<u32> = state.out_buf.download()?;
        assert_eq!(result, exp, "histogram: all 42");
    }
    {
        // All value 255 (bucket 255)
        let d = vec![255u32; 256];
        let exp = histogram_cpu(&d);
        state.in_buf.upload(&d)?;
        state.out_buf.upload(&vec![0u32; BUCKETS])?;
        dispatch_histogram(ctx, &mut state, d.len() as u32)?;
        let result: Vec<u32> = state.out_buf.download()?;
        assert_eq!(result, exp, "histogram: all 255");
    }
    {
        // Alternating 0 and 127 → buckets 0 and 127
        let mut d = vec![0u32; 256];
        for i in (1..256).step_by(2) { d[i] = 127; }
        let exp = histogram_cpu(&d);
        state.in_buf.upload(&d)?;
        state.out_buf.upload(&vec![0u32; BUCKETS])?;
        dispatch_histogram(ctx, &mut state, d.len() as u32)?;
        let result: Vec<u32> = state.out_buf.download()?;
        assert_eq!(result, exp, "histogram: alternating 0/127");
    }
    eprintln!("[bench] histogram: all correctness tests passed");

    let mut rng: ChaCha12Rng = ChaCha12Rng::seed_from_u64(42);
    let data: Vec<u32> = (0..BENCH_ELEMENTS).map(|_| rng.next_u32() % RANGE).collect();

    state.in_buf.upload(&data)?;
    state.out_buf.upload(&vec![0u32; BUCKETS])?;

    dispatch_histogram(ctx, &mut state, BENCH_ELEMENTS as u32)?;
    ctx.reset_query_pool(0, ITERATIONS * 2);

    let start = Instant::now();
    let mut gpu_timestamp_ns: u64 = 0;
    let mut total_invocations: u64 = 0;

    let wg = Dispatcher::workgroup_count_1d(BENCH_ELEMENTS as u32, WG_SIZE);

    for i in 0..ITERATIONS {
        state.out_buf.upload(&vec![0u32; BUCKETS])?;
        profiler.dispatch_profiled(&mut state.dispatcher, ctx, &state.pipeline, state.desc, wg, i * 2, i * 2 + 1, i)?;
        gpu_timestamp_ns += (profiler.get_elapsed_ms(ctx, i * 2)? * 1_000_000.0) as u64;
        total_invocations += profiler.get_invocations(ctx, i)?;
    }

    let gpu_dur: std::time::Duration = start.elapsed();
    let gpu_ms: f64 = gpu_dur.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    let gpu_timestamp_ms: f64 = gpu_timestamp_ns as f64 / 1_000_000.0 / ITERATIONS as f64;

    let cpu_start = Instant::now();
    for _ in 0..ITERATIONS {
        histogram_cpu(&data);
    }

    let cpu_dur: std::time::Duration = cpu_start.elapsed();
    let cpu_ms: f64 = cpu_dur.as_secs_f64() * 1000.0 / ITERATIONS as f64;

    destroy_histogram(ctx, state);

    let bytes_read: f64 = (BENCH_ELEMENTS * 4) as f64;
    let bytes_written: f64 = (BUCKETS * 4) as f64;
    let bytes: f64 = bytes_read + bytes_written;
    let bandwidth_gbps: f64 = bytes / (gpu_ms / 1000.0) / 1e9;
    let avg_invocations: u64 = total_invocations / ITERATIONS as u64;

    let report = BenchmarkReport {
        name: "histogram",
        gpu_ms: (gpu_ms * 100.0).round() / 100.0,
        gpu_timestamp_ms: (gpu_timestamp_ms * 100.0).round() / 100.0,
        invocations: avg_invocations,
        bytes_read,
        bytes_written,
    };

    Ok((json!({
        "histogram": {
            "device": device_name,
            "elements": BENCH_ELEMENTS,
            "buckets": BUCKETS,
            "range": RANGE,
            "workgroup_size": WG_SIZE,
            "gpu_ms": (gpu_ms * 100.0).round() / 100.0,
            "gpu_timestamp_ms": (gpu_timestamp_ms * 100.0).round() / 100.0,
            "cpu_ms": (cpu_ms * 100.0).round() / 100.0,
            "bandwidth_gbps": (bandwidth_gbps * 100.0).round() / 100.0,
            "speedup": (cpu_ms / gpu_ms * 100.0).round() / 100.0,
            "correct": true,
        }
    }), report))
}