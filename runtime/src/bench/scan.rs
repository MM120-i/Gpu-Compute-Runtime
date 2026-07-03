use std::time::Instant;
use serde_json::{json, Value};
use ash::vk::{DescriptorSet, BufferUsageFlags};
use gpu_allocator::MemoryLocation; 

use crate::context::GpuContext;
use crate::error::GpuError;
use crate::gpu::buffer::GpuBuffer;
use crate::gpu::pipeline::{BufferBinding, ComputePipeline};
use crate::gpu::dispatcher::{Dispatcher, WorkgroupCount};
use crate::gpu::profiler::{GpuProfiler, BenchmarkReport};

const WG_SIZE: u32 = 256;
const BENCH_ELEMENTS: usize = 1_048_576;   
const ITERATIONS: u32 = 10;

const PASS1_GLSL: &str = include_str!("../../../kernels/benchmarks/scan_pass1.comp");
const PASS2_GLSL: &str = include_str!("../../../kernels/benchmarks/scan_pass2.comp");
const PASS3_GLSL: &str = include_str!("../../../kernels/benchmarks/scan_pass3.comp");
const PASS1_WARP_GLSL: &str = include_str!("../../../kernels/benchmarks/scan_pass1_warp.comp");
const PASS2_WARP_GLSL: &str = include_str!("../../../kernels/benchmarks/scan_pass2_warp.comp");

pub struct ScanState {
    pipeline1: ComputePipeline,
    pipeline2: ComputePipeline,
    pipeline3: ComputePipeline,
    desc1: DescriptorSet,
    desc2: DescriptorSet,
    desc3: DescriptorSet,
    dispatcher1: Dispatcher,
    dispatcher2: Dispatcher,
    dispatcher3: Dispatcher,
    in_buf: GpuBuffer,
    out_buf: GpuBuffer,
    partial_buf: GpuBuffer,
    n: usize,
    wg_count: u32,
}

pub fn init_scan(ctx: &mut GpuContext, n: usize) -> Result<ScanState, GpuError> {
    let wg_count: u32 = (n as u32 + WG_SIZE - 1) / WG_SIZE;
    let u32_size: u64 = std::mem::size_of::<u32>() as u64;

    let in_buf: GpuBuffer = ctx.create_buffer(
        n as u64 * u32_size,
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST, 
        MemoryLocation::CpuToGpu,
    )?;

    let out_buf: GpuBuffer = ctx.create_buffer(
        n as u64 * u32_size,
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_SRC, 
        MemoryLocation::GpuToCpu,
    )?;

    let partial_buf: GpuBuffer = ctx.create_buffer(
        wg_count as u64 * u32_size,
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST, 
        MemoryLocation::GpuToCpu,
    )?;

    let bindings1: [BufferBinding; 3] = [
        BufferBinding {slot: 0},
        BufferBinding {slot: 1},
        BufferBinding {slot: 2},
    ];

    let warp_shuffle: bool = ctx.subgroup_arithmetic;
    
    let src1: &str = if warp_shuffle {
        PASS1_WARP_GLSL
    }
    else{
        PASS1_GLSL
    };

    let src2: &str = if warp_shuffle {
        PASS2_WARP_GLSL
    }
    else{
        PASS2_GLSL
    };

    let pipeline1: ComputePipeline = ComputePipeline::from_glsl_no_opt(ctx, src1, "main", &bindings1)?;
    let desc1: DescriptorSet = pipeline1.create_descriptor_set(ctx, &[&in_buf, &out_buf, &partial_buf])?;
    let dispatcher1: Dispatcher = Dispatcher::new(ctx)?;

    let bindings2: [BufferBinding; 1] = [BufferBinding {slot: 0}];
    let pipeline2: ComputePipeline = ComputePipeline::from_glsl_no_opt(ctx, src2, "main", &bindings2)?;
    let desc2: DescriptorSet = pipeline2.create_descriptor_set(ctx, &[&partial_buf])?;
    let dispatcher2: Dispatcher = Dispatcher::new(ctx)?;

    let bindings3: [BufferBinding; 2] = [
        BufferBinding {slot: 0},
        BufferBinding {slot: 1},
    ];

    let pipeline3: ComputePipeline = ComputePipeline::from_glsl_no_opt(ctx, PASS3_GLSL, "main", &bindings3)?;
    let desc3: DescriptorSet = pipeline3.create_descriptor_set(ctx, &[&out_buf, &partial_buf])?;
    let dispatcher3: Dispatcher = Dispatcher::new(ctx)?;

    Ok(ScanState {
        pipeline1, pipeline2, pipeline3,
        desc1, desc2, desc3,
        dispatcher1, dispatcher2, dispatcher3,
        in_buf, out_buf, partial_buf,
        n, wg_count,
    })
}

pub fn destroy_scan(ctx: &mut GpuContext, state: ScanState) {
    ctx.destroy_dispatcher(state.dispatcher1);
    ctx.destroy_dispatcher(state.dispatcher2);
    ctx.destroy_dispatcher(state.dispatcher3);
    ctx.destroy_pipeline(state.pipeline1);
    ctx.destroy_pipeline(state.pipeline2);
    ctx.destroy_pipeline(state.pipeline3);
    ctx.destroy_buffer(state.in_buf);
    ctx.destroy_buffer(state.out_buf);
    ctx.destroy_buffer(state.partial_buf);
}

fn dispatch_scan(ctx: &mut GpuContext, state: &mut ScanState) -> Result<(), GpuError>{
    let wg1 = Dispatcher::workgroup_count_1d(state.n as u32, WG_SIZE);
    let wg2 = Dispatcher::workgroup_count_1d(state.wg_count, 256);
    let wg3 = Dispatcher::workgroup_count_1d(state.n as u32, WG_SIZE);

    state.dispatcher1.dispatch(ctx, &state.pipeline1, state.desc1, wg1)?;
    state.dispatcher2.dispatch(ctx, &state.pipeline2, state.desc2, wg2)?;
    state.dispatcher3.dispatch(ctx, &state.pipeline3, state.desc3, wg3)?;

    Ok(())
}

pub fn scan_cpu(data: &[u32]) -> Vec<u32> {
    let mut result: Vec<u32> = Vec::with_capacity(data.len());
    let mut sum: u32 = 0u32;

    for &val in data {
        sum = sum.wrapping_add(val);
        result.push(sum);
    }

    result
}
 
fn test_scan_correctness(ctx: &mut GpuContext, state: &mut ScanState) -> Result<(), GpuError> {
    fn run(data: &[u32], ctx: &mut GpuContext, state: &mut ScanState) -> Result<Vec<u32>, GpuError> {
        state.in_buf.upload(data)?;
        let old_n = state.n;
        let old_wg = state.wg_count;
        state.n = data.len();
        state.wg_count = (data.len() as u32 + WG_SIZE - 1) / WG_SIZE;
        dispatch_scan(ctx, state)?;
        state.n = old_n;
        state.wg_count = old_wg;
        state.out_buf.download()
    }

    // 256 elements, first 4 nonzero
    let mut d1 = vec![0u32; 256];
    d1[..4].copy_from_slice(&[1u32, 2, 3, 4]);
    let r1 = run(&d1, ctx, state)?;
    assert_eq!(&r1[..256], &scan_cpu(&d1), "scan: 4 nonzero + zeros");

    // Single element
    let r2 = run(&[42u32], ctx, state)?;
    assert_eq!(&r2[..1], &scan_cpu(&[42u32]), "scan: single element");

    // Two elements
    let r3 = run(&[5u32, 7u32], ctx, state)?;
    assert_eq!(&r3[..2], &scan_cpu(&[5u32, 7u32]), "scan: two elements");

    // Wrapping: u32::MAX + 1 = 0
    let r4 = run(&[u32::MAX, 1u32, 1u32], ctx, state)?;
    assert_eq!(&r4[..3], &scan_cpu(&[u32::MAX, 1u32, 1u32]), "scan: wrapping add");

    // Non-power-of-2
    let d5: Vec<u32> = (0u32..100).collect();
    let r5 = run(&d5, ctx, state)?;
    assert_eq!(&r5[..100], &scan_cpu(&d5), "scan: 100 elements non-pow2");

    // Random large values against CPU
    use rand::RngExt;
    use rand_chacha::ChaCha12Rng;
    use rand::SeedableRng;
    let mut rng = ChaCha12Rng::seed_from_u64(99);
    let d6: Vec<u32> = (0..512).map(|_| rng.random_range(0u32..u32::MAX)).collect();
    let r6 = run(&d6, ctx, state)?;
    assert_eq!(&r6[..512], &scan_cpu(&d6), "scan: 512 random values");

    eprintln!("[bench] scan: all correctness tests passed");
    Ok(())
}

pub fn run_scan(ctx: &mut GpuContext, profiler: &GpuProfiler) -> Result<(Value, BenchmarkReport), GpuError> {
    let device_name: String = ctx.device_name();
    let mut state: ScanState = init_scan(ctx, BENCH_ELEMENTS)?;

    test_scan_correctness(ctx, &mut state)?;

    let data: Vec<u32> = (0u32..BENCH_ELEMENTS as u32).collect();
    state.in_buf.upload(&data)?;

    let wg1: WorkgroupCount = Dispatcher::workgroup_count_1d(BENCH_ELEMENTS as u32, WG_SIZE);
    let wg2: WorkgroupCount = Dispatcher::workgroup_count_1d(state.wg_count, 256);
    let wg3: WorkgroupCount = Dispatcher::workgroup_count_1d(BENCH_ELEMENTS as u32, WG_SIZE);

    dispatch_scan(ctx, &mut state)?;
    ctx.reset_query_pool(0, ITERATIONS * 6);

    let start: Instant = Instant::now();
    let mut gpu_timestamp_ns: u64 = 0;
    let mut total_invocations: u64 = 0;

    for i in 0..ITERATIONS {
        let base: u32 = i * 6;
        profiler.dispatch_profiled(&mut state.dispatcher1, ctx, &state.pipeline1, state.desc1, wg1, base, base + 1, i * 3)?;
        gpu_timestamp_ns += (profiler.get_elapsed_ms(ctx, base)? * 1_000_000.0) as u64;
        total_invocations += profiler.get_invocations(ctx, i * 3)?;

        profiler.dispatch_profiled(&mut state.dispatcher2, ctx, &state.pipeline2, state.desc2, wg2, base + 2, base + 3, i * 3 + 1)?;
        gpu_timestamp_ns += (profiler.get_elapsed_ms(ctx, base + 2)? * 1_000_000.0) as u64;
        total_invocations += profiler.get_invocations(ctx, i * 3 + 1)?;

        profiler.dispatch_profiled(&mut state.dispatcher3, ctx, &state.pipeline3, state.desc3, wg3, base + 4, base + 5, i * 3 + 2)?;
        gpu_timestamp_ns += (profiler.get_elapsed_ms(ctx, base + 4)? * 1_000_000.0) as u64;
        total_invocations += profiler.get_invocations(ctx, i * 3 + 2)?;
    }

    let gpu_dur: std::time::Duration = start.elapsed();
    let gpu_ms: f64 = gpu_dur.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    let gpu_timestamp_ms: f64 = gpu_timestamp_ns as f64 / 1_000_000.0 / ITERATIONS as f64;

    let cpu_start: Instant = Instant::now();

    for _ in 0..ITERATIONS {
        scan_cpu(&data);
    }

    let cpu_dur: std::time::Duration = cpu_start.elapsed();
    let cpu_ms: f64 = cpu_dur.as_secs_f64() * 1000.0 / ITERATIONS as f64;

    destroy_scan(ctx, state);

    let bytes_read: f64 = (BENCH_ELEMENTS * 4) as f64;
    let bytes_written: f64 = (BENCH_ELEMENTS * 4) as f64;
    let bytes: f64 = bytes_read + bytes_written;
    let bandwidth_gbps: f64 = bytes / (gpu_ms / 1000.0) / 1e9;
    let avg_invocations: u64 = total_invocations / ITERATIONS as u64;

    let report = BenchmarkReport {
        name: "scan",
        gpu_ms: (gpu_ms * 100.0).round() / 100.0,
        gpu_timestamp_ms: (gpu_timestamp_ms * 100.0).round() / 100.0,
        invocations: avg_invocations,
        bytes_read,
        bytes_written,
    };

    Ok((json!({
        "scan": {
            "device": device_name,
            "elements": BENCH_ELEMENTS,
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

