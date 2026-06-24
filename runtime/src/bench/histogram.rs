use std::time::Instant;
use serde_json::{json, Value};
use ash::vk::{DescriptorSet, BufferUsageFlags};
use rand::Rng;
use rand_chacha::ChaCha12Rng;
use rand::SeedableRng;

use crate::context::GpuContext;
use crate::error::GpuError;
use crate::gpu::buffer::GpuBuffer;
use crate::gpu::pipeline::{BufferBinding, ComputePipeline};
use crate::gpu::dispatcher::Dispatcher;

const WG_SIZE: u32 = 256;
const BUCKETS: usize = 256;
const BENCH_ELEMENTS: usize = 1_048_576;
const ITERATIONS: u32 = 10;
const RANGE: u32 = 1024;

const HISTOGRAM_GLSL: &str = r#"#version 460
layout(local_size_x = 256) in;
layout(binding = 0) buffer Input { uint data[]; };
layout(binding = 1) buffer Output { uint hist[]; };

shared uint smem[256];

void main() {
    uint tid = gl_LocalInvocationIndex;
    uint gid = gl_GlobalInvocationID.x;

    smem[tid] = 0u;
    barrier();

    if (gid < data.length()) {
        uint bucket = data[gid] % 256u;
        atomicAdd(smem[bucket], 1u);
    }
    barrier();

    atomicAdd(hist[tid], smem[tid]);
}
"#;

fn histogram_cpu(data: &[u32]) -> Vec<u32> {
    let mut hist: Vec<u32> = vec![0u32; BUCKETS];

    for &val in data {
        let bucket: usize = (val as usize) % BUCKETS;
        hist[bucket] += 1;
    }

    hist
}

fn run_histogram_gpu(ctx: &mut GpuContext, data: &[u32]) -> Result<Vec<u32>, GpuError> {
    let n: usize = data.len();
    let hist_size: usize = BUCKETS * 4;
    let in_buf: GpuBuffer = GpuBuffer::input_u32(ctx, data)?;
    
    let out_buf: GpuBuffer = ctx.create_buffer(
        hist_size as u64, 
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_SRC, 
        gpu_allocator::MemoryLocation::GpuToCpu,
    )?;

    let zeros: Vec<u32> = vec![0u32; BUCKETS];
    out_buf.upload(&zeros)?;

    let bindings: [BufferBinding; 2] = [
        BufferBinding {slot: 0},
        BufferBinding {slot: 1},
    ];

    let pipeline: ComputePipeline = ComputePipeline::from_glsl_no_opt(ctx, HISTOGRAM_GLSL, "main", &bindings)?;
    let desc: DescriptorSet = pipeline.create_descriptor_set(ctx, &[&in_buf, &out_buf])?;
    let mut dispatcher: Dispatcher = Dispatcher::new(ctx)?;
    let wg: crate::gpu::WorkgroupCount = Dispatcher::workgroup_count_1d(n as u32, WG_SIZE);

    dispatcher.dispatch(ctx, &pipeline, desc, wg)?;

    let result: Vec<u32> = out_buf.download()?;

    ctx.destroy_dispatcher(dispatcher);
    ctx.destroy_pipeline(pipeline);
    ctx.destroy_buffer(in_buf);
    ctx.destroy_buffer(out_buf);

    Ok(result)
}

pub fn run_histogram(ctx: &mut GpuContext) -> Result<Value, GpuError> {
    let device_name: String = ctx.device_name();

    let mut rng: ChaCha12Rng = ChaCha12Rng::seed_from_u64(1);
    let small_data: Vec<u32> = (0..256).map(|_| rng.next_u32() % 256).collect();
    let expected: Vec<u32> = histogram_cpu(&small_data);
    let gpu_result: Vec<u32> = run_histogram_gpu(ctx, &small_data)?;
    assert_eq!(gpu_result, expected, "histogram correctness failed");

    let mut rng: ChaCha12Rng = ChaCha12Rng::seed_from_u64(42);
    let data: Vec<u32> = (0..BENCH_ELEMENTS).map(|_| rng.next_u32() % RANGE).collect();
    let _expected: Vec<u32> = histogram_cpu(&data);
    let _ = run_histogram_gpu(ctx, &data)?;
    let start: Instant = Instant::now();

    for _ in 0..ITERATIONS {
        let _ = run_histogram_gpu(ctx, &data)?;
    }

    let gpu_dur: std::time::Duration = start.elapsed();
    let gpu_ms: f64 = gpu_dur.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    let cpu_start: Instant = Instant::now();

    for _ in 0..ITERATIONS {
        histogram_cpu(&data);
    }

    let cpu_dur: std::time::Duration = cpu_start.elapsed();
    let cpu_ms: f64 = cpu_dur.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    let bytes: f64 = (BENCH_ELEMENTS * 4 + BUCKETS * 4) as f64;
    let bandwidth_gbps: f64 = bytes / (gpu_ms / 1000.0) / 1e9;

    Ok(json!({
        "histogram": {
            "device": device_name,
            "elements": BENCH_ELEMENTS,
            "buckets": BUCKETS,
            "range": RANGE,
            "workgroup_size": WG_SIZE,
            "gpu_ms": (gpu_ms * 100.0).round() / 100.0,
            "cpu_ms": (cpu_ms * 100.0).round() / 100.0,
            "bandwidth_gbps": (bandwidth_gbps * 100.0).round() / 100.0,
            "speedup": (cpu_ms / gpu_ms * 100.0).round() / 100.0,
            "correct": true
        }
    }))
}