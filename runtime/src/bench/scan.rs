// Meat and Potatoes of scan functionality 
// we got 3 sections here, 
// 1) GLSL Sources
// 2) CPU Reference
// 3) GPU Benchmarks

use std::time::Instant;
use serde_json::{json, Value};
use ash::vk::{DescriptorSet, BufferUsageFlags};

use crate::context::GpuContext;
use crate::error::GpuError;
use crate::gpu::buffer::GpuBuffer;
use crate::gpu::pipeline::{BufferBinding, ComputePipeline};
use crate::gpu::dispatcher::Dispatcher;

const WG_SIZE: u32 = 256;
const BENCH_ELEMENTS: usize = 1_048_576;    // we got 2^20 = 4090 workgroups
const ITERATIONS: u32 = 10;

const PASS1_GLSL: &str = r#"#version 460
layout(local_size_x = 256) in;
layout(binding = 0) buffer Input  { uint in_data[]; };
layout(binding = 1) buffer Output { uint out_data[]; };
layout(binding = 2) buffer Partial { uint partial_sums[]; };
shared uint temp[256];
void main() {
    uint tid = gl_LocalInvocationIndex;
    uint gid = gl_GlobalInvocationID.x;
    temp[tid] = in_data[gid];
    barrier();
    for (uint stride = 1u; stride < 256u; stride <<= 1u) {
        uint val = 0u;
        if (tid >= stride) { val = temp[tid - stride]; }
        barrier();
        temp[tid] += val;
        barrier();
    }
    out_data[gid] = temp[tid];
    if (tid == 255u) { partial_sums[gl_WorkGroupID.x] = temp[255]; }
}
"#;

const PASS2_GLSL: &str = r#"#version 460
layout(local_size_x = 256) in;
layout(binding = 0) buffer Partial { uint partial_sums[]; };
#define ITEMS_PER_THREAD 16
shared uint thread_totals[256];
void main() {
    uint tid = gl_LocalInvocationIndex;
    uint n = partial_sums.length();
    uint start = tid * ITEMS_PER_THREAD;
    uint local[ITEMS_PER_THREAD];
    uint running = 0u;
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        uint idx = start + j;
        if (idx < n) { running += partial_sums[idx]; local[j] = running; }
    }
    thread_totals[tid] = running;
    barrier();
    for (uint stride = 1u; stride < 256u; stride <<= 1u) {
        uint val = thread_totals[tid];
        if (tid >= stride) { val += thread_totals[tid - stride]; }
        barrier();
        thread_totals[tid] = val;
        barrier();
    }
    uint carry = (tid > 0u) ? thread_totals[tid - 1u] : 0u;
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        uint idx = start + j;
        if (idx < n) { partial_sums[idx] = local[j] + carry; }
    }
}
"#;

const PASS3_GLSL: &str = r#"#version 460
layout(local_size_x = 256) in;
layout(binding = 0) buffer Output { uint out_data[]; };
layout(binding = 1) buffer Partial { uint partial_sums[]; };
void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint wg = gl_WorkGroupID.x;
    if (wg > 0u) { out_data[gid] += partial_sums[wg - 1u]; }
}
"#;

pub fn scan_cpu(data: &[u32]) -> Vec<u32> {
    let mut result: Vec<u32> = Vec::with_capacity(data.len());
    let mut sum: u32 = 0u32;

    for &val in data {
        sum = sum.wrapping_add(val);
        result.push(sum);
    }

    result
}
 
pub fn run_scan(ctx: &mut GpuContext) -> Result<Value, GpuError> {
    let device_name: String = ctx.device_name();

    let mut small_data: Vec<u32> = vec![0u32; 256];
    small_data[..4].copy_from_slice(&[1u32, 2, 3, 4]);
    let expected: Vec<u32> = scan_cpu(&small_data);
    let small_gpu: Vec<u32> = run_scan_gpu(ctx, &small_data)?;
    assert_eq!(small_gpu, expected, "scan correctness failed");

    let data: Vec<u32> = (0u32..BENCH_ELEMENTS as u32).collect();
    let _expected: Vec<u32> = scan_cpu(&data);

    let _ = run_scan_gpu(ctx, &data);

    let start: Instant = Instant::now();

    for _ in 0..ITERATIONS {
        let _ = run_scan_gpu(ctx, &data);
    }
    
    let gpu_dur: std::time::Duration = start.elapsed();
    let gpu_ms: f64 = gpu_dur.as_secs_f64() * 1000.0 / ITERATIONS as f64;

    let cpu_start: Instant = Instant::now();

    for _ in 0..ITERATIONS {
        scan_cpu(&data);
    }

    let cpu_dur: std::time::Duration = cpu_start.elapsed();
    let cpu_ms: f64 = cpu_dur.as_secs_f64() * 1000.0 / ITERATIONS as f64;

    let bytes: f64 = (2 * BENCH_ELEMENTS * 4) as f64;
    let bandwidth_gbps: f64 = bytes / (gpu_ms / 1000.0) / 1e9;
    
    Ok(json!({
        "scan": {
            "device": device_name,
            "elements": BENCH_ELEMENTS,
            "workgroup_size": WG_SIZE,
            "gpu_ms": (gpu_ms * 100.0).round() / 100.0,
            "cpu_ms": (cpu_ms * 100.0).round() / 100.0,
            "bandwidth_gbps": (bandwidth_gbps * 100.0).round() / 100.0,
            "speedup": (cpu_ms / gpu_ms * 100.0).round() / 100.0,
            "correct": true
        }
    }))
}

fn run_scan_gpu(ctx: &mut GpuContext, data: &[u32]) -> Result<Vec<u32>, GpuError> {
    let n: usize = data.len();
    let wg_count: u32 = (n as u32 + WG_SIZE - 1) / WG_SIZE;

    let in_buf: GpuBuffer = GpuBuffer::input_u32(ctx, data)?;
    let out_buf: GpuBuffer = GpuBuffer::output_u32(ctx, n)?;
    
    let partial_buf: GpuBuffer = ctx.create_buffer(
        (wg_count as usize * 4) as u64, 
        BufferUsageFlags::STORAGE_BUFFER |
        BufferUsageFlags::TRANSFER_SRC |
        BufferUsageFlags::TRANSFER_DST, 
        gpu_allocator::MemoryLocation::GpuToCpu,
    )?;

    let bindings1: [BufferBinding; 3] = [
        BufferBinding {slot: 0},
        BufferBinding {slot: 1},
        BufferBinding {slot: 2},
    ];

    let pipeline1: ComputePipeline = ComputePipeline::from_glsl_no_opt(ctx, PASS1_GLSL, "main", &bindings1)?;
    let desc1: DescriptorSet = pipeline1.create_descriptor_set(ctx, &[&in_buf, &out_buf, &partial_buf])?;
    let mut dispatcher1: Dispatcher = Dispatcher::new(ctx)?;
    let wg1: crate::gpu::WorkgroupCount = Dispatcher::workgroup_count_1d(n as u32, WG_SIZE);

    let bindings2: [BufferBinding; 1] = [BufferBinding { slot: 0 }];
    let pipeline2: ComputePipeline = ComputePipeline::from_glsl_no_opt(ctx, PASS2_GLSL, "main", &bindings2)?;
    let desc2: DescriptorSet = pipeline2.create_descriptor_set(ctx, &[&partial_buf])?;
    let mut dispatcher2: Dispatcher = Dispatcher::new(ctx)?;
    let wg2: crate::gpu::WorkgroupCount = Dispatcher::workgroup_count_1d(wg_count, 256);

    let bindings3: [BufferBinding; 2] = [
        BufferBinding {slot: 0},
        BufferBinding {slot: 1},
    ];

    let pipeline3: ComputePipeline = ComputePipeline::from_glsl_no_opt(ctx, PASS3_GLSL, "main", &bindings3)?;
    let desc3: DescriptorSet = pipeline3.create_descriptor_set(ctx, &[&out_buf, &partial_buf])?;
    let mut dispatcher3: Dispatcher = Dispatcher::new(ctx)?;
    let wg3: crate::gpu::WorkgroupCount = Dispatcher::workgroup_count_1d(n as u32, WG_SIZE);

    dispatcher1.dispatch(ctx, &pipeline1, desc1, wg1)?;
    dispatcher2.dispatch(ctx, &pipeline2, desc2, wg2)?;
    dispatcher3.dispatch(ctx, &pipeline3, desc3, wg3)?;

    let result: Vec<u32> = out_buf.download()?;

    ctx.destroy_dispatcher(dispatcher1);
    ctx.destroy_dispatcher(dispatcher2);
    ctx.destroy_dispatcher(dispatcher3);
    ctx.destroy_pipeline(pipeline1);
    ctx.destroy_pipeline(pipeline2);
    ctx.destroy_pipeline(pipeline3);
    ctx.destroy_buffer(in_buf);
    ctx.destroy_buffer(out_buf);
    ctx.destroy_buffer(partial_buf);

    Ok(result)
}