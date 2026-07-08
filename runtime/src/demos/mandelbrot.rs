use std::fs;
use std::time::Instant;
use ash::vk::{DescriptorSet, BufferUsageFlags};
use gpu_allocator::MemoryLocation;
use serde_json::{json, Value};

use crate::context::GpuContext;
use crate::error::GpuError;
use crate::gpu::buffer::GpuBuffer;
use crate::gpu::pipeline::{BufferBinding, ComputePipeline};
use crate::gpu::dispatcher::{Dispatcher, WorkgroupCount};
use crate::gpu::profiler::{GpuProfiler, BenchmarkReport};

const WG: u32 = 16;
const SHADER: &str = include_str!("../../../kernels/demos/mandelbrot.glsl");

#[repr(C)]
#[derive(Clone, Copy)]
struct MandelbrotParams {
    width: u32,
    height: u32,
    max_iters: u32,
    cx: f32,
    cy: f32,
    scale: f32,
}

pub fn render_gpu(
    ctx: &mut GpuContext,
    width: u32,
    height: u32,
    max_iters: u32,
    cx: f32,
    cy: f32,
    scale: f32,
) -> Result<Vec<u32>, GpuError> {
    if width == 0 || height == 0 {
        return Err(GpuError::Buffer("image dimensions must be non-zero"));
    }

    let total: u64 = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(GpuError::Buffer("image dimensions overflow"))?;

    let output_bytes: u64 = total
        .checked_mul(std::mem::size_of::<u32>() as u64)
        .ok_or(GpuError::Buffer("output buffer size overflow"))?;

    let out_buf: GpuBuffer = ctx.create_buffer(
        output_bytes,
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_SRC, 
        MemoryLocation::GpuToCpu,
    )?;

    let params: MandelbrotParams = MandelbrotParams {
        width,
        height,
        max_iters,
        cx,
        cy,
        scale,
    };

    let params_buf: GpuBuffer = ctx.create_buffer(
        std::mem::size_of::<MandelbrotParams>() as u64,
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu,
    )?;

    params_buf.upload(&[params])?;

    let bindings: [BufferBinding; 2] = [BufferBinding {slot: 0}, BufferBinding {slot: 1}];
    let pipeline: ComputePipeline = ComputePipeline::from_glsl_no_opt(ctx, SHADER, "main", &bindings)?;
    let desc: DescriptorSet = pipeline.create_descriptor_set(ctx, &[&out_buf, &params_buf])?;
    let mut dispatcher: Dispatcher = Dispatcher::new(ctx)?;

    let wg_x: u32 = width / WG + u32::from(width % WG != 0);
    let wg_y: u32 = height / WG + u32::from(height % WG != 0);
    let wg: WorkgroupCount = WorkgroupCount { x: wg_x, y: wg_y, z: 1 };

    dispatcher.dispatch(ctx, &pipeline, desc, wg)?;

    let result: Vec<u32> = out_buf.download::<u32>()?;

    ctx.destroy_buffer(out_buf);
    ctx.destroy_buffer(params_buf);
    ctx.destroy_dispatcher(dispatcher);
    ctx.destroy_pipeline(pipeline);

    Ok(result)
}

pub const BENCH_WIDTH: u32 = 1024;
pub const BENCH_HEIGHT: u32 = 1024;
pub const BENCH_ITERS: u32 = 200;
const BENCH_CPU_ITERS: u32 = 10;
const GPUBENCH_ITERS: u32 = 10;

pub fn mandelbrot_cpu(
    width: u32,
    height: u32,
    max_iters: u32,
    cx: f32,
    cy: f32,
    scale: f32,
) -> Vec<u32> {
    let total: usize = (width as usize) * (height as usize);
    let mut pixels: Vec<u32> = vec![0u32; total];
    let aspect: f32 = width as f32 / height as f32;

    for py in 0..height {
        for px in 0..width {
            let x0: f32 = (px as f32 / width as f32 - 0.5) * scale * aspect + cx;
            let y0: f32 = (py as f32 / height as f32 - 0.5) * scale + cy;
            let (mut x, mut y) = (0.0f32, 0.0f32);
            let mut iter: u32 = 0;

            while iter < max_iters && (x * x + y * y) <= 4.0 {
                let xtemp: f32 = x * x - y * y + x0;
                y = 2.0 * x * y + y0;
                x = xtemp;
                iter += 1;
            }

            pixels[(py * width + px) as usize] = iter;
        }
    }

    pixels
}

struct MandelbrotBenchState {
    pipeline: ComputePipeline,
    desc: DescriptorSet,
    dispatcher: Dispatcher,
    out_buf: GpuBuffer,
    params_buf: GpuBuffer,
    wg: WorkgroupCount,
}

fn init_bench_state(
    ctx: &mut GpuContext,
    width: u32,
    height: u32,
    max_iters: u32,
    cx: f32,
    cy: f32,
    scale: f32,
) -> Result<MandelbrotBenchState, GpuError> {
    let total: u64 = u64::from(width).checked_mul(u64::from(height)).ok_or(GpuError::Buffer("image dimensions overflow"))?;
    let output_bytes: u64 = total.checked_mul(std::mem::size_of::<u32>() as u64).ok_or(GpuError::Buffer("output buffer size overflow"))?;

    let out_buf: GpuBuffer = ctx.create_buffer(
        output_bytes,
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::GpuToCpu,
    )?;

    let params_buf: GpuBuffer = ctx.create_buffer(
        std::mem::size_of::<MandelbrotParams>() as u64,
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::CpuToGpu,
    )?;

    let params: MandelbrotParams = MandelbrotParams { width, height, max_iters, cx, cy, scale };
    params_buf.upload(&[params])?;

    let bindings: [BufferBinding; 2] = [BufferBinding { slot: 0 }, BufferBinding { slot: 1 }];
    let pipeline: ComputePipeline = ComputePipeline::from_glsl_no_opt(ctx, SHADER, "main", &bindings)?;
    let desc: DescriptorSet = pipeline.create_descriptor_set(ctx, &[&out_buf, &params_buf])?;
    let dispatcher: Dispatcher = Dispatcher::new(ctx)?;

    let wg_x: u32 = width / WG + u32::from(width % WG != 0);
    let wg_y: u32 = height / WG + u32::from(height % WG != 0);
    let wg: WorkgroupCount = WorkgroupCount { x: wg_x, y: wg_y, z: 1 };

    Ok(MandelbrotBenchState { pipeline, desc, dispatcher, out_buf, params_buf, wg })
}

fn destroy_bench_state(ctx: &mut GpuContext, state: MandelbrotBenchState) {
    ctx.destroy_dispatcher(state.dispatcher);
    ctx.destroy_pipeline(state.pipeline);
    ctx.destroy_buffer(state.out_buf);
    ctx.destroy_buffer(state.params_buf);
}

pub fn bench_mandelbrot(
    ctx: &mut GpuContext,
    profiler: &GpuProfiler,
) -> Result<(Value, BenchmarkReport), GpuError> {
    let device_name: String = ctx.device_name();
    let (cx, cy, scale) = (-0.5, 0.0, 3.5);

    let mut state: MandelbrotBenchState = init_bench_state(ctx, BENCH_WIDTH, BENCH_HEIGHT, BENCH_ITERS, cx, cy, scale)?;

    state.dispatcher.dispatch(ctx, &state.pipeline, state.desc, state.wg)?;

    let gpu_correct: Vec<u32> = state.out_buf.download::<u32>()?;
    let cpu_correct: Vec<u32> = mandelbrot_cpu(BENCH_WIDTH, BENCH_HEIGHT, BENCH_ITERS, cx, cy, scale);

    let mismatches: usize = gpu_correct.iter().zip(cpu_correct.iter()).filter(|(a, b)| {
        let diff: u32 = if *a > *b {
             *a - *b 
        } 
        else { 
            *b - *a 
        };

        diff > 5
    }).count();

    let total: usize = (BENCH_WIDTH * BENCH_HEIGHT) as usize;
    let pct: f64 = mismatches as f64 / total as f64 * 100.0;

    eprintln!("[bench] mandelbrot: {} / {} pixels differ by >5 iterations ({:.2}%)", mismatches, total, pct);
    assert!(pct < 1.0, "mandelbrot: too many pixel mismatches ({:.2}%) — expected <1% due to fp32 boundary divergence", pct);

    ctx.reset_query_pool(0, GPUBENCH_ITERS * 2);

    let start: Instant = Instant::now();
    let mut gpu_timestamp_ns: u64 = 0;
    let mut total_invocations: u64 = 0;

    for i in 0..GPUBENCH_ITERS {
        let slot = i * 2;

        profiler.dispatch_profiled(
            &mut state.dispatcher, ctx, &state.pipeline, state.desc,
            state.wg, slot, slot + 1, i,
        )?;

        gpu_timestamp_ns += (profiler.get_elapsed_ms(ctx, slot)? * 1_000_000.0) as u64;
        total_invocations += profiler.get_invocations(ctx, i)?;
    }

    let gpu_dur: std::time::Duration = start.elapsed();
    let gpu_ms: f64 = gpu_dur.as_secs_f64() * 1000.0 / GPUBENCH_ITERS as f64;
    let gpu_timestamp_ms: f64 = gpu_timestamp_ns as f64 / 1_000_000.0 / GPUBENCH_ITERS as f64;
    let avg_invocations: u64 = total_invocations / GPUBENCH_ITERS as u64;

    let cpu_start: Instant = Instant::now();

    for _ in 0..BENCH_CPU_ITERS {
        mandelbrot_cpu(BENCH_WIDTH, BENCH_HEIGHT, BENCH_ITERS, cx, cy, scale);
    }

    let cpu_dur: std::time::Duration = cpu_start.elapsed();
    let cpu_ms: f64 = cpu_dur.as_secs_f64() * 1000.0 / BENCH_CPU_ITERS as f64;

    destroy_bench_state(ctx, state);

    let gpu_ms_r: f64 = (gpu_ms * 100.0).round() / 100.0;
    let gpu_ts_ms_r: f64 = (gpu_timestamp_ms * 100.0).round() / 100.0;
    let cpu_ms_r: f64 = (cpu_ms * 100.0).round() / 100.0;
    let speedup_r: f64 = (cpu_ms / gpu_ms * 100.0).round() / 100.0;

    let report: BenchmarkReport = BenchmarkReport {
        name: "mandelbrot",
        gpu_ms: gpu_ms_r,
        gpu_timestamp_ms: gpu_ts_ms_r,
        invocations: avg_invocations,
        bytes_read: (BENCH_WIDTH * BENCH_HEIGHT * 4) as f64,
        bytes_written: (BENCH_WIDTH * BENCH_HEIGHT * 4) as f64,
    };

    Ok((json!({
        "mandelbrot": {
            "device": device_name,
            "width": BENCH_WIDTH,
            "height": BENCH_HEIGHT,
            "max_iters": BENCH_ITERS,
            "workgroup_size": WG,
            "gpu_ms": gpu_ms_r,
            "gpu_timestamp_ms": gpu_ts_ms_r,
            "cpu_ms": cpu_ms_r,
            "invocations": avg_invocations,
            "speedup": speedup_r,
            "correct": true,
        }
    }), report))
}

fn write_ppm(path: &str, pixels: &[u32], width: u32, height: u32) -> Result<(), GpuError> {
    let mut contents: Vec<u8> = format!("P6\n{} {}\n255\n", width, height).into_bytes();

    for &iter in pixels {
        let (r, g, b) = if iter == 0 {
            (0u8, 0u8, 0u8)
        }
        else{
            let t: f32 = iter as f32 / 200.0;
            let r: u8 = (t.min(1.0) * 255.0) as u8;
            let g: u8 = ((t * 0.6).min(1.0) * 255.0) as u8;
            let b: u8 = ((t * 0.4).min(1.0) * 255.0) as u8;
            (r, g, b)
        };

        contents.extend_from_slice(&[r, g, b]);
    }

    fs::write(path, contents).map_err(|_| GpuError::Buffer("failed to write PPM"))?;
    Ok(())
}

pub fn render_to_file(
    ctx: &mut GpuContext,
    width: u32,
    height: u32, 
    max_iters: u32,
    cx: f32,
    cy: f32,
    scale: f32,
    path: &str,
) -> Result<(), GpuError>{
    let pixels: Vec<u32> = render_gpu(ctx, width, height, max_iters, cx, cy, scale)?;
    write_ppm(path, &pixels, width, height)?;
    Ok(())
}