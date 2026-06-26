use std::time::Instant;
use serde_json::{json, Value};
use ash::vk::{DescriptorSet, BufferUsageFlags};
use gpu_allocator::MemoryLocation;
use rand::RngExt;
use rand_chacha::ChaCha12Rng;
use rand::SeedableRng;

use crate::context::GpuContext;
use crate::error::GpuError;
use crate::gpu::buffer::GpuBuffer;
use crate::gpu::pipeline::{BufferBinding, ComputePipeline};
use crate::gpu::dispatcher::Dispatcher;

const WG_SIZE: u32 = 256;
const ROWS: usize = 262_144;
const COLS: usize = 262_144;
const NNZ_PER_ROW_MIN: usize = 8;
const NNZ_PER_ROW_MAX: usize = 24;
const ITERATIONS: u32 = 10;

const SPMV_GLSL: &str = include_str!("../../../kernels/benchmarks/spmv.comp");

#[allow(dead_code)]
struct CsrMatrix {
    row_ptrs: Vec<u32>,
    col_indices: Vec<u32>,
    values: Vec<f32>,
    rows: usize,
    cols: usize,
}

fn generate_csr_matrix(rows: usize, cols: usize) -> CsrMatrix {
    let mut rng: ChaCha12Rng = ChaCha12Rng::seed_from_u64(42);
    let mut row_ptrs: Vec<u32> = Vec::with_capacity(rows + 1);
    let mut col_indices: Vec<u32> = Vec::new();
    let mut values: Vec<f32> = Vec::new();
    row_ptrs.push(0u32);

    for _ in 0..rows {
        let nnz: usize = rng.random_range(NNZ_PER_ROW_MIN..=NNZ_PER_ROW_MAX);

        for _ in 0..nnz {
            col_indices.push(rng.random_range(0u32..COLS as u32));
            values.push(rng.random_range(-1.0f32..1.0));
        }

        row_ptrs.push(row_ptrs.last().unwrap() + nnz as u32);
    }

    CsrMatrix { row_ptrs, col_indices, values, rows, cols }
}

struct SpmvState {
    pipeline: ComputePipeline,
    desc: DescriptorSet,
    dispatcher: Dispatcher,
    row_buf: GpuBuffer,
    col_buf: GpuBuffer,
    val_buf: GpuBuffer,
    x_buf: GpuBuffer,
    y_buf: GpuBuffer,
    rows: u32,
}

fn init_spmv(ctx: &mut GpuContext, mat: &CsrMatrix, x: &[f32]) -> Result<SpmvState, GpuError> {
    let row_buf: GpuBuffer = GpuBuffer::input_u32(ctx, &mat.row_ptrs)?;
    let col_buf: GpuBuffer = GpuBuffer::input_u32(ctx, &mat.col_indices)?;
    let val_buf: GpuBuffer = GpuBuffer::input(ctx, &mat.values)?;
    let x_buf: GpuBuffer = GpuBuffer::input(ctx, x)?;

    let y_buf: GpuBuffer = ctx.create_buffer(
        mat.rows as u64 * std::mem::size_of::<f32>() as u64,
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::GpuToCpu,
    )?;

    let bindings: [BufferBinding; 5] = [
        BufferBinding { slot: 0 },
        BufferBinding { slot: 1 },
        BufferBinding { slot: 2 },
        BufferBinding { slot: 3 },
        BufferBinding { slot: 4 },
    ];
    let pipeline: ComputePipeline = ComputePipeline::from_glsl_no_opt(ctx, SPMV_GLSL, "main", &bindings)?;
    let desc: DescriptorSet = pipeline.create_descriptor_set(ctx, &[&row_buf, &col_buf, &val_buf, &x_buf, &y_buf])?;
    let dispatcher: Dispatcher = Dispatcher::new(ctx)?;

    Ok(SpmvState { 
        pipeline, 
        desc, 
        dispatcher, 
        row_buf, 
        col_buf, 
        val_buf, 
        x_buf, 
        y_buf, 
        rows: mat.rows as u32
    })
}

fn destroy_spmv(ctx: &mut GpuContext, state: SpmvState) {
    ctx.destroy_dispatcher(state.dispatcher);
    ctx.destroy_pipeline(state.pipeline);
    ctx.destroy_buffer(state.row_buf);
    ctx.destroy_buffer(state.col_buf);
    ctx.destroy_buffer(state.val_buf);
    ctx.destroy_buffer(state.x_buf);
    ctx.destroy_buffer(state.y_buf);
}

fn dispatch_spmv(ctx: &mut GpuContext, state: &mut SpmvState) -> Result<(), GpuError> {
    let wg: crate::gpu::WorkgroupCount = Dispatcher::workgroup_count_1d(state.rows, WG_SIZE);
    state.dispatcher.dispatch(ctx, &state.pipeline, state.desc, wg)?;
    Ok(())
}

fn spmv_cpu(mat: &CsrMatrix, x: &[f32]) -> Vec<f32> {
    let mut y: Vec<f32> = vec![0.0f32; mat.rows];

    for row in 0..mat.rows {
        let start: usize = mat.row_ptrs[row] as usize;
        let end: usize = mat.row_ptrs[row + 1] as usize;
        let mut sum: f32 = 0.0;

        for i in start..end {
            sum += mat.values[i] * x[mat.col_indices[i] as usize];
        }

        y[row] = sum;
    }

    y
}

pub fn run_spmv(ctx: &mut GpuContext) -> Result<Value, GpuError> {
    let device_name: String = ctx.device_name();

    let small_mat: CsrMatrix = CsrMatrix {
        row_ptrs: vec![0, 2, 3, 5],
        col_indices: vec![0, 2, 1, 0, 2],
        values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        rows: 3,
        cols: 3,
    };

    let small_x: Vec<f32> = vec![1.0f32; 3];
    let mut small_state: SpmvState = init_spmv(ctx, &small_mat, &small_x)?;
    dispatch_spmv(ctx, &mut small_state)?;
    let small_result: Vec<f32> = small_state.y_buf.download()?;
    assert_eq!(small_result, spmv_cpu(&small_mat, &small_x), "spmv: random 3x3");
    destroy_spmv(ctx, small_state);

    // Identity 4×4
    {
        let mat = CsrMatrix {
            row_ptrs: vec![0, 1, 2, 3, 4],
            col_indices: vec![0, 1, 2, 3],
            values: vec![1.0, 2.0, 3.0, 4.0],
            rows: 4, cols: 4,
        };
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut id_state = init_spmv(ctx, &mat, &x)?;
        dispatch_spmv(ctx, &mut id_state)?;
        let r: Vec<f32> = id_state.y_buf.download()?;
        assert_eq!(r, spmv_cpu(&mat, &x), "spmv: diag 4x4");
        destroy_spmv(ctx, id_state);
    }

    // Single row
    {
        let mat = CsrMatrix {
            row_ptrs: vec![0, 5],
            col_indices: vec![0, 1, 2, 3, 4],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            rows: 1, cols: 5,
        };
        let x: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let mut s_state = init_spmv(ctx, &mat, &x)?;
        dispatch_spmv(ctx, &mut s_state)?;
        let r: Vec<f32> = s_state.y_buf.download()?;
        assert_eq!(r, spmv_cpu(&mat, &x), "spmv: single row");
        destroy_spmv(ctx, s_state);
    }
    eprintln!("[bench] spmv: all correctness tests passed");

    let mat: CsrMatrix = generate_csr_matrix(ROWS, COLS);
    let total_nnz: usize = mat.values.len();
    let mut rng: ChaCha12Rng = ChaCha12Rng::seed_from_u64(7);
    let x: Vec<f32> = (0..COLS).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let mut state: SpmvState = init_spmv(ctx, &mat, &x)?;

    dispatch_spmv(ctx, &mut state)?;
    ctx.reset_query_pool(0, ITERATIONS * 2);

    let start: Instant = Instant::now();
    let mut gpu_timestamp_ns: u64 = 0;

    for i in 0..ITERATIONS {
        let elapsed: f64 = state.dispatcher.dispatch_timed(
            ctx, &state.pipeline, state.desc,
            Dispatcher::workgroup_count_1d(state.rows, WG_SIZE),
            i as u32 * 2, i as u32 * 2 + 1,
        )?;
        gpu_timestamp_ns += (elapsed * 1_000_000.0) as u64;
    }

    let gpu_dur: std::time::Duration = start.elapsed();
    let gpu_ms: f64 = gpu_dur.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    let gpu_timestamp_ms: f64 = gpu_timestamp_ns as f64 / 1_000_000.0 / ITERATIONS as f64;

    let cpu_start: Instant = Instant::now();

    for _ in 0..ITERATIONS {
        spmv_cpu(&mat, &x);
    }

    let cpu_dur: std::time::Duration = cpu_start.elapsed();
    let cpu_ms: f64 = cpu_dur.as_secs_f64() * 1000.0 / ITERATIONS as f64;

    destroy_spmv(ctx, state);

    let bytes_read: f64 = (total_nnz * 8 + COLS * 4) as f64;
    let bytes_written: f64 = (ROWS * 4) as f64;
    let bandwidth_gbps: f64 = (bytes_read + bytes_written) / (gpu_ms / 1000.0) / 1e9;

    Ok(json!({
        "spmv": {
            "device": device_name,
            "rows": ROWS,
            "cols": COLS,
            "nnz": total_nnz,
            "workgroup_size": WG_SIZE,
            "gpu_ms": (gpu_ms * 100.0).round() / 100.0,
            "gpu_timestamp_ms": (gpu_timestamp_ms * 100.0).round() / 100.0,
            "cpu_ms": (cpu_ms * 100.0).round() / 100.0,
            "bandwidth_gbps": (bandwidth_gbps * 100.0).round() / 100.0,
            "speedup": (cpu_ms / gpu_ms * 100.0).round() / 100.0,
            "correct": true,
        }
    }))
}