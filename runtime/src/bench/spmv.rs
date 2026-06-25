use std::time::Instant;
use serde_json::{json, Value};

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

const SPMV_GLSL: &str = r#"#version 460
layout(local_size_x = 256) in;
layout(binding = 0) buffer RowPtrs   { uint row_ptrs[]; };
layout(binding = 1) buffer ColIndices { uint col_indices[]; };
layout(binding = 2) buffer Values    { float values[]; };
layout(binding = 3) buffer X         { float x[]; };
layout(binding = 4) buffer Y         { float y[]; };

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint n_rows = row_ptrs.length() - 1u;
    if (row >= n_rows) 
        return;

    uint start = row_ptrs[row];
    uint end   = row_ptrs[row + 1];

    float sum = 0.0;

    for (uint i = start; i < end; i++) 
        sum += values[i] * x[col_indices[i]];

    y[row] = sum;
}
"#;

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
            col_indices.push(rng.random_range(0u32..cols as u32));
            values.push(rng.random_range(-1.0f32..1.0));
        }

        row_ptrs.push(row_ptrs.last().unwrap() + nnz as u32);
    }

    CsrMatrix { row_ptrs, col_indices, values , rows, cols }
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

fn run_spmv_gpu(ctx: &mut GpuContext, mat: &CsrMatrix, x: &[f32]) -> Result<Vec<f32>, GpuError> {
    let rows = mat.rows;

    let row_buf = GpuBuffer::input_u32(ctx, &mat.row_ptrs)?;
    let col_buf = GpuBuffer::input_u32(ctx, &mat.col_indices)?;
    let val_buf = GpuBuffer::input(ctx, &mat.values)?;
    let x_buf   = GpuBuffer::input(ctx, x)?;
    let y_buf   = GpuBuffer::output(ctx, rows)?;

    let zeros = vec![0.0f32; rows];
    y_buf.upload(&zeros)?;

    let bindings = [
        BufferBinding { slot: 0 },
        BufferBinding { slot: 1 },
        BufferBinding { slot: 2 },
        BufferBinding { slot: 3 },
        BufferBinding { slot: 4 },
    ];

    let pipeline = ComputePipeline::from_glsl_no_opt(ctx, SPMV_GLSL, "main", &bindings)?;
    let desc = pipeline.create_descriptor_set(ctx, &[&row_buf, &col_buf, &val_buf, &x_buf, &y_buf])?;
    let mut dispatcher = Dispatcher::new(ctx)?;
    let wg = Dispatcher::workgroup_count_1d(rows as u32, WG_SIZE);

    dispatcher.dispatch(ctx, &pipeline, desc, wg)?;

    let result: Vec<f32> = y_buf.download()?;

    ctx.destroy_dispatcher(dispatcher);
    ctx.destroy_pipeline(pipeline);
    ctx.destroy_buffer(row_buf);
    ctx.destroy_buffer(col_buf);
    ctx.destroy_buffer(val_buf);
    ctx.destroy_buffer(x_buf);
    ctx.destroy_buffer(y_buf);

    Ok(result)
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

    let small_x: Vec<f32> = vec![1.0, 1.0, 1.0];
    let expected: Vec<f32> = spmv_cpu(&small_mat, &small_x);
    let gpu_result: Vec<f32> = run_spmv_gpu(ctx, &small_mat, &small_x)?;
    assert_eq!(gpu_result, expected, "spmv correctness failed on 3x3");

    let mat: CsrMatrix = generate_csr_matrix(ROWS, COLS);
    let total_nnz: usize = mat.values.len();
    let mut rng: ChaCha12Rng = ChaCha12Rng::seed_from_u64(7);
    let x: Vec<f32> = (0..COLS).map(|_| rng.random_range(-1.0f32..1.0)).collect();

    let _expected: Vec<f32> = spmv_cpu(&mat, &x);
    let _ = run_spmv_gpu(ctx, &mat, &x)?;
    let start: Instant = Instant::now();

    for _ in 0..ITERATIONS {
        let _ = run_spmv_gpu(ctx, &mat, &x)?;
    }

    let gpu_dur: std::time::Duration = start.elapsed();
    let gpu_ms: f64 = gpu_dur.as_secs_f64() * 1000.0 / ITERATIONS as f64;

    let cpu_start: Instant = Instant::now();

    for _ in 0..ITERATIONS {
        spmv_cpu(&mat, &x);
    }

    let cpu_dur: std::time::Duration = cpu_start.elapsed();
    let cpu_ms: f64 = cpu_dur.as_secs_f64() * 1000.0 / ITERATIONS as f64;
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
            "cpu_ms": (cpu_ms * 100.0).round() / 100.0,
            "bandwidth_gbps": (bandwidth_gbps * 100.0).round() / 100.0,
            "speedup": (cpu_ms / gpu_ms * 100.0).round() / 100.0,
            "correct": true
        }
    }))
}