use std::fs;
use ash::vk::{DescriptorSet, BufferUsageFlags};
use gpu_allocator::MemoryLocation;

use crate::context::GpuContext;
use crate::error::GpuError;
use crate::gpu::buffer::GpuBuffer;
use crate::gpu::pipeline::{BufferBinding, ComputePipeline};
use crate::gpu::dispatcher::{Dispatcher, WorkgroupCount};

const WG: u32 = 16;
const SHADER: &str = include_str!("../../../kernels/demos/mandelbrot.comp");

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
    let total: u64 = (width * height) as u64;

    let out_buf: GpuBuffer = ctx.create_buffer(
        total * std::mem::size_of::<u32>() as u64, 
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

    let wg_x: u32 = (width + WG - 1) / WG;
    let wg_y: u32 = (height + WG - 1) / WG;
    let wg: WorkgroupCount = WorkgroupCount { x: wg_x, y: wg_y, z: 1 };

    dispatcher.dispatch(ctx, &pipeline, desc, wg)?;

    let result: Vec<u32> = out_buf.download::<u32>()?;

    ctx.destroy_buffer(out_buf);
    ctx.destroy_buffer(params_buf);
    ctx.destroy_dispatcher(dispatcher);
    ctx.destroy_pipeline(pipeline);

    Ok(result)
}

fn write_ppm(path: &str, pixels: &[u32], width: u32, height: u32){
    let mut contents: String = format!("P6\n{} {}\n255\n", width, height);

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

        contents.push(r as char);
        contents.push(g as char);
        contents.push(b as char);
    }

    fs::write(path, &contents).expect("write PPM");
}

pub fn render_to_fil(
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
    write_ppm(path, &pixels, width, height);
    Ok(())
}