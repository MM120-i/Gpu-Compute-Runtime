use runtime::buffer::GpuBuffer;
use runtime::context::GpuContext;
use runtime::dispatcher::{Dispatcher, WorkgroupCount};
use runtime::pipeline::{BufferBinding, ComputePipeline};

#[test]
fn double_values() {
    let mut ctx: GpuContext = GpuContext::new().expect("create GpuContext");
    let input: Vec<f32> = (1..=8).map(|i: i32| i as f32).collect();
    let in_buf: GpuBuffer = GpuBuffer::input(&mut ctx, &input).expect("create input buffer");
    let out_buf: GpuBuffer = GpuBuffer::output(&mut ctx, input.len()).expect("create output buffer");

    let spirv_bytes: &[u8] = include_bytes!(env!("DOUBLE_SPV"));
    let spirv: &[u32] = unsafe {
        std::slice::from_raw_parts(
            spirv_bytes.as_ptr() as *const u32,
            spirv_bytes.len() / 4,
        )
    };

    let bindings: [BufferBinding; 2] = [
        BufferBinding { slot: 0 },
        BufferBinding { slot: 1 },
    ];

    let pipeline: ComputePipeline = ComputePipeline::new(&ctx, spirv, "main", &bindings).expect("create ComputePipeline");
    let descriptor_set: ash::vk::DescriptorSet = pipeline
        .create_descriptor_set(&ctx, &[&in_buf, &out_buf])
        .expect("create descriptor set");

    let mut dispatcher: Dispatcher = Dispatcher::new(&ctx).expect("create Dispatcher");
    let workgroups: WorkgroupCount = Dispatcher::workgroup_count_1d(input.len() as u32, 64);

    dispatcher
        .dispatch(&ctx, &pipeline, descriptor_set, workgroups)
        .expect("dispatch");

    let result: Vec<f32> = out_buf.download().expect("download result");
    let expected: Vec<f32> = input.iter().map(|x| x * 2.0).collect();

    assert_eq!(result, expected, "GPU doubling did not produce expected values");

    ctx.destroy_dispatcher(dispatcher);
    ctx.destroy_pipeline(pipeline);
    ctx.destroy_buffer(out_buf);
    ctx.destroy_buffer(in_buf);
}
