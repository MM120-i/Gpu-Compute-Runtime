use runtime::context::GpuContext;
use std::time::Instant;
use runtime::gpu::buffer::GpuBuffer;
use runtime::gpu::dispatcher::{Dispatcher, WorkgroupCount};
use runtime::gpu::pipeline::{BufferBinding, ComputePipeline};

#[test]
fn timestamp_queries_work() {
    let mut ctx: GpuContext = GpuContext::new().expect("create GpuContext");
    let mut dispatcher: Dispatcher = Dispatcher::new(&ctx).expect("create Dispatcher");

    let shader: &str = "\
        #version 450\n\
        layout(local_size_x=256) in;\n\
        layout(std430, binding=0) buffer D { uint data[]; };\n\
        void main() {\n\
            uint x = gl_GlobalInvocationID.x;\n\
            for (int i = 0; i < 4096; i++) { x = x * 1103515245u + 12345u; }\n\
            data[gl_GlobalInvocationID.x] = x;\n\
        }";

    let bindings: [BufferBinding; 1] = [BufferBinding { slot: 0 }];
    let pipeline: ComputePipeline = ComputePipeline::from_glsl_no_opt(&ctx, shader, "main", &bindings).expect("create pipeline");
    let buf: GpuBuffer = GpuBuffer::output_u32(&mut ctx, 65536).expect("create output buffer");
    let desc_set: ash::vk::DescriptorSet = pipeline.create_descriptor_set(&ctx, &[&buf]).expect("create descriptor set");

    ctx.reset_query_pool(0, 6);

    let ts_small: f64 = dispatcher.dispatch_timed(
        &ctx, &pipeline, desc_set, WorkgroupCount { x: 1, y: 1, z: 1 }, 0, 1,
    ).expect("dispatch_timed small");

    assert!(ts_small > 0.0, "GPU timestamp should be positive, got {}", ts_small);

    let ts_big: f64 = dispatcher.dispatch_timed(
        &ctx, &pipeline, desc_set, WorkgroupCount { x: 256, y: 1, z: 1 }, 2, 3,
    ).expect("dispatch_timed big");
    
    assert!(ts_big > ts_small, "256 workgroups ({:.4} ms) should exceed 1 ({:.4} ms)", ts_big, ts_small);

    let start: Instant = Instant::now();
    let ts_wall: f64 = dispatcher.dispatch_timed(
        &ctx, &pipeline, desc_set, WorkgroupCount { x: 256, y: 1, z: 1 }, 4, 5,
    ).expect("dispatch_timed wall-check");

    let wall_ms: f64 = start.elapsed().as_secs_f64() * 1000.0;
    
    assert!(ts_wall <= wall_ms + 0.01, "GPU timestamp ({:.4} ms) should not exceed wall-clock ({:.4} ms)", ts_wall, wall_ms);

    ctx.destroy_buffer(buf);
}

#[test]
fn workgroup_count_1d_exact() {
    let wg: WorkgroupCount = Dispatcher::workgroup_count_1d(64, 64);
    assert_eq!(wg.x, 1);
    assert_eq!(wg.y, 1);
    assert_eq!(wg.z, 1);
}

#[test]
fn workgroup_count_1d_round_up() {
    let wg: WorkgroupCount = Dispatcher::workgroup_count_1d(65, 64);
    assert_eq!(wg.x, 2);
}

#[test]
fn workgroup_count_1d_one_element() {
    let wg: WorkgroupCount = Dispatcher::workgroup_count_1d(1, 64);
    assert_eq!(wg.x, 1);
}

#[test]
fn workgroup_count_1d_zero_elements() {
    let wg: WorkgroupCount = Dispatcher::workgroup_count_1d(0, 64);
    assert_eq!(wg.x, 0);
}

#[test]
fn workgroup_count_1d_large_input() {
    let wg: WorkgroupCount = Dispatcher::workgroup_count_1d(1_000_000, 256);
    assert_eq!(wg.x, 3907);
}

#[test]
fn workgroup_construct() {
    let wg: WorkgroupCount = WorkgroupCount { x: 8, y: 4, z: 2 };
    assert_eq!(wg.x, 8);
    assert_eq!(wg.y, 4);
    assert_eq!(wg.z, 2);
}
