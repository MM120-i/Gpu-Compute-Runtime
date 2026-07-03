use runtime::context::GpuContext;
use runtime::gpu::buffer::GpuBuffer;
use runtime::gpu::dispatcher::{Dispatcher, WorkgroupCount};
use runtime::gpu::pipeline::{BufferBinding, ComputePipeline};
use runtime::gpu::profiler::{GpuProfiler, BenchmarkReport};

#[test]
fn profiled_dispatch_returns_positive_invocations() {
    let mut ctx: GpuContext = GpuContext::new().expect("create GpuContext");
    let mut dispatcher: Dispatcher = Dispatcher::new(&ctx).expect("create Dispatcher");
    let profiler: GpuProfiler = GpuProfiler::new(&ctx).expect("create GpuProfiler");

    let shader: &str = "\
        #version 450\n\
        layout(local_size_x=256) in;\n\
        layout(std430, binding=0) buffer D { uint data[]; };\n\
        void main() {\n\
            uint x = gl_GlobalInvocationID.x;\n\
            data[gl_GlobalInvocationID.x] = x;\n\
        }";

    let bindings: [BufferBinding; 1] = [BufferBinding { slot: 0 }];
    let pipeline: ComputePipeline = ComputePipeline::from_glsl_no_opt(&ctx, shader, "main", &bindings).expect("create pipeline");
    let buf: GpuBuffer = GpuBuffer::output_u32(&mut ctx, 65536).expect("create output buffer");
    let desc_set: ash::vk::DescriptorSet = pipeline.create_descriptor_set(&ctx, &[&buf]).expect("create descriptor set");

    ctx.reset_query_pool(0, 2);

    profiler.dispatch_profiled(
        &mut dispatcher, &ctx, &pipeline, desc_set,
        WorkgroupCount { x: 1, y: 1, z: 1 },
        0, 1, 0,
    ).expect("dispatch_profiled");

    let invocations: u64 = profiler.get_invocations(&ctx, 0).expect("get_invocations");
    assert!(invocations > 0, "expected positive invocations, got {}", invocations);

    ctx.destroy_buffer(buf);
}

#[test]
fn profiled_timestamps_monotonic() {
    let mut ctx: GpuContext = GpuContext::new().expect("create GpuContext");
    let mut dispatcher: Dispatcher = Dispatcher::new(&ctx).expect("create Dispatcher");
    let profiler: GpuProfiler = GpuProfiler::new(&ctx).expect("create GpuProfiler");

    let shader = "\
        #version 450\n\
        layout(local_size_x=256) in;\n\
        layout(std430, binding=0) buffer D { uint data[]; };\n\
        void main() {\n\
            for (int i = 0; i < 4096; i++) { gl_GlobalInvocationID.x; }\n\
            data[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x;\n\
        }";

    let bindings: [BufferBinding; 1] = [BufferBinding { slot: 0 }];
    let pipeline: ComputePipeline = ComputePipeline::from_glsl_no_opt(&ctx, shader, "main", &bindings).expect("create pipeline");
    let buf: GpuBuffer = GpuBuffer::output_u32(&mut ctx, 65536).expect("create output buffer");
    let desc_set: ash::vk::DescriptorSet = pipeline.create_descriptor_set(&ctx, &[&buf]).expect("create descriptor set");

    ctx.reset_query_pool(0, 6);

    profiler.dispatch_profiled(
        &mut dispatcher, &ctx, &pipeline, desc_set,
        WorkgroupCount { x: 1, y: 1, z: 1 },
        0, 1, 0,
    ).expect("dispatch_profiled small");

    let ts_small: f64 = profiler.get_elapsed_ms(&ctx, 0).expect("get_elapsed_ms small");
    assert!(ts_small > 0.0, "timestamp should be positive, got {}", ts_small);

    profiler.dispatch_profiled(
        &mut dispatcher, &ctx, &pipeline, desc_set,
        WorkgroupCount { x: 256, y: 1, z: 1 },
        2, 3, 1,
    ).expect("dispatch_profiled large");

    let ts_large: f64 = profiler.get_elapsed_ms(&ctx, 2).expect("get_elapsed_ms large");
    assert!(ts_large > ts_small, "256 workgroups ({:.4} ms) should exceed 1 ({:.4} ms)", ts_large, ts_small);

    ctx.destroy_buffer(buf);
}

#[test]
fn bandwidth_utilization_formula() {
    let ctx: GpuContext = GpuContext::new().expect("create GpuContext");
    let profiler: GpuProfiler = GpuProfiler::new(&ctx).expect("create GpuProfiler");

    let bytes_total: f64 = 10_000_000.0;
    let time_ms: f64 = 1.0;
    let bw_pct: f64 = profiler.bandwidth_utilization(bytes_total, time_ms);

    assert!(bw_pct > 0.0, "bandwidth should be > 0%");
    assert!(bw_pct < 100.0, "bandwidth should be < 100% (got {}%)", bw_pct);

    let mut manual: GpuProfiler = profiler;
    manual.set_peak_bandwidth(10.0);
    let bw_10gbps: f64 = manual.bandwidth_utilization(10_000_000_000.0, 1000.0);
    assert!((bw_10gbps - 100.0).abs() < 0.01, "10 GB/s on 10 GB/s peak should be 100%, got {}", bw_10gbps);
}

#[test]
fn format_invocations_used_in_report() {
    let ctx: GpuContext = GpuContext::new().expect("create GpuContext");
    let profiler: GpuProfiler = GpuProfiler::new(&ctx).expect("create GpuProfiler");

    profiler.print_report(&ctx.device_name(), &[
        BenchmarkReport {
            name: "test",
            gpu_ms: 0.5,
            gpu_timestamp_ms: 0.3,
            invocations: 1_234_567,
            bytes_read: 5_000_000.0,
            bytes_written: 2_000_000.0,
        },
    ]);
}

#[test]
fn set_peak_bandwidth_updates_value() {
    let ctx: GpuContext = GpuContext::new().expect("create GpuContext");
    let mut profiler: GpuProfiler = GpuProfiler::new(&ctx).expect("create GpuProfiler");
    let before: f64 = profiler.bandwidth_utilization(1_000_000_000.0, 1000.0);

    profiler.set_peak_bandwidth(1000.0);
    let after: f64 = profiler.bandwidth_utilization(1_000_000_000.0, 1000.0);

    assert!(after < before, "higher peak should give lower utilization");
}
