use ash::vk;

use crate::context::GpuContext;
use crate::gpu::dispatcher::{Dispatcher, WorkgroupCount};
use crate::gpu::pipeline::ComputePipeline;
use crate::error::GpuError;

fn lookup_peak_bandwidth(device_name: &str) -> f64 {
    match device_name {
        // NVIDIA
        n if n.contains("RTX 4090") => 1008.0,
        n if n.contains("RTX 4080") => 736.0,
        n if n.contains("RTX 4070 Ti") => 504.0,
        n if n.contains("RTX 4070") => 504.0,
        n if n.contains("RTX 4060 Ti") => 288.0,
        n if n.contains("RTX 4060") => 272.0,
        n if n.contains("RTX 3090 Ti") => 1008.0,
        n if n.contains("RTX 3090") => 936.0,
        n if n.contains("RTX 3080 Ti") => 912.0,
        n if n.contains("RTX 3080") => 760.0,
        n if n.contains("RTX 3070 Ti") => 608.0,
        n if n.contains("RTX 3070") => 448.0,
        n if n.contains("RTX 3060 Ti") => 448.0,
        n if n.contains("RTX 3060") => 360.0,
        n if n.contains("RTX 3050") => 224.0,
        n if n.contains("RTX 2080 Ti") => 616.0,
        n if n.contains("RTX 2080") => 448.0,
        n if n.contains("RTX 2070") => 448.0,
        n if n.contains("RTX 2060 SUPER") => 448.0,
        n if n.contains("RTX 2060") => 336.0,
        
        // AMD RDNA3
        n if n.contains("RX 7900 XTX") => 960.0,
        n if n.contains("RX 7900 XT") => 800.0,
        n if n.contains("RX 7900 GRE") => 576.0,
        n if n.contains("RX 7800 XT") => 624.0,
        n if n.contains("RX 7700 XT") => 432.0,
        n if n.contains("RX 7600 XT") => 288.0,
        n if n.contains("RX 7600") => 288.0,
       
        // AMD RDNA2
        n if n.contains("RX 6950 XT") => 576.0,
        n if n.contains("RX 6900 XT") => 512.0,
        n if n.contains("RX 6800 XT") => 512.0,
        n if n.contains("RX 6800") => 512.0,
        n if n.contains("RX 6750 XT") => 432.0,
        n if n.contains("RX 6700 XT") => 384.0,
        n if n.contains("RX 6650 XT") => 280.0,
        n if n.contains("RX 6600 XT") => 256.0,
        n if n.contains("RX 6600") => 224.0,
        n if n.contains("RX 6500 XT") => 144.0,
        n if n.contains("RX 6400") => 128.0,
    
        // AMD RDNA1
        n if n.contains("RX 5700 XT") => 448.0,
        n if n.contains("RX 5700") => 448.0,
        n if n.contains("RX 5600 XT") => 288.0,
        n if n.contains("RX 5500 XT") => 224.0,
     
        // Intel Arc
        n if n.contains("Arc A770") => 560.0,
        n if n.contains("Arc A750") => 512.0,
        n if n.contains("Arc A580") => 512.0,
        n if n.contains("Arc A380") => 186.0,
        n if n.contains("Arc A310") => 124.0,
        _ => 100.0,
    }
}

pub struct GpuProfiler {
    stats_pool: vk::QueryPool,
    peak_bandwidth_gbps: f64,
}

pub struct BenchmarkReport {
    pub name: &'static str,
    pub gpu_ms: f64,
    pub gpu_timestamp_ms: f64,
    pub invocations: u64,
    pub bytes_read: f64,
    pub bytes_written: f64,
}

fn format_invocations(n: u64) -> String {
    let s: String = n.to_string();
    let len: usize = s.len();
    let mut out: String = String::with_capacity(len + len / 3);

    for(i, c) in s.chars().enumerate(){
        if i > 0 && (len - i) % 3 == 0{
            out.push(',');
        }

        out.push(c);
    }

    out
}

impl GpuProfiler {
    pub fn new(ctx: &GpuContext) -> Self {
        let pool_info: vk::QueryPoolCreateInfo<'_> = vk::QueryPoolCreateInfo {
            s_type: vk::StructureType::QUERY_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::QueryPoolCreateFlags::empty(),
            query_type: vk::QueryType::PIPELINE_STATISTICS,
            query_count: 32,
            pipeline_statistics: vk::QueryPipelineStatisticFlags::COMPUTE_SHADER_INVOCATIONS,
            _marker: std::marker::PhantomData,
        };

        let stats_pool: vk::QueryPool = unsafe {
            ctx.device().create_query_pool(&pool_info, None).expect("create pipeline statistics query pool")
        };

        unsafe {
            ctx.device().reset_query_pool(stats_pool, 0, 32);
        }

        let peak_bandwidth_gbps: f64 = lookup_peak_bandwidth(&ctx.device_name());

        Self { stats_pool, peak_bandwidth_gbps }
    }

    pub fn set_peak_bandwidth(&mut self, gbps: f64){
        self.peak_bandwidth_gbps = gbps;
    }

    pub fn begin_profile(
        &self,
        ctx: &GpuContext,
        dispatcher: &Dispatcher,
        timestamp_slot: u32,
        stats_slot: u32,
    ) {
        unsafe {
            ctx.device().cmd_write_timestamp(
                dispatcher.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                ctx.timestamp_query_pool,
                timestamp_slot,
            );
        }

        dispatcher.begin_stats_query(ctx, self.stats_pool, stats_slot);
    }

    pub fn end_profile(
        &self,
        ctx: &GpuContext,
        dispatcher: &Dispatcher,
        timestamp_slot: u32,
        stats_slot: u32,
    ) {
        dispatcher.end_stats_query(ctx, self.stats_pool, stats_slot);
        
        unsafe {
            ctx.device().cmd_write_timestamp(
                dispatcher.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                ctx.timestamp_query_pool,
                timestamp_slot,
            );
        }
    }

    pub fn dispatch_profiled(
        &self,
        dispatcher: &mut Dispatcher,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        descriptor_set: vk::DescriptorSet,
        workgroups: WorkgroupCount,
        ts_start: u32,
        ts_end: u32,
        stats_slot: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            ctx.device().reset_command_buffer(dispatcher.command_buffer, vk::CommandBufferResetFlags::empty()).map_err(|e| GpuError::Vk("reset_command_buffer", e))?;
        }

        let begin_info: vk::CommandBufferBeginInfo<'_> = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: std::ptr::null(),
            flags: vk::CommandBufferUsageFlags::empty(),
            p_inheritance_info: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        unsafe {
            ctx.device().begin_command_buffer(dispatcher.command_buffer, &begin_info).map_err(|e| GpuError::Vk("begin_command_buffer", e))?;
        }

        self.begin_profile(ctx, dispatcher, ts_start, stats_slot);

        unsafe {
            ctx.device().cmd_bind_pipeline(dispatcher.command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline.raw_pipeline());
            ctx.device().cmd_bind_descriptor_sets(dispatcher.command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline.raw_layout(), 0, &[descriptor_set], &[]);
            ctx.device().cmd_dispatch(dispatcher.command_buffer, workgroups.x, workgroups.y, workgroups.z);
        }

        self.end_profile(ctx, dispatcher, ts_end, stats_slot);

        unsafe {
            ctx.device().end_command_buffer(dispatcher.command_buffer).map_err(|e| GpuError::Vk("end_command_buffer", e))?;
        }

        let submit_info: vk::SubmitInfo<'_> = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &dispatcher.command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        unsafe {
            ctx.device().queue_submit(ctx.compute_queue, &[submit_info], dispatcher.fence).map_err(|e| GpuError::Vk("queue_submit", e))?;
            ctx.device().wait_for_fences(&[dispatcher.fence], true, u64::MAX).map_err(|e| GpuError::Vk("wait_for_fences", e))?;
            ctx.device().reset_fences(&[dispatcher.fence]).map_err(|e: vk::Result| GpuError::Vk("reset_fences", e))?;
        }

        Ok(())
    }

    pub fn get_elapsed_ms(&self, ctx: &GpuContext, first_slot: u32) -> Result<f64, GpuError> {
        let mut data: [u64; 2] = [0u64; 2];

        unsafe {
            ctx.device().get_query_pool_results(
                ctx.timestamp_query_pool,
                first_slot,
                &mut data,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            ).map_err(|e| GpuError::Vk("get_query_pool_results", e))?;
        }

        let ticks: u64 = data[1] - data[0];

        Ok(ticks as f64 * ctx.timestamp_period / 1_000_000.0)
    }

    pub fn get_invocations(&self, ctx: &GpuContext, slot: u32) -> Result<u64, GpuError> {
        let mut data: [u64; 1] = [0u64; 1];

        unsafe {
            ctx.device().get_query_pool_results(
                self.stats_pool, 
                slot, 
                &mut data, 
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            ).map_err(|e| GpuError::Vk("get_query_pool_results", e))?;
        }
        
        Ok(data[0])
    }

    pub fn bandwidth_utilization(&self, bytes_total: f64, time_ms: f64) -> f64 {
        let actual_gbps: f64 = bytes_total / (time_ms / 1000.0) / 1e9;
        actual_gbps / self.peak_bandwidth_gbps * 100.0
    }

    pub fn destroy(self, ctx: &mut GpuContext) {
        unsafe {
            ctx.device().destroy_query_pool(self.stats_pool, None);
        }
    }
}

impl GpuProfiler {
    pub fn print_report(&self, device_name: &str, entries: &[BenchmarkReport]) {
        let line_width: usize = 15 + 3 + 10 + 3 + 12 + 3 + 11 + 3 + 13;
        let padding: String = " ".repeat((line_width - device_name.len() - 22) / 2);

        println!();
        println!("{}= Profiling Summary -- {} =", padding, device_name);
        println!("{}Peak Bandwidth: {:.2} GB/s", padding, self.peak_bandwidth_gbps);
        println!();

        println!("╔═════════════════╤════════════╤══════════════╤═════════════╤═══════════════╗");
        println!("║ {:15} │ {:10} │ {:12} │ {:11} │ {:13} ║",
            "Benchmark", "GPU (ms)", "GPU ts (ms)", "Invocs", "BW % peak");
        println!("╠═════════════════╪════════════╪══════════════╪═════════════╪═══════════════╣");

        for entry in entries {
            let bytes_total = entry.bytes_read + entry.bytes_written;
            let bw_pct = self.bandwidth_utilization(bytes_total, entry.gpu_timestamp_ms);
            let invocs = format_invocations(entry.invocations);

            println!("║ {:15} │ {:>10.2} │ {:>12.2} │ {:>11} │ {:>12.2}% ║",
                entry.name, entry.gpu_ms, entry.gpu_timestamp_ms, invocs, bw_pct);
        }

        println!("╚═════════════════╧════════════╧══════════════╧═════════════╧═══════════════╝");
        println!();
    }
}