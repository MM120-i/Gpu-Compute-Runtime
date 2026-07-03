use ash::vk;

use crate::context::GpuContext;
use crate::gpu::dispatcher::Dispatcher;
use crate::error::GpuError;

fn lookup_peak_bandwidth(device_name: &str) -> f64 {
    match device_name {
        n if n.contains("RTX 2060 SUPER") => 448.0,
        n if n.contains("RTX 3060") => 360.0,
        n if n.contains("RTX 3070") => 448.0,
        n if n.contains("RTX 3080") => 760.0,
        n if n.contains("RTX 3090") => 936.0,
        n if n.contains("RTX 4060") => 272.0,
        n if n.contains("RTX 4070") => 504.0,
        n if n.contains("RTX 4080") => 736.0,
        n if n.contains("RTX 4090") => 1008.0,
        _ => 100.0,
    }
}

pub struct GpuProfiler {
    stats_pool: vk::QueryPool,
    peak_bandwidth_gbps: f64,
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

            ctx.device().cmd_begin_query(
                dispatcher.command_buffer,
                self.stats_pool,
                stats_slot,
                vk::QueryControlFlags::empty(),
            );
        }
    }

    pub fn end_profile(
        &self,
        ctx: &GpuContext,
        dispatcher: &Dispatcher,
        timestamp_slot: u32,
        stats_slot: u32,
    ) {
        unsafe {
            ctx.device().cmd_end_query(
                dispatcher.command_buffer,
                self.stats_pool,
                stats_slot,
            );

            ctx.device().cmd_write_timestamp(
                dispatcher.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                ctx.timestamp_query_pool,
                timestamp_slot,
            );
        }
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