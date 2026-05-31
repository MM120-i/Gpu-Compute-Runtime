use ash::vk;

use crate::context::GpuContext;
use crate::pipeline::ComputePipeline;
use crate::error::GpuError;

pub struct Dispatcher {
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
}

pub struct WorkgroupCount {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dispatcher {
    pub fn new(ctx: &GpuContext) -> Result<Self, GpuError> {
        let alloc_info: vk::CommandBufferAllocateInfo<'_> = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            command_pool: ctx.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            _marker: std::marker::PhantomData,
        };

        let command_buffer: vk::CommandBuffer = unsafe {
            ctx.device().allocate_command_buffers(&alloc_info).map_err(|e: vk::Result| GpuError::Vk("allocate_command_buffers", e))?
            [0]
        };

        let fence_info: vk::FenceCreateInfo<'_> = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::FenceCreateFlags::empty(),
            _marker: std::marker::PhantomData,
        };

        let fence: vk::Fence = unsafe {
            ctx.device().create_fence(&fence_info, None).map_err(|e: vk::Result| GpuError::Vk("create_fence", e))?
        };

        Ok(Self { command_buffer, fence })
    }

    pub fn dispatch(&mut self, ctx: &GpuContext, pipeline: &ComputePipeline, descriptor_set: vk::DescriptorSet, workgroups: WorkgroupCount) -> Result<(), GpuError> {
        unsafe {
            ctx.device().reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty()).map_err(|e: vk::Result| GpuError::Vk("reset_command_buffer", e))?
        }

        let begin_info: vk::CommandBufferBeginInfo<'_> = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: std::ptr::null(),
            flags: vk::CommandBufferUsageFlags::empty(),
            p_inheritance_info: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        unsafe {
            ctx.device().begin_command_buffer(
                self.command_buffer, 
                &begin_info,
            ).map_err(|e: vk::Result| GpuError::Vk("begin_command_buffer", e))?
        }

        unsafe {
            ctx.device().cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE, 
                pipeline.raw_pipeline()
            );
        }

        unsafe {
            ctx.device().cmd_bind_descriptor_sets(
                self.command_buffer, 
                vk::PipelineBindPoint::COMPUTE, 
                pipeline.raw_layout(), 
                0, 
                &[descriptor_set], 
                &[],
            );
        }

        unsafe {
            ctx.device().cmd_dispatch(
                self.command_buffer, 
                workgroups.x, 
                workgroups.y, 
                workgroups.z
            );
        }

        unsafe {
            ctx.device().end_command_buffer(self.command_buffer).map_err(|e: vk::Result| GpuError::Vk("end_command_buffer", e))?;
        }

        let submit_info: vk::SubmitInfo<'_> = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        unsafe {
            ctx.device().queue_submit(
                ctx.compute_queue,
                &[submit_info],
                self.fence,
            ).map_err(|e: vk::Result| GpuError::Vk("queue_submit", e))?;
        }

        unsafe {
            ctx.device().wait_for_fences(
                &[self.fence], 
                true, 
                u64::MAX
            ).map_err(|e: vk::Result| GpuError::Vk("wait_for_fences", e))?;
        }

        unsafe {
            ctx.device().reset_fences(&[self.fence]).map_err(|e: vk::Result| GpuError::Vk("reset_fences", e))?;
        }
        
        Ok(())
    }

    pub fn workgroup_count_1d(total_elements: u32, local_size: u32) -> WorkgroupCount {
        WorkgroupCount { 
            x: (total_elements + local_size - 1) / local_size, 
            y: 1, 
            z: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workgroup_count_1d_exact() {
        let wg = Dispatcher::workgroup_count_1d(64, 64);
        assert_eq!(wg.x, 1);
        assert_eq!(wg.y, 1);
        assert_eq!(wg.z, 1);
    }

    #[test]
    fn workgroup_count_1d_round_up() {
        let wg = Dispatcher::workgroup_count_1d(65, 64);
        assert_eq!(wg.x, 2);
    }

    #[test]
    fn workgroup_count_1d_one_element() {
        let wg = Dispatcher::workgroup_count_1d(1, 64);
        assert_eq!(wg.x, 1);
    }

    #[test]
    fn workgroup_count_1d_zero_elements() {
        let wg = Dispatcher::workgroup_count_1d(0, 64);
        assert_eq!(wg.x, 0);
    }

    #[test]
    fn workgroup_count_1d_large_input() {
        let wg = Dispatcher::workgroup_count_1d(1_000_000, 256);
        assert_eq!(wg.x, 3907);
    }

    #[test]
    fn workgroup_construct() {
        let wg = WorkgroupCount { x: 8, y: 4, z: 2 };
        assert_eq!(wg.x, 8);
        assert_eq!(wg.y, 4);
        assert_eq!(wg.z, 2);
    }
}