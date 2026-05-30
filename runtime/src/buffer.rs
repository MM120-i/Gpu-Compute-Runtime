// Design decision: Explicit ctx.destroy_buffer(buf). No Drop, no lifetime complexity. Buffer
// holds its internals as pub(crate) so contect can destory it. 
// TODO: No RAII (buffer holds Arc<GpuContext> for auto cleanup), so maybe as a refactor we can add RAII idk

use ash::vk;

use crate::{context::GpuContext, error::GpuError};

pub struct GpuBuffer {
    pub(crate) raw: vk::Buffer,
    pub(crate) allocation: gpu_allocator::vulkan::Allocation,
    pub(crate) size: u64,
}

impl GpuBuffer {
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<(), GpuError>{
        todo!()
    }

    pub fn download<T: Copy>(&self) -> Result<Vec<T>, GpuError>{
        todo!()
    }

    pub fn raw(&self) -> vk::Buffer{
        todo!()
    }

    pub fn size(&self) -> u64 {
        todo!()
    }

    // convenience constructors
    pub fn input(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        todo!()
    }

    pub fn output(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        todo!()
    }
}