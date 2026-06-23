// Design decision: Explicit ctx.destroy_buffer(buf). No Drop, no lifetime complexity. Buffer
// holds its internals as pub(crate) so contect can destory it. 
// TODO: No RAII (buffer holds Arc<GpuContext> for auto cleanup), so maybe as a refactor we can add RAII idk

use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::error::GpuError;
use crate::context::GpuContext;

pub struct GpuBuffer {
    pub(crate) raw: vk::Buffer,
    pub(crate) allocation: gpu_allocator::vulkan::Allocation,
    pub(crate) size: u64,
}

impl GpuBuffer {
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<(), GpuError>{
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>())
        };
        
        if(bytes.len() as u64) > self.size {
            return Err(GpuError::Buffer("data exceeds buffer size"));
        }

        let ptr: std::ptr::NonNull<std::ffi::c_void> = self.allocation.mapped_ptr().ok_or(GpuError::Buffer("Buffer is not host-visible"))?;

        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr() as *mut u8, bytes.len());
        }

        Ok(())
    }

    pub fn download<T: Copy + Default>(&self) -> Result<Vec<T>, GpuError>{
        let count: usize = self.size as usize / std::mem::size_of::<T>();
        let mut result: Vec<T> = vec![T::default(); count];
        let ptr: std::ptr::NonNull<std::ffi::c_void> = self.allocation.mapped_ptr().ok_or(GpuError::Buffer("buffer is not host-visible"))?;

        unsafe {
            std::ptr::copy_nonoverlapping(ptr.as_ptr() as *const u8, result.as_mut_ptr() as *mut u8, self.size as usize);
        }

        Ok(result)
    }

    pub fn raw(&self) -> vk::Buffer{
        self.raw
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    // convenience constructors
    pub fn input(ctx: &mut GpuContext, data: &[f32]) -> Result<Self, GpuError> {
        let size: u64 = (data.len() * std::mem::size_of::<f32>()) as u64;
        let buf: GpuBuffer = ctx.create_buffer(
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,
        )?;

        buf.upload(data)?;
        Ok(buf)
    }

    pub fn output(ctx: &mut GpuContext, count: usize) -> Result<Self, GpuError> {
        let size: u64 = (count * std::mem::size_of::<f32>()) as u64;

        ctx.create_buffer(
            size, 
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC, 
            MemoryLocation::GpuToCpu
        )
    }

    pub fn input_u32(ctx: &mut GpuContext, data: &[u32]) -> Result<Self, GpuError> {
        let size: u64 = (data.len() * std::mem::size_of::<u32>()) as u64;
        
        let buf: GpuBuffer = ctx.create_buffer(
            size, 
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, 
            MemoryLocation::CpuToGpu,
        )?;

        buf.upload(data)?;
        Ok(buf)
    }

    pub fn output_u32(ctx: &mut GpuContext, count: usize) -> Result<Self, GpuError>{
        let size: u64 = (count * std::mem::size_of::<u32>()) as u64;

        ctx.create_buffer(
            size, 
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC, 
            MemoryLocation::GpuToCpu,
        )
    }
}