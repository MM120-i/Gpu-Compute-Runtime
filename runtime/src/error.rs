use std::fmt;

#[derive(Debug)]
pub enum GpuError{
    Init(&'static str),
    Vk(&'static str, ash::vk::Result),
    Alloc(gpu_allocator::AllocationError),
    Io(std::io::Error),
    Shader(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::Init(msg) => write!(f, "Vulkan init: {}", msg),
            GpuError::Vk(msg, result) => write!(f, "Vulkan {}: {}", result, msg),
            GpuError::Alloc(e) => write!(f, "GPU Alloc: {}", e),
            GpuError::Io(e) => write!(f, "I/O: {}", e),
            GpuError::Shader(msg) => write!(f, "Shader: {}", msg),
        }
    }
}

// whats this for?
impl std::error::Error for GpuError {}

impl From<gpu_allocator::AllocationError> for GpuError {
    fn from(value: gpu_allocator::AllocationError) -> Self {
        GpuError::Alloc(value)
    }
}

impl From<std::io::Error> for GpuError{
    fn from(value: std::io::Error) -> Self {
        GpuError::Io(value)
    }
}