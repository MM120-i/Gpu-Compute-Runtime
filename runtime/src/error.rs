use std::fmt;

#[derive(Debug)]
pub enum GpuError{
    Init(&'static str),
    Vk(&'static str, ash::vk::Result),
    Alloc(gpu_allocator::AllocationError),
    Io(std::io::Error),
    Shader(String),
    Buffer(&'static str),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::Init(msg) => write!(f, "Vulkan init: {}", msg),
            GpuError::Vk(msg, result) => write!(f, "Vulkan {}: {}", result, msg),
            GpuError::Alloc(e) => write!(f, "GPU Alloc: {}", e),
            GpuError::Io(e) => write!(f, "I/O: {}", e),
            GpuError::Shader(msg) => write!(f, "Shader: {}", msg),
            GpuError::Buffer(msg) => write!(f, "Buffer: {}", msg),
        }
    }
}

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

// ================================== TEST CASES ==================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_init() {
        let e: GpuError = GpuError::Init("create_instance");
        let s: String = format!("{}", e);
        assert!(s.contains("Vulkan init"));
        assert!(s.contains("create_instance"));
    }

    #[test]
    fn display_vk() {
        let e: GpuError = GpuError::Vk("alloc_buffers", ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        let s: String = format!("{}", e);
        assert!(s.contains("alloc_buffers"));
        assert!(s.contains("Vulkan"));
    }

    #[test]
    fn display_alloc() {
        let e: GpuError = GpuError::Alloc(gpu_allocator::AllocationError::OutOfMemory);
        let s: String = format!("{}", e);
        assert!(s.contains("GPU Alloc"));
    }

    #[test]
    fn display_io() {
        let e: GpuError = GpuError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "file missing"));
        let s: String = format!("{}", e);
        assert!(s.contains("I/O"));
    }

    #[test]
    fn display_shader() {
        let e: GpuError = GpuError::Shader("invalid spirv".to_string());
        let s: String = format!("{}", e);
        assert!(s.contains("Shader"));
        assert!(s.contains("invalid spirv"));
    }

    #[test]
    fn display_buffer() {
        let e: GpuError = GpuError::Buffer("size mismatch");
        let s: String = format!("{}", e);
        assert!(s.contains("Buffer"));
        assert!(s.contains("size mismatch"));
    }

    #[test]
    fn from_alloc_error() {
        let err: GpuError = gpu_allocator::AllocationError::OutOfMemory.into();
        assert!(matches!(err, GpuError::Alloc(_)));
    }

    #[test]
    fn from_io_error() {
        let io_err: std::io::Error = std::io::Error::new(std::io::ErrorKind::Other, "io failure");
        let err: GpuError = io_err.into();
        assert!(matches!(err, GpuError::Io(_)));
    }

    #[test]
    fn error_trait_impl() {
        let e: GpuError = GpuError::Buffer("test");
        let _: &dyn std::error::Error = &e;
    }
}