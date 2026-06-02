use std::ffi::CString;

use crate::error::GpuError;

const MAX_SPIRV_SIZE: usize = 16 * 1024 * 1024;
const MAX_ERROR_SIZE: usize = 4096;

// Rust FFI wrapper
unsafe extern "C" {
    unsafe fn compile_shader(source: *const std::ffi::c_char, out_spirv: *mut u8, out_size: *mut usize, max_size: usize) -> i32;
    unsafe fn compile_shader_with_errors(source: *const std::ffi::c_char, out_spirv: *mut u8, out_size: *mut usize, max_size: usize, out_error: *mut std::ffi::c_char, error_max_size: usize) -> i32;
}

pub fn compile_glsl(source: &str) -> Result<Vec<u8>, GpuError> {
    let mut spirv: Vec<u8> = vec![0u8; MAX_SPIRV_SIZE];  
    let mut out_size: usize = 0;
    let c_source: CString = CString::new(source).map_err(|_| GpuError::Shader("source contains null byte".into()))?;
    
    let ret: i32 = unsafe {
        compile_shader(c_source.as_ptr(), spirv.as_mut_ptr(), &mut out_size, MAX_SPIRV_SIZE)
    };

    if ret != 0 {
        return Err(GpuError::Shader(format!("compilation failed with code {}", ret)));
    }

    spirv.truncate(out_size);

    Ok(spirv)
}

pub fn compile_glsl_with_errors(source: &str) -> Result<(Vec<u8>, String), GpuError> {
    let mut spirv: Vec<u8> = vec![0u8; MAX_SPIRV_SIZE];  
    let mut out_size: usize = 0;
    let mut error_buf: Vec<i8> = vec![0i8; MAX_ERROR_SIZE];
    let c_source: CString = CString::new(source).map_err(|_| GpuError::Shader("source contains null byte".into()))?;
    
    let ret: i32 = unsafe {
        compile_shader_with_errors(c_source.as_ptr(), spirv.as_mut_ptr(), &mut out_size, MAX_SPIRV_SIZE, error_buf.as_mut_ptr(), MAX_ERROR_SIZE)
    };

    let error_message: String = if error_buf[0] != 0 {
        unsafe {
            std::ffi::CStr::from_ptr(error_buf.as_ptr()).to_string_lossy().into_owned()
        }
    }
    else{
        String::new()
    };

    if ret != 0 {
        let detail: String = if error_message.is_empty() {
            format!("compilation failed with code {}", ret)
        }
        else{
            format!("compilation failed({}): {}", ret, error_message)
        };

        return Err(GpuError::Shader(detail));
    }

    spirv.truncate(out_size);

    Ok((spirv, error_message))
}
