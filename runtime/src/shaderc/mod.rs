use std::ffi::CString;

use crate::error::GpuError;

const STATUS_SUCCESS: i32 = 0;

const MAX_SPIRV_SIZE: usize = 16 * 1024 * 1024;
const MAX_ERROR_SIZE: usize = 4096;
const MAX_PREPROCESSOR_SIZE: usize = 1 << 20;
const MAX_UNROLLED_SIZE: usize = 4 << 20;

// Rust FFI wrapper
unsafe extern "C" {
    unsafe fn compile_shader(
        source: *const std::ffi::c_char, 
        out_spirv: *mut u8, 
        out_size: *mut usize, 
        max_size: usize
    ) -> i32;

    unsafe fn compile_shader_with_errors(
        source: *const std::ffi::c_char, 
        out_spirv: *mut u8, 
        out_size: *mut usize, 
        max_size: usize, 
        out_error: *mut std::ffi::c_char, 
        error_max_size: usize
    ) -> i32;

    unsafe fn preprocessor_shader(
        source: *const std::ffi::c_char, 
        source_path: *const std::ffi::c_char, 
        include_dirs: *const std::ffi::c_char,
        defines: *const std::ffi::c_char,
        out_result: *mut std::ffi::c_char,
        result_max: usize,
        out_error: *mut std::ffi::c_char,
        error_max: usize,
    ) -> i32;

    unsafe fn unroll_loops(
        source: *const std::ffi::c_char, 
        out_result: *mut std::ffi::c_char,
        result_max: usize,
        out_error: *mut std::ffi::c_char,
        error_max: usize,
    ) -> i32;
}

fn error_from_status(ret: i32, error_buf: &[i8]) -> GpuError {
    let error_msg: String = if !error_buf.is_empty() && error_buf[0] != 0 {
        unsafe {
            std::ffi::CStr::from_ptr(error_buf.as_ptr())
        }.to_string_lossy().into_owned()
    }
    else{
        String::new()
    };

    let detail: String = if error_msg.is_empty() {
        format!("operation failed with code {}", ret)
    }
    else{
        format!("operation failed ({}): {}", ret, error_msg)
    };

    GpuError::Shader(detail)
}

pub fn compile_glsl(source: &str) -> Result<Vec<u8>, GpuError> {
    let mut spirv: Vec<u8> = vec![0u8; MAX_SPIRV_SIZE];  
    let mut out_size: usize = 0;
    let c_source: CString = CString::new(source).map_err(|_| GpuError::Shader("source contains null byte".into()))?;
    
    let ret: i32 = unsafe {
        compile_shader(c_source.as_ptr(), spirv.as_mut_ptr(), &mut out_size, MAX_SPIRV_SIZE)
    };

    if ret != STATUS_SUCCESS {
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

    let error_message: String = if !error_buf.is_empty() && error_buf[0] != 0 {
        unsafe {
            std::ffi::CStr::from_ptr(error_buf.as_ptr()).to_string_lossy().into_owned()
        }
    }
    else{
        String::new()
    };

    if ret != STATUS_SUCCESS {
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

pub fn preprocess(source: &str, source_path: Option<&str>, include_dirs: Option<&str>, defines: Option<&str>) -> Result<String, GpuError> {
    let c_source:CString = CString::new(source).map_err(|_| GpuError::Shader("source contains null byte".into()))?;

    let c_source_path: Option<CString> = source_path
                    .map(|s: &str| CString::new(s))
                    .transpose()
                    .map_err(|_| GpuError::Shader("source_path contains null byte".into()))?;

    let c_include_dirs: Option<CString> = include_dirs
                    .map(|s: &str| CString::new(s))
                    .transpose()
                    .map_err(|_| GpuError::Shader("include_dirs contains null byte".into()))?;

    let c_defines: Option<CString>= defines 
                 .map(|s: &str| CString::new(s))
                 .transpose()
                 .map_err(|_| GpuError::Shader("defines contains null byte".into()))?;
    
    let mut out_buf: Vec<i8> = vec![0i8; MAX_PREPROCESSOR_SIZE];
    let mut err_buf: Vec<i8> = vec![0i8; MAX_ERROR_SIZE];

    let ret: i32 = unsafe {
        preprocessor_shader(
            c_source.as_ptr(), 
            c_source_path.as_ref().map_or(std::ptr::null(), |s: &CString| s.as_ptr()), 
            c_include_dirs.as_ref().map_or(std::ptr::null(), |s: &CString| s.as_ptr()), 
            c_defines.as_ref().map_or(std::ptr::null(), |s: &CString| s.as_ptr()), 
            out_buf.as_mut_ptr(), 
            MAX_PREPROCESSOR_SIZE, 
            err_buf.as_mut_ptr(), 
            MAX_ERROR_SIZE,
        )
    };

    if ret != STATUS_SUCCESS {
        return Err(error_from_status(ret, &err_buf));
    }

    let output: String = unsafe {
        std::ffi::CStr::from_ptr(out_buf.as_ptr())
    }.to_string_lossy().into_owned();

    Ok(output)
}

pub fn unroll(source: &str) -> Result<String, GpuError> {
    let c_source: CString = CString::new(source).map_err(|_| GpuError::Shader("source contains null byte".into()))?;
    let mut out_buf: Vec<i8> = vec![0i8; MAX_UNROLLED_SIZE];
    let mut err_buf: Vec<i8> = vec![0i8; MAX_ERROR_SIZE];

    let ret: i32 = unsafe {
        unroll_loops(
            c_source.as_ptr(), 
            out_buf.as_mut_ptr(), 
            MAX_UNROLLED_SIZE, 
            err_buf.as_mut_ptr(), 
            MAX_ERROR_SIZE,
        )
    };

    if ret != STATUS_SUCCESS{
        return Err(error_from_status(ret, &err_buf));
    }

    let output: String = unsafe {
        std::ffi::CStr::from_ptr(out_buf.as_ptr())
    }.to_string_lossy().into_owned();

    Ok(output)
}