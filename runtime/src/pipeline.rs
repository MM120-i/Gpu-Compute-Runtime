use std::ffi::CString;
use ash::vk;

use crate::context::GpuContext;
use crate::buffer::GpuBuffer;
use crate::error::GpuError;

pub struct BufferBinding {
    pub slot: u32,
}

pub struct ComputePipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) descriptor_pool: vk::DescriptorPool,
    pub(crate) shader_module: vk::ShaderModule,
}

impl ComputePipeline {
    // Build a compute pipeline from SPIR-V bytes
    pub fn new(ctx: &GpuContext, spirv: &[u32], entry_point: &str, bindings: &[BufferBinding]) -> Result<Self, GpuError> {
        let shader_module_info: vk::ShaderModuleCreateInfo<'_> = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::ShaderModuleCreateFlags::default(),
            code_size: spirv.len() * 4, // cuz 4 bytes per u32 word
            p_code: spirv.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        let shader_module: vk::ShaderModule = unsafe {
            ctx.device().create_shader_module(&shader_module_info, None).map_err(|e: vk::Result| GpuError::Vk("create_shader_module", e))?
        };

        let mut layout_bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = Vec::new();

        for b in bindings {
            layout_bindings.push(vk::DescriptorSetLayoutBinding {
                binding: b.slot,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: std::ptr::null(),
                _marker: std::marker::PhantomData,
            });
        }

        let layout_info: vk::DescriptorSetLayoutCreateInfo<'_> = vk::DescriptorSetLayoutCreateInfo{
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::default(),
            binding_count: layout_bindings.len() as u32,
            p_bindings: layout_bindings.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        let descriptor_set_layout: vk::DescriptorSetLayout = unsafe {
            ctx.device().create_descriptor_set_layout(&layout_info, None).map_err(|e| GpuError::Vk("create_descriptor_set_layout", e))?
        };

        let pipeline_layout_info: vk::PipelineLayoutCreateInfo<'_> = vk::PipelineLayoutCreateInfo{
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::default(),
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        let pipeline_layout: vk::PipelineLayout = unsafe {
            ctx.device().create_pipeline_layout(&pipeline_layout_info, None).map_err(|e| GpuError::Vk("create_pipeline_layout", e))?
        };

        let bindings_count: u32 = bindings.len() as u32;
        let pool_sizes: [vk::DescriptorPoolSize; 1] = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: bindings_count * 32,
            },
        ];

        let pool_info: vk::DescriptorPoolCreateInfo<'_> = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            max_sets: 32,
            pool_size_count: 1,
            p_pool_sizes: pool_sizes.as_ptr(),
            _marker: std::marker::PhantomData,
        };

        let descriptor_pool: vk::DescriptorPool = unsafe {
            ctx.device().create_descriptor_pool(&pool_info, None).map_err(|e| GpuError::Vk("create_descriptor_poll", e))?
        };

        let entry_point_c: CString = CString::new(entry_point).expect("entry point name contains a null byte");
        
        let stage_info: vk::PipelineShaderStageCreateInfo<'_> = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::default(),
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: entry_point_c.as_ptr(),
            p_specialization_info: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        let pipeline_info: vk::ComputePipelineCreateInfo<'_> = vk::ComputePipelineCreateInfo{
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineCreateFlags::default(),
            stage: stage_info,
            layout: pipeline_layout,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
            _marker: std::marker::PhantomData,
        };

        let pipeline:vk::Pipeline  = unsafe {
            ctx.device().create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| GpuError::Vk("create_compute_pipelines", e))?
                [0]
        };
        
        Ok(Self { 
            pipeline, 
            pipeline_layout, 
            descriptor_set_layout, 
            descriptor_pool, 
            shader_module 
        })
    }


    pub fn create_descriptor_set(&self, ctx: &GpuContext, buffers: &[&GpuBuffer]) -> Result<vk::DescriptorSet, GpuError> {
        let alloc_info: vk::DescriptorSetAllocateInfo<'_> = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            descriptor_pool: self.descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &self.descriptor_set_layout,
            _marker: std::marker::PhantomData,
        };

        let descriptor_set: vk::DescriptorSet = unsafe {
            ctx.device().allocate_descriptor_sets(&alloc_info).map_err(|e: vk::Result| GpuError::Vk("allocate_descriptor_sets", e))?
            [0]  
        };

        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();

        for buf in buffers {
            buffer_infos.push(vk::DescriptorBufferInfo {
                buffer: buf.raw(),
                offset: 0,
                range: buf.size(),
            });
        }
        
        let mut writes: Vec<vk::WriteDescriptorSet<'_>> = Vec::new();

        for (i, buffer_info) in buffer_infos.iter().enumerate() {
            writes.push(vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: descriptor_set,
                dst_binding: i as u32,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_image_info: std::ptr::null(),
                p_buffer_info: buffer_info,
                p_texel_buffer_view: std::ptr::null(),
                _marker: std::marker::PhantomData,
            });
        }

        unsafe {
            ctx.device().update_descriptor_sets(&writes, &[]);
        }

        Ok(descriptor_set)
    }

    pub fn from_glsl(ctx: &GpuContext, glsl_source: &str, entry_point: &str, bindings: &[BufferBinding]) -> Result<Self, GpuError> {
        let spirv: Vec<u8> = crate::compiler::compile_glsl(glsl_source)?;
        let words: &[u32] = unsafe {
            std::slice::from_raw_parts(
                spirv.as_ptr() as *const u32, 
                spirv.len() / std::mem::size_of::<u32>()
            )
        };
        
        Self::new(ctx, words, entry_point, bindings)
    }

    pub fn from_glsl_with_errors(ctx: &GpuContext, glsl_source: &str, entry_point: &str, bindings: &[BufferBinding]) -> Result<Self, GpuError> {
        let (spirv, _errors): (Vec<u8>, String) = crate::compiler::compile_glsl_with_errors(glsl_source)?;
        let words: &[u32] = unsafe {
            std::slice::from_raw_parts(
                spirv.as_ptr() as *const u32, 
                spirv.len() / std::mem::size_of::<u32>()
            )
        };
        
        Self::new(ctx, words, entry_point, bindings)
    }

    pub fn raw_pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn raw_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub(crate) fn destroy(&self, ctx: &mut GpuContext) {
        unsafe {
            ctx.device().destroy_descriptor_pool(self.descriptor_pool, None);
            ctx.device().destroy_pipeline(self.pipeline, None);
            ctx.device().destroy_pipeline_layout(self.pipeline_layout, None);
            ctx.device().destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            ctx.device().destroy_shader_module(self.shader_module, None);
        }
    }
}