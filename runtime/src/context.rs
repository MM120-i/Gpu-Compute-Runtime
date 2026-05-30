use std::ffi::{CStr, CString};
use std::mem::ManuallyDrop;
use ash::{vk, Entry};
use ash::ext::debug_utils;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use crate::error::GpuError;
use crate::buffer::GpuBuffer;

#[allow(dead_code)] // fields used once dispatcher.rs exists. For now compiler needs to shut up about unused code
pub struct GpuContext{
    pub(crate) entry: ash::Entry,
    instance: ManuallyDrop<ash::Instance>,
    device: ManuallyDrop<ash::Device>,

    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) physical_device_properties: vk::PhysicalDeviceProperties,
    
    pub(crate) compute_queue: vk::Queue,
    pub(crate) queue_family_index: u32,

    pub(crate) allocator: ManuallyDrop<Allocator>,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) descriptor_pool: vk::DescriptorPool,

    debug_utils_loader: Option<debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl GpuContext {
    pub fn new() -> Result<Self, GpuError> {
        let entry: Entry = unsafe {
            Entry::load().map_err(|_| GpuError::Init("Failed to load vulkan-1.dll. Install the Vulkan SDK"))?
        };

        let (ext_ptrs, layer_ptrs) = Self::setup_debug(&entry)?;
        let app_name: CString = CString::new("GPU Compute Runtime").unwrap();
        let engine_name: CString = CString::new("gpu-compute-runtime").unwrap();

        let app_info: vk::ApplicationInfo<'_> = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 0, 1, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_api_version(0, 0, 1, 0),
            api_version: vk::make_api_version(0, 1, 3, 0),
            _marker: std::marker::PhantomData,
        };

        let instance_create_info: vk::InstanceCreateInfo<'_> = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_layer_count: layer_ptrs.len() as u32,
            pp_enabled_layer_names: layer_ptrs.as_ptr(),
            enabled_extension_count: ext_ptrs.len() as u32,
            pp_enabled_extension_names: ext_ptrs.as_ptr(),
            ..Default::default()
        };

        let instance: ManuallyDrop<ash::Instance> = ManuallyDrop::new(
            unsafe {
                entry.create_instance(&instance_create_info, None)
            }.map_err(|e: vk::Result| GpuError::Vk("create_instance", e))?
        );
        
        // If validation layers are available, create the debug messenger now
        let has_validation: bool = !layer_ptrs.is_empty();
        let (debug_utils_loader, debug_messenger) = if has_validation {
            let debug_loader: debug_utils::Instance = debug_utils::Instance::new(&entry, &instance);
            let create_info: vk::DebugUtilsMessengerCreateInfoEXT<'_> = vk::DebugUtilsMessengerCreateInfoEXT {
                s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                p_next: std::ptr::null(),
                flags: vk::DebugUtilsMessengerCreateFlagsEXT::default(),
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                pfn_user_callback: Some(vulkan_debug_callback),
                p_user_data: std::ptr::null_mut(),
                _marker: std::marker::PhantomData,
            };
            let messenger: vk::DebugUtilsMessengerEXT = unsafe {
                debug_loader.create_debug_utils_messenger(&create_info, None)
                    .map_err(|e: vk::Result| GpuError::Vk("create_debug_utils_messenger", e))?
            };
            (Some(debug_loader), Some(messenger))
        } 
        else {
            (None, None)
        };

        let devices: Vec<vk::PhysicalDevice> = unsafe {
            instance.enumerate_physical_devices()
        }.map_err(|_| GpuError::Init("No Vulkan-capable GPU found"))?;

        let physical_device: vk::PhysicalDevice = Self::pick_physical_device(&devices, &instance)?;
        let physical_device_properties: vk::PhysicalDeviceProperties = unsafe {
            instance.get_physical_device_properties(physical_device)
        };

        let device_name_str: String = {
            let name_slice: &[i8; 256] = &physical_device_properties.device_name;
            let len: usize = name_slice.iter().position(|&c| c == 0).unwrap_or(name_slice.len());
            let name_bytes: &[u8] = unsafe { std::slice::from_raw_parts(name_slice.as_ptr() as *const u8, len) };
            String::from_utf8_lossy(name_bytes).to_string()
        };

        println!("Using GPU: {}", device_name_str);

        let queue_family_index: u32 = Self::find_compute_queue_family(&instance, physical_device).ok_or(GpuError::Init("No queu family with VK_QUEUE_COMPUTE_BIT found"))?;
        
        let queue_priority: [f32; 1] = [1.0f32];
        let queue_create_info: vk::DeviceQueueCreateInfo<'_> = vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: queue_priority.as_ptr(),
            ..Default::default()
        };

        let device_features: vk::PhysicalDeviceFeatures = vk::PhysicalDeviceFeatures::default();

        let device_create_info: vk::DeviceCreateInfo<'_> = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create_info,
            p_enabled_features: &device_features,
            ..Default::default()
        };

        let device: ManuallyDrop<ash::Device> = ManuallyDrop::new(
            unsafe {
                instance.create_device(physical_device, &device_create_info, None)
            }.map_err(|e: vk::Result| GpuError::Vk("create_device", e))?
        );

        let compute_queue: vk::Queue = unsafe {
            device.get_device_queue(queue_family_index, 0)
        };

        let command_pool: vk::CommandPool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index,
                    ..Default::default()
                },
                None,
            )
        }.map_err(|e: vk::Result| GpuError::Vk("create_command_pool", e))?;

        let allocator: ManuallyDrop<Allocator> = ManuallyDrop::new(
            Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance: (*instance).clone(),
                device: (*device).clone(),
                physical_device,
                debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
                buffer_device_address: false,
                allocation_sizes: gpu_allocator::AllocationSizes::default(),
            }).map_err(GpuError::Alloc)?
        );

        let pool_sizes: [vk::DescriptorPoolSize; 1] = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 32,
            },
        ];

        let descriptor_pool: vk::DescriptorPool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
                    max_sets: 32,
                    pool_size_count: pool_sizes.len() as u32,
                    p_pool_sizes: pool_sizes.as_ptr(),
                    ..Default::default()
                },
                None
            )
        }.map_err(|e: vk::Result| GpuError::Vk("create_descriptor_pool", e))?;

        Ok(Self {
            entry,
            instance,
            device,
            physical_device,
            physical_device_properties,
            compute_queue,
            queue_family_index,
            allocator,
            command_pool,
            descriptor_pool,
            debug_utils_loader,
            debug_messenger
        })
    }

    fn find_compute_queue_family(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<u32> {
        let families: Vec<vk::QueueFamilyProperties> = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };
        
        families.iter()
                .position(|qf: &vk::QueueFamilyProperties| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|i: usize| i as u32)
    }

    // Prefer an external gpu (2060 super in my case), fall back to integrated graphics if needed
    fn pick_physical_device(devices: &[vk::PhysicalDevice], instance: &ash::Instance) -> Result<vk::PhysicalDevice, GpuError> {
        for &device in devices {
            let prop: vk::PhysicalDeviceProperties = unsafe {
                instance.get_physical_device_properties(device)
            };

            if prop.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                return Ok(device);
            }
        }

        devices.first().copied().ok_or(GpuError::Init("No GPUs found"))
    }

    fn setup_debug(entry: &ash::Entry) -> Result<(
        Vec<*const i8>,
        Vec<*const i8>,
    ), GpuError>{
        let layer_props: Vec<vk::LayerProperties> = unsafe {
            entry.enumerate_instance_layer_properties()
                .map_err(|e: vk::Result| GpuError::Vk("enumerate_instance_layer_properties", e))?
        };

        let has_validation: bool = layer_props.iter().any(|lp: &vk::LayerProperties| {
            let name: &CStr = unsafe { CStr::from_ptr(lp.layer_name.as_ptr()) };
            name.to_bytes_with_nul() == b"VK_LAYER_KHRONOS_validation\0"
        });

        let mut ext_ptrs: Vec<*const i8> = Vec::new();
        let mut layer_ptrs: Vec<*const i8> = Vec::new();

        if has_validation {
            ext_ptrs.push(debug_utils::NAME.as_ptr());
            layer_ptrs.push(
                unsafe {
                    CStr::from_bytes_with_nul_unchecked(
                        b"VK_LAYER_KHRONOS_validation\0"
                    )
                }.as_ptr()
            );
            println!("Vulkan validation layers enabled");
        } else {
            println!("Vulkan validation layers NOT available - skipping");
        }

        Ok((ext_ptrs, layer_ptrs))
    }

    // Return the GPU name (2060 super in my case)
    pub fn device_name(&self) -> String {
        let name_slice: &[i8; 256] = &self.physical_device_properties.device_name;
        let len: usize = name_slice.iter().position(|&c| c == 0).unwrap_or(name_slice.len());
        let name_bytes: &[u8] = unsafe { std::slice::from_raw_parts(name_slice.as_ptr() as *const u8, len) };
        String::from_utf8_lossy(name_bytes).to_string()
    }

    // Block until the GPU finished all pending work
    pub fn wait_idle(&self) -> Result<(), GpuError> {
        unsafe {
            self.device.device_wait_idle()
        }.map_err(|e: vk::Result| GpuError::Vk("device_wait_idle", e))
    }

    pub(crate) fn device(&self) -> &ash::Device {
        &*self.device
    }

    // cleanup helpers
    pub fn create_buffer(&mut self, size: u64, usage: vk::BufferUsageFlags, location: MemoryLocation) -> Result<GpuBuffer, GpuError> {
        let buffer_info: vk::BufferCreateInfo<'_> = vk::BufferCreateInfo {
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let raw: vk::Buffer = unsafe {
            self.device.create_buffer(&buffer_info, None).map_err(|e| GpuError::Vk("create_buffer", e))?
        };

        let requirements: vk::MemoryRequirements = unsafe {
            self.device.get_buffer_memory_requirements(raw)
        };

        let allocation: gpu_allocator::vulkan::Allocation = self.allocator.allocate(
            &gpu_allocator::vulkan::AllocationCreateDesc {
                name: "GpuBuffer",
                requirements,
                location,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged
            }
        ).map_err(GpuError::Alloc)?;

        unsafe {
            self.device.bind_buffer_memory(raw, allocation.memory(), allocation.offset()).map_err(|e| GpuError::Vk("bind_buffer_memory", e))?
        }
        
        Ok(GpuBuffer { raw, allocation, size })
    }

    pub fn destroy_pipeline(&mut self, pipeline: crate::pipeline::ComputePipeline) {
        pipeline.destroy(self);
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            
            ManuallyDrop::drop(&mut self.allocator);

            if let (Some(ref loader), Some(messenger)) = (&self.debug_utils_loader, self.debug_messenger){
                loader.destroy_debug_utils_messenger(messenger, None);
            }

            ManuallyDrop::drop(&mut self.device);
            ManuallyDrop::drop(&mut self.instance);
        }
    }
}

// When validation layers detect problem, we r gonna call this callback
unsafe extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let level: &str = match severity{
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "ERROR",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "WARNING",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "INFO",
        _ => "VERBOSE",
    };

    if let Some(msg) = data.as_ref() {
        if let Some(msg_str) = msg.p_message.as_ref(){
            println!("[Vulkan-{}] {}", level, msg_str);
        }
    }

    vk::FALSE
}