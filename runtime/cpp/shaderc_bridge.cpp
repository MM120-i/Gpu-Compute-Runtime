#include <shaderc/shaderc.hpp>
#include <cstdint>
#include <cstring>
#include <string>

enum ReturnType {
    SUCCESS = 0,
    NULL_ARGS = -1,
    SMALL_BUFFER = -2,
};

extern "C" {
   
    int compile_shader(const char *source, uint8_t *out_spirv, size_t *out_size, size_t max_size){
        
        if(!source || !out_spirv || !out_size)
            return NULL_ARGS;

        shaderc::Compiler compiler;
        shaderc::CompileOptions options;

        options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
        options.SetOptimizationLevel(shaderc_optimization_level_performance);

        auto result = compiler.CompileGlslToSpv(source, std::strlen(source), shaderc_glsl_compute_shader, "shader.comp", options);
        auto status = result.GetCompilationStatus();

        if(status != shaderc_compilation_status_success)
            return static_cast<int>(status);

        size_t byte_size = (result.end() - result.begin()) * sizeof(uint32_t);

        if(byte_size > max_size)
            return SMALL_BUFFER;

        std::memcpy(out_spirv, result.begin(), byte_size);
        *out_size = byte_size;

        return SUCCESS;
    }

    int compile_shader_with_errors(const char *source, uint8_t *out_spirv, size_t *out_size, size_t max_size, char *out_error, size_t error_max_size){
        
        if(!source || !out_spirv || !out_size || !out_error)
            return NULL_ARGS;

        shaderc::Compiler compiler;
        shaderc::CompileOptions options;

        options.SetTargetEnvironment(shaderc_target_env_vulkan,shaderc_env_version_vulkan_1_3);
        options.SetOptimizationLevel(shaderc_optimization_level_performance);

        auto result = compiler.CompileGlslToSpv(source, std::strlen(source), shaderc_glsl_compute_shader, "shader.comp", options);
        auto status = result.GetCompilationStatus();

        if(status != shaderc_compilation_status_success){
            std::string error_message = result.GetErrorMessage();

            if(!error_message.empty()){
                const size_t copy_len = error_message.length() < error_max_size - 1 ? error_message.length() : error_max_size - 1;
                std::memcpy(out_error, error_message.c_str(), copy_len);
                out_error[copy_len] = '\0';
            }
            else if(error_max_size > 0){
                out_error[0] = '\0';
            }

            return static_cast<int>(status);
        }

        if(error_max_size > 0)
            out_error[0] = '\0';

        size_t byte_size = (result.end() - result.begin()) * sizeof(uint32_t);
        
        if (byte_size > max_size) 
            return SMALL_BUFFER;
        
        std::memcpy(out_spirv, result.begin(), byte_size);
        *out_size = byte_size;

        return SUCCESS;
    }
}