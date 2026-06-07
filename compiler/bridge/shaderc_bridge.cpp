#include <shaderc/shaderc.hpp>
#include <cstdint>
#include <cstring>
#include <string>

#include "../preprocessor/preprocessor.h"
#include "../unroller/loop_unroller.h"

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

    int preprocessor_shader(
        const char *source, 
        const char *source_path,
        const char *include_dirs,
        const char *defines,
        char *out_result,
        size_t result_max,
        char *out_error,
        size_t error_max
    ){
        if(!source || !out_result || !out_error)
            return NULL_ARGS;
        
        std::string src(source);
        std::string sp = source_path ? source_path : "";
        std::string id = include_dirs ? include_dirs : "";
        std::string df = defines ? defines : "";

        auto r = preprocess_glsl(src, sp, id, df);

        if(r.status != SUCCESS){
            if(!r.error.empty() && error_max > 0){
                size_t n = r.error.length() < error_max - 1 ? r.error.length() : error_max - 1;
                std::memcpy(out_error, r.error.c_str(), n);
                out_error[n] = '\0';
            }
            else if(error_max > 0){
                out_error[0] = '\0';
            }

            return r.status;
        }

        if(r.output.length() >= result_max)
            return SMALL_BUFFER;
        
        std::memcpy(out_result, r.output.c_str(), r.output.length() + 1);

        if(error_max > 0)
            out_error[0] = '\0';

        return SUCCESS;
    }

    int unroll_loops(
        const char *source,
        char *out_result,
        size_t result_max,
        char *out_error,
        size_t error_max
    ){
        if(!source || !out_result || !out_error)
            return NULL_ARGS;

        auto r = unroll_glsl_loops(source);

        if(r.status != SUCCESS){
            if(!r.error.empty() && error_max > 0){
                size_t n = r.error.length() < error_max - 1 ? r.error.length() : error_max - 1;
                std::memcpy(out_error, r.error.c_str(), n);
                out_error[n] = '\0';
            }
            else if(error_max > 0){
                out_error[0] = '\0';
            }

            return r.status;
        }

        if(r.output.length() >= result_max)
            return SMALL_BUFFER;
        
        std::memcpy(out_result, r.output.c_str(), r.output.length() + 1);

        if(error_max > 0)
            out_error[0] = '\0';

        return SUCCESS;
    }
}