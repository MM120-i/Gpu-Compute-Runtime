#include <cstring>
#include <memory>
#include <sstream>

#include "ast/ast.h"
#include "ast/glsl_lexer.h"
#include "ast/glsl_parser.h"
#include "ast/emitter.h"
#include "ast/constant_propagation.h"
#include "compiler_pipeline.h"

namespace {
    std::string extract_version_line(const std::string &source, std::string &body){
        std::istringstream stream(source);
        std::string line;

        while(std::getline(stream, line)){
            auto first_non_space = line.find_first_not_of(" \t\r");

            if(first_non_space != std::string::npos && line.compare(first_non_space, 8, "#version") == 0){
                std::ostringstream rest;
                rest << stream.rdbuf();
                body = rest.str();

                if(!body.empty() && body.back() != '\n')
                    body.push_back('\n');

                return line + "\n";
            }
        }

        body = source;
        return {};
    }

    bool run_passes(const std::vector<PassKind> &passes, Program &program, std::string &error){
        for(auto kind : passes){
            switch (kind){
                case PASS_CONSTANT_PROPAGATION:
                    ConstantPropagation constant_prop;
                    constant_prop.fold(program);
                    break;
            }
        }

        return true;
    }   
}

void CompilerPipeline::add_pass(PassKind kind){
    passes_.push_back(kind);
}


/**
* Order matters here, so:
* extract #version -> lex -> parse -> run passes -> emit GLSL -> prepend #version
* Note: Input must be valid GLSL starting with '#version'
*/
PipelineResult CompilerPipeline::run(const std::string &source) const {
    PipelineResult result;

    if(passes_.empty()){
        if(source.empty()){
            result.error = "Empty source";
            return result;
        }

        result.glsl = source;
        result.success = true;
        
        return result;
    }

    std::string body;
    auto version_line = extract_version_line(source, body);

    if(body.empty()){
        result.error = "Empty source after #version extraction";
        return result;
    }

    Lexer lexer(body);
    Parser parser(lexer);
    std::unique_ptr<Program> program;

    try {
        program = parser.parse();
    }
    catch(const std::exception& e){
        result.error = "Parse error: ";
        result.error += e.what();
        return result;
    }
    
    if(!program){
        result.error = "Parser returned null program";
        return result;
    }

    if(!run_passes(passes_, *program, result.error))
        return result;

    Emitter emitter;
    std::string emitted = emitter.emit(*program);

    result.glsl = version_line + emitted;
    result.success = true;

    return result;
}

extern "C" {
    char *run_pipeline(const char *source, int pass_flags){
        if(!source)
            return nullptr;

        CompilerPipeline pipeline;

        if(pass_flags & PASS_CONSTANT_PROPAGATION)
            pipeline.add_pass(PASS_CONSTANT_PROPAGATION);

        auto result = pipeline.run(source);

        if(!result.success)
            return nullptr;

        auto *cstr = static_cast<char *>(std::malloc(result.glsl.size() + 1));

        if(!cstr){
            std::perror("Mem alloc failed");
            return nullptr;
        }

        std::memcpy(cstr, result.glsl.data(), result.glsl.size());
        cstr[result.glsl.size()] = '\0';

        return cstr;
    }
}