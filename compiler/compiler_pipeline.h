#pragma once

#include <string>
#include <vector>

enum PassKind {
    PASS_CONSTANT_PROPAGATION = 1 << 0,
};

struct PipelineResult {
    std::string glsl;
    bool success = false;
    std::string error;
};

/**
 * Class is non-copyable/non-moveable to make ownership clear, even tho 
 * the defaulted special memebers would be fine.
 */
class CompilerPipeline {
private:
    std::vector<PassKind> passes_;

public:
    CompilerPipeline() = default;
    CompilerPipeline(const CompilerPipeline &) = delete;
    CompilerPipeline &operator = (const CompilerPipeline &) = delete;
    CompilerPipeline(CompilerPipeline &&) = delete;
    CompilerPipeline &operator = (CompilerPipeline &&) = delete;

    ~CompilerPipeline() = default;

    void add_pass(PassKind);
    PipelineResult run(const std::string &) const;
};

extern "C" {
    char *run_pipeline(const char *, int);
}