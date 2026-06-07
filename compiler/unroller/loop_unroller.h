#pragma once

#include <string>

struct UnRollResult {
    int status;
    std::string output;
    std::string error;
};

struct ForClauses {
    std::string init;
    std::string cond;
    std::string inc;
};

struct ConstantBound {
    bool valid = false;
    int count = 0;   
};

UnRollResult unroll_glsl_loops(const std::string &);
