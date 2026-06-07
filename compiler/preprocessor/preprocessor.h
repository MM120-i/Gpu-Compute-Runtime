#pragma once

#include <string>

enum Status {
    SUCCESS = 0,
    NULL_ARGS = -1,
    SMALL_BUFFER = -2,
    ERR_INCLUDE_DEPTH = -3,
    ERR_UNMATCHED_COND = -4,
};

#define RETURN_ERROR(line, msg) do {        \
    (result).status = (line);               \
    (result).error  = (msg);                \
    return (result);                        \
} while(0)

enum DirectiveKind {
    DIR_INCLUDE,
    DIR_DEFINE,
    DIR_IFDEF,
    DIR_IFNDEF,
    DIR_ELSE,
    DIR_ENDIF,
    DIR_ERROR,
    DIR_UNKNOWN,
};

struct PreprocessResult {
    int status;
    std::string output;
    std::string error;
};

PreprocessResult preprocess_glsl(const std::string&, const std::string&, const std::string&, const std::string&);
DirectiveKind classify(const std::string &);