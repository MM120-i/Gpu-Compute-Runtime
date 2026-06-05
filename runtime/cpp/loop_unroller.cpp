#include <cctype>

#include "loop_unroller.h"

static void skip_whitespace(const std::string& src, size_t& i) {
    while (i < src.length() && (src[i] == ' ' || src[i] == '\t' || src[i] == '\n'))
        i++;
}

static std::string extract_paren_block(const std::string& src, size_t& i) {
    i++;
    std::string block;
    int depth = 1;

    while (i < src.length() && depth > 0) {
        if (src[i] == '(') 
            depth++;
        else if (src[i] == ')') 
            depth--;

        if (depth > 0) {
            block += src[i];
            i++;
        }
    }

    i++;  

    return block;
}

static ForClauses split_for_clauses(const std::string& paren) {
    auto semi1 = paren.find(';');
    auto semi2 = paren.rfind(';');

    if (semi1 == std::string::npos || semi2 == std::string::npos || semi1 == semi2)
        return {};  

    return {
        paren.substr(0, semi1),
        paren.substr(semi1 + 1, semi2 - semi1 - 1),
        paren.substr(semi2 + 1)
    };
}

static ConstantBound try_parse_constant_bound(const std::string& cond) {
    auto lt = cond.find('<');

    if (lt == std::string::npos)
        return {false, 0};

    size_t start = lt + 1;

    if (start < cond.length() && cond[start] == '=')
        start++;

    std::string val_str = cond.substr(start);
    auto first = val_str.find_first_not_of(" \t");
    auto last  = val_str.find_last_not_of(" \t");

    if (first == std::string::npos)
        return {false, 0};

    val_str = val_str.substr(first, last - first + 1);

    try {
        int count = std::stoi(val_str);

        if (count >= 1 && count <= 64)
            return {true, count};
    } 
    catch (...) {}

    return {false, 0};
}

static std::string extract_loop_body(const std::string& src, size_t& i) {
    skip_whitespace(src, i);

    if (i >= src.length()) 
        return "";

    std::string body;

    if (src[i] == '{') {
        int depth = 1;
        body += src[i];
        i++;

        while (i < src.length() && depth > 0) {
            if(src[i] == '{') 
                depth++;
            else if(src[i] == '}') 
                depth--;

            body += src[i];
            i++;
        }
    } 
    else {
        while (i < src.length() && src[i] != ';' && src[i] != '}') {
            body += src[i];
            i++;
        }

        if (i < src.length() && src[i] == ';') {
            body += ';';
            i++;
        }
    }

    return body;
}

static std::string extract_var_name(const std::string& init) {
    auto eq = init.find('=');

    if (eq == std::string::npos) 
        return "";

    std::string before_eq = init.substr(0, eq);
    auto last_space = before_eq.find_last_of(" \t");

    std::string name;

    if (last_space != std::string::npos)
        name = before_eq.substr(last_space + 1);
    else
        name = before_eq;

    auto vf = name.find_first_not_of(" \t");
    auto vl = name.find_last_not_of(" \t");

    if (vf == std::string::npos) 
        return "";

    return name.substr(vf, vl - vf + 1);
}

static bool contains_break_or_continue(const std::string& body) {
    std::string lower = body;

    for (auto& c : lower) 
        c = std::tolower(static_cast<unsigned char>(c));

    return lower.find("break") != std::string::npos ||
           lower.find("continue") != std::string::npos;
}

static std::string substitute_var(const std::string& text, const std::string& var_name, const std::string& replacement) {
    std::string result = text;
    size_t pos = 0;

    while ((pos = result.find(var_name, pos)) != std::string::npos) {
        bool word_start = pos == 0 ||
            (!std::isalnum(static_cast<unsigned char>(result[pos - 1])) &&
             result[pos - 1] != '_');

        bool word_end = (pos + var_name.length() >= result.length()) ||
            (!std::isalnum(static_cast<unsigned char>(result[pos + var_name.length()])) &&
             result[pos + var_name.length()] != '_');

        if (word_start && word_end) {
            result.replace(pos, var_name.length(), replacement);
            pos += replacement.length();
        } 
        else {
            pos++;
        }
    }

    return result;
}

static std::string generate_unrolled(const std::string& body, const std::string& var_name, int iter_count) {
    std::string result;

    if (body[0] == '{') {
        result = "{\n";
        std::string inner = body.substr(1, body.length() - 2);

        for (int iter = 0; iter < iter_count; iter++) {
            std::string iter_body = substitute_var(inner, var_name, std::to_string(iter));
            result += iter_body + "\n";
        }

        result += "}";
    } 
    else {
        for (int iter = 0; iter < iter_count; iter++) 
            result += substitute_var(body, var_name, std::to_string(iter));
    }

    return result;
}

UnRollResult unroll_glsl_loops(const std::string& source) {
    UnRollResult result{0, "", ""};
    std::string output;
    size_t i = 0;

    while (i < source.length()) {
        if (i + 3 < source.length() &&
            std::tolower(static_cast<unsigned char>(source[i]))     == 'f' &&
            std::tolower(static_cast<unsigned char>(source[i + 1])) == 'o' &&
            std::tolower(static_cast<unsigned char>(source[i + 2])) == 'r')
        {
            size_t for_start = i;
            i += 3;

            skip_whitespace(source, i);

            if (i < source.length() && source[i] == '(') {
                std::string paren_block = extract_paren_block(source, i);
                auto [init, cond, inc] = split_for_clauses(paren_block);
                auto bound = try_parse_constant_bound(cond);

                if (!bound.valid) {
                    output += "for(" + paren_block + ")";
                    continue;
                }

                std::string body = extract_loop_body(source, i);

                if (contains_break_or_continue(body)) {
                    output += "for(" + paren_block + ")";
                    continue;
                }

                std::string var_name = extract_var_name(init);

                if (var_name.empty()) {
                    output += "for(" + paren_block + ")";
                    continue;
                }

                output += generate_unrolled(body, var_name, bound.count);
            } 
            else {
                output += "for";
            }
        } 
        else {
            output += source[i];
            i++;
        }
    }

    result.output = output;

    return result;
}
