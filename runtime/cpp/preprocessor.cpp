#include <sstream>
#include <fstream>
#include <stack>
#include <unordered_map>
#include <cctype>

#include "preprocessor.h"

static DirectiveKind classify_directive(const std::string& dir) {
    if (dir == "#include") 
        return DIR_INCLUDE;
    if (dir == "#define")  
        return DIR_DEFINE;
    if (dir == "#ifdef")   
        return DIR_IFDEF;
    if (dir == "#ifndef")  
        return DIR_IFNDEF;
    if (dir == "#else")    
        return DIR_ELSE;
    if (dir == "#endif")   
        return DIR_ENDIF;
    if (dir == "#error")   
        return DIR_ERROR;

    return DIR_UNKNOWN;
}

static PreprocessResult process(
    const std::string& source,
    const std::string& source_dir,
    const std::vector<std::string>& search_dir,
    const std::string& std_dir,
    const std::unordered_map<std::string, std::string>& defines_map,
    int depth
);

static std::string trimmed(const std::string&);

static PreprocessResult handle_include(
    const std::string& t,
    int line_num,
    const std::string& source_dir,
    const std::vector<std::string>& search_dir,
    const std::string& std_dir,
    const std::unordered_map<std::string, std::string>& defines_map,
    int depth,
    std::ostringstream& output
) {
    PreprocessResult result{0, "", ""};

    char quote = 0;
    size_t start = t.find('\"');

    if (start != std::string::npos) {
        quote = '\"';
    } 
    else {
        start = t.find('<');

        if (start != std::string::npos)
            quote = '>';
    }

    if (quote == 0)
        RETURN_ERROR(line_num, "#include: expected \"file\" or <file>");

    char close = (quote == '<') ? '>' : '\"';
    size_t end = t.find(close, start + 1);

    if (end == std::string::npos)
        RETURN_ERROR(line_num, "#include: missing closing " + std::string(1, close));

    std::string filename = t.substr(start + 1, end - start - 1);
    std::string full_path;
    bool found = false;

    auto try_dirs = [&](const std::vector<std::string>& dirs) -> bool {
        for (const auto& d : dirs) {
            full_path = d + "/" + filename;
            std::ifstream test(full_path);

            if (test.good()) {
                found = true;
                return true;
            }
        }

        return false;
    };

    if (quote == '\"') {
        if (!source_dir.empty() && !try_dirs({source_dir}))
            try_dirs(search_dir);
    } 
    else {
        if (!try_dirs({std_dir}))
            try_dirs(search_dir);
    }

    if (!found)
        RETURN_ERROR(line_num, "#include: '" + filename + "' not found");

    std::ifstream file(full_path);
    std::string content(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    auto sub = process(content, source_dir, search_dir, std_dir, defines_map, depth + 1);

    if (sub.status != SUCCESS)
        return sub;

    output << sub.output;
    return result;
}

static PreprocessResult handle_define(const std::string& rest, std::unordered_map<std::string, std::string>& macros) {
    auto space = rest.find_first_of(" \t");

    if (space != std::string::npos) {
        std::string name  = rest.substr(0, space);
        std::string value = trimmed(rest.substr(space));
        macros[name] = value;
    } 
    else if (!rest.empty()) {
        macros[rest] = "";
    }

    return {0, "", ""};
}

static PreprocessResult handle_ifdef(const std::string& rest, bool parent_active, const std::unordered_map<std::string, std::string>& macros, std::stack<std::pair<bool, bool>>& cond) {
    bool active = parent_active && macros.find(rest) != macros.end();
    cond.push({active, false});
    return {0, "", ""};
}

static PreprocessResult handle_ifndef(const std::string& rest, bool parent_active, const std::unordered_map<std::string, std::string>& macros, std::stack<std::pair<bool, bool>>& cond) {
    bool active = parent_active && macros.find(rest) == macros.end();
    cond.push({active, false});
    return {0, "", ""};
}

static PreprocessResult handle_else( std::stack<std::pair<bool, bool>>& cond) {
    PreprocessResult result{0, "", ""};

    if (cond.size() <= 1)
        RETURN_ERROR(0, "#else without #if");

    auto top = cond.top(); 
    cond.pop();
    bool parent = cond.top().first;
    cond.push({parent && !top.first, false});

    return result;
}

static PreprocessResult handle_endif(std::stack<std::pair<bool, bool>>& cond) {

    if (cond.size() <= 1) {
        PreprocessResult r{0, "", ""};
        r.status = ERR_UNMATCHED_COND;
        r.error  = "unmatched #endif";
        return r;
    }

    cond.pop();
    return {0, "", ""};
}

static PreprocessResult handle_error(const std::string& t, const std::string& rest,int line_num) {
    std::string msg;
    auto qs = t.find('\"');

    if (qs != std::string::npos) {
        auto qe = t.find('\"', qs + 1);

        if (qe != std::string::npos)
            msg = t.substr(qs + 1, qe - qs - 1);
        else
            msg = t.substr(qs + 1);
    } 
    else {
        msg = rest;
    }

    return {line_num, "", msg};
}

static PreprocessResult handle_directive(
    const std::string& directive,
    const std::string& rest,
    const std::string& t,
    int line_num,
    std::unordered_map<std::string, std::string>& macros,
    std::stack<std::pair<bool, bool>>& cond,
    bool parent_active,
    const std::string& source_dir,
    const std::vector<std::string>& search_dir,
    const std::string& std_dir,
    const std::unordered_map<std::string, std::string>& defines_map,
    int depth,
    std::ostringstream& output
) {
    if (!parent_active) 
        return {0, "", ""};  

    PreprocessResult result{0, "", ""};

    switch (classify_directive(directive)) {
        case DIR_INCLUDE:
            return handle_include(t, line_num, source_dir, search_dir, std_dir,
                                  defines_map, depth, output);
        case DIR_DEFINE:
            return handle_define(rest, macros);

        case DIR_IFDEF:
            return handle_ifdef(rest, parent_active, macros, cond);

        case DIR_IFNDEF:
            return handle_ifndef(rest, parent_active, macros, cond);

        case DIR_ELSE:
            return handle_else(cond);

        case DIR_ENDIF:
            return handle_endif(cond);

        case DIR_ERROR:
            return handle_error(t, rest, line_num);

        default:
            output << t << "\n";
            return result;
    }
}

static std::string trimmed(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end   = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

//  Public API
PreprocessResult preprocess_glsl(
    const std::string& source,
    const std::string& source_path,
    const std::string& include_dirs,
    const std::string& defines
) {
    std::unordered_map<std::string, std::string> defines_map;

    if (!defines.empty()) {
        std::istringstream ss(defines);
        std::string entry;

        while (std::getline(ss, entry, ';')) {
            auto eq = entry.find('=');

            if (eq != std::string::npos)
                defines_map[entry.substr(0, eq)] = entry.substr(eq + 1);
            else
                defines_map[entry] = "";
        }
    }

    std::string source_dir;

    if (!source_path.empty()) {
        auto slash = source_path.find_last_of("/\\");

        if (slash != std::string::npos)
            source_dir = source_path.substr(0, slash);
    }

    std::vector<std::string> search_dirs;
    if (!include_dirs.empty()) {
        std::istringstream ss(include_dirs);
        std::string dir;
        while (std::getline(ss, dir, ';'))
            if (!dir.empty())
                search_dirs.push_back(dir);
    }

    return process(source, source_dir, search_dirs, "shaders/std", defines_map, 0);
}

static PreprocessResult process(
    const std::string& source,
    const std::string& source_dir,
    const std::vector<std::string>& search_dir,
    const std::string& std_dir,
    const std::unordered_map<std::string, std::string>& defines_map,
    int depth
) {
    PreprocessResult result{0, "", ""};

    if (depth > 16)
        RETURN_ERROR(ERR_INCLUDE_DEPTH, "#include: maximum include depth (16) exceeded");

    std::unordered_map<std::string, std::string> macros = defines_map;
    std::ostringstream output;
    std::istringstream input(source);
    std::string line;
    int line_num = 0;

    std::stack<std::pair<bool, bool>> cond;
    cond.push({true, false});   

    while (std::getline(input, line)) {
        line_num++;
        std::string t = trimmed(line);

        if (t.empty() || t[0] != '#') {
            if (cond.top().first) {
                std::string expanded = line;

                for (const auto& [name, value] : macros) {
                    size_t pos = 0;

                    while ((pos = expanded.find(name, pos)) != std::string::npos) {
                        bool word_start = pos == 0 || (!isalnum(expanded[pos - 1]) && expanded[pos - 1] != '_');
                        bool word_end = pos + name.length() >= expanded.length() || (!isalnum(expanded[pos + name.length()]) &&expanded[pos + name.length()] != '_');

                        if (word_start && word_end) {
                            expanded.replace(pos, name.length(), value);
                            pos += value.length();
                        } 
                        else {
                            pos++;
                        }
                    }
                }
                output << expanded << "\n";
            }
            continue;
        }

        std::string directive = t.substr(0, t.find_first_of(" \t"));
        std::string rest = trimmed(t.substr(directive.length()));

        auto r = handle_directive(
            directive, rest, t, line_num, macros, cond, cond.top().first,
            source_dir, search_dir, std_dir, defines_map, depth, output
        );

        if (r.status != SUCCESS) {
            result.status = r.status;
            result.output = output.str();
            result.error  = r.error;
            return result;
        }
    }

    if (cond.size() != 1)
        RETURN_ERROR(ERR_UNMATCHED_COND, "unmatched #ifdef/#ifndef");

    result.output = output.str();
    return result;
}
