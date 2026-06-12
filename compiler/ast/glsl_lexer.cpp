#include <cstdio>
#include <unordered_map>

#include "glsl_lexer.h"
#include "emitter.h"

Lexer::Lexer(std::string source): source_(std::move(source)) {}


char Lexer::peek_char(size_t ahead) const {
    size_t index = pos_ + ahead;
    return index < source_.size() ? source_[index] : '\0';
}

char Lexer::advance(){
    char c = source_[pos_++];

    if(c == '\n'){
        line_++;
        column_ = 1;
    }
    else{
        column_++;
    }

    return c;
}

void Lexer::skip_whitespace() {
    while(pos_ < source_.size()){
        char c = source_[pos_];

        if(c == ' ' || c == '\t' || c == '\n' || c == '\r'){
            advance();
        }
        else if(c == '/'){
            if(peek_char(1) == '/')
                skip_line_comment();
            else if(peek_char(1) == '*')
                skip_block_comment();
            else
                break;
        }
        else{
            break;
        }
    }
}

void Lexer::skip_line_comment() {
    while(pos_ < source_.size() && source_[pos_] != '\n')
        advance();
}

void Lexer::skip_block_comment() {
    while(pos_ + 1 < source_.size()){
        if(source_[pos_] == '*' && source_[pos_ + 1] == '/'){
            advance();
            advance();
            return;
        }

        advance();
    }
}

Token Lexer::make_identifier_or_keyword(SourceLoc loc){
    static const std::unordered_map<std::string, TokenKind> keywords = {
        {"void", KW_VOID}, {"float", KW_FLOAT}, {"int", KW_INT},
        {"uint", KW_UINT}, {"bool", KW_BOOL}, {"double", KW_DOUBLE},
        {"vec2", KW_VEC2}, {"vec3", KW_VEC3}, {"vec4", KW_VEC4},
        {"if", KW_IF}, {"else", KW_ELSE}, {"for", KW_FOR},
        {"while", KW_WHILE}, {"return", KW_RETURN}, 
        {"layout", KW_LAYOUT}, {"buffer", KW_BUFFER},
        {"in", KW_IN}, {"out", KW_OUT},
        {"true", KW_TRUE}, {"false", KW_FALSE},
        {"struct", KW_STRUCT},
        {"break", KW_BREAK},
        {"const", KW_CONST},
    };

    token_start_ = pos_;

    while(isalnum(peek_char()) || peek_char() == '_')
        advance();

    auto text = std::string_view(source_).substr(token_start_, pos_ - token_start_);
    auto it = keywords.find(std::string(text));

    TokenKind kind = (it != keywords.end()) ? it->second : IDENTIFIER;

    return {kind, loc, text};
}

Token Lexer::make_number(SourceLoc loc){
    token_start_ = pos_;
    bool is_float = false;

    while(isdigit(peek_char()))
        advance();

    if(peek_char() == '.'){
        is_float = true;
        advance();

        while(isdigit(peek_char()))
            advance();
    }

    if(peek_char() == 'f' || peek_char() == 'F'){
        is_float = true;
        advance();
    }

    auto text = std::string_view(source_).substr(token_start_, pos_ - token_start_);

    return {is_float ? FLOAT_LITERAL : INT_LITERAL, loc, text};
}

Token Lexer::peek(){
    if(!has_peeked_){
        peeked_ = next_token();
        has_peeked_ = true;
    }

    return peeked_;
}

Token Lexer::consume(){
    if(has_peeked_){
        has_peeked_ = false;
        return peeked_;
    }

    return next_token();
}

bool Lexer::eof() const{
    return pos_ >= source_.size();
}

Token Lexer::next_token(){
    skip_whitespace();

    if(pos_ >= source_.size())
        return {END_OF_FILE, {(int)line_, (int)column_}, {}};

    SourceLoc loc{(int)line_, (int)column_};
    token_start_ = pos_;
    char c = source_[pos_];

    if(isalpha(c) || c == '_')
        return make_identifier_or_keyword(loc);
    
    if(isdigit(c))
        return make_number(loc);

    advance();

    switch (c) {
        case '(':
            return {OPEN_PAREN, loc, {}};

        case ')':
            return {CLOSE_PAREN, loc, {}};

        case '{':
            return {OPEN_BRACE, loc, {}};

        case '}':
            return {CLOSE_BRACE, loc, {}};

        case '[':
            return {OPEN_BRACKET, loc, {}};
            
        case ']':
            return {CLOSE_BRACKET, loc, {}};

        case ';':
            return {SEMICOLON, loc, {}};

        case ',':
            return {COMMA, loc, {}};

        case '.':
            return {DOT, loc, {}};

        case '#':
            return {HASH, loc, {}};

        case '+':
            if(peek_char() == '+'){
                advance();
                return {PLUS_PLUS, loc, {}};
            }

            if(peek_char() == '='){
                advance();
                return {PLUS_EQUALS, loc, {}};
            }

            return {PLUS, loc, {}};

        case '-':
            if(peek_char() == '-'){
                advance();
                return {MINUS_MINUS, loc, {}};
            }

            if(peek_char() == '='){
                advance();
                return {MINUS_EQUALS, loc, {}};
            }

            return {MINUS, loc, {}};

        case '*':
            if(peek_char() == '='){
                advance();
                return {STAR_EQUALS, loc, {}};
            }
            return {STAR, loc, {}};

        case '/':
            if(peek_char() == '='){
                advance();
                return {SLASH_EQUALS, loc, {}};
            }
            return {SLASH, loc, {}};

        case '=':
            if(peek_char() == '='){
                advance();
                return {EQUALS_EQUALS, loc, {}};
            }

            return {EQUALS, loc, {}};

        case '!':
            if (peek_char() == '='){
                advance();
                return {NOT_EQUALS, loc, {}};
            }

            return {BANG, loc, {}};

        case '<':
            if(peek_char() == '='){
                advance();
                return {LESS_EQUALS, loc, {}};
            }

            return {LESS, loc, {}};

        case '>':
            if(peek_char() == '='){
                advance();
                return {GREATER_EQUALS, loc, {}};
            }

            return {GREATER, loc, {}};
            
        case '&':
            if(peek_char() == '&'){
                advance();
                return {AND_AND, loc, {}};
            }

            return {AMPERSAND, loc, {}};

        case '|':
            if(peek_char() == '|'){
                advance();
                return {OR_OR, loc, {}};
            }

            return {PIPE, loc, {}};

        case '^':
            return {CARET, loc, {}};

        case '~':
            return {TILDE, loc, {}};

        default:
            return {UNKNOWN, loc, {}};
    }
}

static const char *token_kind_name(TokenKind kind){
    switch (kind) {
        case END_OF_FILE:       return "END_OF_FILE";
        case INT_LITERAL:       return "INT_LITERAL";
        case FLOAT_LITERAL:     return "FLOAT_LITERAL";
        case IDENTIFIER:        return "IDENTIFIER";
        case KW_VOID:           return "KW_VOID";
        case KW_FLOAT:          return "KW_FLOAT";
        case KW_INT:            return "KW_INT";
        case KW_UINT:           return "KW_UINT";
        case KW_BOOL:           return "KW_BOOL";
        case KW_DOUBLE:         return "KW_DOUBLE";
        case KW_VEC2:           return "KW_VEC2";
        case KW_VEC3:           return "KW_VEC3";
        case KW_VEC4:           return "KW_VEC4";
        case KW_IF:             return "KW_IF";
        case KW_ELSE:           return "KW_ELSE";
        case KW_FOR:            return "KW_FOR";
        case KW_WHILE:          return "KW_WHILE";
        case KW_RETURN:         return "KW_RETURN";
        case KW_LAYOUT:         return "KW_LAYOUT";
        case KW_BUFFER:         return "KW_BUFFER";
        case KW_IN:             return "KW_IN";
        case KW_OUT:            return "KW_OUT";
        case KW_TRUE:           return "KW_TRUE";
        case KW_FALSE:          return "KW_FALSE";
        case KW_STRUCT:         return "KW_STRUCT";
        case KW_BREAK:          return "KW_BREAK";
        case KW_CONTINUE:       return "KW_CONTINUE";
        case KW_CONST:          return "KW_CONST";
        case SEMICOLON:         return "SEMICOLON";
        case OPEN_BRACE:        return "OPEN_BRACE";
        case CLOSE_BRACE:       return "CLOSE_BRACE";
        case OPEN_PAREN:        return "OPEN_PAREN";
        case CLOSE_PAREN:       return "CLOSE_PAREN";
        case OPEN_BRACKET:      return "OPEN_BRACKET";
        case CLOSE_BRACKET:     return "CLOSE_BRACKET";
        case COMMA:             return "COMMA";
        case DOT:               return "DOT";
        case HASH:              return "HASH";
        case EQUALS:            return "EQUALS";
        case PLUS_EQUALS:       return "PLUS_EQUALS";
        case MINUS_EQUALS:      return "MINUS_EQUALS";
        case STAR_EQUALS:       return "STAR_EQUALS";
        case SLASH_EQUALS:      return "SLASH_EQUALS";
        case PLUS:              return "PLUS";
        case MINUS:             return "MINUS";
        case STAR:              return "STAR";
        case SLASH:             return "SLASH";
        case PLUS_PLUS:         return "PLUS_PLUS";
        case MINUS_MINUS:       return "MINUS_MINUS";
        case EQUALS_EQUALS:     return "EQUALS_EQUALS";
        case NOT_EQUALS:        return "NOT_EQUALS";
        case LESS:              return "LESS";
        case GREATER:           return "GREATER";
        case LESS_EQUALS:       return "LESS_EQUALS";
        case GREATER_EQUALS:    return "GREATER_EQUALS";
        case AND_AND:           return "AND_AND";
        case OR_OR:             return "OR_OR";
        case BANG:              return "BANG";
        case AMPERSAND:         return "AMPERSAND";
        case PIPE:              return "PIPE";
        case CARET:             return "CARET";
        case TILDE:             return "TILDE";
        default:                return "UNKNOWN";
    }
}

// ========================== EXTERN C FUNCTIONS ==========================

extern "C" int tokenize_glsl(const char *source, char *output, int output_size){
    if(!source || !output || output_size <= 0)
        return static_cast<int>(Error::NULL_ARGS);

    Lexer lexer(source);
    int written = 0;

    while(written < output_size){
        Token t = lexer.consume();
        const char *name = token_kind_name(t.kind);
        int needed = snprintf(output + written, output_size - written, "%s,", name);

        if(needed < 0)
            return static_cast<int>(Error::SMALL_BUFFER);

        written += needed;

        if(t.kind == END_OF_FILE)
            break;
    }

    if(written > 0)
        output[written - 1] = '\0';

    return static_cast<int>(Error::SUCCESS);
}

extern "C" int emit_test_ast(char *output, int output_size){

    if(!output || output_size <= 0)
        return static_cast<int>(Error::NULL_ARGS);

    Program prog;

    auto v = std::make_unique<VarDecl>();

    v->type = {TypeKind::FLOAT};
    v->name = "x";
    v->initializer = std::make_unique<FloatLiteral>();

    static_cast<FloatLiteral *>(v->initializer.get())->value = 3.14f;
    prog.declarations.push_back(std::move(v));

    Emitter emitter;
    std::string result = emitter.emit(prog);

    if((int)result.size() >= output_size)
        return static_cast<int>(Error::SMALL_BUFFER);

    std::strncpy(output, result.c_str(), output_size);

    return static_cast<int>(Error::SUCCESS);
}