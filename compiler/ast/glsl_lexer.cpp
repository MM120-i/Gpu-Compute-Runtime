#include <unordered_map>

#include "glsl_lexer.h"

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
        {"uint", KW_UINT}, {"bool", KW_BOOL},
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
            return {STAR, loc, {}};

        case '/':
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
