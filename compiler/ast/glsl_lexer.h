#pragma once

#include <string>
#include <string_view>

#include "ast.h"

enum TokenKind {
    END_OF_FILE,

    INT_LITERAL,
    FLOAT_LITERAL,
    IDENTIFIER,

    KW_VOID, KW_FLOAT, KW_INT, KW_UINT, KW_BOOL,
    KW_VEC2, KW_VEC3, KW_VEC4,
    KW_IF, KW_ELSE, KW_FOR, KW_WHILE, KW_RETURN,
    KW_LAYOUT, KW_BUFFER, KW_IN, KW_OUT,
    KW_TRUE, KW_FALSE, KW_STRUCT,
    KW_BREAK, KW_CONTINUE, KW_CONST,

    SEMICOLON, OPEN_BRACE, CLOSE_BRACE,
    OPEN_PAREN, CLOSE_PAREN, OPEN_BRACKET, CLOSE_BRACKET,
    COMMA, DOT, HASH,

    EQUALS, PLUS_EQUALS, MINUS_EQUALS,

    PLUS, MINUS, STAR, SLASH,
    PLUS_PLUS, MINUS_MINUS,

    EQUALS_EQUALS, NOT_EQUALS,
    LESS, GREATER, LESS_EQUALS, GREATER_EQUALS,

    AND_AND, OR_OR, BANG,

    AMPERSAND, PIPE, CARET, TILDE,

    UNKNOWN,
};

struct Token {
    TokenKind kind;
    SourceLoc loc;
    std::string_view text;
};

class Lexer {
private:
    std::string source_;
    
    size_t pos_ = 0;
    size_t line_ = 1;
    size_t column_ = 1;
    size_t token_start_ = 0;

    Token peeked_;
    bool has_peeked_ = false;

    char peek_char(size_t ahead = 0) const;
    char advance();

    void skip_whitespace();
    void skip_line_comment();
    void skip_block_comment();

    Token make_identifier_or_keyword(SourceLoc);
    Token make_number(SourceLoc);
    Token next_token();

public:
    explicit Lexer(std::string source);

    Token peek();
    Token consume();
    bool eof() const;
};