#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ast.h"
#include "glsl_lexer.h"

class Parser {
private:
    Lexer &lexer_;
    std::string error_;

    void set_error(const std::string &);
    
    Token peek();
    Token consume();

    bool match(TokenKind);
    void expect(TokenKind, const std::string &);

    Type parse_type();

    std::unique_ptr<Decl> parse_declaration();
    std::unique_ptr<BufferDecl> parse_buffer_decl(const std::string &);
    std::unique_ptr<FunctionDecl> parse_function(Type, const std::string &);
    std::unique_ptr<VarDecl> parse_var_decl(Type, const std::string &);

    std::unique_ptr<Stmt> parse_stmt();
    std::unique_ptr<BlockStmt> parse_block();
    std::unique_ptr<Stmt> parse_for();
    std::unique_ptr<Stmt> parse_if();
    std::unique_ptr<Stmt> parse_return();

    // expressions
    std::unique_ptr<Expr> parse_expr();
    std::unique_ptr<Expr> parse_assignment();
    std::unique_ptr<Expr> parse_logical_or();
    std::unique_ptr<Expr> parse_logical_and();
    std::unique_ptr<Expr> parse_inclusive_or();
    std::unique_ptr<Expr> parse_xor();
    std::unique_ptr<Expr> parse_and();
    std::unique_ptr<Expr> parse_equality();
    std::unique_ptr<Expr> parse_relational();
    std::unique_ptr<Expr> parse_shift();
    std::unique_ptr<Expr> parse_additive();
    std::unique_ptr<Expr> parse_multiplicative();
    std::unique_ptr<Expr> parse_unary();
    std::unique_ptr<Expr> parse_postfix();
    std::unique_ptr<Expr> parse_primary();

    std::unique_ptr<Expr> make_int(int);
    std::unique_ptr<Expr> make_float(float);
    std::unique_ptr<Expr> make_bool(bool);
    std::unique_ptr<Expr> make_var(const std::string &);
    std::unique_ptr<Expr> make_unary(UnaryOp, std::unique_ptr<Expr>);
    std::unique_ptr<Expr> make_binary(BinaryOp, std::unique_ptr<Expr>, std::unique_ptr<Expr>);
    std::unique_ptr<Expr> make_assign(std::string, std::unique_ptr<Expr>, std::unique_ptr<Expr>);
    std::unique_ptr<Expr> make_call(const std::string &, std::vector<std::unique_ptr<Expr>>);
    std::unique_ptr<Expr> make_member(std::unique_ptr<Expr>, const std::string &);
    std::unique_ptr<Expr> make_array(std::unique_ptr<Expr>, std::unique_ptr<Expr>);
    std::unique_ptr<Expr> clone_expr(Expr &);

public:
    explicit Parser(Lexer &);
    std::unique_ptr<Program> parse();

    const std::string &error() const {
        return error_;
    }
};

extern "C" {
    char *parse_and_emit_glsl(const char *, const char**);
    void free_emitted_string(char *);
}