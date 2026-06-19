#pragma once

#include "ast.h"
#include <memory>

class ConstantPropagation {
private:
    void fold_decls(std::vector<std::unique_ptr<Decl>> &);
    void fold_stmt(std::unique_ptr<Stmt> &);
    void fold_block(BlockStmt &);
    void fold_expr(std::unique_ptr<Expr> &);
    void fold_binary(std::unique_ptr<Expr> &, BinaryExpr *);
    void fold_unary(std::unique_ptr<Expr> &, UnaryExpr *);

public:
    void fold(Program &);
};