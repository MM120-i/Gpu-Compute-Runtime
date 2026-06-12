#pragma once

#include <sstream>

#include "ast.h"

class Emitter : public Visitor{
private:
    std::ostringstream out_;
    int indent_ = 0;

    void indent();
    void emit_type(const Type &);
    void emit_op(BinaryOp);

public:
    std::string emit(Program &);

    void visit(Program &) override;
    void visit(VarDecl &) override;
    void visit(BufferDecl &) override;
    void visit(LayoutDecl &) override;
    void visit(FunctionDecl &) override;
    void visit(BlockStmt &) override;
    void visit(ForStmt &) override;
    void visit(IfStmt &) override;
    void visit(ReturnStmt &) override;
    void visit(ExprStmt &) override;
    void visit(DeclStmt&) override;
    void visit(IntLiteral &) override;
    void visit(FloatLiteral &) override;
    void visit(BoolLiteral &) override;
    void visit(Variable &) override;
    void visit(UnaryExpr &) override;
    void visit(BinaryExpr &) override;
    void visit(AssignExpr &) override;
    void visit(CallExpr &) override;
    void visit(MemberExpr &) override;
    void visit(ArrayExpr &) override;
};