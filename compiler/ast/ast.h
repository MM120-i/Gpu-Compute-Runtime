#pragma once

#include <string>
#include <vector>
#include <memory>

// forward declarations
class Visitor;
class Program;
struct Decl;
class VarDecl;
class BufferDecl;
struct BufferMember;
class FunctionDecl;

struct Stmt;
class BlockStmt;
class ForStmt;
class IfStmt;
class ReturnStmt;
class ExprStmt;

struct Expr;
class IntLiteral;
class FloatLiteral;
class BoolLiteral;
class Variable;
class UnaryExpr;
class BinaryExpr;
class AssignExpr;
class CallExpr;
class MemberExpr;
class ArrayExpr;

enum class TypeKind {
    FLOAT,
    INT,
    UINT,
    BOOL,
    VEC2,
    VEC3,
    VEC4,
};

enum class StorageClass {
    NONE,
    BUFFER,
};

enum class UnaryOp {
    NEGATE,
    NOT,
};

enum class BinaryOp {
    ADD,
    SUB,
    MUL,
    DIV,
    LT,
    GT,
    LE,
    GE,
    EQ,
    NE,
    AND,
    OR,
};

struct SourceLoc {
    int line = 0;
    int column = 0;
};

struct Type {
    TypeKind kind = TypeKind::FLOAT;
    int array_size = 0;
};

struct BufferMember {
    Type type;
    std::string name;
};

struct Node {
    SourceLoc loc;
    virtual ~Node() = default;
    virtual void accept(Visitor &) = 0;
};

struct Decl : Node {};
struct Stmt : Node {};
struct Expr : Node {};

class Visitor {
public:
    virtual ~Visitor() = default;

    virtual void visit(Program&);
    virtual void visit(VarDecl&);
    virtual void visit(BufferDecl&);
    virtual void visit(FunctionDecl&);
    virtual void visit(BlockStmt&);
    virtual void visit(ForStmt&);
    virtual void visit(IfStmt&);
    virtual void visit(ReturnStmt&);
    virtual void visit(ExprStmt&);
    virtual void visit(IntLiteral&);
    virtual void visit(FloatLiteral&);
    virtual void visit(BoolLiteral&);
    virtual void visit(Variable&);
    virtual void visit(UnaryExpr&);
    virtual void visit(BinaryExpr&);
    virtual void visit(AssignExpr&);
    virtual void visit(CallExpr&);
    virtual void visit(MemberExpr&);
    virtual void visit(ArrayExpr&);
};

// ======================= Root of our AST tree =======================
class Program : public Decl {
public:
    std::vector<std::unique_ptr<Decl>> declarations;
    
    void accept(Visitor &v) override {
        v.visit(*this);
    }
};

class VarDecl : public Decl {
public:
    Type type;
    std::string name;
    std::unique_ptr<Expr> initializer;

    void accept(Visitor &v) override {
        v.visit(*this);
    }
};

class BufferDecl : public Decl {
public:
    std::string block_name;
    std::string instance_name;
    int binding = 0;
    std::vector<BufferMember> members;

    void accept(Visitor &v) override {
        v.visit(*this);
    }
};

class FunctionDecl : public Decl {
public:
    Type return_type;
    std::string name;
    std::vector<VarDecl> parameters;
    std::unique_ptr<BlockStmt> body;

    void accept(Visitor &v) override {
        v.visit(*this);
    }
};

class BlockStmt : public Stmt {
public:
    std::vector<std::unique_ptr<Stmt>> statements;

    void accept(Visitor &v) override {
        v.visit(*this);
    }
};

class ForStmt : public Stmt {
public:
    std::unique_ptr<Stmt> init;
    std::unique_ptr<Expr> condition;
    std::unique_ptr<Expr> increment;
    std::unique_ptr<Stmt> body;

    void accept(Visitor &v) override {
        v.visit(*this);
    }
};

class IfStmt : public Stmt {
public:
    std::unique_ptr<Expr> condition;
    std::unique_ptr<Stmt> then_branch;
    std::unique_ptr<Stmt> else_branch;

    void accept(Visitor &v) override {
        v.visit(*this);
    }
};

class ReturnStmt : public Stmt {
public:
    std::unique_ptr<Expr> value;

    void accept(Visitor& v) override { 
        v.visit(*this); 
    }
};

class ExprStmt : public Stmt {
public:
    std::unique_ptr<Expr> expression;

    void accept(Visitor& v) override { 
        v.visit(*this); 
    }
};

class IntLiteral : public Expr {
public:
    int value = 0;

    void accept(Visitor& v) override { 
        v.visit(*this); 
    }
};

class FloatLiteral : public Expr {
public:
    float value = 0.0f;

    void accept(Visitor& v) override { 
        v.visit(*this); 
    }
};

class BoolLiteral : public Expr {
public:
    bool value = false;

    void accept(Visitor& v) override { 
        v.visit(*this); 
    }
};

class Variable : public Expr {
public:
    std::string name;

    void accept(Visitor& v) override { 
        v.visit(*this); 
    }
};

class UnaryExpr : public Expr {
public:
    UnaryOp op;
    std::unique_ptr<Expr> operand;

    void accept(Visitor& v) override { 
        v.visit(*this); 
    }
};

class BinaryExpr : public Expr {
public:
    BinaryOp op;
    std::unique_ptr<Expr> left;
    std::unique_ptr<Expr> right;

    void accept(Visitor& v) override { 
        v.visit(*this); 
    }
};

class AssignExpr : public Expr {
public:
    std::unique_ptr<Expr> target;
    std::unique_ptr<Expr> value;

    void accept(Visitor& v) override { 
        v.visit(*this); 
    }
};

class CallExpr : public Expr {
public:
    std::string function;
    std::vector<std::unique_ptr<Expr>> arguments;

    void accept(Visitor& v) override { 
        v.visit(*this); 
    }
};

class MemberExpr : public Expr {
public:
    std::unique_ptr<Expr> object;
    std::string member;

    void accept(Visitor& v) override { 
        v.visit(*this); 
    }
};

class ArrayExpr : public Expr {
public:
    std::unique_ptr<Expr> array;
    std::unique_ptr<Expr> index;

    void accept(Visitor& v) override { 
        v.visit(*this); 
    }
};