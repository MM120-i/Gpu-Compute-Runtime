#include <optional>
#include <cmath>

#include "constant_propagation.h"

/**
 * Entry point.
 * We only fold inside function bodies. Top level declarations
 */
void ConstantPropagation::fold(Program &program){
    fold_decls(program.declarations);
}

/**
 * If a declaration is a function declaration, then fold its body.
 */
void ConstantPropagation::fold_decls(std::vector<std::unique_ptr<Decl>> &decls){
    for(auto &decl : decls)
        if(auto *fd = dynamic_cast<FunctionDecl *>(decl.get()))
            if(fd->body)
                fold_block(*fd->body);
}

/**
 * folds statements depending on the type of statement.
 * 
 * ExprStmt:    fold the expression
 * ReturnStmt:  fold the return value
 * DeclStmt:    if it wraps a VarDecl, then fold its initializer
 * ForStmt:     fold condition, inc, init, and body
 * IfStmt:      fold condition and both branches
 * Blockstmt:   recurse into the block
 */
void ConstantPropagation::fold_stmt(std::unique_ptr<Stmt> &stmt){
    if(!stmt)
        return;

    if(auto *es = dynamic_cast<ExprStmt *>(stmt.get())){
        fold_expr(es->expression);
    }
    else if(auto *rs = dynamic_cast<ReturnStmt *>(stmt.get())){
        fold_expr(rs->value);
    }
    else if(auto *ds = dynamic_cast<DeclStmt *>(stmt.get())){
        if(auto *vd = dynamic_cast<VarDecl *>(ds->declaration.get()))
            fold_expr(vd->initializer);
    }
    else if(auto *fs = dynamic_cast<ForStmt *>(stmt.get())){
        fold_stmt(fs->init);
        fold_expr(fs->condition);
        fold_expr(fs->increment);
        fold_stmt(fs->body);
    }
    else if(auto *is = dynamic_cast<IfStmt *>(stmt.get())){
        fold_expr(is->condition);
        fold_stmt(is->then_branch);
        fold_stmt(is->else_branch);
    }
    else if(auto *bs = dynamic_cast<BlockStmt *>(stmt.get())){
        fold_block(*bs);
    }
}  

/**
 * fold every statement in a block
 */
void ConstantPropagation::fold_block(BlockStmt &block){
    for(auto &stmt : block.statements)
        fold_stmt(stmt);
}

/**
 * Every expression node follows the same 3 step pattern:
 * 1) Recurse into child (fold them first)
 * 2) Check if this node can be simplified
 * 3) If yes, replace `expr` with a simpler node
 * 
 * We take a unique_ptr<Expr> & so we can do std::move child out 
 * and replace the parent. 
 * Ex:  x + 0 = x
 * fold_expr(be->left)  -> nothing (just a variable)
 * fold_expr(be->right) -> nothing (just a integer literal 0)
 * fold_binary() detects ADD + is_zero(right)
 * expr = std::move(be->left)   -> parent now points to x (cuz the result is just x from x + 0)
 * The old BinaryExpr is then destroyed
 */
void ConstantPropagation::fold_expr(std::unique_ptr<Expr> &expr){
    if(!expr)
        return;

    // recurse based on current node type
    if(auto *be = dynamic_cast<BinaryExpr *>(expr.get())){
        fold_expr(be->left);
        fold_expr(be->right);
        fold_binary(expr, be);
    }
    else if(auto *ue = dynamic_cast<UnaryExpr *>(expr.get())){
        fold_expr(ue->operand);
        fold_unary(expr, ue);
    }
    else if(auto *ae = dynamic_cast<AssignExpr *>(expr.get())){
        fold_expr(ae->value);   // we only getting the rvalue
    }
    else if(auto *ce = dynamic_cast<CallExpr *>(expr.get())){
        for(auto &arg : ce->arguments)
            fold_expr(arg);
    }
    else if(auto *mex = dynamic_cast<MemberExpr *>(expr.get())){
        fold_expr(mex->object);
    }
    else if(auto *aex = dynamic_cast<ArrayExpr *>(expr.get())){
        fold_expr(aex->array);
        fold_expr(aex->index);
    }
}

/**
 * evaulate or simplify bin expressions
 * 
 * The order we look at is important, we check if both sizes r literals first 
 * Strongest simplification then identify patterns
 */
void ConstantPropagation::fold_binary(std::unique_ptr<Expr> &expr, BinaryExpr *be){
    if(auto *l = dynamic_cast<IntLiteral *>(be->left.get())){
        if(auto *r = dynamic_cast<IntLiteral *>(be->right.get())){
            if(auto v = eval_int(l->value, r->value, be->op)){
                expr = std::make_unique<IntLiteral>();
                static_cast<IntLiteral *>(expr.get())->value = *v;
                return;
            }
        }
    }

    if (auto *l = dynamic_cast<FloatLiteral *>(be->left.get())) {
        if (auto *r = dynamic_cast<FloatLiteral *>(be->right.get())) {
            if (auto v = eval_float(l->value, r->value, be->op)) {
                expr = std::make_unique<FloatLiteral>();
                static_cast<FloatLiteral *>(expr.get())->value = *v;
                return;
            }
        }
    }

    if (auto *l = dynamic_cast<IntLiteral *>(be->left.get())) {
        if (auto *r = dynamic_cast<FloatLiteral *>(be->right.get())) {
            if (auto v = eval_float(static_cast<float>(l->value), r->value, be->op)) {
                expr = std::make_unique<FloatLiteral>();
                static_cast<FloatLiteral *>(expr.get())->value = *v;
                return;
            }
        }
    }

    if (auto *l = dynamic_cast<FloatLiteral *>(be->left.get())) {
        if (auto *r = dynamic_cast<IntLiteral *>(be->right.get())) {
            if (auto v = eval_float(l->value, static_cast<float>(r->value), be->op)) {
                expr = std::make_unique<FloatLiteral>();
                static_cast<FloatLiteral *>(expr.get())->value = *v;
                return;
            }
        }
    }

    if (auto *l = dynamic_cast<BoolLiteral *>(be->left.get())) {
        if (auto *r = dynamic_cast<BoolLiteral *>(be->right.get())) {
            if (auto v = eval_bool(l->value, r->value, be->op)) {
                expr = std::make_unique<BoolLiteral>();
                static_cast<BoolLiteral *>(expr.get())->value = *v;
                return;
            }
        }
    }

    switch (be->op) {
        case BinaryOp::ADD:
            if(is_zero(*be->right)){
                expr = std::move(be->left);
                return;
            }

            if(is_zero(*be->left)){
                expr = std::move(be->right);
                return;
            }

            break;

        case BinaryOp::SUB:
            if(is_zero(*be->right)){
                expr = std::move(be->left);
                return;
            }

            if(be->left.get() == be->right.get()){
                auto zero = std::make_unique<IntLiteral>();
                zero->value = 0;
                expr = std::move(zero);
                return;
            }

            break;

        case BinaryOp::MUL:
            if(is_zero(*be->right) || is_zero(*be->right)){
                auto zero = std::make_unique<IntLiteral>();
                zero->value = 0;
                expr = std::move(zero);
                return;
            }

            if(is_one(*be->right)){
                expr = std::move(be->left);
                return;
            }

            if(is_one(*be->left)){
                expr = std::move(be->right);
                return;
            }

            break;

        case BinaryOp::DIV:
            if(is_one(*be->right)){
                expr = std::move(be->left);
                return;
            }

            if(be->left.get() == be->right.get()){
                auto one = std::make_unique<IntLiteral>();
                one->value = 1;
                expr = std::move(one);
                return;
            }
            
            break;
    }
}

/**
 * Fold double negation and literal negation
 */
void ConstantPropagation::fold_unary(std::unique_ptr<Expr> &expr, UnaryExpr *ue){
    switch (ue->op){
        case UnaryOp::NEGATE:
            if(auto *inner = dynamic_cast<UnaryExpr *>(ue->operand.get())){
                if(inner->op == UnaryOp::NEGATE){
                    expr = std::move(inner->operand);
                    return;
                }
            }

            if(auto *il = dynamic_cast<IntLiteral *>(ue->operand.get())){
                il->value = -il->value;
                expr = std::move(ue->operand);
                return;
            }

            if(auto *fl = dynamic_cast<FloatLiteral *>(ue->operand.get())){
                fl->value = -fl->value;
                expr = std::move(ue->operand);
                return;
            }

            break;

        case UnaryOp::NOT:
            if(ue->op == UnaryOp::NOT){
                if(auto *inner = dynamic_cast<UnaryExpr *>(ue->operand.get())){
                    if(inner->op == UnaryOp::NOT){
                        expr = std::move(inner->operand);
                        return;
                    }
                }
            }

            if(auto *bl = dynamic_cast<BoolLiteral *>(ue->operand.get())){
                bl->value = !bl->value;
                expr = std::move(ue->operand);
                return;
            }

            break;
    }
}

static std::optional<bool> eval_bool(int l, int r, BinaryOp op){
    switch (op){
        case BinaryOp::AND:
            return l && r;
        
        case BinaryOp::OR:
            return l || r;

        case BinaryOp::EQ:
            return l == r;

        case BinaryOp::NE:
            return l != r;
        
        default:
            return std::nullopt;
    }
}

static std::optional<int> eval_int(int l, int r, BinaryOp op){
    switch (op){
        case BinaryOp::ADD:
            return l + r;

        case BinaryOp::SUB:
            return l - r;

        case BinaryOp::MUL:
            return l * r;

        case BinaryOp::DIV:
            if(r == 0)
                return std::nullopt;
            return l / r;

        case BinaryOp::LT:  
            return l < r  ? 1 : 0;

        case BinaryOp::GT:  
            return l > r  ? 1 : 0;

        case BinaryOp::LE:  
            return l <= r ? 1 : 0;

        case BinaryOp::GE:  
            return l >= r ? 1 : 0;

        case BinaryOp::EQ:  
            return l == r ? 1 : 0;

        case BinaryOp::NE:  
            return l != r ? 1 : 0;
        
        default:
            return std::nullopt;
    }
}

static std::optional<float> eval_float(float l, float r, BinaryOp op){
    switch (op) {
        case BinaryOp::ADD: 
            return l + r;

        case BinaryOp::SUB: 
            return l - r;
            
        case BinaryOp::MUL: 
            return l * r;
            
        case BinaryOp::DIV:
            if (std::abs(r) < 1e-30f) 
                return std::nullopt;
            return l / r;

        case BinaryOp::LT:  
            return l < r  ? 1.0f : 0.0f;

        case BinaryOp::GT:  
            return l > r  ? 1.0f : 0.0f;

        case BinaryOp::LE:  
            return l <= r ? 1.0f : 0.0f;

        case BinaryOp::GE:  
            return l >= r ? 1.0f : 0.0f;

        case BinaryOp::EQ:  
            return std::abs(l - r) < 1e-30f ? 1.0f : 0.0f;

        case BinaryOp::NE:  
            return std::abs(l - r) >= 1e-30f ? 1.0f : 0.0f;

        default: 
            return std::nullopt;
    }
}

static bool is_zero(Expr &e){
    if(auto *il = dynamic_cast<IntLiteral *>(&e))
        return il->value == 0;

    if(auto *fl = dynamic_cast<FloatLiteral *>(&e))
        return std::abs(fl->value) < 1e-30f;

    return false;
}

static bool is_one(Expr &e){
    if(auto *il = dynamic_cast<IntLiteral *>(&e))
        return il->value == 1;
    
    if(auto *fl = dynamic_cast<FloatLiteral *>(&e))
        return std::abs(fl->value - 1.0f) < 1e-30f;

    return false;
}