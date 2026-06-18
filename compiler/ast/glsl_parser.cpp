/**
 * My notes (cuz i forget):
 * A recurisve descent parser reflect the grammer directly.
 * Each grammer rule becomes one method.
 * The lexer feeds tokens, then the parser matches patterns and builds AST nodes.
 */

#include <cstdlib>
#include <cstring>

#include "glsl_parser.h"
#include "emitter.h"

Parser::Parser(Lexer &lexer) : lexer_(lexer) {}

void Parser::set_error(const std::string &msg){
    if(error_.empty())
        error_ = msg;
}

Token Parser::peek() {
    return lexer_.peek();
}

Token Parser::consume() {
    return lexer_.consume();
}

bool Parser::match(TokenKind kind){
    if(!(peek().kind == kind))
        return false;

    consume();
    return true;
}

void Parser::expect(TokenKind kind, const std::string &msg){
    if(peek().kind == kind)
        consume();
    else 
        set_error(msg);
}

Type Parser::parse_type(){
    Type t;

    if (match(KW_VOID))    
        t.kind = TypeKind::VOID;
    else if (match(KW_FLOAT))  
        t.kind = TypeKind::FLOAT;
    else if (match(KW_INT))    
        t.kind = TypeKind::INT;
    else if (match(KW_UINT))   
        t.kind = TypeKind::UINT;
    else if (match(KW_BOOL))   
        t.kind = TypeKind::BOOL;
    else if (match(KW_DOUBLE)) 
        t.kind = TypeKind::DOUBLE;
    else if (match(KW_VEC2))   
        t.kind = TypeKind::VEC2;
    else if (match(KW_VEC3))   
        t.kind = TypeKind::VEC3;
    else if (match(KW_VEC4))   
        t.kind = TypeKind::VEC4;
    else 
        set_error("expected type");

    return t;
}

std::unique_ptr<Program> Parser::parse(){
    auto prog = std::make_unique<Program>();

    while(peek().kind != END_OF_FILE){
        if(peek().kind == HASH){
            consume();

        while (
            peek().kind != END_OF_FILE &&
            peek().kind != KW_VOID && peek().kind != KW_FLOAT && peek().kind != KW_INT &&
            peek().kind != KW_UINT && peek().kind != KW_BOOL && peek().kind != KW_DOUBLE && peek().kind != KW_LAYOUT &&
            peek().kind != KW_IN && peek().kind != KW_OUT && peek().kind != KW_CONST && peek().kind != KW_STRUCT
        ){
            consume();
        }

        continue;
        }

        auto decl = parse_declaration();

        if(!decl)
            break;

        prog->declarations.push_back(std::move(decl));
    }

    return prog;
}

std::unique_ptr<Decl> Parser::parse_declaration(){
    if(match(KW_LAYOUT)){
        expect(OPEN_PAREN, "expected '(' after layout");
        std::string quals;
        int depth = 1;

        while(depth > 0 && peek().kind != END_OF_FILE){
            Token t = consume();

            if(t.kind == OPEN_PAREN)
                depth++;

            if(t.kind == CLOSE_PAREN)
                depth--;

            if(depth > 0){
                if(!quals.empty())
                    quals += ' ';

                quals += std::string(t.text);
            }
        }

        if(match(KW_BUFFER))
            return parse_buffer_decl(quals);

        auto ld = std::make_unique<LayoutDecl>();
        ld->qualifiers = std::move(quals);

        if(match(KW_IN))
            ld->storage = "in";
        else if(match(KW_OUT))
            ld->storage = "out";
        else if(match(KW_BUFFER))
            ld->storage = "buffer";
        
        expect(SEMICOLON, "expected ';' after layout decl");

        return ld;
    }

    Type t = parse_type();

    if(!error_.empty())
        return nullptr;

    Token name = consume();

    if(peek().kind == OPEN_PAREN)
        return parse_function(t, std::string(name.text));

    return parse_var_decl(t, std::string(name.text));
}

std::unique_ptr<BufferDecl> Parser::parse_buffer_decl(const std::string &layout_str){
    auto bd = std::make_unique<BufferDecl>();
    bd->layout = layout_str;
    bd->block_name = std::string(consume().text);

    expect(OPEN_BRACE, "expected '{' in buffer decl");

    while (peek().kind != CLOSE_BRACE && peek().kind != END_OF_FILE){
        Type mt = parse_type();

        if(!error_.empty())
            return nullptr;
        
        BufferMember mem;
        mem.type = mt;
        mem.name = std::string(consume().text);

        if(match(OPEN_BRACKET)){
            if(match(CLOSE_BRACKET)){
                mem.type.array_size = -1;
            }
            else{
                while(peek().kind != CLOSE_BRACKET && peek().kind != END_OF_FILE)
                    consume();
                consume();
            }
        }

        expect(SEMICOLON, "expect ';' after buffer member");
        bd->members.push_back(std::move(mem));
    }

    expect(CLOSE_BRACE, "expected '}' in buffer decl");

    if(peek().kind == SEMICOLON){
        consume();
    }
    else{
        bd->instance_name = std::string(consume().text);
        expect(SEMICOLON, "expected ';' after buffer decl");
    }

    return bd;
}

std::unique_ptr<FunctionDecl> Parser::parse_function(Type ret_type, const std::string &name){
    auto fd = std::make_unique<FunctionDecl>();
    fd->return_type = ret_type;
    fd->name = name;

    expect(OPEN_PAREN, "expected '(' in function decl");

    if(!match(CLOSE_PAREN)){
        if(peek().kind == KW_VOID && peek().kind != END_OF_FILE){
            consume();
            expect(CLOSE_PAREN, "expected ')' after void param");
        }
        else{
            do {
                Type pt = parse_type();

                if(!error_.empty())
                    return nullptr;

                VarDecl param;
                param.type = pt;
                param.name = std::string(consume().text);

                if(match(OPEN_BRACKET)){
                    while(peek().kind != CLOSE_BRACKET && peek().kind != END_OF_FILE)
                        consume();
                    consume();
                }

                fd->parameters.push_back(std::move(param));
            } 
            while (match(COMMA));
            
            expect(CLOSE_PAREN, "expected ')' in function decl");
        }
    }

    if(peek().kind == OPEN_BRACE)
        fd->body = parse_block();
    else
        expect(SEMICOLON, "expect '{' or ';' after function decl");

    return fd;
}

std::unique_ptr<VarDecl> Parser::parse_var_decl(Type type, const std::string &name){
    auto vd = std::make_unique<VarDecl>();
    vd->type = type;
    vd->name = name;

    if(match(OPEN_BRACKET)){
        if(!match(CLOSE_BRACKET)){
            while(peek().kind != CLOSE_BRACKET && peek().kind != END_OF_FILE)
                consume();
            consume();
        }
    }

    if(match(EQUALS))
        vd->initializer = parse_expr();

    expect(SEMICOLON, "expected ';' after variable declaration");

    return vd;
}

std::unique_ptr<Stmt> Parser::parse_stmt(){
    switch (peek().kind){
        case OPEN_BRACE:
            return parse_block();
        case KW_FOR:
            return parse_for();
        case KW_IF:
            return parse_if();
        case KW_RETURN:
            return parse_return();
    }

    if (peek().kind == KW_VOID || peek().kind == KW_FLOAT || peek().kind == KW_INT ||
        peek().kind == KW_UINT || peek().kind == KW_BOOL || peek().kind == KW_DOUBLE ||
        peek().kind == KW_VEC2 || peek().kind == KW_VEC3 || peek().kind == KW_VEC4) {
    
            Type t = parse_type();

            if(!error_.empty())
                return nullptr;

            std::string vname = std::string(consume().text);
            auto vd = parse_var_decl(t, vname);
            auto ds = std::make_unique<DeclStmt>();
            ds->declaration = std::move(vd);
            
            return ds;
    }

    if(match(KW_BREAK) || match(KW_CONTINUE)){
        expect(SEMICOLON, "expected ';' after break/continue");
        return nullptr;
    }

    auto expr = parse_expr();
    expect(SEMICOLON, "expected ';'");
    auto es = std::make_unique<ExprStmt>();
    es->expression = std::move(expr);

    return es;
}

std::unique_ptr<BlockStmt> Parser::parse_block(){
    auto block = std::make_unique<BlockStmt>();
    expect(OPEN_BRACE, "expected '{'");

    while(peek().kind != CLOSE_BRACE && peek().kind != END_OF_FILE){
        auto s = parse_stmt();

        if(s)
            block->statements.push_back(std::move(s));

        if(!error_.empty())
            return nullptr;
    }

    expect(CLOSE_BRACE, "expected '}'");

    return block;
}

std::unique_ptr<Stmt> Parser::parse_for(){
    consume();
    expect(OPEN_PAREN, "expected '(' after for");

    std::unique_ptr<Stmt> init;

    if(peek().kind == SEMICOLON){
        consume();
    }
    else if(peek().kind == KW_VOID || peek().kind == KW_FLOAT || peek().kind == KW_INT || peek().kind == KW_UINT ||
            peek().kind == KW_BOOL || peek().kind == KW_DOUBLE || peek().kind == KW_VEC2 || peek().kind == KW_VEC3 ||
            peek().kind == KW_VEC4) {
        Type it = parse_type();
        
        if(!error_.empty())
            return nullptr;
        
        std::string iname = std::string(consume().text);
        auto init_var = std::make_unique<VarDecl>();
        init_var->type = it;
        init_var->name = iname;
        
        if(match(OPEN_BRACKET)){
            while(peek().kind != CLOSE_BRACKET && peek().kind != END_OF_FILE)
                consume();
            consume();
        }

        if(match(EQUALS))
            init_var->initializer = parse_expr();

        expect(SEMICOLON, "expected ';' after for-loop init decl");
        auto ds = std::make_unique<DeclStmt>();
        ds->declaration = std::move(init_var);
        init = std::move(ds);
    }
    else{
        auto e = parse_expr();
        expect(SEMICOLON, "expected ';' after for loop init expr");
        auto es = std::make_unique<ExprStmt>();
        es->expression = std::move(e);
        init = std::move(es);
    }

    std::unique_ptr<Expr> cond;

    if(peek().kind != SEMICOLON)
        cond = parse_expr();

    expect(SEMICOLON, "expected ';' after for-loop condition");
    
    std::unique_ptr<Expr> update;
    
    if(peek().kind != CLOSE_PAREN)
        update = parse_expr();

    expect(CLOSE_PAREN, "expected ')' after for-loop update");

    auto body = parse_stmt();
    auto fs = std::make_unique<ForStmt>();
    
    fs->init = std::move(init);
    fs->condition = std::move(cond);
    fs->increment = std::move(update);
    fs->body = std::move(body);

    return fs;
}

std::unique_ptr<Stmt> Parser::parse_if(){
    consume();
    expect(OPEN_PAREN, "expected '(' after if");
    auto cond = parse_expr();
    expect(CLOSE_PAREN, "expected ')' after if condition");
    auto then_branch = parse_stmt();
    std::unique_ptr<Stmt> else_branch;

    if(match(KW_ELSE))
        else_branch = parse_stmt();

    auto is = std::make_unique<IfStmt>();

    is->condition = std::move(cond);
    is->then_branch = std::move(then_branch);
    is->else_branch = std::move(else_branch);

    return is;
}

std::unique_ptr<Stmt> Parser::parse_return(){
    consume();
    std::unique_ptr<Expr> val;
    
    if(peek().kind != SEMICOLON)
        val = parse_expr();

    expect(SEMICOLON, "expected ';' after return");

    auto rs = std::make_unique<ReturnStmt>();
    rs->value = std::move(val);

    return rs;
}

std::unique_ptr<Expr> Parser::parse_expr(){
    return parse_assignment();
}

std::unique_ptr<Expr> Parser::parse_assignment(){
    auto left = parse_logical_or();

    if(match(EQUALS))       
        return make_assign("=",  std::move(left), parse_assignment());

    if(match(PLUS_EQUALS))  
        return make_assign("+=", std::move(left), parse_assignment());

    if(match(MINUS_EQUALS)) 
        return make_assign("-=", std::move(left), parse_assignment());

    if(match(STAR_EQUALS))  
        return make_assign("*=", std::move(left), parse_assignment());

    if(match(SLASH_EQUALS)) 
        return make_assign("/=", std::move(left), parse_assignment());

    return left;
}

std::unique_ptr<Expr> Parser::parse_logical_or(){
    auto left = parse_logical_and();

    while(match(OR_OR))
        left = make_binary(BinaryOp::OR, std::move(left), parse_logical_and());

    return left;
}

std::unique_ptr<Expr> Parser::parse_logical_and(){
    auto left = parse_inclusive_or();
    while(match(AND_AND))
        left = make_binary(BinaryOp::AND, std::move(left), parse_inclusive_or());
    return left;
}

std::unique_ptr<Expr> Parser::parse_inclusive_or(){
    auto left = parse_xor();

    while(match(PIPE))
        left = make_binary(BinaryOp::OR, std::move(left), parse_xor());

    return left;
}

std::unique_ptr<Expr> Parser::parse_xor(){
    auto left = parse_and();

    while(match(CARET))
        left = make_binary(BinaryOp::OR, std::move(left), parse_and());

    return left;
}

std::unique_ptr<Expr> Parser::parse_and(){
    auto left = parse_equality();

    while(match(AMPERSAND))
        left = make_binary(BinaryOp::AND, std::move(left), parse_equality());

    return left;
}

std::unique_ptr<Expr> Parser::parse_equality(){
    auto left = parse_relational();

    while(true){
        if(match(EQUALS_EQUALS))
            left = make_binary(BinaryOp::EQ, std::move(left), parse_relational());
        else if(match(NOT_EQUALS))
            left = make_binary(BinaryOp::NE, std::move(left), parse_relational());
        else 
            break;
    }

    return left;
}

std::unique_ptr<Expr> Parser::parse_relational(){
    auto left = parse_shift();

    while(true){
        if(match(LESS))
            left = make_binary(BinaryOp::LT, std::move(left), parse_shift());
        else if(match(GREATER))
            left = make_binary(BinaryOp::GT, std::move(left), parse_shift());
        else if(match(LESS_EQUALS))
            left = make_binary(BinaryOp::LE, std::move(left), parse_shift());
        else if(match(GREATER_EQUALS))
            left = make_binary(BinaryOp::GE, std::move(left), parse_shift());
        else
            break;
    }

    return left;
}

// << and >> r not supposed by our lexer yet
std::unique_ptr<Expr> Parser::parse_shift(){
    return parse_additive();
}

std::unique_ptr<Expr> Parser::parse_additive(){
    auto left = parse_multiplicative();

    while(true){
        if(match(PLUS))
            left = make_binary(BinaryOp::ADD, std::move(left), parse_multiplicative());
        else if(match(MINUS))
            left = make_binary(BinaryOp::SUB, std::move(left), parse_multiplicative());
        else
            break;
    }

    return left;
}

std::unique_ptr<Expr> Parser::parse_multiplicative(){
    auto left = parse_unary();

    while(true){
        if(match(STAR))
            left = make_binary(BinaryOp::MUL, std::move(left), parse_unary());
        else if(match(SLASH))
            left = make_binary(BinaryOp::DIV, std::move(left), parse_unary());
        else
            break;
    }

    return left;
}

std::unique_ptr<Expr> Parser::parse_unary(){
    if(match(MINUS))
        return make_unary(UnaryOp::NEGATE, parse_unary());
    
    if(match(BANG))
        return make_unary(UnaryOp::NOT, parse_unary());

    if(match(PLUS))
        return parse_unary();

    if(match(TILDE))
        return parse_unary();

    return parse_postfix();
}

std::unique_ptr<Expr> Parser::parse_postfix(){
    auto expr = parse_primary();

    while(true){
        if(match(OPEN_BRACKET)){
            auto index = parse_expr();
            expect(CLOSE_BRACKET, "expected ']");
            expr = make_array(std::move(expr), std::move(index));
        }
        else if(match(DOT)){
            expr = make_member(std::move(expr), std::string(consume().text));
        }
        else if(match(OPEN_PAREN)){
            std::vector<std::unique_ptr<Expr>> args;

            if(peek().kind != CLOSE_PAREN){
                args.push_back(parse_expr());

                while(match(COMMA))
                    args.push_back(parse_expr());
            }

            expect(CLOSE_PAREN, "expected ')");
            std::string fn;

            if(auto *v = dynamic_cast<Variable *>(expr.get()))
                fn = v->name;
            
            expr = make_call(fn, std::move(args));
        }
        else if(match(PLUS_PLUS)){
            auto one = make_int(1);
            auto rhs = make_binary(BinaryOp::ADD, clone_expr(*expr), std::move(one));
            expr = make_assign("=", std::move(expr), std::move(rhs));
        }
        else if(match(MINUS_MINUS)){
            auto one = make_int(1);
            auto rhs = make_binary(BinaryOp::SUB, clone_expr(*expr), std::move(one));
            expr = make_assign("=", std::move(expr), std::move(rhs));
        }
        else{
            break;
        }
    }

    return expr;
}

std::unique_ptr<Expr> Parser::parse_primary(){
    if(peek().kind == INT_LITERAL)
        return make_int(std::atoi(std::string(consume().text).c_str()));

    if(peek().kind == FLOAT_LITERAL)
        return make_float(static_cast<float>(std::atof(std::string(consume().text).c_str())));

    if(match(KW_TRUE))
        return make_bool(true);

    if(match(KW_FALSE))
        return make_bool(false);

    if(peek().kind == IDENTIFIER)
        return make_var(std::string(consume().text));

    if(peek().kind == KW_FLOAT || peek().kind == KW_INT || peek().kind == KW_UINT  || peek().kind == KW_BOOL || peek().kind == KW_DOUBLE || peek().kind == KW_VEC2 || peek().kind == KW_VEC3  || peek().kind == KW_VEC4)
        return make_var(std::string(consume().text));

    if(match(OPEN_PAREN)){
        auto inner = parse_expr();
        expect(CLOSE_PAREN, "expect ')'");
        return inner;
    }

    set_error("unexpected token in expression");
    
    return nullptr;
}

std::unique_ptr<Expr> Parser::make_int(int v){
    auto n = std::make_unique<IntLiteral>();
    n->value = v;
    return n;
}

std::unique_ptr<Expr> Parser::make_float(float v){
    auto n = std::make_unique<FloatLiteral>();
    n->value = v;
    return n;
}

std::unique_ptr<Expr> Parser::make_bool(bool v){
    auto n = std::make_unique<BoolLiteral>();
    n->value = v;
    return n;
}

std::unique_ptr<Expr> Parser::make_var(const std::string &n) {
    auto v = std::make_unique<Variable>();
    v->name = n;
    return v;
}

std::unique_ptr<Expr> Parser::make_unary(UnaryOp op, std::unique_ptr<Expr> o) {
    auto u = std::make_unique<UnaryExpr>();
    u->op = op;
    u->operand = std::move(o);
    return u;
}

std::unique_ptr<Expr> Parser::make_binary(BinaryOp op, std::unique_ptr<Expr> l, std::unique_ptr<Expr> r){
    auto b = std::make_unique<BinaryExpr>();
    b->op = op;
    b->left = std::move(l);
    b->right = std::move(r);
    return b;
}

std::unique_ptr<Expr> Parser::make_call(const std::string &fn, std::vector<std::unique_ptr<Expr>> args){
    auto c = std::make_unique<CallExpr>();
    c->function = fn;
    c->arguments = std::move(args);
    return c;
}

std::unique_ptr<Expr> Parser::make_member(std::unique_ptr<Expr> obj, const std::string &m){
    auto me = std::make_unique<MemberExpr>();
    me->object = std::move(obj);
    me->member = m;
    return me;
}

std::unique_ptr<Expr> Parser::make_array(std::unique_ptr<Expr> arr, std::unique_ptr<Expr> index){
    auto a = std::make_unique<ArrayExpr>();
    a->array = std::move(arr);
    a->index = std::move(index);
    return a;
}

std::unique_ptr<Expr> Parser::clone_expr(Expr &e){
    if(auto *v = dynamic_cast<Variable *>(&e))
        return make_var(v->name);
    
    if(auto *il = dynamic_cast<IntLiteral *>(&e))
        return make_int(il->value);

    if (auto *fl = dynamic_cast<FloatLiteral *>(&e))
        return make_float(fl->value);

    if (auto *bl = dynamic_cast<BoolLiteral *>(&e))
        return make_bool(bl->value);

    set_error("cannot clone expression");
    return nullptr;
}

extern "C" char *parse_and_emit_glsl(const char *source, const char **error_out){
    if(!source){
        if(error_out)
            *error_out = _strdup("null source");
        return nullptr;
    }

    Lexer lexer{std::string(source)};
    Parser parser{lexer};
    auto program = parser.parse();

    if(!program || !parser.error().empty()){
        if(error_out)
            *error_out = _strdup(parser.error().c_str());
        return nullptr;
    }

    Emitter emitter;
    std::string result = emitter.emit(*program);
    char *out = (char *)std::malloc(result.size() + 1);

    if(!out){
        std::perror("Memory allocation failed: parse_and_emit_glsl");
        return nullptr;
    }

    std::memcpy(out, result.c_str(), result.size() + 1);

    return out;
}

extern "C" void free_emitted_string(char *s){
    std::free(s);
}