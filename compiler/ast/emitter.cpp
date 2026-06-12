#include <string>

#include "emitter.h"

std::string Emitter::emit(Program &prog) {
    visit(prog);
    return out_.str();
}

void Emitter::indent(){
    out_ << std::string(indent_ * 4, ' ');
}

void Emitter::emit_type(const Type &t){
    switch (t.kind){
        case TypeKind::FLOAT: 
            out_ << "float"; 
            break;

        case TypeKind::INT:   
            out_ << "int";   
            break;

        case TypeKind::UINT:  
            out_ << "uint";  
            break;

        case TypeKind::BOOL:  
            out_ << "bool";  
            break;

        case TypeKind::DOUBLE:
            out_ << "double";
            break;

        case TypeKind::VOID:
            out_ << "void";
            break;

        case TypeKind::VEC2:  
            out_ << "vec2";  
            break;

        case TypeKind::VEC3:  
            out_ << "vec3";  
            break;

        case TypeKind::VEC4:  
            out_ << "vec4";  
            break;
    }

    if(t.array_size == -1)
        out_ << "[]";
    else if(t.array_size > 0)
        out_ << "[" << t.array_size << "]";
}

void Emitter::emit_op(BinaryOp op){
    switch (op){
        case BinaryOp::ADD: 
            out_ << "+";  
            break;
            
        case BinaryOp::SUB: 
            out_ << "-";  
            break;

        case BinaryOp::MUL: 
            out_ << "*";  
            break;

        case BinaryOp::DIV: 
            out_ << "/";  
            break;

        case BinaryOp::LT:  
            out_ << "<";  
            break;

        case BinaryOp::GT:  
            out_ << ">";  
            break;

        case BinaryOp::LE:  
            out_ << "<="; 
            break;

        case BinaryOp::GE:  
            out_ << ">="; 
            break;

        case BinaryOp::EQ:  
            out_ << "=="; 
            break;

        case BinaryOp::NE:  
            out_ << "!="; 
            break;

        case BinaryOp::AND: 
            out_ << "&&"; 
            break;

        case BinaryOp::OR:  
            out_ << "||"; 
            break;
    }
}


void Emitter::visit(Program &prog){
    for(auto &decl : prog.declarations)
        decl->accept(*this);
}

void Emitter::visit(VarDecl &vd){
    emit_type(vd.type);
    out_ << " " << vd.name;

    if(vd.initializer){
        out_ << " = ";
        vd.initializer->accept(*this);
    }

    out_ << ";\n";
}

void Emitter::visit(BufferDecl &bd){
    out_ << "layout(" << bd.layout << ") buffer " << bd.block_name << " {\n";
    indent_++;

    for(auto &member : bd.members){
        indent();
        emit_type(member.type);
        out_ << " " << member.name << ";\n";
    }

    indent_--;
    indent();
    out_ << "} " << bd.instance_name << ";\n";
}

void Emitter::visit(LayoutDecl &ld){
    out_ << "layout(" << ld.qualifiers << ") " << ld.storage << ";\n";
}

void Emitter::visit(FunctionDecl &fd){
    emit_type(fd.return_type);
    out_ << " " << fd.name << "(";

    for(size_t i = 0; i < fd.parameters.size(); i++){
        if(i > 0)
            out_ << ", ";

        emit_type(fd.parameters[i].type);
        out_ << " " << fd.parameters[i].name;
    }

    out_ << ")";

    if(fd.body){
        out_ << " ";
        fd.body->accept(*this);
    }
    else{
        out_ << ";\n";
    }
}

void Emitter::visit(BlockStmt &bs){
    out_ << "{\n";
    indent_++;

    for(auto &stmt : bs.statements){
        indent();
        stmt->accept(*this);
    }

    indent_--;
    indent();
    out_ << "}\n";
}

void Emitter::visit(ForStmt &fs){
    out_ << "for(";

    if(fs.init){
        if(auto *es = dynamic_cast<ExprStmt *>(fs.init.get())){
            if(es->expression)
                es->expression->accept(*this);
        }
        else if(auto *ds = dynamic_cast<DeclStmt *>(fs.init.get())){
            if(auto *vd = dynamic_cast<VarDecl *>(ds->declaration.get())){
                emit_type(vd->type);
                out_ << " " << vd->name;

                if(vd->initializer){
                    out_ << " = ";
                    vd->initializer->accept(*this);
                }
            }
        }
    }

    out_ << "; ";

    if(fs.condition)
        fs.condition->accept(*this);

    out_ << "; ";

    if(fs.increment)
        fs.increment->accept(*this);

    out_ << ") ";

    if(fs.body)
        fs.body->accept(*this);
}

void Emitter::visit(IfStmt &is){
    out_ << "if(";

    if(is.condition)
        is.condition->accept(*this);

    out_ << ") ";

    if(is.then_branch)
        is.then_branch->accept(*this);

    if(is.else_branch){
        out_ << " else ";
        is.else_branch->accept(*this);
    }
}

void Emitter::visit(ReturnStmt &rs){
    out_ << "return";

    if(rs.value){
        out_ << " ";
        rs.value->accept(*this);
    }

    out_ << ";\n";
}

void Emitter::visit(ExprStmt &es){
    if(es.expression)
        es.expression->accept(*this);

    out_ << ";\n";
}

void Emitter::visit(DeclStmt &ds){
    if(ds.declaration)
        ds.declaration->accept(*this);
}

void Emitter::visit(IntLiteral &il){
    out_ << il.value;
}

void Emitter::visit(FloatLiteral &fl){
    std::ostringstream ss;
    ss << fl.value;
    std::string s = ss.str();

    if(s.find('.') == std::string::npos)
        s += ".0";

    out_ << s;
}

void Emitter::visit(BoolLiteral &bl){
    out_ << (bl.value ? "true" : "false");
}

void Emitter::visit(Variable &v){
    out_ << v.name;
}

void Emitter::visit(UnaryExpr &ue){
    out_ << "(";

    switch (ue.op){
        case UnaryOp::NEGATE:
            out_ << "-";
            break;

        case UnaryOp::NOT:
            out_ << "!";
            break;
    }

    if(ue.operand)
        ue.operand->accept(*this);

    out_ << ")";
}

void Emitter::visit(BinaryExpr &be){
    out_ << "(";

    if(be.left)
        be.left->accept(*this);

    out_ << " ";
    emit_op(be.op);
    out_ << " ";

    if(be.right)
        be.right->accept(*this);

    out_ << ")";
}

void Emitter::visit(AssignExpr &ae){
    out_ << "(";

    if(ae.target)
        ae.target->accept(*this);

    out_ << " " << ae.op << " ";

    if(ae.value)
        ae.value->accept(*this);

    out_ << ")";
}

void Emitter::visit(CallExpr &ce){
    out_ << ce.function << "(";

    for(size_t i = 0; i < ce.arguments.size(); i++){
        if(i > 0)
            out_ << ", ";
        
        ce.arguments[i]->accept(*this);
    }

    out_ << ")";
}

void Emitter::visit(MemberExpr &me){
    if(me.object)
        me.object->accept(*this);

    out_ << "." << me.member;
}

void Emitter::visit(ArrayExpr &ae){
    if(ae.array)
        ae.array->accept(*this);

    out_ << "[";

    if(ae.index)
        ae.index->accept(*this);;

    out_ << "]";
}