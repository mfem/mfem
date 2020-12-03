// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#include <array>
#include <iostream>
#include <algorithm>

#define MFEM_DEBUG_COLOR 118

using std::string;
using std::ostream;

#include "xfl.Y.hpp"
#include "xfc.hpp"

// *****************************************************************************
namespace yy { extern std::array<bool, yyruletype::yynrules> rules; }
extern std::array<Node*,8> nodes;

namespace mfem
{

namespace internal
{

/** @cond */ // Doxygen warning: documented symbol was not declared or defined

// *****************************************************************************
/// AST Qfunction dump
// *****************************************************************************
void Qfunc::postfix_expr_pow_expr_d(Rule*) const { out << "kernels::Pow("; }
void Qfunc::postfix_expr_pow_expr_u(Rule*) const { out << ")"; }
void Qfunc::token_POW(Token*) const { out << ","; }
void Qfunc::token_INNER_OP(Token*) const { out << "kernels::Dot"; }
void Qfunc::token_DOT_OP(Token*) const { out << "*"; }
void Qfunc::token_GRAD_OP(Token*) const { out << "kernels::Grad"; }

// *****************************************************************************
/// AST code generation
// *****************************************************************************
void Code::entry_point_statements_d(Rule*) const
{
   out << "#include \"xfm.hpp\"" << std:: endl;
   out << "int main(int argc, char* argv[]){" << std:: endl;
   out << "\tint status = 0;" << std:: endl;
}
void Code::entry_point_statements_u(Rule*) const
{
   out << "\treturn status;\n}" << std:: endl;
}

// *****************************************************************************
void Code::decl_domain_assign_op_expr_d(Rule*) const { }
void Code::decl_domain_assign_op_expr_u(Rule*) const
{
   out << ";" << std::endl;
}

// *****************************************************************************
// ... = 0.0
// Should check if the variable is already known before launching a kernel
static bool KernelCheck(xfl &ufl, Node *n)
{
   assert(n->child);
   assert(n->child->next);
   assert(n->child->next->IsRule());
   assert(n->Number() == decl_id_list_assign_op_expr);
   Node *expr = n->child->next->next;
   assert(expr->Number() == expr_assign_expr);
   const bool only_real = ufl.OnlyToken(TOK::REAL, expr);
   if (only_real && ufl.HitRule(primary_expr_constant, expr)) { return true; }
   return false;
}

// *****************************************************************************
void Code::decl_id_list_assign_op_expr_d(Rule *n) const
{
   //dbg("decl_id_list_assign_op_expr_t");
   const bool kernel = KernelCheck(ufl, n);
   out << "\t";
   ufl.ctx.type = -1;
   if (ufl.HitToken(TOK::TEST_FUNCTION, n))
   {
      out << "xfl::TestFunction ";
      ufl.ctx.type = TOK::TEST_FUNCTION;
   }
   else if (ufl.HitToken(TOK::TRIAL_FUNCTION, n))
   {
      out << "xfl::TrialFunction ";
      ufl.ctx.type = TOK::TRIAL_FUNCTION;
   }
   else if (ufl.HitToken(TOK::FUNCTION, n))
   {
      out << "xfl::Function ";
      ufl.ctx.type = TOK::FUNCTION;
   }
   else if (ufl.HitToken(TOK::CONSTANT_API, n))
   {
      out << "xfl::Constant ";
      ufl.ctx.type = TOK::CONSTANT_API;
   }
   else if (!kernel) { out << "auto "; }
   else { /* nothing */ }
   if (ufl.HitToken(TOK::MESH, n)) { out << "&";}
   if (ufl.HitToken(TOK::DEVICE, n)) { out << "&&";}
}
void Code::decl_id_list_assign_op_expr_u(Rule*) const
{
   out << ";" << std:: endl;
}

// *****************************************************************************
void Code::decl_direct_declarator_d(Rule*) const
{
   dbg();
   dbg("LHS=false");
   yy::rules.at(lhs_lhs) = false;
   out << "\t";
}
void Code::decl_direct_declarator_u(Rule*) const
{
   out << ";" << std:: endl;
}

// *****************************************************************************
void Code::postfix_id_primary_id_d(Rule*) const { }
void Code::postfix_id_primary_id_u(Rule*) const { }

// *****************************************************************************
void Code::primary_id_identifier_d(Rule *n) const
{
   //dbg("\033[33mprimary_id_identifier_t");
   // LHS declaration
   if (yy::rules.at(lhs_lhs) &&
       yy::rules.at(decl_id_list_assign_op_expr) && ufl.ctx.type > 0)
   {
      dbg("\033[33mAdding variable: %s:%d",n->child->Name().c_str(), ufl.ctx.type);
      const std::string &name = n->child->Name();
      auto res = ufl.ctx.vars.emplace(name, xfl::var{name, ufl.ctx.type, xfl::NONE});
      if (res.second == false)
      {
         auto &m = res.first->second;
         dbg("Variable already present with type: %d", m.type);
         assert(false);
      }
      //for (const auto &p : ufl.ctx.vars) { DBG("%s:%d", p.first.c_str(), p.second.type); }
   }
}
void Code::primary_id_identifier_u(Rule*) const { }

// *****************************************************************************
void Code::assign_expr_postfix_expr_assign_op_assign_expr_d(Rule*) const { }
void Code::assign_expr_postfix_expr_assign_op_assign_expr_u(Rule*) const { }

// *****************************************************************************
void Code::primary_expr_identifier_d(Rule *n) const
{
   dbg();
   // If we are inside a argument list, skip it
   constexpr int in_arg_list = assign_expr_postfix_expr_assign_op_assign_expr;
   if (yy::rules.at(in_arg_list)) { return; }
   // RHS & VARDOM_DX => inputs
   if (!yy::rules.at(lhs_lhs) &&
       yy::rules.at(extra_status_rule_var_xt) &&
       !yy::rules.at(postfix_expr_grad_op_lp_additive_expr_rp))
   {
      const std::string &name = n->child->Name();
      if (ufl.ctx.vars.find(name) != ufl.ctx.vars.end())
      {
         auto &var = ufl.ctx.vars.at(name);
         var.mode |= xfl::INTERP;
         dbg("\033[31m%s:%d:%d", name.c_str(), var.type, var.mode);
      }
      //else { assert(false); }
   }
}
void Code::primary_expr_identifier_u(Rule*) const { }

// *****************************************************************************
void Code::assign_op_eq_d(Rule*n) const
{
   // If we are inside a argument list, skip it
   constexpr int in_arg_list = assign_expr_postfix_expr_assign_op_assign_expr;
   if (yy::rules.at(in_arg_list)) { n->dfs.down = false; return; }

   dbg("LHS=false");
   yy::rules.at(lhs_lhs) = false;
}
void Code::assign_op_eq_u(Rule*) const { }


// *****************************************************************************
void Code::primary_expr_api_d(Rule *n) const
{
   if (ufl.HitToken(TOK::CONSTANT_API, n)) { }
   out << "xfl::";
}
void Code::primary_expr_api_u(Rule*) const { }

// *****************************************************************************
void Code::postfix_expr_pow_expr_d(Rule*) const { out << "xfl::math::Pow("; }
void Code::postfix_expr_pow_expr_u(Rule*) const { out << ")"; }

// *****************************************************************************
void Code::id_list_postfix_ids_d(Rule *n) const
{
   if (!yy::rules.at(id_list_id_list_coma_postfix_ids)) { return; }
   assert(n);
   assert(n->next);
   assert(n->next->IsToken());
   assert(n->next->Number() == TOK::COMA);
   assert(n->root);
   assert(n->root->next);
   if (n->root->next->Number() != assign_op_eq) { return; }
   Node *extra = n->root->next;
   ufl.ctx.extra = extra;
}
void Code::id_list_postfix_ids_u(Rule*) const { }

// *****************************************************************************
void Code::if_statement_if_lp_expr_rp_expr_d(Rule*) const { out << "\t"; }
void Code::if_statement_if_lp_expr_rp_expr_u(Rule*) const
{
   out << ";" << std:: endl;
}

// *****************************************************************************
void Code::decl_iteration_statement_d(Rule*) const { out << "\t"; }
void Code::decl_iteration_statement_u(Rule*) const { out << ";" << std:: endl; }

// *****************************************************************************
void Code::decl_api_statement_d(Rule*) const
{
   out << "\tstatus |= ";
   dbg("LHS=false");
   yy::rules.at(lhs_lhs) = false;
}
void Code::decl_api_statement_u(Rule*) const
{
   out << ";" << std:: endl;
}

// *****************************************************************************
void Code::decl_function_d(Rule*) const
{
   out << "\tauto ";
}
void Code::decl_function_u(Rule*) const
{
   out << ";};" << std:: endl;
}

// *****************************************************************************
void Code::args_expr_list_assign_expr_d(Rule *n) const
{
   dbg();
   if (yy::rules.at(decl_function) &&
       !yy::rules.at(args_expr_list_args_expr_list_coma_assign_expr))
   {
      assert(n);
      assert(n->child);
      nodes[0] = n->child;
      n->dfs.down = false; // don't continue down
   }
}
void Code::args_expr_list_assign_expr_u(Rule*) const { }

// *****************************************************************************
void Code::args_expr_list_args_expr_list_coma_assign_expr_d(Rule *n) const
{
   dbg();
   if (yy::rules.at(decl_function)&&
       !yy::rules.at(args_expr_list_args_expr_list_coma_assign_expr))
   {
      assert(n);
      assert(n->child);
      nodes[0] = n->child;
      n->dfs.down = false; // don't continue down
   }
}
void Code::args_expr_list_args_expr_list_coma_assign_expr_u(Rule*) const { }

// *****************************************************************************
void Code::def_statement_nl_d(Rule*n) const
{
   // Only one param for the lambda yet
   out << "[&] (auto ";
   assert(nodes[0]);
   ufl.dfs(nodes[0],me);
   // We delayed the lhs => rhs (from COLON), do it now
   dbg("LHS=false");
   yy::rules.at(lhs_lhs) = false;
   out << ") {";
   n->dfs.down = false;
}
void Code::def_statement_nl_u(Rule*) const { }

// *****************************************************************************
void Code::statement_decl_nl_d(Rule *n) const
{
   dbg("LHS=true");
   yy::rules.at(lhs_lhs) = true;
}
void Code::statement_decl_nl_u(Rule*) const { }

// *****************************************************************************
void Code::def_empty_empty_d(Rule *n) const { def_statement_nl_d(n); }
void Code::def_empty_empty_u(Rule*) const { }

// *****************************************************************************
void Code::extra_status_rule_var_xt_d(Rule*n) const
{
   dbg();
   assert(n->next);
   assert(n->next->child);
   out << "[&]() {\n\t\tauto qf = [](/*";
}
void Code::extra_status_rule_var_xt_u(Rule*n) const
{
   dbg();
   out << "*/";
   // Add variables to the qfunc lambda
   int i = 0;
   for (auto &p : ufl.ctx.vars)
   {
      xfl::var &var = p.second;
      if (var.mode != xfl::NONE)
      {
         out << ((i++>0)?", ":"") << "auto &" << var.name;//, var.type, var.mode);
      }
   }

   out << ", const bool eval){";
   int types = 0;
   for (auto &p : ufl.ctx.vars)
   {
      xfl::var &var = p.second;
      if (var.mode != xfl::NONE)
      {
         types <<= 4;
         types |= var.mode;
      }
   }
   out << " const int types = 0x" << std::hex << types << std::dec << ";";

   out << " if (eval) {(void)(";

   // qf body with the dom_xt will be added next
}

// *****************************************************************************
void Code::extra_status_rule_dom_xt_d(Rule*) const { dbg(); }
void Code::extra_status_rule_dom_xt_u(Rule*n) const
{
   dbg();
   out << ");} return types; };\n\t\treturn (";
   ufl.dfs(n->child, me);
   out << ") * mfem::xfl::QForm<int,decltype(qf)";

   // Add signature of the qfunc lambda
   for (auto &p : ufl.ctx.vars)
   {
      xfl::var &var = p.second;
      if (var.mode != xfl::NONE)
      {
         out << ", xfl::";
         if (var.type == TOK::TEST_FUNCTION) { out << "TestFunction"; }
         if (var.type == TOK::TRIAL_FUNCTION) { out << "TrialFunction"; }
         if (var.type == TOK::FUNCTION) { out << "Function"; }
         if (var.type == TOK::CONSTANT_API) { out << "Constant"; }
         //if (var.mode == xfl::INTERP) { out << "_Interp"; }
         //if (var.mode == xfl::GRAD) { out << "_Grad"; }
         out << "_q &";
      }
   }

   out << ", const bool>(qf);\n\t}";
   out << "()"; // force lambda evaluation to object

   // Add QFunction lambda as argument to the to the xfl::Form
   if (ufl.ctx.qfunc)
   {
      mfem::internal::Qfunc dump(ufl, out);
      ufl.dfs(ufl.ctx.qfunc, dump);
      ufl.ctx.qfunc = nullptr;
   }

   // Flush variables of this dom_dx
   for (auto &p : ufl.ctx.vars)
   {
      xfl::var &var = p.second;
      var.mode = xfl::NONE;
      //dbg("\033[31m%s:%d:%d", var.name.c_str(), var.type, var.mode);
   }
}

// *****************************************************************************
void Code::postfix_expr_grad_op_lp_additive_expr_rp_d(Rule *n) const
{
   assert(n->child);
   assert(n->child->next);
   assert(n->child->next->n == TOK::LP);
   assert(n->child->next->next->next->n == TOK::RP);
   Node *id = ufl.GetToken(TOK::IDENTIFIER, n); assert(id);
   const std::string &name = id->Name();
   if (ufl.ctx.vars.find(name) != ufl.ctx.vars.end())
   {
      xfl::var &var = ufl.ctx.vars.at(name);
      var.mode |= xfl::GRAD;
      dbg("\033[31m%s:%d:%d",name.c_str(), var.type, var.mode);
   }
   //else { assert(false); }
}
void Code::postfix_expr_grad_op_lp_additive_expr_rp_u(Rule*) const { }

// *****************************************************************************
// Specialized Code backend tokens
void Code::token_NL(Token*) const { /* empty */ }

void Code::token_QUOTE(Token *tok) const
{
   std::replace(tok->name.begin(), tok->name.end(), '\'', '\"');
   out << tok->name;
}

void Code::token_DOM_DX(Token*) const { /*out << " xfl::Form()";*/ }

void Code::token_POW(Token*) const { out << ","; }

void Code::token_INNER_OP(Token*) const { out << "dot"; }

void Code::token_EQ(Token*) const { out << " = "; }

void Code::token_FOR(Token*) const { out << "for(auto &&"; }

void Code::token_IN(Token*) const { out << ": cpp::Range("; }

void Code::token_RANGE(Token*) const {  }

void Code::token_COLON(Token*) const
{
   if (yy::rules.at(decl_iteration_statement)) {  out << ")) "; return; }
   if (yy::rules.at(decl_function)) { out << " = "; return; }
   out << ":";
}

void Code::token_DEF(Token*) const
{
   dbg("LHS=true");
   yy::rules.at(lhs_lhs) = true;
}

void Code::token_LP(Token*) const
{
   if (yy::rules.at(decl_function) && yy::rules.at(lhs_lhs)) { return; }
   out << "(";
}

void Code::token_RP(Token*) const
{
   if (yy::rules.at(decl_function) && yy::rules.at(lhs_lhs)) { return; }
   out << ")";
}

void Code::token_RETURN(Token*) const { out << "return "; }

void Code::token_COMA(Token*) const
{
   dbg();
   const bool lhs = yy::rules.at(lhs_lhs);
   const bool decl_func = lhs and yy::rules.at(decl_function);
   if (decl_func) { out << ", auto "; return; }
   out << ", ";
}

void Code::token_AND_AND(Token*) const { out << " && "; }

void Code::token_OR_OR(Token*) const { out << " || "; }

void Code::token_CONSTANT_API(Token*) const { out << "Constant"; }

void Code::token_DOT_OP(Token*) const { out << "*"; }

void Code::token_GRAD_OP(Token*) const { out << "grad"; }

/** @endcond */

} // internal

} // mfem
