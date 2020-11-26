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

using std::string;
using std::ostream;

#include "xfl.Y.hpp"
#include "xfc.hpp"

// *****************************************************************************
namespace yy { extern std::array<bool, yyruletype::yynrules> rules; }
extern std::array<Node*,8> nodes;
using Node_p = Node*;

namespace mfem
{

namespace internal
{

// *****************************************************************************
bool Code::entry_point_statements_t(Node_p&) const
{
   out << "#include \"xfm.hpp\"" << std:: endl;
   out << "int main(int argc, char* argv[]){" << std:: endl;
   out << "\tint status = 0;" << std:: endl;
   return true;
}
bool Code::entry_point_statements_f(Node_p&) const
{
   out << "\treturn status;\n}" << std:: endl;
   return true;
}

// *****************************************************************************
bool Code::decl_domain_assign_op_expr_t(Node_p&) const { return true; }
bool Code::decl_domain_assign_op_expr_f(Node_p&) const
{
   out << ";" << std::endl;
   return true;
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
bool Code::decl_id_list_assign_op_expr_t(Node_p &n) const
{
   const bool kernel = KernelCheck(ufl, n);
   out << "\t";
   if (!kernel) { out << "auto "; }
   if (ufl.HitToken(TOK::MESH, n)) { out << "&";}
   if (ufl.HitToken(TOK::DEVICE, n)) { out << "&&";}
   return true;
}
bool Code::decl_id_list_assign_op_expr_f(Node_p&) const
{
   out << ";" << std:: endl;
   return true;
}

// *****************************************************************************
bool Code::decl_direct_declarator_t(Node_p&) const
{
   out << "\t";
   return true;
}
bool Code::decl_direct_declarator_f(Node_p&) const
{
   out << ";" << std:: endl;
   return true;
}

// *****************************************************************************
bool Code::postfix_id_primary_id_t(Node_p&) const { return true; }
bool Code::postfix_id_primary_id_f(Node_p&) const { return true; }

// *****************************************************************************
bool Code::assign_expr_postfix_expr_assign_op_assign_expr_t(Node_p&) const
{
   return true;
}
bool Code::assign_expr_postfix_expr_assign_op_assign_expr_f(Node_p&) const
{
   return true;
}

// *****************************************************************************
bool Code::primary_expr_identifier_t(Node_p&) const
{
   // If we are inside a argument list, skip it
   if (yy::rules.at(assign_expr_postfix_expr_assign_op_assign_expr)) { return false; }
   return true;
}
bool Code::primary_expr_identifier_f(Node_p&) const { return true; }

// *****************************************************************************
bool Code::assign_op_eq_t(Node_p&) const
{
   // If we are inside a argument list, skip it
   if (yy::rules.at(assign_expr_postfix_expr_assign_op_assign_expr)) { return false; }
   return true;
}
bool Code::assign_op_eq_f(Node_p&) const { return true; }


// *****************************************************************************
bool Code::primary_expr_api_t(Node_p &n) const
{
   if (ufl.HitToken(TOK::CONSTANT_API, n)) { return true; }
   out << "xfl::";
   return true;
}
bool Code::primary_expr_api_f(Node_p&) const { return true; }

// *****************************************************************************
bool Code::postfix_expr_pow_expr_t(Node_p&) const
{
   out << "xfl::math::Pow(";
   return true;
}
bool Code::postfix_expr_pow_expr_f(Node_p&) const
{
   out << ")";
   return true;
}

// *****************************************************************************
bool Code::id_list_postfix_ids_t(Node_p &n) const
{
   if (!yy::rules.at(id_list_id_list_coma_postfix_ids)) { return true; }
   assert(n);
   assert(n->next);
   assert(n->next->IsToken());
   assert(n->next->Number() == TOK::COMA);
   assert(n->root);
   assert(n->root->next);
   if (n->root->next->Number() != assign_op_eq) { return true; }
   Node *eq = n->root->next;
   n = eq;
   return true;
}
bool Code::id_list_postfix_ids_f(Node_p&) const { return true; }

// *****************************************************************************
bool Code::if_statement_if_lp_expr_rp_expr_t(Node_p&) const
{
   out << "\t";
   return true;
}
bool Code::if_statement_if_lp_expr_rp_expr_f(Node_p&) const
{
   out << ";" << std:: endl;
   return true;
}

// *****************************************************************************
bool Code::decl_iteration_statement_t(Node_p&) const
{
   out << "\t";
   return true;
}
bool Code::decl_iteration_statement_f(Node_p&) const
{
   out << ";" << std:: endl;
   return true;
}

// *****************************************************************************
bool Code::decl_api_statement_t(Node_p&) const
{
   out << "\tstatus |= ";
   return true;
}
bool Code::decl_api_statement_f(Node_p&) const
{
   out << ";" << std:: endl;
   return true;
}

// *****************************************************************************
bool Code::decl_function_t(Node_p&) const
{
   out << "\tauto ";
   return true;
}
bool Code::decl_function_f(Node_p&) const
{
   out << ";};" << std:: endl;
   return true;
}

// *****************************************************************************
bool Code::args_expr_list_assign_expr_t(Node_p &n) const
{
   if (yy::rules.at(decl_function) &&
       !yy::rules.at(args_expr_list_args_expr_list_coma_assign_expr))
   {
      assert(n);
      assert(n->child);
      nodes[0] = n->child;
      return false; // don't continue down
   }
   return true;
}
bool Code::args_expr_list_assign_expr_f(Node_p&) const
{
   return true;
}

// *****************************************************************************
bool Code::args_expr_list_args_expr_list_coma_assign_expr_t(Node_p &n) const
{
   if (yy::rules.at(decl_function)&&
       !yy::rules.at(args_expr_list_args_expr_list_coma_assign_expr))
   {
      assert(n);
      assert(n->child);
      nodes[0] = n->child;
      return false; // don't continue down
   }
   return true;
}
bool Code::args_expr_list_args_expr_list_coma_assign_expr_f(Node_p&) const
{
   return true;
}

// *****************************************************************************
bool Code::def_statement_nl_t(Node_p&) const
{
   // Only one param for the lambda yet
   out << "[&] (auto ";
   assert(nodes[0]);
   ufl.dfs(nodes[0],me);
   // We delayed the lhs => rhs (from COLON), do it now
   yy::rules.at(lhs_lhs) = false;
   out << ") {";
   return false;
}
bool Code::def_statement_nl_f(Node_p&) const { return true; }

// *****************************************************************************
bool Code::def_empty_empty_t(Node_p &n) const
{
   return def_statement_nl_t(n);
}
bool Code::def_empty_empty_f(Node_p&) const { return true; }

// *****************************************************************************
namespace str
{
static const char *replace(const char *str, const char a, const char b)
{
   for (char c, *p = const_cast<char*>(str); (c=*p)!=0; *p++ = c==a ? b : c);
   return str;
}
} // namespace str

// *****************************************************************************
// Specialized Code backend tokens
void Code::token_NL(string) const { /* empty */ }

void Code::token_QUOTE(string text) const
{
   str::replace(text.c_str(), '\'','\"');
   out << text;
}

void Code::token_DOM_DX(string) const { out << " xfl::Form()"; }

void Code::token_POW(string) const { out << ","; }

void Code::token_INNER_OP(string) const { out << "dot"; }

void Code::token_EQ(string) const { out << " = "; }

void Code::token_FOR(string) const { out << "for(auto &&"; }

void Code::token_IN(string) const { out << ": cpp::Range("; }

void Code::token_RANGE(string) const {  }

void Code::token_COLON(string) const
{
   if (yy::rules.at(decl_iteration_statement)) {  out << ")) "; return; }
   if (yy::rules.at(decl_function))
   {
      out << " = ";
      return;
   }
   out << ":";
}

void Code::token_DEF(string) const
{
   yy::rules.at(lhs_lhs) = true;
}

void Code::token_LP(string) const
{
   if (yy::rules.at(decl_function) && yy::rules.at(lhs_lhs)) { return; }
   out << "(";
}

void Code::token_RP(string) const
{
   if (yy::rules.at(decl_function) && yy::rules.at(lhs_lhs)) { return; }
   out << ")";
}

void Code::token_RETURN(string) const { out << "return "; }

void Code::token_COMA(string) const
{
   if (yy::rules.at(decl_function) && yy::rules.at(lhs_lhs))
   {
      out << ", auto "; return;
   }
   out << ", ";
}

void Code::token_AND_AND(string) const { out << " && "; }

void Code::token_OR_OR(string) const { out << " || "; }

void Code::token_CONSTANT_API(std::string) const
{
   out << " /** new*/ xfl::Constant";
}

} // internal

} // mfem
