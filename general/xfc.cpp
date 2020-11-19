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

#include "xfl.Y.hpp"
#include "xfc.hpp"

// *****************************************************************************
namespace yy { extern std::array<bool, yyruletype::yynrules> rules; }
extern std::array<Node*,8> nodes;

namespace mfem
{

namespace internal
{

using std::string;
using std::ostream;
using IR = struct Middlend;
using OS = std::ostream;

// *****************************************************************************
template<> bool entry_point_statements_cpu<true>(IR&, OS &out, Node**)
{
   out << "#include \"xfm.hpp\"" << std:: endl;
   out << "int main(int argc, char* argv[]){" << std:: endl;
   out << "\tint status = 0;" << std:: endl;
   return true;
}

template<> bool entry_point_statements_cpu<false>(IR&, OS &out, Node**)
{
   out << "\treturn status;\n}" << std:: endl;
   return true;
}

// *****************************************************************************
template<> bool decl_domain_assign_op_expr_cpu<true>(IR&, OS&, Node**)
{
   //out << "\tDBG(\"" << __FUNCTION__ << "\");" << std::endl;
   return true;
}
template<> bool decl_domain_assign_op_expr_cpu<false>(IR&, OS &out, Node**)
{
   out << ";" << std::endl;
   return true;
}

// *****************************************************************************
// ... = 0.0
// Should check if the variable is already known before launching a kernel
static bool KernelCheck(Node *n)
{
   assert(n->child);
   assert(n->child->next);
   assert(n->child->next->IsRule());
   assert(n->Number() == decl_id_list_assign_op_expr);
   Node *expr = n->child->next->next;
   assert(expr->Number() == expr_assign_expr);
   const bool only_real = OnlyToken(TOK::REAL, expr);
   if (only_real && HitRule(primary_expr_constant, expr)) { return true; }
   return false;
}

// *****************************************************************************
template<> bool decl_id_list_assign_op_expr_cpu<true>(IR&, OS &out, Node **n)
{
   const bool kernel = KernelCheck(*n);
   out << "\t";
   if (!kernel) { out << "auto "; }
   if (HitToken(TOK::MESH, *n)) { out << "&";}
   if (HitToken(TOK::DEVICE, *n)) { out << "&&";}
   return true;
}
template<> bool decl_id_list_assign_op_expr_cpu<false>(IR& ir, OS &out, Node**)
{
   out << ";" << std:: endl;
   return true;
}

// *****************************************************************************
template<> bool decl_direct_declarator_cpu<true>(IR& ir, OS &out, Node**)
{
   out << "\t";
   return true;
}
template<> bool decl_direct_declarator_cpu<false>(IR& ir, OS &out, Node**)
{
   out << ";" << std:: endl;
   return true;
}

// *****************************************************************************
template<> bool postfix_id_primary_id_cpu<true>(IR& ir, OS&, Node**) { return true; }
template<> bool postfix_id_primary_id_cpu<false>(IR& ir, OS&, Node**) { return true; }

// *****************************************************************************
template<>
bool assign_expr_postfix_expr_assign_op_assign_expr_cpu<true>
(IR& ir, OS &out, Node**)
{
   //assert(!rule.at(args_expr_list_args_expr_list_coma_assign_expr));
   return true;
}
template<>
bool assign_expr_postfix_expr_assign_op_assign_expr_cpu<false>(IR& ir,
                                                               OS &out,
                                                               Node**)
{
   return true;
}

// *****************************************************************************
template<> bool primary_expr_identifier_cpu<true>(IR& ir, OS &out, Node**)
{
   // If we are inside a argument list, skip it
   if (yy::rules.at(assign_expr_postfix_expr_assign_op_assign_expr)) { return false; }
   return true;
}
template<> bool primary_expr_identifier_cpu<false>(IR& ir, OS &out,
                                                   Node**) { return true; }

// *****************************************************************************
template<> bool assign_op_eq_cpu<true>(IR& ir, OS &out, Node**)
{
   // If we are inside a argument list, skip it
   if (yy::rules.at(assign_expr_postfix_expr_assign_op_assign_expr)) { return false; }
   return true;
}
template<> bool assign_op_eq_cpu<false>(IR& ir, OS &out, Node**) { return true; }


// *****************************************************************************
template<> bool primary_expr_api_cpu<true>(IR& ir, OS &out, Node **n)
{
   if (HitToken(TOK::CONSTANT_API, *n)) { return true; }
   out << "xfl::";
   return true;
}
template<> bool primary_expr_api_cpu<false>(IR& ir, OS &out, Node**) { return true; }

// *****************************************************************************
template<> bool postfix_expr_pow_expr_cpu<true>(IR& ir, OS &out, Node**)
{
   out << "xfl::math::Pow(";
   return true;
}
template<> bool postfix_expr_pow_expr_cpu<false>(IR& ir, OS &out, Node**)
{
   out << ")";
   return true;
}

// *****************************************************************************
template<> bool id_list_postfix_ids_cpu<true>(IR& ir, OS &, Node **n_addr)
{
   if (!yy::rules.at(id_list_id_list_coma_postfix_ids)) { return true; }
   Node *n = *n_addr;
   assert(n);
   assert(n->next);
   assert(n->next->IsToken());
   assert(n->next->Number() == TOK::COMA);
   assert(n->root);
   assert(n->root->next);
   if (n->root->next->Number() != assign_op_eq) { return true; }
   Node *eq = n->root->next;
   *n_addr = eq;
   return true;
}
template<> bool id_list_postfix_ids_cpu<false>(IR& ir, OS&, Node**) { return true; }

// *****************************************************************************
template<> bool if_statement_if_lp_expr_rp_expr_cpu<true>(IR& ir, OS &out,
                                                          Node **)
{
   out << "\t";
   return true;
}
template<> bool if_statement_if_lp_expr_rp_expr_cpu<false>(IR& ir, OS &out,
                                                           Node**)
{
   out << ";" << std:: endl;
   return true;
}

// *****************************************************************************
template<> bool decl_iteration_statement_cpu<true>(IR& ir, OS &out,
                                                   Node **)
{
   out << "\t";
   return true;
}
template<> bool decl_iteration_statement_cpu<false>(IR& ir, OS &out,
                                                    Node**)
{
   out << ";" << std:: endl;
   return true;
}

// *****************************************************************************
template<> bool decl_api_statement_cpu<true>(IR& ir, OS &out, Node **)
{
   out << "\tstatus |= ";
   return true;
}
template<> bool decl_api_statement_cpu<false>(IR& ir, OS &out, Node**)
{
   out << ";" << std:: endl;
   return true;
}

// *****************************************************************************
// Specialized XPU backend rules
#define RULE(name) \
template<> \
void XPU::rule<name>(bool &dfs, Node **extra) const {\
   if (dfs) { dfs = name##_cpu <true>(ir, out, extra); return; }\
   dfs = name##_cpu <false>(ir, out, extra);}

RULE(entry_point_statements)
RULE(decl_domain_assign_op_expr)
RULE(decl_id_list_assign_op_expr)
RULE(decl_direct_declarator)
RULE(postfix_id_primary_id)
RULE(assign_expr_postfix_expr_assign_op_assign_expr)
RULE(primary_expr_identifier)
RULE(assign_op_eq)
RULE(primary_expr_api)
RULE(postfix_expr_pow_expr)
RULE(id_list_postfix_ids)
RULE(if_statement_if_lp_expr_rp_expr)
RULE(decl_iteration_statement)
RULE(decl_api_statement)

// *****************************************************************************
template<> bool decl_function_cpu<true>(IR& ir, OS &out, Node **)
{
   out << "\tauto ";
   return true;
}
template<> bool decl_function_cpu<false>(IR& ir, OS &out, Node**)
{
   out << ";};" << std:: endl;
   return true;
}
RULE(decl_function)

// *****************************************************************************
template<> bool args_expr_list_assign_expr_cpu<true>(IR&, OS &out,
                                                     Node **n_addr)
{
   if (yy::rules.at(decl_function) &&
       !yy::rules.at(args_expr_list_args_expr_list_coma_assign_expr))
   {
      Node *n = *n_addr;
      assert(n);
      assert(n->child);
      nodes[0] = n->child;
      return false; // don't continue down
   }
   return true;
}
template<> bool args_expr_list_assign_expr_cpu<false>(IR& ir, OS &out,
                                                      Node**)
{
   return true;
}
RULE(args_expr_list_assign_expr)

// *****************************************************************************
template<>
bool args_expr_list_args_expr_list_coma_assign_expr_cpu<true>
(IR&, OS&out, Node **n_addr)
{
   if (yy::rules.at(decl_function)&&
       !yy::rules.at(args_expr_list_args_expr_list_coma_assign_expr))
   {
      Node *n = *n_addr;
      assert(n);
      assert(n->child);
      nodes[0] = n->child;
      return false; // don't continue down
   }
   return true;
}
template<>
bool args_expr_list_args_expr_list_coma_assign_expr_cpu<false>
(IR&, OS &out, Node **) { return true;}
RULE(args_expr_list_args_expr_list_coma_assign_expr)

// *****************************************************************************
template<> bool def_statement_nl_cpu<true>(IR& ir, OS &out, Node **na)
{
   // Only one param for the lambda yet
   out << "[&] (auto ";
   assert(nodes[0]);
   dfs(nodes[0], ir);
   // We delayed the lhs => rhs (from COLON), do it now
   yy::rules.at(lhs_lhs) = false;
   out << ") {";
   return false;
}
template<> bool def_statement_nl_cpu<false>(IR&, OS&, Node**) { return true; }
RULE(def_statement_nl)

// *****************************************************************************
template<> bool def_empty_empty_cpu<true>(IR& ir, OS &out, Node **na)
{ return def_statement_nl_cpu<true>(ir,out,na); }
template<> bool def_empty_empty_cpu<false>(IR&, OS&, Node**) { return true; }
RULE(def_empty_empty)


// *****************************************************************************
template<class ForwardIt, class T>
void txt_replace(ForwardIt first, ForwardIt last,
                 const T& old_value, const T& new_value)
{
   for (; first != last; ++first)
   {
      if (*first == old_value) { *first = new_value; }
   }
}

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
// Specialized XPU backend tokens
template<> void XPU::token<TOK::NL>(string) const { /* empty */ }

template<> void XPU::token<TOK::QUOTE>(string text) const
{
   str::replace(text.c_str(), '\'','\"');
   out << text;
}

template<> void XPU::token<TOK::DOM_DX>(string) const { out << " xfl::Form()"; }

template<> void XPU::token<TOK::POW>(string) const { out << ","; }

template<> void XPU::token<TOK::INNER_OP>(string) const { out << "dot"; }

template<> void XPU::token<TOK::EQ>(string) const { out << " = "; }

template<> void XPU::token<TOK::FOR>(string) const { out << "for(auto &&"; }

template<> void XPU::token<TOK::IN>(string) const { out << ": cpp::Range("; }

template<> void XPU::token<TOK::RANGE>(string) const {  }

template<> void XPU::token<TOK::COLON>(string) const
{
   if (yy::rules.at(decl_iteration_statement)) {  out << ")) "; return; }
   if (yy::rules.at(decl_function))
   {
      out << " = ";
      return;
   }
   out << ":";
}

template<> void XPU::token<TOK::DEF>(string) const
{
   yy::rules.at(lhs_lhs) = true;
}

template<> void XPU::token<TOK::LP>(string) const
{
   if (yy::rules.at(decl_function) && yy::rules.at(lhs_lhs)) { return; }
   out << "(";
}

template<> void XPU::token<TOK::RP>(string) const
{
   if (yy::rules.at(decl_function) && yy::rules.at(lhs_lhs)) { return; }
   out << ")";
}

template<> void XPU::token<TOK::RETURN>(string) const { out << "return "; }

template<> void XPU::token<TOK::COMA>(string) const
{
   if (yy::rules.at(decl_function) && yy::rules.at(lhs_lhs))
   {
      out << ", auto "; return;
   }
   out << ", ";
}

template<> void XPU::token<TOK::AND_AND>(string) const { out << " && "; }
template<> void XPU::token<TOK::OR_OR>(string) const { out << " || "; }

template<> void XPU::token<TOK::CONSTANT_API>(std::string) const
{
   out << " /** new*/ xfl::Constant";
}

} // internal

} // mfem
