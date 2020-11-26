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
#ifndef XFL_XFC_HPP
#define XFL_XFC_HPP

#include "xfl.hpp"
#include <memory>

using TOK = yy::parser::token;

namespace yy { extern std::array<bool, yyruletype::yynrules> rules; }

namespace mfem
{

namespace internal
{

// *****************************************************************************
/// AST transformation which adds DOM_DX prefix/postfix nodes
// *****************************************************************************
class DomDx : public Middlend
{
protected:
   std::ostream &out;
public:
   DomDx(xfl &ufl, std::ostream &out): Middlend(ufl), out(out) {}
   void Visit(Rule& rule)
   {
      if (rule.dfs.down &&
          rule.n == multiplicative_expr_multiplicative_expr_mul_cast_expr &&
          ufl.HitToken(TOK::DOM_DX, rule.child->next->next))
      {
         //DBG("multiplicative_expr_multiplicative_expr_mul_cast_expr\n");
         assert(rule.child->next->n == TOK::MUL);
         assert(rule.child->next->next->n == cast_expr_unary_expr);
         Node *child = rule.child;
         Node *mult = rule.child->next;
         Node *cast = rule.child->next->next;
         const int PX = TOK::DOM_DX_PREFIX;
         Node *prefix = ufl.astAddNode(std::make_shared<Token>(PX, "xfl::NewForm("));
         const int RP = TOK::DOM_DX_POSTFIX;
         Node *postfix = ufl.astAddNode(std::make_shared<Token>(RP, ")"));
         // flush state
         yy::rules.at(rule.n) = false;
         // prefix => child => postfix => mult => cast
         (rule.child = prefix, rule.dfs.n = prefix);
         prefix->next = child;
         child->next = postfix;
         postfix->next = mult;
         mult->next = cast;
         rule.dfs.down = false;
      }
   }
   void Visit(Token&) { /* Nothing to do */  }
};

// *****************************************************************************
/// AST code generation
// *****************************************************************************
class Code : public Middlend
{
   Middlend &me;
   std::ostream &out;
public:
   Code(xfl &ufl, std::ostream &out): Middlend(ufl), me(*this), out(out) {}
#define DECL_RULE(name) \
   void name##_r(bool &d, Node*&n) const { d=d?name##_t(n):name##_f(n); }\
   bool name##_t(Node*&) const;\
   bool name##_f(Node*&) const
   DECL_RULE(entry_point_statements);
   DECL_RULE(decl_domain_assign_op_expr);
   DECL_RULE(decl_id_list_assign_op_expr);
   DECL_RULE(decl_direct_declarator);
   DECL_RULE(postfix_id_primary_id);
   DECL_RULE(assign_expr_postfix_expr_assign_op_assign_expr);
   DECL_RULE(primary_expr_identifier);
   DECL_RULE(assign_op_eq);
   DECL_RULE(primary_expr_api);
   DECL_RULE(postfix_expr_pow_expr);
   DECL_RULE(id_list_postfix_ids);
   DECL_RULE(if_statement_if_lp_expr_rp_expr);
   DECL_RULE(decl_iteration_statement);
   DECL_RULE(decl_api_statement);
   DECL_RULE(decl_function);
   DECL_RULE(args_expr_list_assign_expr);
   DECL_RULE(args_expr_list_args_expr_list_coma_assign_expr);
   DECL_RULE(def_statement_nl);
   DECL_RULE(def_empty_empty);
#undef DECL_RULE

   void Visit(Rule& r)
   {
#define CASE_RULE(nm) case (nm): return nm##_r(r.dfs.down, r.dfs.n);
      switch (r.n)
      {
            CASE_RULE(entry_point_statements);
            CASE_RULE(decl_domain_assign_op_expr);
            CASE_RULE(decl_id_list_assign_op_expr);
            CASE_RULE(decl_direct_declarator);
            CASE_RULE(postfix_id_primary_id);
            CASE_RULE(assign_expr_postfix_expr_assign_op_assign_expr);
            CASE_RULE(primary_expr_identifier);
            CASE_RULE(assign_op_eq);
            CASE_RULE(primary_expr_api);
            CASE_RULE(postfix_expr_pow_expr);
            CASE_RULE(id_list_postfix_ids);
            CASE_RULE(if_statement_if_lp_expr_rp_expr);
            CASE_RULE(decl_iteration_statement);
            CASE_RULE(decl_api_statement);
            CASE_RULE(decl_function);
            CASE_RULE(args_expr_list_assign_expr);
            CASE_RULE(args_expr_list_args_expr_list_coma_assign_expr);
            CASE_RULE(def_statement_nl);
            CASE_RULE(def_empty_empty);
         default: /* Nothing to do */ ;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(std::string) const
   DECL_TOKEN(NL);
   DECL_TOKEN(QUOTE);
   DECL_TOKEN(DOM_DX);
   DECL_TOKEN(POW);
   DECL_TOKEN(INNER_OP);
   DECL_TOKEN(EQ);
   DECL_TOKEN(FOR);
   DECL_TOKEN(IN);
   DECL_TOKEN(RANGE);
   DECL_TOKEN(COLON);
   DECL_TOKEN(DEF);
   DECL_TOKEN(LP);
   DECL_TOKEN(RP);
   DECL_TOKEN(RETURN);
   DECL_TOKEN(COMA);
   DECL_TOKEN(AND_AND);
   DECL_TOKEN(OR_OR);
   DECL_TOKEN(CONSTANT_API);
#undef DECL_TOKEN

   void Visit(Token& tk)
   {
#define CASE_TOKEN(TK) case TOK::TK : return token_##TK(tk.Name())
      switch (tk.n)
      {
            CASE_TOKEN(NL);
            CASE_TOKEN(QUOTE);
            CASE_TOKEN(DOM_DX);
            CASE_TOKEN(POW);
            CASE_TOKEN(INNER_OP);
            CASE_TOKEN(EQ);
            CASE_TOKEN(FOR);
            CASE_TOKEN(IN);
            CASE_TOKEN(RANGE);
            CASE_TOKEN(COLON);
            CASE_TOKEN(DEF);
            CASE_TOKEN(LP);
            CASE_TOKEN(RP);
            CASE_TOKEN(RETURN);
            CASE_TOKEN(COMA);
            CASE_TOKEN(AND_AND);
            CASE_TOKEN(OR_OR);
            CASE_TOKEN(CONSTANT_API);
         default: out << tk.Name();
      }
#undef CASE_TOKEN
   }
};

} // internal

} // mfem

#endif // XFL_XFC_HPP
