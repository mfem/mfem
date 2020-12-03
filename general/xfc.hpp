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
/// AST Qfunction dump
// *****************************************************************************
class Qfunc : public Middlend
{
protected:
   std::ostream &out;
public:
   Qfunc(xfl &ufl, std::ostream &out): Middlend(ufl), out(out) {}
#define DECL_RULE(name) \
   void name##_r(Rule*n) const { n->dfs.down ? name##_d(n) : name##_u(n); }\
   void name##_d(Rule*) const;\
   void name##_u(Rule*) const
   DECL_RULE(postfix_expr_pow_expr);
#undef DECL_RULE

   void Visit(Rule& rule)
   {
#define CASE_RULE(nm) case (nm): return nm##_r(&rule);
      switch (rule.n)
      {
            CASE_RULE(postfix_expr_pow_expr);
         default: /* Nothing to do */ ;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token*) const
   DECL_TOKEN(POW);
   DECL_TOKEN(INNER_OP);
   DECL_TOKEN(DOT_OP);
   DECL_TOKEN(GRAD_OP);
   void Visit(Token& token)
   {
#define CASE_TOKEN(TK) case TOK::TK : return token_##TK(&token)
      switch (token.n)
      {
            CASE_TOKEN(POW);
            CASE_TOKEN(INNER_OP);
            CASE_TOKEN(DOT_OP);
            CASE_TOKEN(GRAD_OP);
         default: out << token.Name();
      }
#undef CASE_TOKEN
   }
};

// *****************************************************************************
/// AST transformation which adds the extra dom_xt/var_xt rules
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
         //dbg("multiplicative_expr_multiplicative_expr_mul_cast_expr");
         if (rule.child->next->n != TOK::MUL)
         {
            // tests/ufl/QuadratureElement.ufl unsuported dx(i)*dx
            return;
         }
         assert(rule.child->next->next->n == cast_expr_unary_expr);

         Node *child = rule.child; assert(child);
         Node *cast_expr = rule.child->next->next;

         constexpr int vx = extra_status_rule_var_xt;
         Node *var_xt = ufl.astAddNode(std::make_shared<Rule>(vx, "var_xt"));

         constexpr int dx = extra_status_rule_dom_xt;
         Node *dom_xt = ufl.astAddNode(std::make_shared<Rule>(dx, "dom_xt"));

         rule.child = var_xt;
         var_xt->root = static_cast<Node*>(&rule); assert(var_xt->root);
         var_xt->child = child; // set the child to be able to do the dfs
         var_xt->next = dom_xt;

         dom_xt->root = static_cast<Node*>(&rule); assert(dom_xt->root);
         dom_xt->next = cast_expr;
         dom_xt->child = child;
         child->root = dom_xt;
         child->next = nullptr;

      }
   }
   void Visit(Token&) { /* Nothing to do */  }
};

// *****************************************************************************
/// AST code generation
// *****************************************************************************
class Code : public Middlend
{
public:
   Middlend &me;
   std::ostream &out;
public:
   Code(xfl &ufl, std::ostream &out): Middlend(ufl), me(*this), out(out) {}
#define DECL_RULE(name) \
   void name##_r(Rule*n) const { n->dfs.down ? name##_d(n) : name##_u(n); }\
   void name##_d(Rule*) const;\
   void name##_u(Rule*) const
   DECL_RULE(entry_point_statements);
   DECL_RULE(decl_domain_assign_op_expr);
   DECL_RULE(decl_id_list_assign_op_expr);
   DECL_RULE(decl_direct_declarator);
   DECL_RULE(postfix_id_primary_id);
   DECL_RULE(primary_id_identifier);
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
   DECL_RULE(statement_decl_nl);
   DECL_RULE(def_empty_empty);
   DECL_RULE(extra_status_rule_var_xt);
   DECL_RULE(extra_status_rule_dom_xt);
   DECL_RULE(postfix_expr_grad_op_lp_additive_expr_rp);
#undef DECL_RULE

   void Visit(Rule& rule)
   {
#define CASE_RULE(nm) case (nm): return nm##_r(&rule);
      switch (rule.n)
      {
            CASE_RULE(entry_point_statements);
            CASE_RULE(decl_domain_assign_op_expr);
            CASE_RULE(decl_id_list_assign_op_expr);
            CASE_RULE(decl_direct_declarator);
            CASE_RULE(postfix_id_primary_id);
            CASE_RULE(primary_id_identifier);
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
            CASE_RULE(statement_decl_nl);
            CASE_RULE(def_empty_empty);
            CASE_RULE(extra_status_rule_var_xt);
            CASE_RULE(extra_status_rule_dom_xt);
            CASE_RULE(postfix_expr_grad_op_lp_additive_expr_rp);
         default: /* Nothing to do */ ;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token*) const
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
   DECL_TOKEN(DOT_OP);
   DECL_TOKEN(GRAD_OP);
#undef DECL_TOKEN

   void Visit(Token& token)
   {
#define CASE_TOKEN(TK) case TOK::TK : return token_##TK(&token)
      switch (token.n)
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
            CASE_TOKEN(DOT_OP);
            CASE_TOKEN(GRAD_OP);
         default: out << token.Name();
      }
#undef CASE_TOKEN
   }
};

} // internal

} // mfem

#endif // XFL_XFC_HPP
