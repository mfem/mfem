// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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

#include <cassert>
#include <memory>
#include <sstream>
#include <stack>
#include <vector>

#include "xfl.hpp"

using TOK = yy::parser::token;

namespace yy
{
extern std::array<bool, yyruletype::yynrules> rules;
}

namespace mfem
{

namespace internal
{

// *****************************************************************************
/// AST eval conditional expr
// *****************************************************************************
class EvalCondExpr : public Middlend
{
protected:
   std::ostream &out;
   mutable std::vector<Node *> v;

public:
   EvalCondExpr(xfl &ufl, std::ostream &out) : Middlend(ufl), out(out)
   {
      ufl.ctx.eval = false;
   }

   void Visit(Rule &) {}

   void Visit(Token &n)
   {
      const int T = n.n;
      if (T == TOK::EQ_EQ)
      {
         assert(v.size() == 2);
         assert(v.at(0)->n == TOK::NATURAL);
         assert(v.at(1)->n == TOK::NATURAL);
         // dbg("%s ?= %s", v.at(0)->Name().c_str(), v.at(1)->Name().c_str());
         const bool eval = v.at(0)->Name() == v.at(1)->Name();
         ufl.ctx.eval = eval;
      }
      if (T == TOK::NATURAL)
      {
         v.push_back(&n);
      }
      if (T == TOK::IDENTIFIER)
      {
         const std::string &name = n.Name();
         assert(ufl.ctx.var.find(name) != ufl.ctx.var.end());
         assert(ufl.ctx.N.find(name) != ufl.ctx.N.end());
         const xfl::var &var = ufl.ctx.var.at(name);
         const int N = ufl.ctx.N.at(name);
         // dbg("Found IDENTIFIER %s = %d", name.c_str(), N);
         assert(var.name == name);
         constexpr int tk = TOK::NATURAL;
         std::string value = std::to_string(N);
         Node *val_n = ufl.NewToken(tk, value.c_str());
         v.push_back(val_n);
      }
      // out << n.Name();
   }
};

// *****************************************************************************
/// AST transformation which turns the postfix id w/o grad to an eval one
// *****************************************************************************
class EvalOp : public Middlend
{
protected:
   std::ostream &out;

public:
   EvalOp(xfl &ufl, std::ostream &out) : Middlend(ufl), out(out) {}
   void Visit(Rule &) {}
   void Visit(Token &token)
   {
      if (token.dfs.down && token.Number() == TOK::IDENTIFIER)
      {
         // we don't have the variables yet which comes in xfl::code
         dbg("\033[33mtoken: %s", token.Name().c_str());
         Node *n = token.root;
         while (n && n->Number() != extra_status_rule_dot_xt)
         {
            dbg("%s", n->Name().c_str());
            // If we hit a grad, stop looking
            if (n->Number() == grad_expr_grad_op_form_args)
            {
               return;
            }
            n = n->root;
         }

         // if we skipped
         if (!n)
         {
            return;
         }

         // skip unknowns places
         if (n->Number() != extra_status_rule_dot_xt)
         {
            return;
         }

         Node *primary_expr = token.root;
         assert(primary_expr->n == primary_expr_identifier);

         constexpr int ex = extra_status_rule_eval_xt;
         Node *eval_xt = ufl.NewRule(ex, "B_xt");
         ufl.InsertNode(eval_xt, primary_expr);
      }
   }
};

// *****************************************************************************
/// AST transformation which turns forms inner to PA mult rules
// *****************************************************************************
class DotPA : public Middlend
{
protected:
   std::ostream &out;

public:
   DotPA(xfl &ufl, std::ostream &out) : Middlend(ufl), out(out) {}
   void Visit(Rule &rule)
   {
      // dbg("%s",rule.Name().c_str());
      if (rule.dfs.down && rule.child && rule.child->next &&
          rule.n == multiplicative_expr_multiplicative_expr_mul_cast_expr &&
          (ufl.HitToken(TOK::DOM_DX, rule.child->next->next) ||
           ufl.HitToken(TOK::EXT_DS, rule.child->next->next) ||
           ufl.HitToken(TOK::INT_DS, rule.child->next->next)))
      {
         dbg();
         assert(rule.child->child);
         assert(rule.child->next->n == TOK::MUL);
         assert(rule.child->next->next->n == cast_expr_unary_expr);

         const Node *inner = ufl.NextToken(rule.child->child);

         if (inner && inner->n == TOK::INNER_OP)
         {
            Node *postfix_expr = inner->root;
            assert(postfix_expr->n ==
                   postfix_expr_inner_op_lp_additive_expr_coma_additive_expr_rp);
            postfix_expr->child = inner->next;
            postfix_expr->n = extra_status_rule_dot_xt;
            postfix_expr->name = "dot_xt";

            assert(postfix_expr->child);  // LP
            assert(postfix_expr->child->n == TOK::LP);
            assert(postfix_expr->child->next);  // additive expr

            Node *coma = postfix_expr->child->next->next;
            assert(coma);  // COMA
            assert(coma->n == TOK::COMA);

            Node *test_expr = coma->next;
            assert(test_expr);

            constexpr int tx = extra_status_rule_transpose_xt;
            Node *transpose_mul_xt = ufl.NewRule(tx, "T_xt");
            transpose_mul_xt->root = postfix_expr;

            transpose_mul_xt->next = test_expr->next;
            test_expr->next = nullptr;

            coma->next = transpose_mul_xt;
            transpose_mul_xt->child = test_expr;
            test_expr->root = transpose_mul_xt;
         }

         mfem::internal::EvalOp eo(ufl, out);
         ufl.DfsPreOrder(rule.child->child, eo);
      }
   }
   void Visit(Token &)   /* Nothing to do */
   {
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
   DomDx(xfl &ufl, std::ostream &out) : Middlend(ufl), out(out) {}
   void Visit(Rule &rule)
   {
      if (rule.dfs.down &&
          rule.n == multiplicative_expr_multiplicative_expr_mul_cast_expr &&
          rule.child->next && ufl.HitToken(TOK::DOM_DX, rule.child->next->next))
      {
         // dbg("multiplicative_expr_multiplicative_expr_mul_cast_expr");
         if (rule.child->next->n != TOK::MUL)
         {
            // tests/ufl/QuadratureElement.ufl unsuported dx(i)*dx
            return;
         }
         assert(rule.child->next->next->n == cast_expr_unary_expr);

         Node *child = rule.child;
         assert(child);
         Node *cast_expr = rule.child->next->next;

         constexpr int dx = extra_status_rule_dom_xt;
         Node *dom_xt = ufl.NewRule(dx, "dom_xt");

         rule.child = dom_xt;
         dom_xt->root = static_cast<Node *>(&rule);
         assert(dom_xt->root);
         dom_xt->child = child;  // set the child to be able to do the dfs
         dom_xt->next = cast_expr;

         child->root = dom_xt;
         child->next = nullptr;
      }
   }
   void Visit(Token &) { /* Nothing to do */ }
};

// *****************************************************************************
/// AST code generation
// *****************************************************************************
class Code : public Middlend
{
private:
   void GenKernelBody(const int) const;
   void GenKernelCode(Rule *) const;

public:
   Middlend &me;
   std::ostringstream &out;
   std::stringstream dev_null;
   mutable std::streambuf *dev_bkp {nullptr};
   const bool ceed_benchmark;

public:
   Code(xfl &ufl, std::ostringstream &out, const bool ceed_benchmark)
      : Middlend(ufl), me(*this), out(out), ceed_benchmark(ceed_benchmark) {}
#define DECL_RULE(name)                                                     \
  void name##_r(Rule *n) const { n->dfs.down ? name##_d(n) : name##_u(n); } \
  void name##_d(Rule *) const;                                              \
  void name##_u(Rule *) const
   DECL_RULE(entry_point_statements);
   DECL_RULE(decl_domain_assign_op_expr);
   DECL_RULE(decl_id_list_assign_op_expr);
   DECL_RULE(decl_direct_declarator);
   DECL_RULE(postfix_id_primary_id);
   DECL_RULE(primary_id_identifier);
   DECL_RULE(assign_expr_postfix_expr_assign_op_assign_expr);
   DECL_RULE(primary_expr_identifier);
   DECL_RULE(assign_op_eq);
   DECL_RULE(postfix_expr_api);
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
   DECL_RULE(extra_status_rule_dom_xt);
   DECL_RULE(extra_status_rule_dot_xt);
   DECL_RULE(grad_expr_grad_op_form_args);
   DECL_RULE(extra_status_rule_transpose_xt);
   DECL_RULE(shift_expr_additive_expr);
#undef DECL_RULE

   void Visit(Rule &rule)
   {
#define CASE_RULE(nm) \
  case (nm):          \
    return nm##_r(&rule);
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
            CASE_RULE(postfix_expr_api);
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
            CASE_RULE(extra_status_rule_dom_xt);
            CASE_RULE(extra_status_rule_dot_xt);
            CASE_RULE(grad_expr_grad_op_form_args);
            CASE_RULE(extra_status_rule_transpose_xt);
            CASE_RULE(shift_expr_additive_expr);
         default: /* Nothing to do */;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token *) const
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
   DECL_TOKEN(ADD);
   DECL_TOKEN(EXPRESSION);
   DECL_TOKEN(BOOL);
#undef DECL_TOKEN

   void Visit(Token &token)
   {
#define CASE_TOKEN(TK) \
  case TOK::TK:        \
    return token_##TK(&token)
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
            CASE_TOKEN(ADD);
            CASE_TOKEN(EXPRESSION);
            CASE_TOKEN(BOOL);
         default:
            out << token.Name();
      }
#undef CASE_TOKEN
   }
};

}  // namespace internal

}  // namespace mfem

#endif  // XFL_XFC_HPP
