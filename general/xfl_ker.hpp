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
#ifndef XFL_XFK_HPP
#define XFL_XFK_HPP

#include <memory>
#include <sstream>
#include <stack>
#include <vector>

#include "xfl.Y.hpp"
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
/// AST dump static kernels
// *****************************************************************************
class StaticKernels : public Middlend
{
   std::ostringstream &out;
   void Dump(Node &n)
   {
      if (n.dfs.down && !n.out.empty())
      {
         out << n.out;
      }
   }

public:
   StaticKernels(xfl &ufl, std::ostringstream &out) : Middlend(ufl), out(out) {}
   void Visit(Rule &n) { Dump(n); }
   void Visit(Token &n) { Dump(n); }
};

// *****************************************************************************
/// AST kernel operations dump
// *****************************************************************************
class KernelOperations : public Middlend
{
protected:
   std::ostream &out;

public:
   KernelOperations(xfl &ufl, std::ostream &out) : Middlend(ufl), out(out) {}
   void Visit(Rule &n)
   {
      const int N = n.n;
      if (N == extra_status_rule_dot_xt)
      {
         out << "*,Ye";
      }
      if (N == extra_status_rule_eval_xt)
      {
         out << "B,";
      }
      if (N == grad_expr_grad_op_form_args)
      {
         out << "G,";
      }
      if (N == extra_status_rule_transpose_xt)
      {
         out << "T,";
      }
   }
   void Visit(Token &n)
   {
      const int T = n.n;
      if (T == TOK::LP)
      {
         return;
      }
      if (T == TOK::RP)
      {
         return;
      }
      if (T == TOK::COMA)
      {
         out << "*,D,*,";
         return;
      }
      if (T == TOK::GRAD_OP)
      {
         return;
      }
      if (T == TOK::IDENTIFIER)
      {
         out << n.Name();
      }
      if (T == TOK::MUL)
      {
         out << "*,";
         return;
      }
      out << ",";  // << "(" << n.Name() << ")";
   }
};

// *****************************************************************************
/// AST Kernel Setup Shape Extractor
// *****************************************************************************
class KernelShapes : public Middlend
{
private:
   mutable std::array<int, 2> shapes{{2, 2}};
   mutable bool lhs{true};
   mutable std::stack<xfl::var *> var_stack;
   mutable std::stack<int> ops_stack;

protected:
   Node *root;
   std::ostringstream &out;
   const xfl::fes &fes;
   const xfl::fec &fec;
   const int dim;  //, vdim, ndofs, ndof, ne;
public:
   KernelShapes(const int K, xfl &ufl, Node *root, std::ostringstream &out,
                const xfl::fes &fes, const xfl::fec &fec);
   std::array<int, 2> GetShapes() const { return shapes; }
#define DECL_RULE(name) void rule_##name(Rule &) const
   DECL_RULE(extra_status_rule_eval_xt);
   DECL_RULE(grad_expr_grad_op_form_args);
#undef DECL_RULE
   void Visit(Rule &rule)
   {
#define CASE_RULE(name) \
  case (name):          \
    return rule_##name(rule);
      switch (rule.n)
      {
            CASE_RULE(extra_status_rule_eval_xt);
            CASE_RULE(grad_expr_grad_op_form_args);
         default: /* Nothing to do */;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token &) const
   DECL_TOKEN(COMA);
   DECL_TOKEN(IDENTIFIER);
#undef DECL_TOKEN
   void Visit(Token &token)
   {
#define CASE_TOKEN(TK) \
  case TOK::TK:        \
    return token_##TK(token)
      switch (token.n)
      {
            CASE_TOKEN(COMA);
            CASE_TOKEN(IDENTIFIER);
         default:;  // out << token.Name();
      }
#undef CASE_TOKEN
   }
};

// *****************************************************************************
/// AST Kernel Setup Code Generator
// *****************************************************************************
class KernelSetup : public Middlend
{
private:
   mutable bool lhs{true};
   mutable std::stack<xfl::var *> var_stack;
   mutable std::stack<int> ops_stack;

protected:
   Node *root;
   std::ostringstream &out;
   const xfl::fes &fes;
   const xfl::fec &fec;
   const int dim;  //, vdim, ndofs, ndof, ne;
public:
   KernelSetup(const int K, xfl &ufl, Node *root, std::ostringstream &out,
               const xfl::fes &fes, const xfl::fec &fec);
   ~KernelSetup();
#define DECL_RULE(name) void rule_##name(Rule &) const
   DECL_RULE(extra_status_rule_eval_xt);
   DECL_RULE(grad_expr_grad_op_form_args);
#undef DECL_RULE
   void Visit(Rule &rule)
   {
#define CASE_RULE(name) \
  case (name):          \
    return rule_##name(rule);
      switch (rule.n)
      {
            CASE_RULE(extra_status_rule_eval_xt);
            CASE_RULE(grad_expr_grad_op_form_args);
         default: /* Nothing to do */;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token &) const
   DECL_TOKEN(COMA);
   DECL_TOKEN(IDENTIFIER);
#undef DECL_TOKEN
   void Visit(Token &token)
   {
#define CASE_TOKEN(TK) \
  case TOK::TK:        \
    return token_##TK(token)
      switch (token.n)
      {
            CASE_TOKEN(COMA);
            CASE_TOKEN(IDENTIFIER);
         default:;  // out << token.Name();
      }
#undef CASE_TOKEN
   }
};

// *****************************************************************************
/// AST Kernel Mult Code Generator
// *****************************************************************************
class KernelMult : public Middlend
{
private:
   mutable bool lhs{true};
   mutable std::stack<xfl::var *> var_stack;
   mutable std::stack<int> ops_stack;

protected:
   Node *root;
   std::ostringstream &out;
   const xfl::fes &fes;
   const xfl::fec &fec;
   const int dim;  //, vdim, ndofs, ndof, ne;
public:
   KernelMult(const int K, xfl &ufl, Node *root, std::ostringstream &out,
              const xfl::fes &fes, const xfl::fec &fec);
   ~KernelMult();
#define DECL_RULE(name) void rule_##name(Rule &) const
   DECL_RULE(extra_status_rule_dom_xt);
   DECL_RULE(extra_status_rule_dot_xt);
   DECL_RULE(extra_status_rule_eval_xt);
   DECL_RULE(grad_expr_grad_op_form_args);
   DECL_RULE(extra_status_rule_transpose_xt);
#undef DECL_RULE
   void Visit(Rule &rule)
   {
#define CASE_RULE(name) \
  case (name):          \
    return rule_##name(rule);
      switch (rule.n)
      {
            CASE_RULE(extra_status_rule_dom_xt);
            CASE_RULE(extra_status_rule_dot_xt);
            CASE_RULE(extra_status_rule_eval_xt);
            CASE_RULE(grad_expr_grad_op_form_args);
            CASE_RULE(extra_status_rule_transpose_xt);
         default: /* Nothing to do */;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token &) const
   DECL_TOKEN(LP);
   DECL_TOKEN(RP);
   DECL_TOKEN(COMA);
   DECL_TOKEN(GRAD_OP);
   DECL_TOKEN(IDENTIFIER);
#undef DECL_TOKEN
   void Visit(Token &token)
   {
#define CASE_TOKEN(TK) \
  case TOK::TK:        \
    return token_##TK(token)
      switch (token.n)
      {
            CASE_TOKEN(LP);
            CASE_TOKEN(RP);
            CASE_TOKEN(COMA);
            CASE_TOKEN(GRAD_OP);
            CASE_TOKEN(IDENTIFIER);
         default:;  // out << token.Name();
      }
#undef CASE_TOKEN
   }
};

// *****************************************************************************
/// AST Kernel Setup Code Generator
// *****************************************************************************
class KerSimdSetup : public Middlend
{
private:
   mutable bool lhs{true};
   mutable std::stack<xfl::var *> var_stack;
   mutable std::stack<int> ops_stack;

protected:
   Node *root;
   std::ostringstream &out;
   const xfl::fes &fes;
   const xfl::fec &fec;
   const int dim;  //, vdim, ndofs, ndof, ne;
public:
   KerSimdSetup(const int K, xfl &ufl, Node *root, std::ostringstream &out,
                const xfl::fes &fes, const xfl::fec &fec);
   ~KerSimdSetup();
#define DECL_RULE(name) void rule_##name(Rule &) const
   DECL_RULE(extra_status_rule_eval_xt);
   DECL_RULE(grad_expr_grad_op_form_args);
#undef DECL_RULE
   void Visit(Rule &rule)
   {
#define CASE_RULE(name) \
  case (name):          \
    return rule_##name(rule);
      switch (rule.n)
      {
            CASE_RULE(extra_status_rule_eval_xt);
            CASE_RULE(grad_expr_grad_op_form_args);
         default: /* Nothing to do */;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token &) const
   DECL_TOKEN(COMA);
   DECL_TOKEN(IDENTIFIER);
#undef DECL_TOKEN
   void Visit(Token &token)
   {
#define CASE_TOKEN(TK) \
  case TOK::TK:        \
    return token_##TK(token)
      switch (token.n)
      {
            CASE_TOKEN(COMA);
            CASE_TOKEN(IDENTIFIER);
         default:;  // out << token.Name();
      }
#undef CASE_TOKEN
   }
};

// *****************************************************************************
/// AST OpenMP + SIMD Kernel Mult Code Generator
// *****************************************************************************
class KerSimdMult : public Middlend
{
private:
   mutable bool lhs{true};
   mutable std::stack<xfl::var *> var_stack;
   mutable std::stack<int> ops_stack;

protected:
   Node *root;
   std::ostringstream &out;
   const xfl::fes &fes;
   const xfl::fec &fec;
   const int dim;

public:
   KerSimdMult(const int K, xfl &ufl, Node *root, std::ostringstream &out,
               const xfl::fes &fes, const xfl::fec &fec);
   ~KerSimdMult();
#define DECL_RULE(name) void rule_##name(Rule &) const
   DECL_RULE(extra_status_rule_eval_xt);
   DECL_RULE(grad_expr_grad_op_form_args);
#undef DECL_RULE
   void Visit(Rule &rule)
   {
#define CASE_RULE(name) \
  case (name):          \
    return rule_##name(rule);
      switch (rule.n)
      {
            CASE_RULE(extra_status_rule_eval_xt);
            CASE_RULE(grad_expr_grad_op_form_args);
         default:;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token &) const
   DECL_TOKEN(COMA);
   DECL_TOKEN(IDENTIFIER);
#undef DECL_TOKEN
   void Visit(Token &token)
   {
#define CASE_TOKEN(TK) \
  case TOK::TK:        \
    return token_##TK(token)
      switch (token.n)
      {
            CASE_TOKEN(COMA);
            CASE_TOKEN(IDENTIFIER);
         default:;
      }
#undef CASE_TOKEN
   }
};

// *****************************************************************************
/// AST Kernel Setup Code Generator
// *****************************************************************************
class KerRegsSetup : public Middlend
{
private:
   mutable bool lhs{true};
   mutable std::stack<xfl::var *> var_stack;
   mutable std::stack<int> ops_stack;

protected:
   Node *root;
   std::ostringstream &out;
   const xfl::fes &fes;
   const xfl::fec &fec;
   const int dim;  //, vdim, ndofs, ndof, ne;
public:
   KerRegsSetup(const int K, xfl &ufl, Node *root, std::ostringstream &out,
                const xfl::fes &fes, const xfl::fec &fec);
   ~KerRegsSetup();
#define DECL_RULE(name) void rule_##name(Rule &) const
   DECL_RULE(extra_status_rule_eval_xt);
   DECL_RULE(grad_expr_grad_op_form_args);
#undef DECL_RULE
   void Visit(Rule &rule)
   {
#define CASE_RULE(name) \
  case (name):          \
    return rule_##name(rule);
      switch (rule.n)
      {
            CASE_RULE(extra_status_rule_eval_xt);
            CASE_RULE(grad_expr_grad_op_form_args);
         default: /* Nothing to do */;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token &) const
   DECL_TOKEN(COMA);
   DECL_TOKEN(IDENTIFIER);
#undef DECL_TOKEN
   void Visit(Token &token)
   {
#define CASE_TOKEN(TK) \
  case TOK::TK:        \
    return token_##TK(token)
      switch (token.n)
      {
            CASE_TOKEN(COMA);
            CASE_TOKEN(IDENTIFIER);
         default:;  // out << token.Name();
      }
#undef CASE_TOKEN
   }
};

// *****************************************************************************
/// AST + Register Kernel Mult Code Generator
// *****************************************************************************
class KerRegsMult : public Middlend
{
private:
   mutable bool lhs{true};
   mutable std::stack<xfl::var *> var_stack;
   mutable std::stack<int> ops_stack;

protected:
   Node *root;
   std::ostringstream &out;
   const xfl::fes &fes;
   const xfl::fec &fec;
   const int dim;
   int D1D, Q1D;

public:
   KerRegsMult(const int K, xfl &ufl, Node *root, std::ostringstream &out,
               const xfl::fes &fes, const xfl::fec &fec);
   ~KerRegsMult();
#define DECL_RULE(name) void rule_##name(Rule &) const
   DECL_RULE(grad_expr_grad_op_form_args);
#undef DECL_RULE
   void Visit(Rule &rule)
   {
#define CASE_RULE(name) \
  case (name):          \
    return rule_##name(rule);
      switch (rule.n)
      {
            CASE_RULE(grad_expr_grad_op_form_args);
         default:;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token &) const
   DECL_TOKEN(COMA);
   DECL_TOKEN(IDENTIFIER);
#undef DECL_TOKEN
   void Visit(Token &token)
   {
#define CASE_TOKEN(TK) \
  case TOK::TK:        \
    return token_##TK(token)
      switch (token.n)
      {
            CASE_TOKEN(COMA);
            CASE_TOKEN(IDENTIFIER);
         default:;
      }
#undef CASE_TOKEN
   }
};

// *****************************************************************************
/// AST Kernel Setup Code Generator
// *****************************************************************************
class KerOreoSetup : public Middlend
{
private:
   mutable bool lhs{true};
   mutable std::stack<xfl::var *> var_stack;
   mutable std::stack<int> ops_stack;

protected:
   Node *root;
   std::ostringstream &out;
   const xfl::fes &fes;
   const xfl::fec &fec;
   const int dim;  //, vdim, ndofs, ndof, ne;
public:
   KerOreoSetup(const int K, xfl &ufl, Node *root, std::ostringstream &out,
                const xfl::fes &fes, const xfl::fec &fec);
   ~KerOreoSetup();
#define DECL_RULE(name) void rule_##name(Rule &) const
   DECL_RULE(extra_status_rule_eval_xt);
   DECL_RULE(grad_expr_grad_op_form_args);
#undef DECL_RULE
   void Visit(Rule &rule)
   {
#define CASE_RULE(name) \
  case (name):          \
    return rule_##name(rule);
      switch (rule.n)
      {
            CASE_RULE(extra_status_rule_eval_xt);
            CASE_RULE(grad_expr_grad_op_form_args);
         default: /* Nothing to do */;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token &) const
   DECL_TOKEN(COMA);
   DECL_TOKEN(IDENTIFIER);
#undef DECL_TOKEN
   void Visit(Token &token)
   {
#define CASE_TOKEN(TK) \
  case TOK::TK:        \
    return token_##TK(token)
      switch (token.n)
      {
            CASE_TOKEN(COMA);
            CASE_TOKEN(IDENTIFIER);
         default:;  // out << token.Name();
      }
#undef CASE_TOKEN
   }
};

// *****************************************************************************
/// AST + Register, SIMD, Threaded Kernel Mult Code Generator
// *****************************************************************************
class KerOreoMult : public Middlend
{
private:
   mutable bool lhs{true};
   mutable std::stack<xfl::var *> var_stack;
   mutable std::stack<int> ops_stack;

protected:
   Node *root;
   std::ostringstream &out;
   const xfl::fes &fes;
   const xfl::fec &fec;
   const int dim;
   int D1D, Q1D;

public:
   KerOreoMult(const int K, xfl &ufl, Node *root, std::ostringstream &out,
               const xfl::fes &fes, const xfl::fec &fec);
   ~KerOreoMult();
#define DECL_RULE(name) void rule_##name(Rule &) const
   DECL_RULE(grad_expr_grad_op_form_args);
#undef DECL_RULE
   void Visit(Rule &rule)
   {
#define CASE_RULE(name) \
  case (name):          \
    return rule_##name(rule);
      switch (rule.n)
      {
            CASE_RULE(grad_expr_grad_op_form_args);
         default:;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token &) const
   DECL_TOKEN(COMA);
   DECL_TOKEN(IDENTIFIER);
#undef DECL_TOKEN
   void Visit(Token &token)
   {
#define CASE_TOKEN(TK) \
  case TOK::TK:        \
    return token_##TK(token)
      switch (token.n)
      {
            CASE_TOKEN(COMA);
            CASE_TOKEN(IDENTIFIER);
         default:;
      }
#undef CASE_TOKEN
   }
};

// *****************************************************************************
/// AST Kernel Setup Code Generator
// *****************************************************************************
class KerLeanSetup : public Middlend
{
private:
   mutable bool lhs{true};
   mutable std::stack<xfl::var *> var_stack;
   mutable std::stack<int> ops_stack;

protected:
   Node *root;
   std::ostringstream &out;
   const xfl::fes &fes;
   const xfl::fec &fec;
   const int dim;

public:
   KerLeanSetup(const int K, xfl &ufl, Node *root, std::ostringstream &out,
                const xfl::fes &fes, const xfl::fec &fec);
   ~KerLeanSetup();
#define DECL_RULE(name) void rule_##name(Rule &) const
   DECL_RULE(extra_status_rule_eval_xt);
   DECL_RULE(grad_expr_grad_op_form_args);
#undef DECL_RULE
   void Visit(Rule &rule)
   {
#define CASE_RULE(name) \
  case (name):          \
    return rule_##name(rule);
      switch (rule.n)
      {
            CASE_RULE(extra_status_rule_eval_xt);
            CASE_RULE(grad_expr_grad_op_form_args);
         default: /* Nothing to do */;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token &) const
   DECL_TOKEN(COMA);
   DECL_TOKEN(IDENTIFIER);
#undef DECL_TOKEN
   void Visit(Token &token)
   {
#define CASE_TOKEN(TK) \
  case TOK::TK:        \
    return token_##TK(token)
      switch (token.n)
      {
            CASE_TOKEN(COMA);
            CASE_TOKEN(IDENTIFIER);
         default:;  // out << token.Name();
      }
#undef CASE_TOKEN
   }
};

// *****************************************************************************
/// AST OpenMP + SIMD Kernel Mult Code Generator
// *****************************************************************************
class KerLeanMult : public Middlend
{
private:
   mutable bool lhs{true};
   mutable std::stack<xfl::var *> var_stack;
   mutable std::stack<int> ops_stack;

protected:
   Node *root;
   std::ostringstream &out;
   const xfl::fes &fes;
   const xfl::fec &fec;
   const int dim;
   int D1D, Q1D;

public:
   KerLeanMult(const int K, xfl &ufl, Node *root, std::ostringstream &out,
               const xfl::fes &fes, const xfl::fec &fec);
   ~KerLeanMult();
#define DECL_RULE(name) void rule_##name(Rule &) const
   DECL_RULE(grad_expr_grad_op_form_args);
#undef DECL_RULE
   void Visit(Rule &rule)
   {
#define CASE_RULE(name) \
  case (name):          \
    return rule_##name(rule);
      switch (rule.n)
      {
            CASE_RULE(grad_expr_grad_op_form_args);
         default:;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token &) const
   DECL_TOKEN(COMA);
   DECL_TOKEN(IDENTIFIER);
#undef DECL_TOKEN
   void Visit(Token &token)
   {
#define CASE_TOKEN(TK) \
  case TOK::TK:        \
    return token_##TK(token)
      switch (token.n)
      {
            CASE_TOKEN(COMA);
            CASE_TOKEN(IDENTIFIER);
         default:;
      }
#undef CASE_TOKEN
   }
};

// *****************************************************************************
/// AST libParanumal-like Kernel Mult Code Generator
// *****************************************************************************
class KerLibPMult : public Middlend
{
private:
   mutable bool lhs{true};
   mutable std::stack<xfl::var *> var_stack;
   mutable std::stack<int> ops_stack;

protected:
   Node *root;
   std::ostringstream &out;
   const xfl::fes &fes;
   const xfl::fec &fec;
   const int dim;

public:
   KerLibPMult(const int K, xfl &ufl, Node *root, std::ostringstream &out,
               const xfl::fes &fes, const xfl::fec &fec);
   ~KerLibPMult();
#define DECL_RULE(name) void rule_##name(Rule &) const
   DECL_RULE(extra_status_rule_eval_xt);
   DECL_RULE(grad_expr_grad_op_form_args);
#undef DECL_RULE
   void Visit(Rule &rule)
   {
#define CASE_RULE(name) \
  case (name):          \
    return rule_##name(rule);
      switch (rule.n)
      {
            CASE_RULE(extra_status_rule_eval_xt);
            CASE_RULE(grad_expr_grad_op_form_args);
         default:;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token &) const
   DECL_TOKEN(COMA);
   DECL_TOKEN(IDENTIFIER);
#undef DECL_TOKEN
   void Visit(Token &token)
   {
#define CASE_TOKEN(TK) \
  case TOK::TK:        \
    return token_##TK(token)
      switch (token.n)
      {
            CASE_TOKEN(COMA);
            CASE_TOKEN(IDENTIFIER);
         default:;
      }
#undef CASE_TOKEN
   }
};

// *****************************************************************************
/// AST libCEED-like Kernel Mult Code Generator
// *****************************************************************************
class KerCeedMult : public Middlend
{
private:
   mutable bool lhs{true};
   mutable std::stack<xfl::var *> var_stack;
   mutable std::stack<int> ops_stack;

protected:
   Node *root;
   std::ostringstream &out;
   const xfl::fes &fes;
   const xfl::fec &fec;
   const int dim;

public:
   KerCeedMult(const int K, xfl &ufl, Node *root, std::ostringstream &out,
               const xfl::fes &fes, const xfl::fec &fec);
   ~KerCeedMult();
#define DECL_RULE(name) void rule_##name(Rule &) const
   DECL_RULE(extra_status_rule_eval_xt);
   DECL_RULE(grad_expr_grad_op_form_args);
#undef DECL_RULE
   void Visit(Rule &rule)
   {
#define CASE_RULE(name) \
  case (name):          \
    return rule_##name(rule);
      switch (rule.n)
      {
            CASE_RULE(extra_status_rule_eval_xt);
            CASE_RULE(grad_expr_grad_op_form_args);
         default:;
      }
#undef CASE_RULE
   }

#define DECL_TOKEN(TK) void token_##TK(Token &) const
   DECL_TOKEN(COMA);
   DECL_TOKEN(IDENTIFIER);
#undef DECL_TOKEN
   void Visit(Token &token)
   {
#define CASE_TOKEN(TK) \
  case TOK::TK:        \
    return token_##TK(token)
      switch (token.n)
      {
            CASE_TOKEN(COMA);
            CASE_TOKEN(IDENTIFIER);
         default:;
      }
#undef CASE_TOKEN
   }
};

}  // namespace internal

}  // namespace mfem

#endif  // XFL_XFK_HPP
