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

namespace mfem
{

namespace internal
{

using TOK = yy::parser::token;

// *****************************************************************************
template<typename D> struct Backend
{
   std::ostream &out;
   Backend(std::ostream &out): out(out) {}
   D const& that() const { return static_cast<const D&>(*this); }

   template<int TK>
   void token(std::string text) const { that().template token<TK>(text); }

   template<int RN>
   void rule(bool &dfs, Node **x) const { that().template rule<RN>(dfs, x); }
};

// *****************************************************************************
struct XPU: Backend<XPU>
{
   struct Middlend &ir;
   XPU(std::ostream &out, struct Middlend &ir): Backend<XPU>(out), ir(ir) {}
   template<int> void rule(bool&, Node**) const { }
   template<int> void token(std::string text) const { out << text; }
};

// *****************************************************************************
struct Middlend
{
   Backend<XPU> &xpu;

   explicit Middlend(Backend<XPU> &dev): xpu(dev) {}

   template<int SN> void middlend(Token<SN> *t) const noexcept
   { xpu.template token<SN>(t->Name()); }

   template<int RN> void middlend(Rule<RN>*, bool &dfs,
                                  Node **extra) const noexcept
   { xpu.template rule<RN>(dfs, extra); }
};

// *****************************************************************************
struct XIR: Middlend
{
   XPU dev;
   XIR(std::ostream &out): Middlend(dev), dev(out, *this) {}
};

// *****************************************************************************
#define RULE(name) \
    template<> void XPU::rule<name>(bool&, Node**) const;\
    template<bool> bool name##_cpu(struct Middlend&, std::ostream&, Node**)
RULE(entry_point_statements);
RULE(decl_domain_assign_op_expr);
RULE(decl_id_list_assign_op_expr);
RULE(decl_direct_declarator);
RULE(postfix_id_primary_id);
RULE(assign_expr_postfix_expr_assign_op_assign_expr);
RULE(primary_expr_identifier);
RULE(assign_op_eq);
RULE(primary_expr_api);
RULE(postfix_expr_pow_expr);
RULE(id_list_postfix_ids);
RULE(if_statement_if_lp_expr_rp_expr);
RULE(decl_iteration_statement);
RULE(decl_api_statement);
RULE(decl_function);
RULE(args_expr_list_assign_expr);
RULE(args_expr_list_args_expr_list_coma_assign_expr);
RULE(def_statement_nl);
RULE(def_empty_empty);
#undef RULE

// *****************************************************************************
template<> void XPU::token<TOK::NL>(std::string) const;
template<> void XPU::token<TOK::QUOTE>(std::string) const;
template<> void XPU::token<TOK::DOM_DX>(std::string) const;
template<> void XPU::token<TOK::POW>(std::string) const;
template<> void XPU::token<TOK::INNER_OP>(std::string) const;
template<> void XPU::token<TOK::EQ>(std::string) const;
template<> void XPU::token<TOK::FOR>(std::string) const;
template<> void XPU::token<TOK::IN>(std::string) const;
template<> void XPU::token<TOK::RANGE>(std::string) const;
template<> void XPU::token<TOK::COLON>(std::string) const;
template<> void XPU::token<TOK::DEF>(std::string) const;
template<> void XPU::token<TOK::LP>(std::string) const;
template<> void XPU::token<TOK::RP>(std::string) const;
template<> void XPU::token<TOK::RETURN>(std::string) const;
template<> void XPU::token<TOK::COMA>(std::string) const;
template<> void XPU::token<TOK::AND_AND>(std::string) const;
template<> void XPU::token<TOK::OR_OR>(std::string) const;

} // internal

} // mfem

#endif // XFL_XFC_HPP
