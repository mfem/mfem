#ifndef XFL_CPU_HPP
#define XFL_CPU_HPP

#include "xfl.hpp"
#ifndef XFL_C
using TOK = yy::parser::token;
#else
using TOK = yytokentype;
#endif

// *****************************************************************************
template<typename D> struct Backend
{
   std::ostream &out;
   Backend(std::ostream &out): out(out) {}
   D const& that() const { return static_cast<const D&>(*this); }

   template<int TK>
   void token(std::string text) const { that().template token<TK>(text); }

   template<int RN>
   void rule(bool &dfs, Node **extra) const { that().template rule<RN>(dfs, extra); }
};

// *****************************************************************************
struct CPU: Backend<CPU>
{
   CPU(std::ostream &out): Backend<CPU>(out) {}
   template<int> void rule(bool&, Node**) const { }
   template<int> void token(std::string text) const { out << " " << text; }
};

#define RULE(name) \
template<> void CPU::rule<name>(bool &dfs, Node **extra) const;
RULE(entry_point_statements)
RULE(decl_domain_assign_op_expr)
RULE(decl_id_list_assign_op_expr)
RULE(decl_direct_declarator)
RULE(postfix_id_identifier)
RULE(assign_expr_postfix_expr_assign_op_assign_expr)
RULE(primary_expr_identifier)
RULE(assign_op_eq)
RULE(primary_expr_api)
RULE(postfix_expr_pow_expr)
RULE(id_list_postfix_id)
#undef RULE

template<> void CPU::token<TOK::NL>(std::string) const;
template<> void CPU::token<TOK::QUOTE>(std::string) const;
template<> void CPU::token<TOK::DOM_DX>(std::string) const;
template<> void CPU::token<TOK::POW>(std::string) const;
template<> void CPU::token<TOK::INNER_OP>(std::string) const;

// *****************************************************************************
struct Middlend
{
   Backend<CPU> &cpu;

   explicit Middlend(Backend<CPU> &dev): cpu(dev) {}

   template<int SN> void middlend(Token<SN> *t) const noexcept
   {
#ifndef XFL_C
      { cpu.template token<SN>(t->Name()); }
#else
      { cpu.template token<SN>(t->Text()); } // #warning Tesxt vs Name
#endif
   }

   template<int RN> void middlend(Rule<RN>*, bool &dfs,
                                  Node **extra) const noexcept
   {
      { cpu.template rule<RN>(dfs, extra); }
   }
};

// *****************************************************************************
struct cpu: Middlend
{
   CPU dev;
   cpu(std::ostream &out): Middlend(dev), dev(out) {}
};

#endif // XFL_CPU_HPP
