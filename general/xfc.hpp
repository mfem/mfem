#ifndef XFL_CPU_HPP
#define XFL_CPU_HPP

#include "xfl.hpp"

#ifndef YYTOKENTYPE
#include "xfl.Y.hpp"
#endif

#define FORWARD_RULE(RN) template<> void CPU::rule<RN>(bool&, Node**) const;
FORWARD_RULE(entry_point_statements)
FORWARD_RULE(decl_domain_assign_op_expr)
FORWARD_RULE(decl_id_list_assign_op_expr)
FORWARD_RULE(decl_direct_declarator)
FORWARD_RULE(postfix_id_identifier)
FORWARD_RULE(assign_expr_postfix_expr_assign_op_assign_expr)
FORWARD_RULE(primary_expr_identifier)
FORWARD_RULE(assign_op_eq)
FORWARD_RULE(primary_expr_api)
FORWARD_RULE(postfix_expr_pow_expr)
FORWARD_RULE(id_list_postfix_id)

#define FORWARD_TOKEN(T) \
    template<> void CPU::token<T>(std::string) const;
FORWARD_TOKEN(NL)
FORWARD_TOKEN(QUOTE)
FORWARD_TOKEN(DOM_DX)
FORWARD_TOKEN(POW)
FORWARD_TOKEN(INNER_OP)

#endif // XFL_CPU_HPP
