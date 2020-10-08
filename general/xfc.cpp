#include "xfc.hpp"
#include <iostream>

using std::string;
using std::ostream;

// *****************************************************************************
template<bool> bool entry_point_statements_cpu(std::ostream&, Node**);
template<bool> bool decl_domain_assign_op_expr_cpu(std::ostream&, Node**);
template<bool> bool decl_id_list_assign_op_expr_cpu(std::ostream&, Node**);
template<bool> bool decl_direct_declarator_cpu(std::ostream&, Node**);
template<bool> bool postfix_id_identifier_cpu(std::ostream&, Node**);
template<bool>
bool assign_expr_postfix_expr_assign_op_assign_expr_cpu(std::ostream&, Node**);
template<bool> bool primary_expr_identifier_cpu(std::ostream&, Node**);
template<bool> bool assign_op_eq_cpu(std::ostream&, Node**);
template<bool> bool primary_expr_api_cpu(std::ostream&, Node**);
template<bool> bool postfix_expr_pow_expr_cpu(std::ostream&, Node**);
template<bool> bool id_list_postfix_id_cpu(std::ostream&, Node**);

// *****************************************************************************
namespace yy { extern std::array<bool, yyruletype::yynrules> rules; }

// *****************************************************************************
template<> bool entry_point_statements_cpu<true>(ostream &out, Node**)
{
   out << "#include \"xfm.hpp\"" << std:: endl;
   out << "int main(int argc, char* argv[]){" << std:: endl;
   return true;
}

template<> bool entry_point_statements_cpu<false>(ostream &out, Node**)
{
   out << "\treturn 0;\n}" << std:: endl;
   return true;
}

// *****************************************************************************
template<> bool decl_domain_assign_op_expr_cpu<true>(ostream &out, Node**)
{
   //out << "\tDBG(\"" << __FUNCTION__ << "\");" << std::endl;
   return true;
}
template<> bool decl_domain_assign_op_expr_cpu<false>(ostream &out, Node**)
{
   out << ";" << std::endl;
   return true;
}

// *****************************************************************************
template<> bool decl_id_list_assign_op_expr_cpu<true>(ostream &out, Node**)
{
   out << "\tauto";
   return true;
}
template<> bool decl_id_list_assign_op_expr_cpu<false>(ostream &out, Node**)
{
   out << ";" << std:: endl;
   return true;
}

// *****************************************************************************
template<> bool decl_direct_declarator_cpu<true>(ostream &out, Node**)
{
   out << "\t";
   return true;
}
template<> bool decl_direct_declarator_cpu<false>(ostream &out, Node**)
{
   out << ";" << std:: endl;
   return true;
}

// *****************************************************************************
template<> bool postfix_id_identifier_cpu<true>(ostream&, Node**) { return true; }
template<> bool postfix_id_identifier_cpu<false>(ostream&, Node**) { return true; }

// *****************************************************************************
template<>
bool assign_expr_postfix_expr_assign_op_assign_expr_cpu<true>
(ostream &out, Node**)
{
   //assert(!rule.at(args_expr_list_args_expr_list_coma_assign_expr));
   return true;
}
template<>
bool assign_expr_postfix_expr_assign_op_assign_expr_cpu<false>(ostream &out,
                                                               Node**)
{
   return true;
}

// *****************************************************************************
template<> bool primary_expr_identifier_cpu<true>(ostream &out, Node**)
{
   // If we are inside a argument list, skip it
   if (yy::rules.at(assign_expr_postfix_expr_assign_op_assign_expr)) { return false; }
   return true;
}
template<> bool primary_expr_identifier_cpu<false>(ostream &out, Node**) { return true; }

// *****************************************************************************
template<> bool assign_op_eq_cpu<true>(ostream &out, Node**)
{
   // If we are inside a argument list, skip it
   if (yy::rules.at(assign_expr_postfix_expr_assign_op_assign_expr)) { return false; }
   return true;
}
template<> bool assign_op_eq_cpu<false>(ostream &out, Node**) { return true; }


// *****************************************************************************
template<> bool primary_expr_api_cpu<true>(ostream &out, Node**)
{
   out << " xfl::"; return true;
}
template<> bool primary_expr_api_cpu<false>(ostream &out, Node**) { return true; }

// *****************************************************************************
template<> bool postfix_expr_pow_expr_cpu<true>(ostream &out, Node**)
{
   out << " xfl::math::Pow(";
   return true;
}
template<> bool postfix_expr_pow_expr_cpu<false>(ostream &out, Node**)
{
   out << ")";
   return true;
}

// *****************************************************************************
template<> bool id_list_postfix_id_cpu<true>(ostream &, Node **n_addr)
{
   if (!yy::rules.at(id_list_id_list_coma_postfix_id)) { return true; }
   Node *n = *n_addr;
   assert(n);
   assert(n->next);
   assert(n->next->IsToken());
   assert(n->next->Number() == COMA);
   assert(n->parent);
   assert(n->parent->next);
   if (n->parent->next->Number() != assign_op_eq) { return true; }
   Node *eq = n->parent->next;
   *n_addr = eq;
   return true;
}
template<> bool id_list_postfix_id_cpu<false>(ostream&, Node**) { return true; }

// *****************************************************************************
// Specialized CPU backend rules
#define RULE(name) \
template<> void CPU::rule<name>(bool &dfs, Node **extra) const {\
   if (dfs) { dfs = name##_cpu <true>(out, extra); return; }\
   dfs = name##_cpu <false>(out, extra);}

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
// Specialized CPU backend tokens
template<> void CPU::token<NL>(string) const { /* empty */ }

template<> void CPU::token<QUOTE>(string text) const
{
   str::replace(text.c_str(), '\'','\"');
   out << text;
}

template<> void CPU::token<DOM_DX>(string) const { out << " xfl::Form()"; }

template<> void CPU::token<POW>(string) const { out << ","; }

template<> void CPU::token<INNER_OP>(string) const { out << "dot"; }
