// A Bison parser, made by GNU Bison 3.7.5.

// Skeleton interface for Bison LALR(1) parsers in C++

// Copyright (C) 2002-2015, 2018-2021 Free Software Foundation, Inc.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// As a special exception, you may create a larger work that contains
// part or all of the Bison parser skeleton and distribute that work
// under terms of your choice, so long as that work isn't itself a
// parser generator using the skeleton or a modified version thereof
// as a parser skeleton.  Alternatively, if you modify or redistribute
// the parser skeleton itself, you may (at your option) remove this
// special exception, which will cause the skeleton and the resulting
// Bison output files to be licensed under the GNU General Public
// License without this special exception.

// This special exception was added by the Free Software Foundation in
// version 2.2 of Bison.


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

// C++ LALR(1) parser skeleton written by Akim Demaille.

// DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
// especially those whose name start with YY_ or yy_.  They are
// private implementation details that can be changed or removed.

#ifndef YY_YY_USERS_CAMIER1_HOME_SAWMILL_UFL_XFL_BUILD_SRC_XFL_Y_HPP_INCLUDED
# define YY_YY_USERS_CAMIER1_HOME_SAWMILL_UFL_XFL_BUILD_SRC_XFL_Y_HPP_INCLUDED
// "%code requires" blocks.
class xfl; struct Node;


# include <cassert>
# include <cstdlib> // std::abort
# include <iostream>
# include <stdexcept>
# include <string>
# include <vector>

#if defined __cplusplus
# define YY_CPLUSPLUS __cplusplus
#else
# define YY_CPLUSPLUS 199711L
#endif

// Support move semantics when possible.
#if 201103L <= YY_CPLUSPLUS
# define YY_MOVE           std::move
# define YY_MOVE_OR_COPY   move
# define YY_MOVE_REF(Type) Type&&
# define YY_RVREF(Type)    Type&&
# define YY_COPY(Type)     Type
#else
# define YY_MOVE
# define YY_MOVE_OR_COPY   copy
# define YY_MOVE_REF(Type) Type&
# define YY_RVREF(Type)    const Type&
# define YY_COPY(Type)     const Type&
#endif

// Support noexcept when possible.
#if 201103L <= YY_CPLUSPLUS
# define YY_NOEXCEPT noexcept
# define YY_NOTHROW
#else
# define YY_NOEXCEPT
# define YY_NOTHROW throw ()
#endif

// Support constexpr when possible.
#if 201703 <= YY_CPLUSPLUS
# define YY_CONSTEXPR constexpr
#else
# define YY_CONSTEXPR
#endif



#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                            \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif
/* XFL Rules */
enum yyruletype
{
   entry_point_statements = 2,
   statements_statement = 3,
   statements_statements_statement = 4,
   statements_extra_status_rule = 5,
   statement_nl = 6,
   statement_decl_nl = 7,
   decl_function = 8,
   decl_domain_assign_op_expr = 9,
   decl_id_list_assign_op_expr = 10,
   decl_lp_id_list_rp_assign_op_expr = 11,
   decl_lb_id_list_rb_assign_op_expr = 12,
   decl_if_statement = 13,
   decl_api_statement = 14,
   decl_iteration_statement = 15,
   decl_direct_declarator = 16,
   primary_id_identifier = 17,
   postfix_id_primary_id = 18,
   postfix_id_postfix_id_lp_rp = 19,
   postfix_id_postfix_id_lp_expr_rp = 20,
   postfix_id_postfix_id_dot_primary_id = 21,
   postfix_ids_postfix_id = 22,
   postfix_ids_postfix_ids_postfix_id = 23,
   id_list_postfix_ids = 24,
   id_list_id_list_coma_postfix_ids = 25,
   extra_status_rule_lhs = 26,
   extra_status_rule_expr_quote = 27,
   extra_status_rule_transpose_xt = 28,
   extra_status_rule_dot_xt = 29,
   extra_status_rule_eval_xt = 30,
   extra_status_rule_var_xt = 31,
   extra_status_rule_dom_xt = 32,
   lhs_lhs = 33,
   dot_xt_dot_xt = 34,
   eval_xt_eval_xt = 35,
   transpose_xt_transpose_xt = 36,
   var_xt_var_xt = 37,
   dom_xt_dom_xt = 38,
   expr_quote_expr_quote = 39,
   function_def_identifier_lp_args_expr_list_rp_colon_def_empty_return_math_expr = 40,
   function_def_identifier_lp_args_expr_list_rp_colon_def_statements_return_math_expr = 41,
   def_empty_empty = 42,
   def_statements_def_statement = 43,
   def_statements_def_statements_def_statement = 44,
   def_statement_nl = 45,
   def_statement_id_list_assign_op_expr_nl = 46,
   def_statement_lp_id_list_rp_assign_op_expr_nl = 47,
   iteration_statement_for_identifier_in_range_lp_identifier_rp_colon_nl_expr = 48,
   iteration_statement_for_identifier_in_range_lp_identifier_rp_colon_expr = 49,
   if_statement_if_lp_expr_rp_expr = 50,
   api_statement_plot_lp_expr_rp = 51,
   api_statement_save_lp_expr_rp = 52,
   api_statement_solve_lp_expr_rp = 53,
   api_statement_project_lp_expr_rp = 54,
   api_statement_benchmark_lp_expr_rp = 55,
   direct_declarator_postfix_id = 56,
   domain_dom_dx = 57,
   domain_ext_ds = 58,
   domain_int_ds = 59,
   constant_natural = 60,
   constant_real = 61,
   constant_bool = 62,
   strings_string = 63,
   strings_strings_string = 64,
   id_n_conditional_expr = 65,
   fes_args_identifier_coma_identifier = 66,
   fes_args_identifier_coma_identifier_coma_id_n = 67,
   fes_args_identifier_coma_quote_coma_id_n = 68,
   element_type_point = 69,
   element_type_triangle = 70,
   element_type_quadrilateral = 71,
   element_type_tetrahedron = 72,
   element_type_hexahedron = 73,
   element_type_wedge = 74,
   api_device = 75,
   api_mesh = 76,
   api_function = 77,
   api_unit_hex_mesh = 78,
   api_unit_square_mesh = 79,
   api_finite_element = 80,
   api_function_space_lp_fes_args_rp = 81,
   api_vector_function_space = 82,
   api_expression = 83,
   api_dirichlet_bc = 84,
   api_trial_function = 85,
   api_test_function = 86,
   api_constant_api = 87,
   api_api_statement = 88,
   api_element_type = 89,
   primary_expr_identifier = 90,
   primary_expr_constant = 91,
   primary_expr_domain = 92,
   primary_expr_quote = 93,
   primary_expr_strings = 94,
   primary_expr_lp_expr_rp = 95,
   primary_expr_lb_expr_rb = 96,
   primary_expr_ls_coords_rs = 97,
   form_args_lp_additive_expr_rp = 98,
   grad_expr_grad_op_form_args = 99,
   transpose_expr_transpose_op_form_args = 100,
   pow_expr_postfix_expr_pow_constant = 101,
   pow_expr_postfix_expr_pow_identifier = 102,
   postfix_expr_primary_expr = 103,
   postfix_expr_pow_expr = 104,
   postfix_expr_inner_op_lp_additive_expr_coma_additive_expr_rp = 105,
   postfix_expr_postfix_expr_lb_expr_rb = 106,
   postfix_expr_postfix_expr_lp_rp = 107,
   postfix_expr_postfix_expr_lp_args_expr_list_rp = 108,
   postfix_expr_postfix_expr_dot_primary_expr = 109,
   postfix_expr_grad_expr = 110,
   postfix_expr_transpose_expr = 111,
   postfix_expr_lhs_lp_identifier_rp = 112,
   postfix_expr_rhs_lp_identifier_rp = 113,
   postfix_expr_api = 114,
   unary_expr_postfix_expr = 115,
   unary_expr_unary_op_cast_expr = 116,
   unary_op_mul = 117,
   unary_op_add = 118,
   unary_op_sub = 119,
   unary_op_tilde = 120,
   unary_op_not = 121,
   cast_expr_unary_expr = 122,
   multiplicative_expr_cast_expr = 123,
   multiplicative_expr_multiplicative_expr_mul_cast_expr = 124,
   multiplicative_expr_multiplicative_expr_div_cast_expr = 125,
   multiplicative_expr_multiplicative_expr_mod_cast_expr = 126,
   dot_expr_multiplicative_expr = 127,
   dot_expr_dot_expr_dot_op_multiplicative_expr = 128,
   additive_expr_dot_expr = 129,
   additive_expr_additive_expr_add_dot_expr = 130,
   additive_expr_additive_expr_sub_dot_expr = 131,
   shift_expr_additive_expr = 132,
   shift_expr_shift_expr_left_shift_additive_expr = 133,
   shift_expr_shift_expr_right_shift_additive_expr = 134,
   relational_expr_shift_expr = 135,
   relational_expr_relational_expr_lt_shift_expr = 136,
   relational_expr_relational_expr_gt_shift_expr = 137,
   relational_expr_relational_expr_lt_eq_shift_expr = 138,
   relational_expr_relational_expr_gt_eq_shift_expr = 139,
   equality_expr_relational_expr = 140,
   equality_expr_equality_expr_eq_eq_relational_expr = 141,
   equality_expr_equality_expr_not_eq_relational_expr = 142,
   and_expr_equality_expr = 143,
   and_expr_and_expr_and_equality_expr = 144,
   exclusive_or_expr_and_expr = 145,
   exclusive_or_expr_exclusive_or_expr_xor_and_expr = 146,
   inclusive_or_expr_exclusive_or_expr = 147,
   inclusive_or_expr_inclusive_or_expr_or_exclusive_or_expr = 148,
   logical_and_expr_inclusive_or_expr = 149,
   logical_and_expr_logical_and_expr_and_and_inclusive_or_expr = 150,
   logical_or_expr_logical_and_expr = 151,
   logical_or_expr_logical_or_expr_or_or_logical_and_expr = 152,
   conditional_expr_logical_or_expr = 153,
   conditional_expr_logical_or_expr_question_expr_colon_conditional_expr = 154,
   assign_expr_conditional_expr = 155,
   assign_expr_postfix_expr_assign_op_assign_expr = 156,
   assign_op_eq = 157,
   assign_op_add_eq = 158,
   assign_op_sub_eq = 159,
   assign_op_mul_eq = 160,
   assign_op_div_eq = 161,
   assign_op_mod_eq = 162,
   assign_op_xor_eq = 163,
   assign_op_and_eq = 164,
   assign_op_or_eq = 165,
   assign_op_left_eq = 166,
   assign_op_right_eq = 167,
   expr_assign_expr = 168,
   expr_expr_coma_assign_expr = 169,
   args_expr_list_assign_expr = 170,
   args_expr_list_args_expr_list_coma_assign_expr = 171,
   args_expr_list_args_expr_list_coma_nl_assign_expr = 172,
   coord_lp_constant_coma_constant_rp = 173,
   coords_coord = 174,
   coords_coords_colon_coord = 175,
   primary_math_expr_identifier = 176,
   primary_math_expr_constant = 177,
   primary_math_expr_domain = 178,
   primary_math_expr_quote = 179,
   primary_math_expr_lp_math_expr_rp = 180,
   primary_math_expr_lb_id_list_rb = 181,
   dot_math_expr_inner_op_lp_additive_expr_coma_additive_expr_rp = 182,
   postfix_math_expr_primary_math_expr = 183,
   postfix_math_expr_postfix_math_expr_lb_math_expr_rb = 184,
   postfix_math_expr_postfix_math_expr_lp_argument_math_expr_list_rp = 185,
   postfix_math_expr_postfix_math_expr_dot_identifier = 186,
   postfix_math_expr_postfix_math_expr_inc_op = 187,
   postfix_math_expr_postfix_math_expr_dec_op = 188,
   postfix_math_expr_postfix_math_expr_pow_constant = 189,
   postfix_math_expr_dot_math_expr = 190,
   postfix_math_expr_grad_op_lp_additive_math_expr_rp = 191,
   argument_math_expr_list_assign_math_expr = 192,
   argument_math_expr_list_argument_math_expr_list_coma_assign_math_expr = 193,
   unary_math_expr_postfix_math_expr = 194,
   unary_math_expr_inc_op_unary_math_expr = 195,
   unary_math_expr_dec_op_unary_math_expr = 196,
   unary_math_expr_mod_unary_math_expr = 197,
   unary_math_expr_unary_math_op_unary_math_expr = 198,
   unary_math_op_mul = 199,
   unary_math_op_add = 200,
   unary_math_op_sub = 201,
   unary_math_op_and = 202,
   unary_math_op_tilde = 203,
   unary_math_op_not = 204,
   multiplicative_math_expr_unary_math_expr = 205,
   multiplicative_math_expr_multiplicative_math_expr_mul_unary_math_expr = 206,
   multiplicative_math_expr_multiplicative_math_expr_div_unary_math_expr = 207,
   multiplicative_math_expr_multiplicative_math_expr_mod_unary_math_expr = 208,
   additive_math_expr_multiplicative_math_expr = 209,
   additive_math_expr_additive_math_expr_add_multiplicative_math_expr = 210,
   additive_math_expr_additive_math_expr_sub_multiplicative_math_expr = 211,
   shift_math_expr_additive_math_expr = 212,
   shift_math_expr_shift_math_expr_left_shift_additive_math_expr = 213,
   shift_math_expr_shift_math_expr_right_shift_additive_math_expr = 214,
   relational_math_expr_shift_math_expr = 215,
   relational_math_expr_relational_math_expr_lt_shift_math_expr = 216,
   relational_math_expr_relational_math_expr_gt_shift_math_expr = 217,
   relational_math_expr_relational_math_expr_lt_eq_shift_math_expr = 218,
   relational_math_expr_relational_math_expr_gt_eq_shift_math_expr = 219,
   equality_math_expr_relational_math_expr = 220,
   equality_math_expr_equality_math_expr_eq_eq_relational_math_expr = 221,
   equality_math_expr_equality_math_expr_not_eq_relational_math_expr = 222,
   and_math_expr_equality_math_expr = 223,
   and_math_expr_and_math_expr_and_equality_math_expr = 224,
   exclusive_or_math_expr_and_math_expr = 225,
   exclusive_or_math_expr_exclusive_or_math_expr_xor_and_math_expr = 226,
   inclusive_or_math_expr_exclusive_or_math_expr = 227,
   inclusive_or_math_expr_inclusive_or_math_expr_or_exclusive_or_math_expr = 228,
   logical_and_math_expr_inclusive_or_math_expr = 229,
   logical_and_math_expr_logical_and_math_expr_and_and_inclusive_or_math_expr = 230,
   logical_or_math_expr_logical_and_math_expr = 231,
   logical_or_math_expr_logical_or_math_expr_or_or_logical_and_math_expr = 232,
   conditional_math_expr_logical_or_math_expr = 233,
   assign_math_expr_conditional_math_expr = 234,
   assign_math_expr_unary_math_expr_assign_math_op_assign_math_expr = 235,
   assign_math_op_eq = 236,
   assign_math_op_add_eq = 237,
   assign_math_op_sub_eq = 238,
   assign_math_op_mul_eq = 239,
   assign_math_op_div_eq = 240,
   assign_math_op_mod_eq = 241,
   assign_math_op_xor_eq = 242,
   assign_math_op_and_eq = 243,
   assign_math_op_or_eq = 244,
   assign_math_op_left_eq = 245,
   assign_math_op_right_eq = 246,
   math_expr_assign_math_expr = 247,
   math_expr_math_expr_coma_assign_math_expr = 248,
   yynrules // should be the same as YYNRULES + 1
};
typedef enum yyruletype yyrule_kind_t;

/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif

namespace yy
{


/// A point in a source file.
class position
{
public:
   /// Type for file name.
   typedef const std::string filename_type;
   /// Type for line and column numbers.
   typedef int counter_type;

   /// Construct a position.
   explicit position (filename_type* f = YY_NULLPTR,
                      counter_type l = 1,
                      counter_type c = 1)
      : filename (f)
      , line (l)
      , column (c)
   {}


   /// Initialization.
   void initialize (filename_type* fn = YY_NULLPTR,
                    counter_type l = 1,
                    counter_type c = 1)
   {
      filename = fn;
      line = l;
      column = c;
   }

   /** \name Line and Column related manipulators
    ** \{ */
   /// (line related) Advance to the COUNT next lines.
   void lines (counter_type count = 1)
   {
      if (count)
      {
         column = 1;
         line = add_ (line, count, 1);
      }
   }

   /// (column related) Advance to the COUNT next columns.
   void columns (counter_type count = 1)
   {
      column = add_ (column, count, 1);
   }
   /** \} */

   /// File name to which this position refers.
   filename_type* filename;
   /// Current line number.
   counter_type line;
   /// Current column number.
   counter_type column;

private:
   /// Compute max (min, lhs+rhs).
   static counter_type add_ (counter_type lhs, counter_type rhs, counter_type min)
   {
      return lhs + rhs < min ? min : lhs + rhs;
   }
};

/// Add \a width columns, in place.
inline position&
operator+= (position& res, position::counter_type width)
{
   res.columns (width);
   return res;
}

/// Add \a width columns.
inline position
operator+ (position res, position::counter_type width)
{
   return res += width;
}

/// Subtract \a width columns, in place.
inline position&
operator-= (position& res, position::counter_type width)
{
   return res += -width;
}

/// Subtract \a width columns.
inline position
operator- (position res, position::counter_type width)
{
   return res -= width;
}

/** \brief Intercept output stream redirection.
 ** \param ostr the destination output stream
 ** \param pos a reference to the position to redirect
 */
template <typename YYChar>
std::basic_ostream<YYChar>&
operator<< (std::basic_ostream<YYChar>& ostr, const position& pos)
{
   if (pos.filename)
   {
      ostr << *pos.filename << ':';
   }
   return ostr << pos.line << '.' << pos.column;
}

/// Two points in a source file.
class location
{
public:
   /// Type for file name.
   typedef position::filename_type filename_type;
   /// Type for line and column numbers.
   typedef position::counter_type counter_type;

   /// Construct a location from \a b to \a e.
   location (const position& b, const position& e)
      : begin (b)
      , end (e)
   {}

   /// Construct a 0-width location in \a p.
   explicit location (const position& p = position ())
      : begin (p)
      , end (p)
   {}

   /// Construct a 0-width location in \a f, \a l, \a c.
   explicit location (filename_type* f,
                      counter_type l = 1,
                      counter_type c = 1)
      : begin (f, l, c)
      , end (f, l, c)
   {}


   /// Initialization.
   void initialize (filename_type* f = YY_NULLPTR,
                    counter_type l = 1,
                    counter_type c = 1)
   {
      begin.initialize (f, l, c);
      end = begin;
   }

   /** \name Line and Column related manipulators
    ** \{ */
public:
   /// Reset initial location to final location.
   void step ()
   {
      begin = end;
   }

   /// Extend the current location to the COUNT next columns.
   void columns (counter_type count = 1)
   {
      end += count;
   }

   /// Extend the current location to the COUNT next lines.
   void lines (counter_type count = 1)
   {
      end.lines (count);
   }
   /** \} */


public:
   /// Beginning of the located region.
   position begin;
   /// End of the located region.
   position end;
};

/// Join two locations, in place.
inline location&
operator+= (location& res, const location& end)
{
   res.end = end.end;
   return res;
}

/// Join two locations.
inline location
operator+ (location res, const location& end)
{
   return res += end;
}

/// Add \a width columns to the end position, in place.
inline location&
operator+= (location& res, location::counter_type width)
{
   res.columns (width);
   return res;
}

/// Add \a width columns to the end position.
inline location
operator+ (location res, location::counter_type width)
{
   return res += width;
}

/// Subtract \a width columns to the end position, in place.
inline location&
operator-= (location& res, location::counter_type width)
{
   return res += -width;
}

/// Subtract \a width columns to the end position.
inline location
operator- (location res, location::counter_type width)
{
   return res -= width;
}

/** \brief Intercept output stream redirection.
 ** \param ostr the destination output stream
 ** \param loc a reference to the location to redirect
 **
 ** Avoid duplicate information.
 */
template <typename YYChar>
std::basic_ostream<YYChar>&
operator<< (std::basic_ostream<YYChar>& ostr, const location& loc)
{
   location::counter_type end_col
      = 0 < loc.end.column ? loc.end.column - 1 : 0;
   ostr << loc.begin;
   if (loc.end.filename
       && (!loc.begin.filename
           || *loc.begin.filename != *loc.end.filename))
   {
      ostr << '-' << loc.end.filename << ':' << loc.end.line << '.' << end_col;
   }
   else if (loc.begin.line < loc.end.line)
   {
      ostr << '-' << loc.end.line << '.' << end_col;
   }
   else if (loc.begin.column < end_col)
   {
      ostr << '-' << end_col;
   }
   return ostr;
}


/// A Bison parser.
class parser
{
public:
#ifndef YYSTYPE
   /// Symbol semantic values.
   typedef  Node*  semantic_type;
#else
   typedef YYSTYPE semantic_type;
#endif
   /// Symbol locations.
   typedef location location_type;

   /// Syntax errors thrown from user actions.
   struct syntax_error : std::runtime_error
   {
      syntax_error (const location_type& l, const std::string& m)
         : std::runtime_error (m)
         , location (l)
      {}

      syntax_error (const syntax_error& s)
         : std::runtime_error (s.what ())
         , location (s.location)
      {}

      ~syntax_error () YY_NOEXCEPT YY_NOTHROW;

      location_type location;
   };

   /// Token kinds.
   struct token
   {
      enum token_kind_type
      {
         YYEMPTY = -2,
         YYEOF = 0,                     // "end of file"
         YYerror = 1,                   // error
         YYUNDEF = 2,                   // "invalid token"
         LL_SHIFT = 3,                  // LL_SHIFT
         NL = 4,                        // NL
         AS = 5,                        // AS
         DEF = 6,                       // DEF
         FROM = 7,                      // FROM
         IMPORT = 8,                    // IMPORT
         RETURN = 9,                    // RETURN
         PLOT = 10,                     // PLOT
         BENCHMARK = 11,                // BENCHMARK
         SAVE = 12,                     // SAVE
         SOLVE = 13,                    // SOLVE
         PROJECT = 14,                  // PROJECT
         STRING = 15,                   // STRING
         QUOTE = 16,                    // QUOTE
         IF = 17,                       // IF
         FOR = 18,                      // FOR
         IN = 19,                       // IN
         RANGE = 20,                    // RANGE
         DOT_OP = 21,                   // DOT_OP
         INNER_OP = 22,                 // INNER_OP
         GRAD_OP = 23,                  // GRAD_OP
         TRANSPOSE_OP = 24,             // TRANSPOSE_OP
         LHS = 25,                      // LHS
         RHS = 26,                      // RHS
         DEVICE = 27,                   // DEVICE
         MESH = 28,                     // MESH
         FINITE_ELEMENT = 29,           // FINITE_ELEMENT
         UNIT_SQUARE_MESH = 30,         // UNIT_SQUARE_MESH
         UNIT_HEX_MESH = 31,            // UNIT_HEX_MESH
         FUNCTION = 32,                 // FUNCTION
         FUNCTION_SPACE = 33,           // FUNCTION_SPACE
         VECTOR_FUNCTION_SPACE = 34,    // VECTOR_FUNCTION_SPACE
         EXPRESSION = 35,               // EXPRESSION
         DIRICHLET_BC = 36,             // DIRICHLET_BC
         TRIAL_FUNCTION = 37,           // TRIAL_FUNCTION
         TEST_FUNCTION = 38,            // TEST_FUNCTION
         CONSTANT_API = 39,             // CONSTANT_API
         POINT = 40,                    // POINT
         SEGMENT = 41,                  // SEGMENT
         TRIANGLE = 42,                 // TRIANGLE
         QUADRILATERAL = 43,            // QUADRILATERAL
         TETRAHEDRON = 44,              // TETRAHEDRON
         HEXAHEDRON = 45,               // HEXAHEDRON
         WEDGE = 46,                    // WEDGE
         OR_OR = 47,                    // OR_OR
         AND_AND = 48,                  // AND_AND
         DOM_DX = 49,                   // DOM_DX
         EXT_DS = 50,                   // EXT_DS
         INT_DS = 51,                   // INT_DS
         EQ_EQ = 52,                    // EQ_EQ
         ADD_EQ = 53,                   // ADD_EQ
         SUB_EQ = 54,                   // SUB_EQ
         MUL_EQ = 55,                   // MUL_EQ
         DIV_EQ = 56,                   // DIV_EQ
         MOD_EQ = 57,                   // MOD_EQ
         XOR_EQ = 58,                   // XOR_EQ
         AND_EQ = 59,                   // AND_EQ
         OR_EQ = 60,                    // OR_EQ
         LEFT_EQ = 61,                  // LEFT_EQ
         RIGHT_EQ = 62,                 // RIGHT_EQ
         NATURAL = 63,                  // NATURAL
         REAL = 64,                     // REAL
         BOOL = 65,                     // BOOL
         IDENTIFIER = 66,               // IDENTIFIER
         GT = 67,                       // GT
         LT = 68,                       // LT
         EQ = 69,                       // EQ
         ADD = 70,                      // ADD
         SUB = 71,                      // SUB
         MUL = 72,                      // MUL
         DIV = 73,                      // DIV
         POW = 74,                      // POW
         LS = 75,                       // LS
         RS = 76,                       // RS
         LP = 77,                       // LP
         RP = 78,                       // RP
         LB = 79,                       // LB
         RB = 80,                       // RB
         COMA = 81,                     // COMA
         APOSTROPHE = 82,               // APOSTROPHE
         COLON = 83,                    // COLON
         DOT = 84,                      // DOT
         MOD = 85,                      // MOD
         TILDE = 86,                    // TILDE
         LEFT_SHIFT = 87,               // LEFT_SHIFT
         RIGHT_SHIFT = 88,              // RIGHT_SHIFT
         LT_EQ = 89,                    // LT_EQ
         GT_EQ = 90,                    // GT_EQ
         NOT_EQ = 91,                   // NOT_EQ
         AND = 92,                      // AND
         XOR = 93,                      // XOR
         OR = 94,                       // OR
         QUESTION = 95,                 // QUESTION
         NOT = 96,                      // NOT
         INC_OP = 97,                   // INC_OP
         DEC_OP = 98,                   // DEC_OP
         TRANSPOSE_XT = 99,             // TRANSPOSE_XT
         DOT_XT = 100,                  // DOT_XT
         EVAL_XT = 101,                 // EVAL_XT
         GRAD_XT = 102,                 // GRAD_XT
         VAR_XT = 103,                  // VAR_XT
         DOM_XT = 104,                  // DOM_XT
         EXPR_QUOTE = 105,              // EXPR_QUOTE
         EMPTY = 106                    // EMPTY
      };
      /// Backward compatibility alias (Bison 3.6).
      typedef token_kind_type yytokentype;
   };

   /// Token kind, as returned by yylex.
   typedef token::yytokentype token_kind_type;

   /// Backward compatibility alias (Bison 3.6).
   typedef token_kind_type token_type;

   /// Symbol kinds.
   struct symbol_kind
   {
      enum symbol_kind_type
      {
         YYNTOKENS = 107, ///< Number of tokens.
         S_YYEMPTY = -2,
         S_YYEOF = 0,                             // "end of file"
         S_YYerror = 1,                           // error
         S_YYUNDEF = 2,                           // "invalid token"
         S_LL_SHIFT = 3,                          // LL_SHIFT
         S_NL = 4,                                // NL
         S_AS = 5,                                // AS
         S_DEF = 6,                               // DEF
         S_FROM = 7,                              // FROM
         S_IMPORT = 8,                            // IMPORT
         S_RETURN = 9,                            // RETURN
         S_PLOT = 10,                             // PLOT
         S_BENCHMARK = 11,                        // BENCHMARK
         S_SAVE = 12,                             // SAVE
         S_SOLVE = 13,                            // SOLVE
         S_PROJECT = 14,                          // PROJECT
         S_STRING = 15,                           // STRING
         S_QUOTE = 16,                            // QUOTE
         S_IF = 17,                               // IF
         S_FOR = 18,                              // FOR
         S_IN = 19,                               // IN
         S_RANGE = 20,                            // RANGE
         S_DOT_OP = 21,                           // DOT_OP
         S_INNER_OP = 22,                         // INNER_OP
         S_GRAD_OP = 23,                          // GRAD_OP
         S_TRANSPOSE_OP = 24,                     // TRANSPOSE_OP
         S_LHS = 25,                              // LHS
         S_RHS = 26,                              // RHS
         S_DEVICE = 27,                           // DEVICE
         S_MESH = 28,                             // MESH
         S_FINITE_ELEMENT = 29,                   // FINITE_ELEMENT
         S_UNIT_SQUARE_MESH = 30,                 // UNIT_SQUARE_MESH
         S_UNIT_HEX_MESH = 31,                    // UNIT_HEX_MESH
         S_FUNCTION = 32,                         // FUNCTION
         S_FUNCTION_SPACE = 33,                   // FUNCTION_SPACE
         S_VECTOR_FUNCTION_SPACE = 34,            // VECTOR_FUNCTION_SPACE
         S_EXPRESSION = 35,                       // EXPRESSION
         S_DIRICHLET_BC = 36,                     // DIRICHLET_BC
         S_TRIAL_FUNCTION = 37,                   // TRIAL_FUNCTION
         S_TEST_FUNCTION = 38,                    // TEST_FUNCTION
         S_CONSTANT_API = 39,                     // CONSTANT_API
         S_POINT = 40,                            // POINT
         S_SEGMENT = 41,                          // SEGMENT
         S_TRIANGLE = 42,                         // TRIANGLE
         S_QUADRILATERAL = 43,                    // QUADRILATERAL
         S_TETRAHEDRON = 44,                      // TETRAHEDRON
         S_HEXAHEDRON = 45,                       // HEXAHEDRON
         S_WEDGE = 46,                            // WEDGE
         S_OR_OR = 47,                            // OR_OR
         S_AND_AND = 48,                          // AND_AND
         S_DOM_DX = 49,                           // DOM_DX
         S_EXT_DS = 50,                           // EXT_DS
         S_INT_DS = 51,                           // INT_DS
         S_EQ_EQ = 52,                            // EQ_EQ
         S_ADD_EQ = 53,                           // ADD_EQ
         S_SUB_EQ = 54,                           // SUB_EQ
         S_MUL_EQ = 55,                           // MUL_EQ
         S_DIV_EQ = 56,                           // DIV_EQ
         S_MOD_EQ = 57,                           // MOD_EQ
         S_XOR_EQ = 58,                           // XOR_EQ
         S_AND_EQ = 59,                           // AND_EQ
         S_OR_EQ = 60,                            // OR_EQ
         S_LEFT_EQ = 61,                          // LEFT_EQ
         S_RIGHT_EQ = 62,                         // RIGHT_EQ
         S_NATURAL = 63,                          // NATURAL
         S_REAL = 64,                             // REAL
         S_BOOL = 65,                             // BOOL
         S_IDENTIFIER = 66,                       // IDENTIFIER
         S_GT = 67,                               // GT
         S_LT = 68,                               // LT
         S_EQ = 69,                               // EQ
         S_ADD = 70,                              // ADD
         S_SUB = 71,                              // SUB
         S_MUL = 72,                              // MUL
         S_DIV = 73,                              // DIV
         S_POW = 74,                              // POW
         S_LS = 75,                               // LS
         S_RS = 76,                               // RS
         S_LP = 77,                               // LP
         S_RP = 78,                               // RP
         S_LB = 79,                               // LB
         S_RB = 80,                               // RB
         S_COMA = 81,                             // COMA
         S_APOSTROPHE = 82,                       // APOSTROPHE
         S_COLON = 83,                            // COLON
         S_DOT = 84,                              // DOT
         S_MOD = 85,                              // MOD
         S_TILDE = 86,                            // TILDE
         S_LEFT_SHIFT = 87,                       // LEFT_SHIFT
         S_RIGHT_SHIFT = 88,                      // RIGHT_SHIFT
         S_LT_EQ = 89,                            // LT_EQ
         S_GT_EQ = 90,                            // GT_EQ
         S_NOT_EQ = 91,                           // NOT_EQ
         S_AND = 92,                              // AND
         S_XOR = 93,                              // XOR
         S_OR = 94,                               // OR
         S_QUESTION = 95,                         // QUESTION
         S_NOT = 96,                              // NOT
         S_INC_OP = 97,                           // INC_OP
         S_DEC_OP = 98,                           // DEC_OP
         S_TRANSPOSE_XT = 99,                     // TRANSPOSE_XT
         S_DOT_XT = 100,                          // DOT_XT
         S_EVAL_XT = 101,                         // EVAL_XT
         S_GRAD_XT = 102,                         // GRAD_XT
         S_VAR_XT = 103,                          // VAR_XT
         S_DOM_XT = 104,                          // DOM_XT
         S_EXPR_QUOTE = 105,                      // EXPR_QUOTE
         S_EMPTY = 106,                           // EMPTY
         S_YYACCEPT = 107,                        // $accept
         S_entry_point = 108,                     // entry_point
         S_statements = 109,                      // statements
         S_statement = 110,                       // statement
         S_decl = 111,                            // decl
         S_primary_id = 112,                      // primary_id
         S_postfix_id = 113,                      // postfix_id
         S_postfix_ids = 114,                     // postfix_ids
         S_id_list = 115,                         // id_list
         S_extra_status_rule = 116,               // extra_status_rule
         S_lhs = 117,                             // lhs
         S_dot_xt = 118,                          // dot_xt
         S_eval_xt = 119,                         // eval_xt
         S_transpose_xt = 120,                    // transpose_xt
         S_var_xt = 121,                          // var_xt
         S_dom_xt = 122,                          // dom_xt
         S_expr_quote = 123,                      // expr_quote
         S_function = 124,                        // function
         S_def_empty = 125,                       // def_empty
         S_def_statements = 126,                  // def_statements
         S_def_statement = 127,                   // def_statement
         S_iteration_statement = 128,             // iteration_statement
         S_if_statement = 129,                    // if_statement
         S_api_statement = 130,                   // api_statement
         S_direct_declarator = 131,               // direct_declarator
         S_domain = 132,                          // domain
         S_constant = 133,                        // constant
         S_strings = 134,                         // strings
         S_id_n = 135,                            // id_n
         S_fes_args = 136,                        // fes_args
         S_element_type = 137,                    // element_type
         S_api = 138,                             // api
         S_primary_expr = 139,                    // primary_expr
         S_form_args = 140,                       // form_args
         S_grad_expr = 141,                       // grad_expr
         S_transpose_expr = 142,                  // transpose_expr
         S_pow_expr = 143,                        // pow_expr
         S_postfix_expr = 144,                    // postfix_expr
         S_unary_expr = 145,                      // unary_expr
         S_unary_op = 146,                        // unary_op
         S_cast_expr = 147,                       // cast_expr
         S_multiplicative_expr = 148,             // multiplicative_expr
         S_dot_expr = 149,                        // dot_expr
         S_additive_expr = 150,                   // additive_expr
         S_shift_expr = 151,                      // shift_expr
         S_relational_expr = 152,                 // relational_expr
         S_equality_expr = 153,                   // equality_expr
         S_and_expr = 154,                        // and_expr
         S_exclusive_or_expr = 155,               // exclusive_or_expr
         S_inclusive_or_expr = 156,               // inclusive_or_expr
         S_logical_and_expr = 157,                // logical_and_expr
         S_logical_or_expr = 158,                 // logical_or_expr
         S_conditional_expr = 159,                // conditional_expr
         S_assign_expr = 160,                     // assign_expr
         S_assign_op = 161,                       // assign_op
         S_expr = 162,                            // expr
         S_args_expr_list = 163,                  // args_expr_list
         S_coord = 164,                           // coord
         S_coords = 165,                          // coords
         S_primary_math_expr = 166,               // primary_math_expr
         S_dot_math_expr = 167,                   // dot_math_expr
         S_postfix_math_expr = 168,               // postfix_math_expr
         S_argument_math_expr_list = 169,         // argument_math_expr_list
         S_unary_math_expr = 170,                 // unary_math_expr
         S_unary_math_op = 171,                   // unary_math_op
         S_multiplicative_math_expr = 172,        // multiplicative_math_expr
         S_additive_math_expr = 173,              // additive_math_expr
         S_shift_math_expr = 174,                 // shift_math_expr
         S_relational_math_expr = 175,            // relational_math_expr
         S_equality_math_expr = 176,              // equality_math_expr
         S_and_math_expr = 177,                   // and_math_expr
         S_exclusive_or_math_expr = 178,          // exclusive_or_math_expr
         S_inclusive_or_math_expr = 179,          // inclusive_or_math_expr
         S_logical_and_math_expr = 180,           // logical_and_math_expr
         S_logical_or_math_expr = 181,            // logical_or_math_expr
         S_conditional_math_expr = 182,           // conditional_math_expr
         S_assign_math_expr = 183,                // assign_math_expr
         S_assign_math_op = 184,                  // assign_math_op
         S_math_expr = 185                        // math_expr
      };
   };

   /// (Internal) symbol kind.
   typedef symbol_kind::symbol_kind_type symbol_kind_type;

   /// The number of tokens.
   static const symbol_kind_type YYNTOKENS = symbol_kind::YYNTOKENS;

   /// A complete symbol.
   ///
   /// Expects its Base type to provide access to the symbol kind
   /// via kind ().
   ///
   /// Provide access to semantic value and location.
   template <typename Base>
   struct basic_symbol : Base
   {
      /// Alias to Base.
      typedef Base super_type;

      /// Default constructor.
      basic_symbol ()
         : value ()
         , location ()
      {}

#if 201103L <= YY_CPLUSPLUS
      /// Move constructor.
      basic_symbol (basic_symbol&& that)
         : Base (std::move (that))
         , value (std::move (that.value))
         , location (std::move (that.location))
      {}
#endif

      /// Copy constructor.
      basic_symbol (const basic_symbol& that);
      /// Constructor for valueless symbols.
      basic_symbol (typename Base::kind_type t,
                    YY_MOVE_REF (location_type) l);

      /// Constructor for symbols with semantic value.
      basic_symbol (typename Base::kind_type t,
                    YY_RVREF (semantic_type) v,
                    YY_RVREF (location_type) l);

      /// Destroy the symbol.
      ~basic_symbol ()
      {
         clear ();
      }

      /// Destroy contents, and record that is empty.
      void clear () YY_NOEXCEPT
      {
         Base::clear ();
      }

      /// The user-facing name of this symbol.
      const char *name () const YY_NOEXCEPT
      {
         return parser::symbol_name (this->kind ());
      }

      /// Backward compatibility (Bison 3.6).
      symbol_kind_type type_get () const YY_NOEXCEPT;

      /// Whether empty.
      bool empty () const YY_NOEXCEPT;

      /// Destructive move, \a s is emptied into this.
      void move (basic_symbol& s);

      /// The semantic value.
      semantic_type value;

      /// The location.
      location_type location;

   private:
#if YY_CPLUSPLUS < 201103L
      /// Assignment operator.
      basic_symbol& operator= (const basic_symbol& that);
#endif
   };

   /// Type access provider for token (enum) based symbols.
   struct by_kind
   {
      /// Default constructor.
      by_kind ();

#if 201103L <= YY_CPLUSPLUS
      /// Move constructor.
      by_kind (by_kind&& that);
#endif

      /// Copy constructor.
      by_kind (const by_kind& that);

      /// The symbol kind as needed by the constructor.
      typedef token_kind_type kind_type;

      /// Constructor from (external) token numbers.
      by_kind (kind_type t);

      /// Record that this symbol is empty.
      void clear () YY_NOEXCEPT;

      /// Steal the symbol kind from \a that.
      void move (by_kind& that);

      /// The (internal) type number (corresponding to \a type).
      /// \a empty when empty.
      symbol_kind_type kind () const YY_NOEXCEPT;

      /// Backward compatibility (Bison 3.6).
      symbol_kind_type type_get () const YY_NOEXCEPT;

      /// The symbol kind.
      /// \a S_YYEMPTY when empty.
      symbol_kind_type kind_;
   };

   /// Backward compatibility for a private implementation detail (Bison 3.6).
   typedef by_kind by_type;

   /// "External" symbols: returned by the scanner.
   struct symbol_type : basic_symbol<by_kind>
   {};

   /// Build a parser object.
   parser (xfl &ufl_yyarg);
   virtual ~parser ();

#if 201103L <= YY_CPLUSPLUS
   /// Non copyable.
   parser (const parser&) = delete;
   /// Non copyable.
   parser& operator= (const parser&) = delete;
#endif

   /// Parse.  An alias for parse ().
   /// \returns  0 iff parsing succeeded.
   int operator() ();

   /// Parse.
   /// \returns  0 iff parsing succeeded.
   virtual int parse ();

#if YYDEBUG
   /// The current debugging stream.
   std::ostream& debug_stream () const YY_ATTRIBUTE_PURE;
   /// Set the current debugging stream.
   void set_debug_stream (std::ostream &);

   /// Type for debugging levels.
   typedef int debug_level_type;
   /// The current debugging level.
   debug_level_type debug_level () const YY_ATTRIBUTE_PURE;
   /// Set the current debugging level.
   void set_debug_level (debug_level_type l);
#endif

   /// Report a syntax error.
   /// \param loc    where the syntax error is found.
   /// \param msg    a description of the syntax error.
   virtual void error (const location_type& loc, const std::string& msg);

   /// Report a syntax error.
   void error (const syntax_error& err);

   /// The user-facing name of the symbol whose (internal) number is
   /// YYSYMBOL.  No bounds checking.
   static const char *symbol_name (symbol_kind_type yysymbol);



   class context
   {
   public:
      context (const parser& yyparser, const symbol_type& yyla);
      const symbol_type& lookahead () const YY_NOEXCEPT { return yyla_; }
      symbol_kind_type token () const YY_NOEXCEPT { return yyla_.kind (); }
      const location_type& location () const YY_NOEXCEPT { return yyla_.location; }

      /// Put in YYARG at most YYARGN of the expected tokens, and return the
      /// number of tokens stored in YYARG.  If YYARG is null, return the
      /// number of expected tokens (guaranteed to be less than YYNTOKENS).
      int expected_tokens (symbol_kind_type yyarg[], int yyargn) const;

   private:
      const parser& yyparser_;
      const symbol_type& yyla_;
   };

private:
#if YY_CPLUSPLUS < 201103L
   /// Non copyable.
   parser (const parser&);
   /// Non copyable.
   parser& operator= (const parser&);
#endif

   /// Check the lookahead yytoken.
   /// \returns  true iff the token will be eventually shifted.
   bool yy_lac_check_ (symbol_kind_type yytoken) const;
   /// Establish the initial context if no initial context currently exists.
   /// \returns  true iff the token will be eventually shifted.
   bool yy_lac_establish_ (symbol_kind_type yytoken);
   /// Discard any previous initial lookahead context because of event.
   /// \param event  the event which caused the lookahead to be discarded.
   ///               Only used for debbuging output.
   void yy_lac_discard_ (const char* event);

   /// Stored state numbers (used for stacks).
   typedef short state_type;

   /// The arguments of the error message.
   int yy_syntax_error_arguments_ (const context& yyctx,
                                   symbol_kind_type yyarg[], int yyargn) const;

   /// Generate an error message.
   /// \param yyctx     the context in which the error occurred.
   virtual std::string yysyntax_error_ (const context& yyctx) const;
   /// Compute post-reduction state.
   /// \param yystate   the current state
   /// \param yysym     the nonterminal to push on the stack
   static state_type yy_lr_goto_state_ (state_type yystate, int yysym);

   /// Whether the given \c yypact_ value indicates a defaulted state.
   /// \param yyvalue   the value to check
   static bool yy_pact_value_is_default_ (int yyvalue);

   /// Whether the given \c yytable_ value indicates a syntax error.
   /// \param yyvalue   the value to check
   static bool yy_table_value_is_error_ (int yyvalue);

   static const short yypact_ninf_;
   static const signed char yytable_ninf_;

   /// Convert a scanner token kind \a t to a symbol kind.
   /// In theory \a t should be a token_kind_type, but character literals
   /// are valid, yet not members of the token_type enum.
   static symbol_kind_type yytranslate_ (int t);



   // Tables.
   // YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   // STATE-NUM.
   static const short yypact_[];

   // YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   // Performed when YYTABLE does not specify something else to do.  Zero
   // means the default is an error.
   static const unsigned char yydefact_[];

   // YYPGOTO[NTERM-NUM].
   static const short yypgoto_[];

   // YYDEFGOTO[NTERM-NUM].
   static const short yydefgoto_[];

   // YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   // positive, shift that token.  If negative, reduce the rule whose
   // number is the opposite.  If YYTABLE_NINF, syntax error.
   static const short yytable_[];

   static const short yycheck_[];

   // YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   // symbol of state STATE-NUM.
   static const unsigned char yystos_[];

   // YYR1[YYN] -- Symbol number of symbol that rule YYN derives.
   static const unsigned char yyr1_[];

   // YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.
   static const signed char yyr2_[];


#if YYDEBUG
   // YYRLINE[YYN] -- Source line where rule number YYN was defined.
   static const short yyrline_[];
   /// Report on the debug stream that the rule \a r is going to be reduced.
   virtual void yy_reduce_print_ (int r) const;
   /// Print the state stack on the debug stream.
   virtual void yy_stack_print_ () const;

   /// Debugging level.
   int yydebug_;
   /// Debug stream.
   std::ostream* yycdebug_;

   /// \brief Display a symbol kind, value and location.
   /// \param yyo    The output stream.
   /// \param yysym  The symbol.
   template <typename Base>
   void yy_print_ (std::ostream& yyo, const basic_symbol<Base>& yysym) const;
#endif

   /// \brief Reclaim the memory associated to a symbol.
   /// \param yymsg     Why this token is reclaimed.
   ///                  If null, print nothing.
   /// \param yysym     The symbol.
   template <typename Base>
   void yy_destroy_ (const char* yymsg, basic_symbol<Base>& yysym) const;

private:
   /// Type access provider for state based symbols.
   struct by_state
   {
      /// Default constructor.
      by_state () YY_NOEXCEPT;

      /// The symbol kind as needed by the constructor.
      typedef state_type kind_type;

      /// Constructor.
      by_state (kind_type s) YY_NOEXCEPT;

      /// Copy constructor.
      by_state (const by_state& that) YY_NOEXCEPT;

      /// Record that this symbol is empty.
      void clear () YY_NOEXCEPT;

      /// Steal the symbol kind from \a that.
      void move (by_state& that);

      /// The symbol kind (corresponding to \a state).
      /// \a symbol_kind::S_YYEMPTY when empty.
      symbol_kind_type kind () const YY_NOEXCEPT;

      /// The state number used to denote an empty symbol.
      /// We use the initial state, as it does not have a value.
      enum { empty_state = 0 };

      /// The state.
      /// \a empty when empty.
      state_type state;
   };

   /// "Internal" symbol: element of the stack.
   struct stack_symbol_type : basic_symbol<by_state>
   {
      /// Superclass.
      typedef basic_symbol<by_state> super_type;
      /// Construct an empty symbol.
      stack_symbol_type ();
      /// Move or copy construction.
      stack_symbol_type (YY_RVREF (stack_symbol_type) that);
      /// Steal the contents from \a sym to build this.
      stack_symbol_type (state_type s, YY_MOVE_REF (symbol_type) sym);
#if YY_CPLUSPLUS < 201103L
      /// Assignment, needed by push_back by some old implementations.
      /// Moves the contents of that.
      stack_symbol_type& operator= (stack_symbol_type& that);

      /// Assignment, needed by push_back by other implementations.
      /// Needed by some other old implementations.
      stack_symbol_type& operator= (const stack_symbol_type& that);
#endif
   };

   /// A stack with random access from its top.
   template <typename T, typename S = std::vector<T> >
   class stack
   {
   public:
      // Hide our reversed order.
      typedef typename S::iterator iterator;
      typedef typename S::const_iterator const_iterator;
      typedef typename S::size_type size_type;
      typedef typename std::ptrdiff_t index_type;

      stack (size_type n = 200)
         : seq_ (n)
      {}

#if 201103L <= YY_CPLUSPLUS
      /// Non copyable.
      stack (const stack&) = delete;
      /// Non copyable.
      stack& operator= (const stack&) = delete;
#endif

      /// Random access.
      ///
      /// Index 0 returns the topmost element.
      const T&
      operator[] (index_type i) const
      {
         return seq_[size_type (size () - 1 - i)];
      }

      /// Random access.
      ///
      /// Index 0 returns the topmost element.
      T&
      operator[] (index_type i)
      {
         return seq_[size_type (size () - 1 - i)];
      }

      /// Steal the contents of \a t.
      ///
      /// Close to move-semantics.
      void
      push (YY_MOVE_REF (T) t)
      {
         seq_.push_back (T ());
         operator[] (0).move (t);
      }

      /// Pop elements from the stack.
      void
      pop (std::ptrdiff_t n = 1) YY_NOEXCEPT
      {
         for (; 0 < n; --n)
         {
            seq_.pop_back ();
         }
      }

      /// Pop all elements from the stack.
      void
      clear () YY_NOEXCEPT
      {
         seq_.clear ();
      }

      /// Number of elements on the stack.
      index_type
      size () const YY_NOEXCEPT
      {
         return index_type (seq_.size ());
      }

      /// Iterator on top of the stack (going downwards).
      const_iterator
      begin () const YY_NOEXCEPT
      {
         return seq_.begin ();
      }

      /// Bottom of the stack.
      const_iterator
      end () const YY_NOEXCEPT
      {
         return seq_.end ();
      }

      /// Present a slice of the top of a stack.
      class slice
      {
      public:
         slice (const stack& stack, index_type range)
            : stack_ (stack)
            , range_ (range)
         {}

         const T&
         operator[] (index_type i) const
         {
            return stack_[range_ - i];
         }

      private:
         const stack& stack_;
         index_type range_;
      };

   private:
#if YY_CPLUSPLUS < 201103L
      /// Non copyable.
      stack (const stack&);
      /// Non copyable.
      stack& operator= (const stack&);
#endif
      /// The wrapped container.
      S seq_;
   };


   /// Stack type.
public: // XFL
   typedef stack<stack_symbol_type> stack_type;
private: // XFL

   /// The stack.
   stack_type yystack_;
   /// The stack for LAC.
   /// Logically, the yy_lac_stack's lifetime is confined to the function
   /// yy_lac_check_. We just store it as a member of this class to hold
   /// on to the memory and to avoid frequent reallocations.
   /// Since yy_lac_check_ is const, this member must be mutable.
   mutable std::vector<state_type> yylac_stack_;
   /// Whether an initial LAC context was established.
   bool yy_lac_established_;


   /// Push a new state on the stack.
   /// \param m    a debug message to display
   ///             if null, no trace is output.
   /// \param sym  the symbol
   /// \warning the contents of \a s.value is stolen.
   void yypush_ (const char* m, YY_MOVE_REF (stack_symbol_type) sym);

   /// Push a new look ahead token on the state on the stack.
   /// \param m    a debug message to display
   ///             if null, no trace is output.
   /// \param s    the state
   /// \param sym  the symbol (for its value and location).
   /// \warning the contents of \a sym.value is stolen.
   void yypush_ (const char* m, state_type s, YY_MOVE_REF (symbol_type) sym);

   /// Pop \a n symbols from the stack.
   void yypop_ (int n = 1);

   /// Constants.
   enum
   {
      yylast_ = 922,     ///< Last index in yytable_.
      yynnts_ = 79,  ///< Number of nonterminal symbols.
      yyfinal_ = 56 ///< Termination state number.
   };


   // User arguments.
   xfl &ufl;

};


} // yy


// "%code provides" blocks.

#include "xfl_mid.hpp"
#define YY_DECL int yylex(Node* *yylval, yy::location*, xfl &ufl)



#endif // !YY_YY_USERS_CAMIER1_HOME_SAWMILL_UFL_XFL_BUILD_SRC_XFL_Y_HPP_INCLUDED
