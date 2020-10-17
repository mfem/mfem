// A Bison parser, made by GNU Bison 3.7.3.

// Skeleton interface for Bison LALR(1) parsers in C++

// Copyright (C) 2002-2015, 2018-2020 Free Software Foundation, Inc.

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


/**
 ** \file /Users/camier1/home/sawmill/ufl/xfl/build/src/xfl.Y.hpp
 ** Define the yy::parser class.
 */

// C++ LALR(1) parser skeleton written by Akim Demaille.

// DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
// especially those whose name start with YY_ or yy_.  They are
// private implementation details that can be changed or removed.

#ifndef YY_YY_USERS_CAMIER1_HOME_SAWMILL_UFL_XFL_BUILD_SRC_XFL_Y_HPP_INCLUDED
# define YY_YY_USERS_CAMIER1_HOME_SAWMILL_UFL_XFL_BUILD_SRC_XFL_Y_HPP_INCLUDED
// "%code requires" blocks.
 
#ifndef XFL_C
class xfl;
#endif


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
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
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
enum yyruletype {
   entry_point_statements = 2,
   statements_statement = 3,
   statements_statements_statement = 4,
   statement_nl = 5,
   statement_decl_nl = 6,
   decl_function = 7,
   decl_domain_assign_op_expr = 8,
   decl_id_list_assign_op_expr = 9,
   decl_lp_id_list_rp_assign_op_expr = 10,
   decl_lb_id_list_rb_assign_op_expr = 11,
   decl_iteration_statement = 12,
   decl_direct_declarator = 13,
   postfix_id_identifier = 14,
   id_list_postfix_id = 15,
   id_list_id_list_coma_postfix_id = 16,
   function_def_identifier_lp_args_expr_list_rp_colon_def_statements_return_math_expr = 17,
   def_statements_def_statement = 18,
   def_statements_def_statements_def_statement = 19,
   def_statement_nl = 20,
   def_statement_id_list_assign_op_expr_nl = 21,
   def_statement_lp_id_list_rp_assign_op_expr_nl = 22,
   direct_declarator_postfix_id = 23,
   direct_declarator_direct_declarator_lp_rp = 24,
   direct_declarator_direct_declarator_lp_expr_rp = 25,
   iteration_statement_for_identifier_in_range_lp_identifier_rp_colon_nl_expr = 26,
   domain_dom_dx = 27,
   domain_ext_ds = 28,
   domain_int_ds = 29,
   constant_natural = 30,
   constant_real = 31,
   strings_string = 32,
   strings_strings_string = 33,
   api_unit_square_mesh = 34,
   api_function = 35,
   api_function_space = 36,
   api_expression = 37,
   api_dirichlet_bc = 38,
   api_trial_function = 39,
   api_test_function = 40,
   api_constant_api = 41,
   primary_expr_identifier = 42,
   primary_expr_constant = 43,
   primary_expr_domain = 44,
   primary_expr_quote = 45,
   primary_expr_strings = 46,
   primary_expr_lp_expr_rp = 47,
   primary_expr_lb_expr_rb = 48,
   primary_expr_ls_coords_rs = 49,
   primary_expr_api = 50,
   pow_expr_postfix_expr_pow_constant = 51,
   pow_expr_postfix_expr_pow_identifier = 52,
   dot_expr_dot_op_lp_additive_expr_coma_additive_expr_rp = 53,
   dot_expr_inner_op_lp_additive_expr_coma_additive_expr_rp = 54,
   postfix_expr_primary_expr = 55,
   postfix_expr_pow_expr = 56,
   postfix_expr_postfix_expr_lb_expr_rb = 57,
   postfix_expr_postfix_expr_lp_rp = 58,
   postfix_expr_postfix_expr_lp_args_expr_list_rp = 59,
   postfix_expr_postfix_expr_dot_primary_expr = 60,
   postfix_expr_dot_expr = 61,
   postfix_expr_grad_op_lp_additive_expr_rp = 62,
   postfix_expr_lhs_lp_identifier_rp = 63,
   postfix_expr_rhs_lp_identifier_rp = 64,
   unary_expr_postfix_expr = 65,
   unary_expr_unary_op_cast_expr = 66,
   unary_op_mul = 67,
   unary_op_add = 68,
   unary_op_sub = 69,
   unary_op_tilde = 70,
   unary_op_not = 71,
   cast_expr_unary_expr = 72,
   multiplicative_expr_cast_expr = 73,
   multiplicative_expr_multiplicative_expr_mul_cast_expr = 74,
   multiplicative_expr_multiplicative_expr_div_cast_expr = 75,
   multiplicative_expr_multiplicative_expr_mod_cast_expr = 76,
   additive_expr_multiplicative_expr = 77,
   additive_expr_additive_expr_add_multiplicative_expr = 78,
   additive_expr_additive_expr_sub_multiplicative_expr = 79,
   shift_expr_additive_expr = 80,
   shift_expr_shift_expr_left_shift_additive_expr = 81,
   shift_expr_shift_expr_right_shift_additive_expr = 82,
   relational_expr_shift_expr = 83,
   relational_expr_relational_expr_lt_shift_expr = 84,
   relational_expr_relational_expr_gt_shift_expr = 85,
   relational_expr_relational_expr_lt_eq_shift_expr = 86,
   relational_expr_relational_expr_gt_eq_shift_expr = 87,
   equality_expr_relational_expr = 88,
   equality_expr_equality_expr_eq_eq_relational_expr = 89,
   equality_expr_equality_expr_not_eq_relational_expr = 90,
   and_expr_equality_expr = 91,
   and_expr_and_expr_and_equality_expr = 92,
   exclusive_or_expr_and_expr = 93,
   exclusive_or_expr_exclusive_or_expr_xor_and_expr = 94,
   inclusive_or_expr_exclusive_or_expr = 95,
   inclusive_or_expr_inclusive_or_expr_or_exclusive_or_expr = 96,
   logical_and_expr_inclusive_or_expr = 97,
   logical_and_expr_logical_and_expr_and_and_inclusive_or_expr = 98,
   logical_or_expr_logical_and_expr = 99,
   logical_or_expr_logical_or_expr_or_or_logical_and_expr = 100,
   conditional_expr_logical_or_expr = 101,
   conditional_expr_logical_or_expr_question_expr_colon_conditional_expr = 102,
   assign_expr_conditional_expr = 103,
   assign_expr_postfix_expr_assign_op_assign_expr = 104,
   assign_op_eq = 105,
   assign_op_add_eq = 106,
   assign_op_sub_eq = 107,
   assign_op_mul_eq = 108,
   assign_op_div_eq = 109,
   assign_op_mod_eq = 110,
   assign_op_xor_eq = 111,
   assign_op_and_eq = 112,
   assign_op_or_eq = 113,
   assign_op_left_eq = 114,
   assign_op_right_eq = 115,
   expr_assign_expr = 116,
   expr_expr_coma_assign_expr = 117,
   args_expr_list_assign_expr = 118,
   args_expr_list_args_expr_list_coma_assign_expr = 119,
   args_expr_list_args_expr_list_coma_nl_assign_expr = 120,
   coord_lp_constant_coma_constant_rp = 121,
   coords_coord = 122,
   coords_coords_colon_coord = 123,
   primary_math_expr_identifier = 124,
   primary_math_expr_constant = 125,
   primary_math_expr_domain = 126,
   primary_math_expr_quote = 127,
   primary_math_expr_lp_math_expr_rp = 128,
   primary_math_expr_lb_id_list_rb = 129,
   dot_math_expr_dot_op_lp_additive_expr_coma_additive_expr_rp = 130,
   dot_math_expr_inner_op_lp_additive_expr_coma_additive_expr_rp = 131,
   postfix_math_expr_primary_math_expr = 132,
   postfix_math_expr_postfix_math_expr_lb_math_expr_rb = 133,
   postfix_math_expr_postfix_math_expr_lp_argument_math_expr_list_rp = 134,
   postfix_math_expr_postfix_math_expr_dot_identifier = 135,
   postfix_math_expr_postfix_math_expr_inc_op = 136,
   postfix_math_expr_postfix_math_expr_dec_op = 137,
   postfix_math_expr_postfix_math_expr_pow_constant = 138,
   postfix_math_expr_dot_math_expr = 139,
   postfix_math_expr_grad_op_lp_additive_math_expr_rp = 140,
   argument_math_expr_list_assign_math_expr = 141,
   argument_math_expr_list_argument_math_expr_list_coma_assign_math_expr = 142,
   unary_math_expr_postfix_math_expr = 143,
   unary_math_expr_inc_op_unary_math_expr = 144,
   unary_math_expr_dec_op_unary_math_expr = 145,
   unary_math_expr_mod_unary_math_expr = 146,
   unary_math_expr_unary_math_op_unary_math_expr = 147,
   unary_math_op_mul = 148,
   unary_math_op_add = 149,
   unary_math_op_sub = 150,
   unary_math_op_and = 151,
   unary_math_op_tilde = 152,
   unary_math_op_not = 153,
   multiplicative_math_expr_unary_math_expr = 154,
   multiplicative_math_expr_multiplicative_math_expr_mul_unary_math_expr = 155,
   multiplicative_math_expr_multiplicative_math_expr_div_unary_math_expr = 156,
   multiplicative_math_expr_multiplicative_math_expr_mod_unary_math_expr = 157,
   additive_math_expr_multiplicative_math_expr = 158,
   additive_math_expr_additive_math_expr_add_multiplicative_math_expr = 159,
   additive_math_expr_additive_math_expr_sub_multiplicative_math_expr = 160,
   shift_math_expr_additive_math_expr = 161,
   shift_math_expr_shift_math_expr_left_shift_additive_math_expr = 162,
   shift_math_expr_shift_math_expr_right_shift_additive_math_expr = 163,
   relational_math_expr_shift_math_expr = 164,
   relational_math_expr_relational_math_expr_lt_shift_math_expr = 165,
   relational_math_expr_relational_math_expr_gt_shift_math_expr = 166,
   relational_math_expr_relational_math_expr_lt_eq_shift_math_expr = 167,
   relational_math_expr_relational_math_expr_gt_eq_shift_math_expr = 168,
   equality_math_expr_relational_math_expr = 169,
   equality_math_expr_equality_math_expr_eq_eq_relational_math_expr = 170,
   equality_math_expr_equality_math_expr_not_eq_relational_math_expr = 171,
   and_math_expr_equality_math_expr = 172,
   and_math_expr_and_math_expr_and_equality_math_expr = 173,
   exclusive_or_math_expr_and_math_expr = 174,
   exclusive_or_math_expr_exclusive_or_math_expr_xor_and_math_expr = 175,
   inclusive_or_math_expr_exclusive_or_math_expr = 176,
   inclusive_or_math_expr_inclusive_or_math_expr_or_exclusive_or_math_expr = 177,
   logical_and_math_expr_inclusive_or_math_expr = 178,
   logical_and_math_expr_logical_and_math_expr_and_and_inclusive_or_math_expr = 179,
   logical_or_math_expr_logical_and_math_expr = 180,
   logical_or_math_expr_logical_or_math_expr_or_or_logical_and_math_expr = 181,
   conditional_math_expr_logical_or_math_expr = 182,
   assign_math_expr_conditional_math_expr = 183,
   assign_math_expr_unary_math_expr_assign_math_op_assign_math_expr = 184,
   assign_math_op_eq = 185,
   assign_math_op_add_eq = 186,
   assign_math_op_sub_eq = 187,
   assign_math_op_mul_eq = 188,
   assign_math_op_div_eq = 189,
   assign_math_op_mod_eq = 190,
   assign_math_op_xor_eq = 191,
   assign_math_op_and_eq = 192,
   assign_math_op_or_eq = 193,
   assign_math_op_left_eq = 194,
   assign_math_op_right_eq = 195,
   math_expr_assign_math_expr = 196,
   math_expr_math_expr_coma_assign_math_expr = 197,
   yynrules // should be the same as YYNRULES + 1
};
typedef enum yyruletype yyrule_kind_t;

/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif

namespace yy {


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
      ostr << *pos.filename << ':';
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
      ostr << '-' << loc.end.filename << ':' << loc.end.line << '.' << end_col;
    else if (loc.begin.line < loc.end.line)
      ostr << '-' << loc.end.line << '.' << end_col;
    else if (loc.begin.column < end_col)
      ostr << '-' << end_col;
    return ostr;
  }


  /// A Bison parser.
  class parser
  {
  public:
#ifndef YYSTYPE
    /// Symbol semantic values.
    typedef struct Node* semantic_type;
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
    STRING = 10,                   // STRING
    QUOTE = 11,                    // QUOTE
    FOR = 12,                      // FOR
    IN = 13,                       // IN
    RANGE = 14,                    // RANGE
    DOT_OP = 15,                   // DOT_OP
    INNER_OP = 16,                 // INNER_OP
    GRAD_OP = 17,                  // GRAD_OP
    LHS = 18,                      // LHS
    RHS = 19,                      // RHS
    UNIT_SQUARE_MESH = 20,         // UNIT_SQUARE_MESH
    FUNCTION = 21,                 // FUNCTION
    FUNCTION_SPACE = 22,           // FUNCTION_SPACE
    EXPRESSION = 23,               // EXPRESSION
    DIRICHLET_BC = 24,             // DIRICHLET_BC
    TRIAL_FUNCTION = 25,           // TRIAL_FUNCTION
    TEST_FUNCTION = 26,            // TEST_FUNCTION
    CONSTANT_API = 27,             // CONSTANT_API
    DOM_DX = 28,                   // DOM_DX
    EXT_DS = 29,                   // EXT_DS
    INT_DS = 30,                   // INT_DS
    EQ_EQ = 31,                    // EQ_EQ
    ADD_EQ = 32,                   // ADD_EQ
    SUB_EQ = 33,                   // SUB_EQ
    MUL_EQ = 34,                   // MUL_EQ
    DIV_EQ = 35,                   // DIV_EQ
    MOD_EQ = 36,                   // MOD_EQ
    XOR_EQ = 37,                   // XOR_EQ
    AND_EQ = 38,                   // AND_EQ
    OR_EQ = 39,                    // OR_EQ
    LEFT_EQ = 40,                  // LEFT_EQ
    RIGHT_EQ = 41,                 // RIGHT_EQ
    NATURAL = 42,                  // NATURAL
    REAL = 43,                     // REAL
    IDENTIFIER = 44,               // IDENTIFIER
    GT = 45,                       // GT
    LT = 46,                       // LT
    EQ = 47,                       // EQ
    ADD = 48,                      // ADD
    SUB = 49,                      // SUB
    MUL = 50,                      // MUL
    DIV = 51,                      // DIV
    POW = 52,                      // POW
    LS = 53,                       // LS
    RS = 54,                       // RS
    LP = 55,                       // LP
    RP = 56,                       // RP
    LB = 57,                       // LB
    RB = 58,                       // RB
    COMA = 59,                     // COMA
    APOSTROPHE = 60,               // APOSTROPHE
    COLON = 61,                    // COLON
    DOT = 62,                      // DOT
    MOD = 63,                      // MOD
    TILDE = 64,                    // TILDE
    LEFT_SHIFT = 65,               // LEFT_SHIFT
    RIGHT_SHIFT = 66,              // RIGHT_SHIFT
    LT_EQ = 67,                    // LT_EQ
    GT_EQ = 68,                    // GT_EQ
    NOT_EQ = 69,                   // NOT_EQ
    AND = 70,                      // AND
    XOR = 71,                      // XOR
    OR = 72,                       // OR
    AND_AND = 73,                  // AND_AND
    OR_OR = 74,                    // OR_OR
    QUESTION = 75,                 // QUESTION
    NOT = 76,                      // NOT
    INC_OP = 77,                   // INC_OP
    DEC_OP = 78,                   // DEC_OP
    EMPTY = 79                     // EMPTY
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
        YYNTOKENS = 80, ///< Number of tokens.
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
        S_STRING = 10,                           // STRING
        S_QUOTE = 11,                            // QUOTE
        S_FOR = 12,                              // FOR
        S_IN = 13,                               // IN
        S_RANGE = 14,                            // RANGE
        S_DOT_OP = 15,                           // DOT_OP
        S_INNER_OP = 16,                         // INNER_OP
        S_GRAD_OP = 17,                          // GRAD_OP
        S_LHS = 18,                              // LHS
        S_RHS = 19,                              // RHS
        S_UNIT_SQUARE_MESH = 20,                 // UNIT_SQUARE_MESH
        S_FUNCTION = 21,                         // FUNCTION
        S_FUNCTION_SPACE = 22,                   // FUNCTION_SPACE
        S_EXPRESSION = 23,                       // EXPRESSION
        S_DIRICHLET_BC = 24,                     // DIRICHLET_BC
        S_TRIAL_FUNCTION = 25,                   // TRIAL_FUNCTION
        S_TEST_FUNCTION = 26,                    // TEST_FUNCTION
        S_CONSTANT_API = 27,                     // CONSTANT_API
        S_DOM_DX = 28,                           // DOM_DX
        S_EXT_DS = 29,                           // EXT_DS
        S_INT_DS = 30,                           // INT_DS
        S_EQ_EQ = 31,                            // EQ_EQ
        S_ADD_EQ = 32,                           // ADD_EQ
        S_SUB_EQ = 33,                           // SUB_EQ
        S_MUL_EQ = 34,                           // MUL_EQ
        S_DIV_EQ = 35,                           // DIV_EQ
        S_MOD_EQ = 36,                           // MOD_EQ
        S_XOR_EQ = 37,                           // XOR_EQ
        S_AND_EQ = 38,                           // AND_EQ
        S_OR_EQ = 39,                            // OR_EQ
        S_LEFT_EQ = 40,                          // LEFT_EQ
        S_RIGHT_EQ = 41,                         // RIGHT_EQ
        S_NATURAL = 42,                          // NATURAL
        S_REAL = 43,                             // REAL
        S_IDENTIFIER = 44,                       // IDENTIFIER
        S_GT = 45,                               // GT
        S_LT = 46,                               // LT
        S_EQ = 47,                               // EQ
        S_ADD = 48,                              // ADD
        S_SUB = 49,                              // SUB
        S_MUL = 50,                              // MUL
        S_DIV = 51,                              // DIV
        S_POW = 52,                              // POW
        S_LS = 53,                               // LS
        S_RS = 54,                               // RS
        S_LP = 55,                               // LP
        S_RP = 56,                               // RP
        S_LB = 57,                               // LB
        S_RB = 58,                               // RB
        S_COMA = 59,                             // COMA
        S_APOSTROPHE = 60,                       // APOSTROPHE
        S_COLON = 61,                            // COLON
        S_DOT = 62,                              // DOT
        S_MOD = 63,                              // MOD
        S_TILDE = 64,                            // TILDE
        S_LEFT_SHIFT = 65,                       // LEFT_SHIFT
        S_RIGHT_SHIFT = 66,                      // RIGHT_SHIFT
        S_LT_EQ = 67,                            // LT_EQ
        S_GT_EQ = 68,                            // GT_EQ
        S_NOT_EQ = 69,                           // NOT_EQ
        S_AND = 70,                              // AND
        S_XOR = 71,                              // XOR
        S_OR = 72,                               // OR
        S_AND_AND = 73,                          // AND_AND
        S_OR_OR = 74,                            // OR_OR
        S_QUESTION = 75,                         // QUESTION
        S_NOT = 76,                              // NOT
        S_INC_OP = 77,                           // INC_OP
        S_DEC_OP = 78,                           // DEC_OP
        S_EMPTY = 79,                            // EMPTY
        S_YYACCEPT = 80,                         // $accept
        S_entry_point = 81,                      // entry_point
        S_statements = 82,                       // statements
        S_statement = 83,                        // statement
        S_decl = 84,                             // decl
        S_postfix_id = 85,                       // postfix_id
        S_id_list = 86,                          // id_list
        S_function = 87,                         // function
        S_def_statements = 88,                   // def_statements
        S_def_statement = 89,                    // def_statement
        S_direct_declarator = 90,                // direct_declarator
        S_iteration_statement = 91,              // iteration_statement
        S_domain = 92,                           // domain
        S_constant = 93,                         // constant
        S_strings = 94,                          // strings
        S_api = 95,                              // api
        S_primary_expr = 96,                     // primary_expr
        S_pow_expr = 97,                         // pow_expr
        S_dot_expr = 98,                         // dot_expr
        S_postfix_expr = 99,                     // postfix_expr
        S_unary_expr = 100,                      // unary_expr
        S_unary_op = 101,                        // unary_op
        S_cast_expr = 102,                       // cast_expr
        S_multiplicative_expr = 103,             // multiplicative_expr
        S_additive_expr = 104,                   // additive_expr
        S_shift_expr = 105,                      // shift_expr
        S_relational_expr = 106,                 // relational_expr
        S_equality_expr = 107,                   // equality_expr
        S_and_expr = 108,                        // and_expr
        S_exclusive_or_expr = 109,               // exclusive_or_expr
        S_inclusive_or_expr = 110,               // inclusive_or_expr
        S_logical_and_expr = 111,                // logical_and_expr
        S_logical_or_expr = 112,                 // logical_or_expr
        S_conditional_expr = 113,                // conditional_expr
        S_assign_expr = 114,                     // assign_expr
        S_assign_op = 115,                       // assign_op
        S_expr = 116,                            // expr
        S_args_expr_list = 117,                  // args_expr_list
        S_coord = 118,                           // coord
        S_coords = 119,                          // coords
        S_primary_math_expr = 120,               // primary_math_expr
        S_dot_math_expr = 121,                   // dot_math_expr
        S_postfix_math_expr = 122,               // postfix_math_expr
        S_argument_math_expr_list = 123,         // argument_math_expr_list
        S_unary_math_expr = 124,                 // unary_math_expr
        S_unary_math_op = 125,                   // unary_math_op
        S_multiplicative_math_expr = 126,        // multiplicative_math_expr
        S_additive_math_expr = 127,              // additive_math_expr
        S_shift_math_expr = 128,                 // shift_math_expr
        S_relational_math_expr = 129,            // relational_math_expr
        S_equality_math_expr = 130,              // equality_math_expr
        S_and_math_expr = 131,                   // and_math_expr
        S_exclusive_or_math_expr = 132,          // exclusive_or_math_expr
        S_inclusive_or_math_expr = 133,          // inclusive_or_math_expr
        S_logical_and_math_expr = 134,           // logical_and_math_expr
        S_logical_or_math_expr = 135,            // logical_or_math_expr
        S_conditional_math_expr = 136,           // conditional_math_expr
        S_assign_math_expr = 137,                // assign_math_expr
        S_assign_math_op = 138,                  // assign_math_op
        S_math_expr = 139                        // math_expr
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
      void clear ()
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
      void clear ();

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
      const symbol_type& lookahead () const { return yyla_; }
      symbol_kind_type token () const { return yyla_.kind (); }
      const location_type& location () const { return yyla_.location; }

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
          seq_.pop_back ();
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
      yylast_ = 649,     ///< Last index in yytable_.
      yynnts_ = 60,  ///< Number of nonterminal symbols.
      yyfinal_ = 25 ///< Termination state number.
    };


    // User arguments.
    xfl &ufl;

  };


} // yy


// "%code provides" blocks.

#ifndef XFL_C
using TOK = yy::parser::token;
#else
using TOK = yytokentype;
#endif

constexpr int YYTOKENSHIFT(int T) { return T - TOK::LL_SHIFT + 3; }
constexpr int YYTOKENUNSHIFT(int T) { return T + TOK::LL_SHIFT - 3;}
#ifndef XFL_C
#define YY_DECL int yylex (Node* *yylval, yy::location*, xfl &ker)
#endif



#endif // !YY_YY_USERS_CAMIER1_HOME_SAWMILL_UFL_XFL_BUILD_SRC_XFL_Y_HPP_INCLUDED
