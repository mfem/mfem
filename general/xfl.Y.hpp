/* A Bison parser, made by GNU Bison 3.7.2. */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2020 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>. */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison. */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed. */

#ifndef YY_YY_USERS_CAMIER1_HOME_SAWMILL_UFL_XFL_BUILD_SRC_XFL_Y_HPP_INCLUDED
# define YY_YY_USERS_CAMIER1_HOME_SAWMILL_UFL_XFL_BUILD_SRC_XFL_Y_HPP_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file" */
    YYerror = 256,                 /* error */
    YYUNDEF = 257,                 /* "invalid token" */
    LL_SHIFT = 258,                /* LL_SHIFT */
    NL = 259,                      /* NL */
    AS = 260,                      /* AS */
    DEF = 261,                     /* DEF */
    FROM = 262,                    /* FROM */
    IMPORT = 263,                  /* IMPORT */
    RETURN = 264,                  /* RETURN */
    STRING = 265,                  /* STRING */
    QUOTE = 266,                   /* QUOTE */
    FOR = 267,                     /* FOR */
    IN = 268,                      /* IN */
    RANGE = 269,                   /* RANGE */
    DOT_OP = 270,                  /* DOT_OP */
    INNER_OP = 271,                /* INNER_OP */
    GRAD_OP = 272,                 /* GRAD_OP */
    LHS = 273,                     /* LHS */
    RHS = 274,                     /* RHS */
    UNIT_SQUARE_MESH = 275,        /* UNIT_SQUARE_MESH */
    FUNCTION = 276,                /* FUNCTION */
    FUNCTION_SPACE = 277,          /* FUNCTION_SPACE */
    EXPRESSION = 278,              /* EXPRESSION */
    DIRICHLET_BC = 279,            /* DIRICHLET_BC */
    TRIAL_FUNCTION = 280,          /* TRIAL_FUNCTION */
    TEST_FUNCTION = 281,           /* TEST_FUNCTION */
    CONSTANT_API = 282,            /* CONSTANT_API */
    DOM_DX = 283,                  /* DOM_DX */
    EXT_DS = 284,                  /* EXT_DS */
    INT_DS = 285,                  /* INT_DS */
    EQ_EQ = 286,                   /* EQ_EQ */
    ADD_EQ = 287,                  /* ADD_EQ */
    SUB_EQ = 288,                  /* SUB_EQ */
    MUL_EQ = 289,                  /* MUL_EQ */
    DIV_EQ = 290,                  /* DIV_EQ */
    MOD_EQ = 291,                  /* MOD_EQ */
    XOR_EQ = 292,                  /* XOR_EQ */
    AND_EQ = 293,                  /* AND_EQ */
    OR_EQ = 294,                   /* OR_EQ */
    LEFT_EQ = 295,                 /* LEFT_EQ */
    RIGHT_EQ = 296,                /* RIGHT_EQ */
    NATURAL = 297,                 /* NATURAL */
    REAL = 298,                    /* REAL */
    IDENTIFIER = 299,              /* IDENTIFIER */
    GT = 300,                      /* GT */
    LT = 301,                      /* LT */
    EQ = 302,                      /* EQ */
    ADD = 303,                     /* ADD */
    SUB = 304,                     /* SUB */
    MUL = 305,                     /* MUL */
    DIV = 306,                     /* DIV */
    POW = 307,                     /* POW */
    LS = 308,                      /* LS */
    RS = 309,                      /* RS */
    LP = 310,                      /* LP */
    RP = 311,                      /* RP */
    LB = 312,                      /* LB */
    RB = 313,                      /* RB */
    COMA = 314,                    /* COMA */
    APOSTROPHE = 315,              /* APOSTROPHE */
    COLON = 316,                   /* COLON */
    DOT = 317,                     /* DOT */
    MOD = 318,                     /* MOD */
    TILDE = 319,                   /* TILDE */
    LEFT_SHIFT = 320,              /* LEFT_SHIFT */
    RIGHT_SHIFT = 321,             /* RIGHT_SHIFT */
    LT_EQ = 322,                   /* LT_EQ */
    GT_EQ = 323,                   /* GT_EQ */
    NOT_EQ = 324,                  /* NOT_EQ */
    AND = 325,                     /* AND */
    XOR = 326,                     /* XOR */
    OR = 327,                      /* OR */
    AND_AND = 328,                 /* AND_AND */
    OR_OR = 329,                   /* OR_OR */
    QUESTION = 330,                /* QUESTION */
    NOT = 331,                     /* NOT */
    INC_OP = 332,                  /* INC_OP */
    DEC_OP = 333,                  /* DEC_OP */
    EMPTY = 334                    /* EMPTY */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Rules */
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

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef struct Node* YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (YYSTYPE *root);
/* "%code provides" blocks. */

#define YYTOKENSHIFT(T) (T - LL_SHIFT + 3)
#define YYTOKENUNSHIFT(T) (T + LL_SHIFT - 3)


#endif /* !YY_YY_USERS_CAMIER1_HOME_SAWMILL_UFL_XFL_BUILD_SRC_XFL_Y_HPP_INCLUDED */
