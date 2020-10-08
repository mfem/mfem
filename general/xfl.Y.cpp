/* A Bison parser, made by GNU Bison 3.7.2. */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed. */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.7.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "../data/yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue. */

#include "xfl.hpp"
#define _ yy::rhs<YYN>(&yyval, yyn, yyvsp);

// %token-table: This feature is obsolescent
// %defines
// %define parse.error verbose
// %warning: %token order has to be sync'ed with the lexer's one


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

#include "xfl.Y.hpp"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file" */
  YYSYMBOL_YYerror = 1,                    /* error */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token" */
  YYSYMBOL_LL_SHIFT = 3,                   /* LL_SHIFT */
  YYSYMBOL_NL = 4,                         /* NL */
  YYSYMBOL_AS = 5,                         /* AS */
  YYSYMBOL_DEF = 6,                        /* DEF */
  YYSYMBOL_FROM = 7,                       /* FROM */
  YYSYMBOL_IMPORT = 8,                     /* IMPORT */
  YYSYMBOL_RETURN = 9,                     /* RETURN */
  YYSYMBOL_STRING = 10,                    /* STRING */
  YYSYMBOL_QUOTE = 11,                     /* QUOTE */
  YYSYMBOL_FOR = 12,                       /* FOR */
  YYSYMBOL_IN = 13,                        /* IN */
  YYSYMBOL_RANGE = 14,                     /* RANGE */
  YYSYMBOL_DOT_OP = 15,                    /* DOT_OP */
  YYSYMBOL_INNER_OP = 16,                  /* INNER_OP */
  YYSYMBOL_GRAD_OP = 17,                   /* GRAD_OP */
  YYSYMBOL_LHS = 18,                       /* LHS */
  YYSYMBOL_RHS = 19,                       /* RHS */
  YYSYMBOL_UNIT_SQUARE_MESH = 20,          /* UNIT_SQUARE_MESH */
  YYSYMBOL_FUNCTION = 21,                  /* FUNCTION */
  YYSYMBOL_FUNCTION_SPACE = 22,            /* FUNCTION_SPACE */
  YYSYMBOL_EXPRESSION = 23,                /* EXPRESSION */
  YYSYMBOL_DIRICHLET_BC = 24,              /* DIRICHLET_BC */
  YYSYMBOL_TRIAL_FUNCTION = 25,            /* TRIAL_FUNCTION */
  YYSYMBOL_TEST_FUNCTION = 26,             /* TEST_FUNCTION */
  YYSYMBOL_CONSTANT_API = 27,              /* CONSTANT_API */
  YYSYMBOL_DOM_DX = 28,                    /* DOM_DX */
  YYSYMBOL_EXT_DS = 29,                    /* EXT_DS */
  YYSYMBOL_INT_DS = 30,                    /* INT_DS */
  YYSYMBOL_EQ_EQ = 31,                     /* EQ_EQ */
  YYSYMBOL_ADD_EQ = 32,                    /* ADD_EQ */
  YYSYMBOL_SUB_EQ = 33,                    /* SUB_EQ */
  YYSYMBOL_MUL_EQ = 34,                    /* MUL_EQ */
  YYSYMBOL_DIV_EQ = 35,                    /* DIV_EQ */
  YYSYMBOL_MOD_EQ = 36,                    /* MOD_EQ */
  YYSYMBOL_XOR_EQ = 37,                    /* XOR_EQ */
  YYSYMBOL_AND_EQ = 38,                    /* AND_EQ */
  YYSYMBOL_OR_EQ = 39,                     /* OR_EQ */
  YYSYMBOL_LEFT_EQ = 40,                   /* LEFT_EQ */
  YYSYMBOL_RIGHT_EQ = 41,                  /* RIGHT_EQ */
  YYSYMBOL_NATURAL = 42,                   /* NATURAL */
  YYSYMBOL_REAL = 43,                      /* REAL */
  YYSYMBOL_IDENTIFIER = 44,                /* IDENTIFIER */
  YYSYMBOL_GT = 45,                        /* GT */
  YYSYMBOL_LT = 46,                        /* LT */
  YYSYMBOL_EQ = 47,                        /* EQ */
  YYSYMBOL_ADD = 48,                       /* ADD */
  YYSYMBOL_SUB = 49,                       /* SUB */
  YYSYMBOL_MUL = 50,                       /* MUL */
  YYSYMBOL_DIV = 51,                       /* DIV */
  YYSYMBOL_POW = 52,                       /* POW */
  YYSYMBOL_LS = 53,                        /* LS */
  YYSYMBOL_RS = 54,                        /* RS */
  YYSYMBOL_LP = 55,                        /* LP */
  YYSYMBOL_RP = 56,                        /* RP */
  YYSYMBOL_LB = 57,                        /* LB */
  YYSYMBOL_RB = 58,                        /* RB */
  YYSYMBOL_COMA = 59,                      /* COMA */
  YYSYMBOL_APOSTROPHE = 60,                /* APOSTROPHE */
  YYSYMBOL_COLON = 61,                     /* COLON */
  YYSYMBOL_DOT = 62,                       /* DOT */
  YYSYMBOL_MOD = 63,                       /* MOD */
  YYSYMBOL_TILDE = 64,                     /* TILDE */
  YYSYMBOL_LEFT_SHIFT = 65,                /* LEFT_SHIFT */
  YYSYMBOL_RIGHT_SHIFT = 66,               /* RIGHT_SHIFT */
  YYSYMBOL_LT_EQ = 67,                     /* LT_EQ */
  YYSYMBOL_GT_EQ = 68,                     /* GT_EQ */
  YYSYMBOL_NOT_EQ = 69,                    /* NOT_EQ */
  YYSYMBOL_AND = 70,                       /* AND */
  YYSYMBOL_XOR = 71,                       /* XOR */
  YYSYMBOL_OR = 72,                        /* OR */
  YYSYMBOL_AND_AND = 73,                   /* AND_AND */
  YYSYMBOL_OR_OR = 74,                     /* OR_OR */
  YYSYMBOL_QUESTION = 75,                  /* QUESTION */
  YYSYMBOL_NOT = 76,                       /* NOT */
  YYSYMBOL_INC_OP = 77,                    /* INC_OP */
  YYSYMBOL_DEC_OP = 78,                    /* DEC_OP */
  YYSYMBOL_EMPTY = 79,                     /* EMPTY */
  YYSYMBOL_YYACCEPT = 80,                  /* $accept */
  YYSYMBOL_entry_point = 81,               /* entry_point */
  YYSYMBOL_statements = 82,                /* statements */
  YYSYMBOL_statement = 83,                 /* statement */
  YYSYMBOL_decl = 84,                      /* decl */
  YYSYMBOL_postfix_id = 85,                /* postfix_id */
  YYSYMBOL_id_list = 86,                   /* id_list */
  YYSYMBOL_function = 87,                  /* function */
  YYSYMBOL_def_statements = 88,            /* def_statements */
  YYSYMBOL_def_statement = 89,             /* def_statement */
  YYSYMBOL_direct_declarator = 90,         /* direct_declarator */
  YYSYMBOL_iteration_statement = 91,       /* iteration_statement */
  YYSYMBOL_domain = 92,                    /* domain */
  YYSYMBOL_constant = 93,                  /* constant */
  YYSYMBOL_strings = 94,                   /* strings */
  YYSYMBOL_api = 95,                       /* api */
  YYSYMBOL_primary_expr = 96,              /* primary_expr */
  YYSYMBOL_pow_expr = 97,                  /* pow_expr */
  YYSYMBOL_dot_expr = 98,                  /* dot_expr */
  YYSYMBOL_postfix_expr = 99,              /* postfix_expr */
  YYSYMBOL_unary_expr = 100,               /* unary_expr */
  YYSYMBOL_unary_op = 101,                 /* unary_op */
  YYSYMBOL_cast_expr = 102,                /* cast_expr */
  YYSYMBOL_multiplicative_expr = 103,      /* multiplicative_expr */
  YYSYMBOL_additive_expr = 104,            /* additive_expr */
  YYSYMBOL_shift_expr = 105,               /* shift_expr */
  YYSYMBOL_relational_expr = 106,          /* relational_expr */
  YYSYMBOL_equality_expr = 107,            /* equality_expr */
  YYSYMBOL_and_expr = 108,                 /* and_expr */
  YYSYMBOL_exclusive_or_expr = 109,        /* exclusive_or_expr */
  YYSYMBOL_inclusive_or_expr = 110,        /* inclusive_or_expr */
  YYSYMBOL_logical_and_expr = 111,         /* logical_and_expr */
  YYSYMBOL_logical_or_expr = 112,          /* logical_or_expr */
  YYSYMBOL_conditional_expr = 113,         /* conditional_expr */
  YYSYMBOL_assign_expr = 114,              /* assign_expr */
  YYSYMBOL_assign_op = 115,                /* assign_op */
  YYSYMBOL_expr = 116,                     /* expr */
  YYSYMBOL_args_expr_list = 117,           /* args_expr_list */
  YYSYMBOL_coord = 118,                    /* coord */
  YYSYMBOL_coords = 119,                   /* coords */
  YYSYMBOL_primary_math_expr = 120,        /* primary_math_expr */
  YYSYMBOL_dot_math_expr = 121,            /* dot_math_expr */
  YYSYMBOL_postfix_math_expr = 122,        /* postfix_math_expr */
  YYSYMBOL_argument_math_expr_list = 123,  /* argument_math_expr_list */
  YYSYMBOL_unary_math_expr = 124,          /* unary_math_expr */
  YYSYMBOL_unary_math_op = 125,            /* unary_math_op */
  YYSYMBOL_multiplicative_math_expr = 126, /* multiplicative_math_expr */
  YYSYMBOL_additive_math_expr = 127,       /* additive_math_expr */
  YYSYMBOL_shift_math_expr = 128,          /* shift_math_expr */
  YYSYMBOL_relational_math_expr = 129,     /* relational_math_expr */
  YYSYMBOL_equality_math_expr = 130,       /* equality_math_expr */
  YYSYMBOL_and_math_expr = 131,            /* and_math_expr */
  YYSYMBOL_exclusive_or_math_expr = 132,   /* exclusive_or_math_expr */
  YYSYMBOL_inclusive_or_math_expr = 133,   /* inclusive_or_math_expr */
  YYSYMBOL_logical_and_math_expr = 134,    /* logical_and_math_expr */
  YYSYMBOL_logical_or_math_expr = 135,     /* logical_or_math_expr */
  YYSYMBOL_conditional_math_expr = 136,    /* conditional_math_expr */
  YYSYMBOL_assign_math_expr = 137,         /* assign_math_expr */
  YYSYMBOL_assign_math_op = 138,           /* assign_math_op */
  YYSYMBOL_math_expr = 139                 /* math_expr */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;



/* Unqualified %code blocks. */
 #include "xfc.hpp" 


#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int16 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
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


#ifdef NDEBUG
# define YY_ASSERT(E) ((void) (0 && (E)))
#else
# include <assert.h> /* INFRINGES ON USER NAME SPACE */
# define YY_ASSERT(E) assert (E)
#endif


#if 1

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
# define YYCOPY_NEEDED 1
#endif /* 1 */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  25
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   649

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  80
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  60
/* YYNRULES -- Number of rules.  */
#define YYNRULES  197
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  352

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   334


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined. */
static constexpr yytype_int16 yyrline[] =
{
       0,    55,    55,    57,    57,    59,    59,    61,    62,    63,
      64,    65,    66,    67,    69,    73,    73,    77,    79,    79,
      80,    81,    82,    86,    87,    88,    91,    94,    94,    94,
      97,    97,    99,    99,   102,   103,   104,   105,   106,   107,
     108,   109,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   122,   123,   126,   127,   129,   130,   131,   132,   133,
     134,   135,   136,   137,   138,   140,   140,   142,   142,   142,
     142,   142,   144,   146,   147,   148,   149,   151,   152,   153,
     155,   156,   157,   159,   160,   161,   162,   163,   165,   166,
     167,   169,   170,   172,   173,   175,   176,   178,   179,   181,
     182,   184,   185,   187,   188,   190,   191,   191,   191,   191,
     191,   192,   192,   192,   193,   193,   195,   196,   198,   199,
     200,   202,   204,   204,   209,   210,   211,   212,   213,   214,
     217,   218,   221,   222,   223,   224,   225,   226,   227,   228,
     229,   232,   233,   236,   237,   238,   239,   240,   242,   242,
     242,   242,   242,   242,   244,   245,   246,   247,   249,   250,
     251,   253,   254,   255,   257,   258,   259,   260,   261,   263,
     264,   265,   267,   268,   270,   271,   273,   274,   276,   277,
     279,   280,   282,   284,   285,   287,   288,   288,   288,   288,
     288,   289,   289,   289,   290,   290,   292,   293
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if 1
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  static const char *const yy_sname[] =
  {
  "end of file", "error", "invalid token", "LL_SHIFT", "NL", "AS", "DEF",
  "FROM", "IMPORT", "RETURN", "STRING", "QUOTE", "FOR", "IN", "RANGE",
  "DOT_OP", "INNER_OP", "GRAD_OP", "LHS", "RHS", "UNIT_SQUARE_MESH",
  "FUNCTION", "FUNCTION_SPACE", "EXPRESSION", "DIRICHLET_BC",
  "TRIAL_FUNCTION", "TEST_FUNCTION", "CONSTANT_API", "DOM_DX", "EXT_DS",
  "INT_DS", "EQ_EQ", "ADD_EQ", "SUB_EQ", "MUL_EQ", "DIV_EQ", "MOD_EQ",
  "XOR_EQ", "AND_EQ", "OR_EQ", "LEFT_EQ", "RIGHT_EQ", "NATURAL", "REAL",
  "IDENTIFIER", "GT", "LT", "EQ", "ADD", "SUB", "MUL", "DIV", "POW", "LS",
  "RS", "LP", "RP", "LB", "RB", "COMA", "APOSTROPHE", "COLON", "DOT",
  "MOD", "TILDE", "LEFT_SHIFT", "RIGHT_SHIFT", "LT_EQ", "GT_EQ", "NOT_EQ",
  "AND", "XOR", "OR", "AND_AND", "OR_OR", "QUESTION", "NOT", "INC_OP",
  "DEC_OP", "EMPTY", "$accept", "entry_point", "statements", "statement",
  "decl", "postfix_id", "id_list", "function", "def_statements",
  "def_statement", "direct_declarator", "iteration_statement", "domain",
  "constant", "strings", "api", "primary_expr", "pow_expr", "dot_expr",
  "postfix_expr", "unary_expr", "unary_op", "cast_expr",
  "multiplicative_expr", "additive_expr", "shift_expr", "relational_expr",
  "equality_expr", "and_expr", "exclusive_or_expr", "inclusive_or_expr",
  "logical_and_expr", "logical_or_expr", "conditional_expr", "assign_expr",
  "assign_op", "expr", "args_expr_list", "coord", "coords",
  "primary_math_expr", "dot_math_expr", "postfix_math_expr",
  "argument_math_expr_list", "unary_math_expr", "unary_math_op",
  "multiplicative_math_expr", "additive_math_expr", "shift_math_expr",
  "relational_math_expr", "equality_math_expr", "and_math_expr",
  "exclusive_or_math_expr", "inclusive_or_math_expr",
  "logical_and_math_expr", "logical_or_math_expr", "conditional_math_expr",
  "assign_math_expr", "assign_math_op", "math_expr", YY_NULLPTR
  };
  return yy_sname[yysymbol];
}
#endif

#ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_int16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334
};
#endif

#define YYPACT_NINF (-264)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-24)

#define yytable_value_is_error(Yyn) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM. */
static constexpr yytype_int16 yypact[] =
{
     119,  -264,   -13,    -3,  -264,  -264,  -264,  -264,    18,    18,
      58,   119,  -264,    42,    12,   410,  -264,   -53,  -264,   240,
      26,    62,  -264,    59,    34,  -264,  -264,  -264,  -264,  -264,
    -264,  -264,  -264,  -264,  -264,  -264,  -264,  -264,  -264,    18,
     535,   411,   535,   535,    73,   240,   240,  -264,  -264,  -264,
      56,    88,   105,   111,   118,  -264,  -264,  -264,  -264,  -264,
    -264,  -264,  -264,  -264,  -264,  -264,  -264,  -264,  -264,   130,
     535,   535,  -264,  -264,  -264,  -264,   117,  -264,  -264,  -264,
    -264,   534,  -264,   535,  -264,   -33,    72,   147,    28,    11,
      27,   134,   149,   146,   194,  -264,  -264,   170,  -264,    80,
     170,  -264,    94,   176,   535,   535,   535,   535,   535,   196,
     206,   241,  -264,   -29,   186,   258,  -264,   136,   473,   535,
     592,   535,   120,  -264,   535,   535,   535,   535,   535,   535,
     535,   535,   535,   535,   535,   535,   535,   535,   535,   535,
     535,   535,   535,   535,  -264,   191,   304,   218,   170,   170,
     103,   122,   -35,   255,   257,   259,  -264,   130,  -264,  -264,
    -264,  -264,  -264,   190,   297,  -264,  -264,  -264,  -264,  -264,
     -33,   -33,    72,    72,   147,   147,   147,   147,    28,    28,
      11,    27,   134,   149,   146,    46,  -264,    16,   535,  -264,
     293,   535,   535,  -264,  -264,  -264,   241,  -264,  -264,  -264,
     535,  -264,    18,   410,    15,  -264,  -264,   290,   108,   169,
     311,  -264,   226,   535,   347,  -264,   356,  -264,  -264,  -264,
     240,     2,  -264,   319,   330,   333,  -264,  -264,  -264,  -264,
     347,    18,   347,  -264,  -264,  -264,   347,   347,  -264,  -264,
    -264,  -264,    67,   303,   347,    35,   317,   305,    87,    14,
     299,   323,   326,   327,   325,  -264,  -264,   342,   535,   535,
    -264,   535,   535,   347,   251,   314,  -264,  -264,  -264,   241,
     347,   347,   359,  -264,  -264,  -264,  -264,  -264,  -264,  -264,
    -264,  -264,  -264,  -264,  -264,  -264,   347,  -264,   347,   347,
     347,   347,   347,   347,   347,   347,   347,   347,   347,   347,
     347,   347,   347,   347,   347,   347,   347,   170,     4,   161,
     167,  -264,   179,  -264,  -264,  -264,   253,  -264,   320,  -264,
    -264,  -264,  -264,  -264,    35,    35,   317,   317,   305,   305,
     305,   305,    87,    87,    14,   299,   323,   326,   327,  -264,
    -264,   535,   535,  -264,  -264,   347,  -264,   185,   199,  -264,
    -264,  -264
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error. */
static constexpr yytype_uint8 yydefact[] =
{
       0,     5,     0,     0,    27,    28,    29,    14,     0,     0,
       0,     2,     3,     0,    15,     0,     7,    13,    12,     0,
       0,     0,    15,     0,     0,     1,     4,     6,   106,   107,
     108,   109,   110,   111,   112,   113,   114,   115,   105,     0,
       0,     0,     0,     0,     0,     0,     0,    16,    32,    45,
       0,     0,     0,     0,     0,    34,    35,    36,    37,    38,
      39,    40,    41,    30,    31,    42,    68,    69,    67,     0,
       0,     0,    70,    71,    44,    43,    46,    50,    55,    56,
      61,    65,    72,     0,    73,    77,    80,    83,    88,    91,
      93,    95,    97,    99,   101,   103,   116,     9,    24,     0,
       8,   118,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   122,     0,     0,     0,    33,     0,     0,     0,
       0,     0,    65,    66,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    25,     0,     0,     0,    10,    11,
       0,     0,     0,     0,     0,     0,    49,     0,    47,    48,
      52,    51,    58,     0,     0,    60,   104,    74,    75,    76,
      78,    79,    81,    82,    85,    84,    86,    87,    89,    90,
      92,    94,    96,    98,   100,     0,   117,     0,     0,   119,
       0,     0,     0,    62,    63,    64,     0,   123,    59,    57,
       0,    20,     0,     0,     0,    18,   120,     0,     0,     0,
       0,   102,     0,     0,     0,    19,     0,    53,    54,   121,
       0,     0,   127,     0,     0,     0,   124,   149,   150,   148,
       0,     0,     0,   152,   151,   153,     0,     0,   126,   125,
     132,   139,   143,   154,     0,   158,   161,   164,   169,   172,
     174,   176,   178,   180,   182,   183,   196,    17,     0,     0,
      21,     0,     0,     0,     0,     0,   146,   144,   145,     0,
       0,     0,     0,   136,   137,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   185,     0,   147,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    26,     0,     0,
       0,   154,     0,   128,   129,   138,     0,   141,     0,   135,
     184,   155,   156,   157,   159,   160,   162,   163,   166,   165,
     167,   168,   170,   171,   173,   175,   177,   179,   181,   197,
      22,     0,     0,   140,   134,     0,   133,     0,     0,   142,
     130,   131
};

  /* YYPGOTO[NTERM-NUM]. */
static constexpr yytype_int16 yypgoto[] =
{
    -264,  -264,  -264,   394,  -264,    33,    20,  -264,  -264,   202,
    -264,  -264,     0,  -102,  -264,  -264,   287,  -264,  -264,    -5,
    -264,  -264,   -71,   254,  -103,   127,   248,   271,   274,   270,
     273,   275,  -264,   214,   -42,     3,    -2,   300,   262,  -264,
    -264,  -264,  -264,  -264,  -180,  -264,    95,  -216,   -31,    93,
     114,   150,   153,   116,   157,  -264,  -264,  -263,  -264,  -220
};

  /* YYDEFGOTO[NTERM-NUM]. */
static constexpr yytype_int16 yydefgoto[] =
{
      -1,    10,    11,    12,    13,    22,    15,    16,   204,   205,
      17,    18,    74,    75,    76,    77,    78,    79,    80,   122,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,    96,    40,    97,   102,   112,   113,
     240,   241,   242,   316,   311,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   286,   257
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error. */
static constexpr yytype_int16 yytable[] =
{
      19,   101,    41,   150,   151,   152,   260,   317,   340,   155,
     264,    19,   123,   127,   128,   161,   -23,   124,   125,   201,
     201,   193,    42,   320,   214,   156,   172,   173,    23,    24,
     126,    20,   157,    14,   243,    81,    81,    81,    81,    99,
     100,    21,   135,   339,    14,   299,    27,   312,   104,   105,
     243,   318,   266,   167,   168,   169,   267,   268,    25,     7,
       7,   143,     7,   143,   287,    81,    81,   -23,   114,   115,
     202,   202,    47,   131,   132,    44,   101,   326,   327,   166,
     136,    43,   349,   300,   121,   288,   289,   103,   208,   209,
     243,   243,    46,    39,   210,   133,   134,   137,   290,    81,
      81,   186,   148,   149,   189,   143,   243,   200,   321,   322,
     323,   106,   239,    81,    81,    45,    81,   164,    39,   269,
     127,   128,   270,     1,   271,     2,   243,   116,   239,   272,
     239,     3,   295,   296,   239,   239,   144,    81,    81,   143,
     185,    81,   239,   107,   273,   274,   206,     4,     5,     6,
     145,   127,   128,   146,   297,   298,   127,   128,   309,   310,
     108,   239,   191,     7,   217,   243,   109,   315,   239,   239,
     127,   128,   117,   110,     8,   118,     9,   119,    63,    64,
     160,   192,   120,    81,   239,   111,   239,   239,   239,   239,
     239,   239,   239,   239,   239,   239,   239,   239,   239,   239,
     239,   239,   239,   239,   239,   138,   213,   203,    81,   127,
     128,   221,   129,   130,   238,   127,   128,   127,   128,   140,
     341,   139,   212,   259,   203,   218,   342,   291,   292,   143,
     238,   147,   238,   127,   128,   343,   238,   238,   347,   348,
     153,   350,   158,   239,   238,   143,   198,   127,   128,   146,
     154,   265,   187,    81,    81,   351,   307,   308,   174,   175,
     176,   177,   190,   238,   328,   329,   330,   331,   141,   142,
     238,   238,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,   220,    63,    64,    39,   238,    38,   238,   238,
     238,   238,   238,   238,   238,   238,   238,   238,   238,   238,
     238,   238,   238,   238,   238,   238,   238,   313,   188,   344,
     306,   194,   345,   195,    48,    49,   159,   143,   196,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,     4,     5,     6,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   238,    63,    64,    65,   207,
     285,   216,    66,    67,    68,   199,   143,    69,   222,    70,
     258,    71,   223,   224,   225,   291,   292,   219,    72,   301,
     293,   294,   314,    39,   261,     4,     5,     6,   346,   306,
      73,   170,   171,   178,   179,   262,   324,   325,   263,    63,
      64,   226,   332,   333,   302,   227,   228,   229,   303,   305,
     304,   306,   230,   319,   231,    26,   215,   165,   180,   182,
     232,   233,   181,   183,   211,   334,   184,   234,   163,   197,
     337,    48,    49,   235,   236,   237,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    60,    61,    62,     4,
       5,     6,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,   335,    63,    64,    65,   336,    38,     0,    66,
      67,    68,   338,     0,    69,     0,    70,    98,    71,    39,
       0,     0,     0,     0,     0,    72,     0,     0,     0,     0,
       0,     0,     0,    48,    49,     0,     0,    73,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      62,     4,     5,     6,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    63,    64,    65,     0,     0,
       0,    66,    67,    68,     0,     0,    69,     0,    70,   162,
      71,     0,     0,     0,     0,     0,     0,    72,     0,     0,
       0,     0,     0,     0,     0,    48,    49,     0,     0,    73,
      50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
      60,    61,    62,     4,     5,     6,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,     0,    63,    64,    65,
       0,    38,     0,    66,    67,    68,   117,     0,    69,   118,
      70,   119,    71,     0,     0,     0,   120,     0,     0,    72,
       0,     0,    48,    49,     0,     0,     0,     0,     0,     0,
       0,    73,    55,    56,    57,    58,    59,    60,    61,    62,
       4,     5,     6,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    63,    64,    65,     0,     0,     0,
       0,     0,     0,     0,     0,    69,     0,    70,     0,    71
};

static constexpr yytype_int16 yycheck[] =
{
       0,    43,    55,   106,   107,   108,     4,   270,     4,   111,
     230,    11,    83,    48,    49,   117,     4,    50,    51,     4,
       4,    56,    19,   286,     9,    54,   129,   130,     8,     9,
      63,    44,    61,     0,   214,    40,    41,    42,    43,    41,
      42,    44,    31,   306,    11,    31,     4,   263,    45,    46,
     230,   271,   232,   124,   125,   126,   236,   237,     0,    44,
      44,    59,    44,    59,   244,    70,    71,    55,    70,    71,
      55,    55,    39,    45,    46,    13,   118,   293,   294,   121,
      69,    55,   345,    69,    81,    50,    51,    14,   191,   192,
     270,   271,    58,    59,   196,    67,    68,    70,    63,   104,
     105,   143,   104,   105,   146,    59,   286,    61,   288,   289,
     290,    55,   214,   118,   119,    56,   121,   119,    59,    52,
      48,    49,    55,     4,    57,     6,   306,    10,   230,    62,
     232,    12,    45,    46,   236,   237,    56,   142,   143,    59,
     142,   146,   244,    55,    77,    78,   188,    28,    29,    30,
      56,    48,    49,    59,    67,    68,    48,    49,   261,   262,
      55,   263,    59,    44,    56,   345,    55,   269,   270,   271,
      48,    49,    52,    55,    55,    55,    57,    57,    42,    43,
      44,    59,    62,   188,   286,    55,   288,   289,   290,   291,
     292,   293,   294,   295,   296,   297,   298,   299,   300,   301,
     302,   303,   304,   305,   306,    71,   203,   187,   213,    48,
      49,   213,    65,    66,   214,    48,    49,    48,    49,    73,
      59,    72,   202,   220,   204,    56,    59,    48,    49,    59,
     230,    55,   232,    48,    49,    56,   236,   237,   341,   342,
      44,    56,    56,   345,   244,    59,    56,    48,    49,    59,
      44,   231,    61,   258,   259,    56,   258,   259,   131,   132,
     133,   134,    44,   263,   295,   296,   297,   298,    74,    75,
     270,   271,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    41,    56,    42,    43,    59,   286,    47,   288,   289,
     290,   291,   292,   293,   294,   295,   296,   297,   298,   299,
     300,   301,   302,   303,   304,   305,   306,    56,     4,    56,
      59,    56,    59,    56,    10,    11,    58,    59,    59,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,   345,    42,    43,    44,    56,
      47,    61,    48,    49,    50,    58,    59,    53,    11,    55,
       4,    57,    15,    16,    17,    48,    49,    56,    64,    70,
      65,    66,    58,    59,    55,    28,    29,    30,    58,    59,
      76,   127,   128,   135,   136,    55,   291,   292,    55,    42,
      43,    44,   299,   300,    71,    48,    49,    50,    72,    74,
      73,    59,    55,    44,    57,    11,   204,   120,   137,   139,
      63,    64,   138,   140,   200,   301,   141,    70,   118,   157,
     304,    10,    11,    76,    77,    78,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    41,   302,    42,    43,    44,   303,    47,    -1,    48,
      49,    50,   305,    -1,    53,    -1,    55,    56,    57,    59,
      -1,    -1,    -1,    -1,    -1,    64,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    10,    11,    -1,    -1,    76,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    42,    43,    44,    -1,    -1,
      -1,    48,    49,    50,    -1,    -1,    53,    -1,    55,    56,
      57,    -1,    -1,    -1,    -1,    -1,    -1,    64,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    10,    11,    -1,    -1,    76,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    -1,    42,    43,    44,
      -1,    47,    -1,    48,    49,    50,    52,    -1,    53,    55,
      55,    57,    57,    -1,    -1,    -1,    62,    -1,    -1,    64,
      -1,    -1,    10,    11,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    76,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    42,    43,    44,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    53,    -1,    55,    -1,    57
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM. */
static constexpr yytype_uint8 yystos[] =
{
       0,     4,     6,    12,    28,    29,    30,    44,    55,    57,
      81,    82,    83,    84,    85,    86,    87,    90,    91,    92,
      44,    44,    85,    86,    86,     0,    83,     4,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    47,    59,
     115,    55,   115,    55,    13,    56,    58,    85,    10,    11,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    42,    43,    44,    48,    49,    50,    53,
      55,    57,    64,    76,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,   111,   112,   113,   114,   116,    56,   116,
     116,   114,   117,    14,   115,   115,    55,    55,    55,    55,
      55,    55,   118,   119,   116,   116,    10,    52,    55,    57,
      62,   115,    99,   102,    50,    51,    63,    48,    49,    65,
      66,    45,    46,    67,    68,    31,    69,    70,    71,    72,
      73,    74,    75,    59,    56,    56,    59,    55,   116,   116,
     104,   104,   104,    44,    44,    93,    54,    61,    56,    58,
      44,    93,    56,   117,   116,    96,   114,   102,   102,   102,
     103,   103,   104,   104,   105,   105,   105,   105,   106,   106,
     107,   108,   109,   110,   111,   116,   114,    61,     4,   114,
      44,    59,    59,    56,    56,    56,    59,   118,    56,    58,
      61,     4,    55,    86,    88,    89,   114,    56,   104,   104,
      93,   113,    86,   115,     9,    89,    61,    56,    56,    56,
      56,   116,    11,    15,    16,    17,    44,    48,    49,    50,
      55,    57,    63,    64,    70,    76,    77,    78,    92,    93,
     120,   121,   122,   124,   125,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   139,     4,   115,
       4,    55,    55,    55,   139,    86,   124,   124,   124,    52,
      55,    57,    62,    77,    78,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    47,   138,   124,    50,    51,
      63,    48,    49,    65,    66,    45,    46,    67,    68,    31,
      69,    70,    71,    72,    73,    74,    59,   116,   116,   104,
     104,   124,   127,    56,    58,    93,   123,   137,   139,    44,
     137,   124,   124,   124,   126,   126,   127,   127,   128,   128,
     128,   128,   129,   129,   130,   131,   132,   133,   134,   137,
       4,    59,    59,    56,    56,    59,    58,   104,   104,   137,
      56,    56
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives. */
static constexpr yytype_uint8 yyr1[] =
{
       0,    80,    81,    82,    82,    83,    83,    84,    84,    84,
      84,    84,    84,    84,    85,    86,    86,    87,    88,    88,
      89,    89,    89,    90,    90,    90,    91,    92,    92,    92,
      93,    93,    94,    94,    95,    95,    95,    95,    95,    95,
      95,    95,    96,    96,    96,    96,    96,    96,    96,    96,
      96,    97,    97,    98,    98,    99,    99,    99,    99,    99,
      99,    99,    99,    99,    99,   100,   100,   101,   101,   101,
     101,   101,   102,   103,   103,   103,   103,   104,   104,   104,
     105,   105,   105,   106,   106,   106,   106,   106,   107,   107,
     107,   108,   108,   109,   109,   110,   110,   111,   111,   112,
     112,   113,   113,   114,   114,   115,   115,   115,   115,   115,
     115,   115,   115,   115,   115,   115,   116,   116,   117,   117,
     117,   118,   119,   119,   120,   120,   120,   120,   120,   120,
     121,   121,   122,   122,   122,   122,   122,   122,   122,   122,
     122,   123,   123,   124,   124,   124,   124,   124,   125,   125,
     125,   125,   125,   125,   126,   126,   126,   126,   127,   127,
     127,   128,   128,   128,   129,   129,   129,   129,   129,   130,
     130,   130,   131,   131,   132,   132,   133,   133,   134,   134,
     135,   135,   136,   137,   137,   138,   138,   138,   138,   138,
     138,   138,   138,   138,   138,   138,   139,   139
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN. */
static constexpr yytype_int8 yyr2[] =
{
       0,     2,     1,     1,     2,     1,     2,     1,     3,     3,
       5,     5,     1,     1,     1,     1,     3,     9,     1,     2,
       1,     4,     6,     1,     3,     4,    10,     1,     1,     1,
       1,     1,     1,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     3,     3,
       1,     3,     3,     6,     6,     1,     1,     4,     3,     4,
       3,     1,     4,     4,     4,     1,     2,     1,     1,     1,
       1,     1,     1,     1,     3,     3,     3,     1,     3,     3,
       1,     3,     3,     1,     3,     3,     3,     3,     1,     3,
       3,     1,     3,     1,     3,     1,     3,     1,     3,     1,
       3,     1,     5,     1,     3,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     1,     3,
       4,     5,     1,     3,     1,     1,     1,     1,     3,     3,
       6,     6,     1,     4,     4,     3,     2,     2,     3,     1,
       4,     1,     3,     1,     2,     2,     2,     2,     1,     1,
       1,     1,     1,     1,     1,     3,     3,     3,     1,     3,
       3,     1,     3,     3,     1,     3,     3,     3,     3,     1,
       3,     3,     1,     3,     1,     3,     1,     3,     1,     3,
       1,     3,     1,     1,     3,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     3
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        YY_LAC_DISCARD ("YYBACKUP");                              \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (root, YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
# ifndef YY_LOCATION_PRINT
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif


# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value, root); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYSTYPE *root)
{
  FILE *yyoutput = yyo;
  YYUSE (yyoutput);
  YYUSE (root);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYSTYPE *root)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep, root);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule, YYSTYPE *root)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)], root);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule, root); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


/* Given a state stack such that *YYBOTTOM is its bottom, such that
   *YYTOP is either its top or is YYTOP_EMPTY to indicate an empty
   stack, and such that *YYCAPACITY is the maximum number of elements it
   can hold without a reallocation, make sure there is enough room to
   store YYADD more elements.  If not, allocate a new stack using
   YYSTACK_ALLOC, copy the existing elements, and adjust *YYBOTTOM,
   *YYTOP, and *YYCAPACITY to reflect the new capacity and memory
   location.  If *YYBOTTOM != YYBOTTOM_NO_FREE, then free the old stack
   using YYSTACK_FREE.  Return 0 if successful or if no reallocation is
   required.  Return YYENOMEM if memory is exhausted.  */
static int
yy_lac_stack_realloc (YYPTRDIFF_T *yycapacity, YYPTRDIFF_T yyadd,
#if YYDEBUG
                      char const *yydebug_prefix,
                      char const *yydebug_suffix,
#endif
                      yy_state_t **yybottom,
                      yy_state_t *yybottom_no_free,
                      yy_state_t **yytop, yy_state_t *yytop_empty)
{
  YYPTRDIFF_T yysize_old =
    *yytop == yytop_empty ? 0 : *yytop - *yybottom + 1;
  YYPTRDIFF_T yysize_new = yysize_old + yyadd;
  if (*yycapacity < yysize_new)
    {
      YYPTRDIFF_T yyalloc = 2 * yysize_new;
      yy_state_t *yybottom_new;
      /* Use YYMAXDEPTH for maximum stack size given that the stack
         should never need to grow larger than the main state stack
         needs to grow without LAC.  */
      if (YYMAXDEPTH < yysize_new)
        {
          YYDPRINTF ((stderr, "%smax size exceeded%s", yydebug_prefix,
                      yydebug_suffix));
          return YYENOMEM;
        }
      if (YYMAXDEPTH < yyalloc)
        yyalloc = YYMAXDEPTH;
      yybottom_new =
        YY_CAST (yy_state_t *,
                 YYSTACK_ALLOC (YY_CAST (YYSIZE_T,
                                         yyalloc * YYSIZEOF (*yybottom_new))));
      if (!yybottom_new)
        {
          YYDPRINTF ((stderr, "%srealloc failed%s", yydebug_prefix,
                      yydebug_suffix));
          return YYENOMEM;
        }
      if (*yytop != yytop_empty)
        {
          YYCOPY (yybottom_new, *yybottom, yysize_old);
          *yytop = yybottom_new + (yysize_old - 1);
        }
      if (*yybottom != yybottom_no_free)
        YYSTACK_FREE (*yybottom);
      *yybottom = yybottom_new;
      *yycapacity = yyalloc;
    }
  return 0;
}

/* Establish the initial context for the current lookahead if no initial
   context is currently established.

   We define a context as a snapshot of the parser stacks.  We define
   the initial context for a lookahead as the context in which the
   parser initially examines that lookahead in order to select a
   syntactic action.  Thus, if the lookahead eventually proves
   syntactically unacceptable (possibly in a later context reached via a
   series of reductions), the initial context can be used to determine
   the exact set of tokens that would be syntactically acceptable in the
   lookahead's place.  Moreover, it is the context after which any
   further semantic actions would be erroneous because they would be
   determined by a syntactically unacceptable token.

   YY_LAC_ESTABLISH should be invoked when a reduction is about to be
   performed in an inconsistent state (which, for the purposes of LAC,
   includes consistent states that don't know they're consistent because
   their default reductions have been disabled).  Iff there is a
   lookahead token, it should also be invoked before reporting a syntax
   error.  This latter case is for the sake of the debugging output.

   For parse.lac=full, the implementation of YY_LAC_ESTABLISH is as
   follows.  If no initial context is currently established for the
   current lookahead, then check if that lookahead can eventually be
   shifted if syntactic actions continue from the current context.
   Report a syntax error if it cannot.  */
#define YY_LAC_ESTABLISH                                                \
do {                                                                    \
  if (!yy_lac_established)                                              \
    {                                                                   \
      YYDPRINTF ((stderr,                                               \
                  "LAC: initial context established for %s\n",          \
                  yysymbol_name (yytoken)));                            \
      yy_lac_established = 1;                                           \
      switch (yy_lac (yyesa, &yyes, &yyes_capacity, yyssp, yytoken))    \
        {                                                               \
        case YYENOMEM:                                                  \
          goto yyexhaustedlab;                                          \
        case 1:                                                         \
          goto yyerrlab;                                                \
        }                                                               \
    }                                                                   \
} while (0)

/* Discard any previous initial lookahead context because of Event,
   which may be a lookahead change or an invalidation of the currently
   established initial context for the current lookahead.

   The most common example of a lookahead change is a shift.  An example
   of both cases is syntax error recovery.  That is, a syntax error
   occurs when the lookahead is syntactically erroneous for the
   currently established initial context, so error recovery manipulates
   the parser stacks to try to find a new initial context in which the
   current lookahead is syntactically acceptable.  If it fails to find
   such a context, it discards the lookahead.  */
#if YYDEBUG
# define YY_LAC_DISCARD(Event)                                           \
do {                                                                     \
  if (yy_lac_established)                                                \
    {                                                                    \
      YYDPRINTF ((stderr, "LAC: initial context discarded due to "       \
                  Event "\n"));                                          \
      yy_lac_established = 0;                                            \
    }                                                                    \
} while (0)
#else
# define YY_LAC_DISCARD(Event) yy_lac_established = 0
#endif

/* Given the stack whose top is *YYSSP, return 0 iff YYTOKEN can
   eventually (after perhaps some reductions) be shifted, return 1 if
   not, or return YYENOMEM if memory is exhausted.  As preconditions and
   postconditions: *YYES_CAPACITY is the allocated size of the array to
   which *YYES points, and either *YYES = YYESA or *YYES points to an
   array allocated with YYSTACK_ALLOC.  yy_lac may overwrite the
   contents of either array, alter *YYES and *YYES_CAPACITY, and free
   any old *YYES other than YYESA.  */
static int
yy_lac (yy_state_t *yyesa, yy_state_t **yyes,
        YYPTRDIFF_T *yyes_capacity, yy_state_t *yyssp, yysymbol_kind_t yytoken)
{
  yy_state_t *yyes_prev = yyssp;
  yy_state_t *yyesp = yyes_prev;
  /* Reduce until we encounter a shift and thereby accept the token.  */
  YYDPRINTF ((stderr, "LAC: checking lookahead %s:", yysymbol_name (yytoken)));
  if (yytoken == YYSYMBOL_YYUNDEF)
    {
      YYDPRINTF ((stderr, " Always Err\n"));
      return 1;
    }
  while (1)
    {
      int yyrule = yypact[+*yyesp];
      if (yypact_value_is_default (yyrule)
          || (yyrule += yytoken) < 0 || YYLAST < yyrule
          || yycheck[yyrule] != yytoken)
        {
          /* Use the default action.  */
          yyrule = yydefact[+*yyesp];
          if (yyrule == 0)
            {
              YYDPRINTF ((stderr, " Err\n"));
              return 1;
            }
        }
      else
        {
          /* Use the action from yytable.  */
          yyrule = yytable[yyrule];
          if (yytable_value_is_error (yyrule))
            {
              YYDPRINTF ((stderr, " Err\n"));
              return 1;
            }
          if (0 < yyrule)
            {
              YYDPRINTF ((stderr, " S%d\n", yyrule));
              return 0;
            }
          yyrule = -yyrule;
        }
      /* By now we know we have to simulate a reduce.  */
      YYDPRINTF ((stderr, " R%d", yyrule - 1));
      {
        /* Pop the corresponding number of values from the stack.  */
        YYPTRDIFF_T yylen = yyr2[yyrule];
        /* First pop from the LAC stack as many tokens as possible.  */
        if (yyesp != yyes_prev)
          {
            YYPTRDIFF_T yysize = yyesp - *yyes + 1;
            if (yylen < yysize)
              {
                yyesp -= yylen;
                yylen = 0;
              }
            else
              {
                yyesp = yyes_prev;
                yylen -= yysize;
              }
          }
        /* Only afterwards look at the main stack.  */
        if (yylen)
          yyesp = yyes_prev -= yylen;
      }
      /* Push the resulting state of the reduction.  */
      {
        yy_state_fast_t yystate;
        {
          const int yylhs = yyr1[yyrule] - YYNTOKENS;
          const int yyi = yypgoto[yylhs] + *yyesp;
          yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyesp
                     ? yytable[yyi]
                     : yydefgoto[yylhs]);
        }
        if (yyesp == yyes_prev)
          {
            yyesp = *yyes;
            YY_IGNORE_USELESS_CAST_BEGIN
            *yyesp = YY_CAST (yy_state_t, yystate);
            YY_IGNORE_USELESS_CAST_END
          }
        else
          {
            if (yy_lac_stack_realloc (yyes_capacity, 1,
#if YYDEBUG
                                      " (", ")",
#endif
                                      yyes, yyesa, &yyesp, yyes_prev))
              {
                YYDPRINTF ((stderr, "\n"));
                return YYENOMEM;
              }
            YY_IGNORE_USELESS_CAST_BEGIN
            *++yyesp = YY_CAST (yy_state_t, yystate);
            YY_IGNORE_USELESS_CAST_END
          }
        YYDPRINTF ((stderr, " G%d", yystate));
      }
    }
}

/* Context of a parse error.  */
typedef struct
{
  yy_state_t *yyssp;
  yy_state_t *yyesa;
  yy_state_t **yyes;
  YYPTRDIFF_T *yyes_capacity;
  yysymbol_kind_t yytoken;
} yypcontext_t;

/* Put in YYARG at most YYARGN of the expected tokens given the
   current YYCTX, and return the number of tokens stored in YYARG.  If
   YYARG is null, return the number of expected tokens (guaranteed to
   be less than YYNTOKENS).  Return YYENOMEM on memory exhaustion.
   Return 0 if there are more than YYARGN expected tokens, yet fill
   YYARG up to YYARGN. */
static int
yypcontext_expected_tokens (const yypcontext_t *yyctx,
                            yysymbol_kind_t yyarg[], int yyargn)
{
  /* Actual size of YYARG. */
  int yycount = 0;

  int yyx;
  for (yyx = 0; yyx < YYNTOKENS; ++yyx)
    {
      yysymbol_kind_t yysym = YY_CAST (yysymbol_kind_t, yyx);
      if (yysym != YYSYMBOL_YYerror && yysym != YYSYMBOL_YYUNDEF)
        switch (yy_lac (yyctx->yyesa, yyctx->yyes, yyctx->yyes_capacity, yyctx->yyssp, yysym))
          {
          case YYENOMEM:
            return YYENOMEM;
          case 1:
            continue;
          default:
            if (!yyarg)
              ++yycount;
            else if (yycount == yyargn)
              return 0;
            else
              yyarg[yycount++] = yysym;
          }
    }
  if (yyarg && yycount == 0 && 0 < yyargn)
    yyarg[0] = YYSYMBOL_YYEMPTY;
  return yycount;
}




#ifndef yystrlen
# if defined __GLIBC__ && defined _STRING_H
#  define yystrlen(S) (YY_CAST (YYPTRDIFF_T, strlen (S)))
# else
/* Return the length of YYSTR.  */
static YYPTRDIFF_T
yystrlen (const char *yystr)
{
  YYPTRDIFF_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
# endif
#endif

#ifndef yystpcpy
# if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#  define yystpcpy stpcpy
# else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
# endif
#endif



static int
yy_syntax_error_arguments (const yypcontext_t *yyctx,
                           yysymbol_kind_t yyarg[], int yyargn)
{
  /* Actual size of YYARG. */
  int yycount = 0;
  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
       In the first two cases, it might appear that the current syntax
       error should have been detected in the previous state when yy_lac
       was invoked.  However, at that time, there might have been a
       different syntax error that discarded a different initial context
       during error recovery, leaving behind the current lookahead.
  */
  if (yyctx->yytoken != YYSYMBOL_YYEMPTY)
    {
      int yyn;
      YYDPRINTF ((stderr, "Constructing syntax error message\n"));
      if (yyarg)
        yyarg[yycount] = yyctx->yytoken;
      ++yycount;
      yyn = yypcontext_expected_tokens (yyctx,
                                        yyarg ? yyarg + 1 : yyarg, yyargn - 1);
      if (yyn == YYENOMEM)
        return YYENOMEM;
      else if (yyn == 0)
        YYDPRINTF ((stderr, "No expected tokens.\n"));
      else
        yycount += yyn;
    }
  return yycount;
}

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.  In order to see if a particular token T is a
   valid looakhead, invoke yy_lac (YYESA, YYES, YYES_CAPACITY, YYSSP, T).

   Return 0 if *YYMSG was successfully written.  Return -1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return YYENOMEM if the
   required number of bytes is too large to store or if
   yy_lac returned YYENOMEM.  */
static int
yysyntax_error (YYPTRDIFF_T *yymsg_alloc, char **yymsg,
                const yypcontext_t *yyctx)
{
  enum { YYARGS_MAX = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat: reported tokens (one for the "unexpected",
     one per "expected"). */
  yysymbol_kind_t yyarg[YYARGS_MAX];
  /* Cumulated lengths of YYARG.  */
  YYPTRDIFF_T yysize = 0;

  /* Actual size of YYARG. */
  int yycount = yy_syntax_error_arguments (yyctx, yyarg, YYARGS_MAX);
  if (yycount == YYENOMEM)
    return YYENOMEM;

  switch (yycount)
    {
#define YYCASE_(N, S)                       \
      case N:                               \
        yyformat = S;                       \
        break
    default: /* Avoid compiler warnings. */
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
    }

  /* Compute error message size.  Don't count the "%s"s, but reserve
     room for the terminator.  */
  yysize = yystrlen (yyformat) - 2 * yycount + 1;
  {
    int yyi;
    for (yyi = 0; yyi < yycount; ++yyi)
      {
        YYPTRDIFF_T yysize1
          = yysize + yystrlen (yysymbol_name (yyarg[yyi]));
        if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
          yysize = yysize1;
        else
          return YYENOMEM;
      }
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return -1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp = yystpcpy (yyp, yysymbol_name (yyarg[yyi++]));
          yyformat += 2;
        }
      else
        {
          ++yyp;
          ++yyformat;
        }
  }
  return 0;
}


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep, YYSTYPE *root)
{
  YYUSE (yyvaluep);
  YYUSE (root);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (YYSTYPE *root)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

    yy_state_t yyesa[20];
    yy_state_t *yyes = yyesa;
    YYPTRDIFF_T yyes_capacity = 20 < YYMAXDEPTH ? 20 : YYMAXDEPTH;

  /* Whether LAC context is established.  A Boolean.  */
  int yy_lac_established = 0;
  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYPTRDIFF_T yymsg_alloc = sizeof yymsgbuf;

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    goto yyexhaustedlab;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    {
      YY_LAC_ESTABLISH;
      goto yydefault;
    }
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      YY_LAC_ESTABLISH;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  YY_LAC_DISCARD ("shift");
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  {
    int yychar_backup = yychar;
    switch (yyn)
      {
  case 2:{  /* entry_point: statements */
#define entry_point_statements 2
   constexpr int YYN = 2;
   { _ *root = yyval; }
   break;}

  case 3:{  /* statements: statement */
#define statements_statement 3
   constexpr int YYN = 3;
   {_}
   break;}

  case 4:{  /* statements: statements statement */
#define statements_statements_statement 4
   constexpr int YYN = 4;
   {_}
   break;}

  case 5:{  /* statement: NL */
#define statement_nl 5
   constexpr int YYN = 5;
   {_}
   break;}

  case 6:{  /* statement: decl NL */
#define statement_decl_nl 6
   constexpr int YYN = 6;
   {_}
   break;}

  case 7:{  /* decl: function */
#define decl_function 7
   constexpr int YYN = 7;
   {_}
   break;}

  case 8:{  /* decl: domain assign_op expr */
#define decl_domain_assign_op_expr 8
   constexpr int YYN = 8;
   {_}
   break;}

  case 9:{  /* decl: id_list assign_op expr */
#define decl_id_list_assign_op_expr 9
   constexpr int YYN = 9;
   {_}
   break;}

  case 10:{  /* decl: LP id_list RP assign_op expr */
#define decl_lp_id_list_rp_assign_op_expr 10
   constexpr int YYN = 10;
   {_}
   break;}

  case 11:{  /* decl: LB id_list RB assign_op expr */
#define decl_lb_id_list_rb_assign_op_expr 11
   constexpr int YYN = 11;
   {_}
   break;}

  case 12:{  /* decl: iteration_statement */
#define decl_iteration_statement 12
   constexpr int YYN = 12;
   {_}
   break;}

  case 13:{  /* decl: direct_declarator */
#define decl_direct_declarator 13
   constexpr int YYN = 13;
   {_}
   break;}

  case 14:{  /* postfix_id: IDENTIFIER */
#define postfix_id_identifier 14
   constexpr int YYN = 14;
   {_}
   break;}

  case 15:{  /* id_list: postfix_id */
#define id_list_postfix_id 15
   constexpr int YYN = 15;
   {_}
   break;}

  case 16:{  /* id_list: id_list COMA postfix_id */
#define id_list_id_list_coma_postfix_id 16
   constexpr int YYN = 16;
   {_}
   break;}

  case 17:{  /* function: DEF IDENTIFIER LP args_expr_list RP COLON def_statements RETURN math_expr */
#define function_def_identifier_lp_args_expr_list_rp_colon_def_statements_return_math_expr 17
   constexpr int YYN = 17;
   {_}
   break;}

  case 18:{  /* def_statements: def_statement */
#define def_statements_def_statement 18
   constexpr int YYN = 18;
   {_}
   break;}

  case 19:{  /* def_statements: def_statements def_statement */
#define def_statements_def_statements_def_statement 19
   constexpr int YYN = 19;
   {_}
   break;}

  case 20:{  /* def_statement: NL */
#define def_statement_nl 20
   constexpr int YYN = 20;
   {_}
   break;}

  case 21:{  /* def_statement: id_list assign_op expr NL */
#define def_statement_id_list_assign_op_expr_nl 21
   constexpr int YYN = 21;
   {_}
   break;}

  case 22:{  /* def_statement: LP id_list RP assign_op expr NL */
#define def_statement_lp_id_list_rp_assign_op_expr_nl 22
   constexpr int YYN = 22;
   {_}
   break;}

  case 23:{  /* direct_declarator: postfix_id */
#define direct_declarator_postfix_id 23
   constexpr int YYN = 23;
   {_}
   break;}

  case 24:{  /* direct_declarator: direct_declarator LP RP */
#define direct_declarator_direct_declarator_lp_rp 24
   constexpr int YYN = 24;
   {_}
   break;}

  case 25:{  /* direct_declarator: direct_declarator LP expr RP */
#define direct_declarator_direct_declarator_lp_expr_rp 25
   constexpr int YYN = 25;
   {_}
   break;}

  case 26:{  /* iteration_statement: FOR IDENTIFIER IN RANGE LP IDENTIFIER RP COLON NL expr */
#define iteration_statement_for_identifier_in_range_lp_identifier_rp_colon_nl_expr 26
   constexpr int YYN = 26;
   {_}
   break;}

  case 27:{  /* domain: DOM_DX */
#define domain_dom_dx 27
   constexpr int YYN = 27;
   {_}
   break;}

  case 28:{  /* domain: EXT_DS */
#define domain_ext_ds 28
   constexpr int YYN = 28;
   {_}
   break;}

  case 29:{  /* domain: INT_DS */
#define domain_int_ds 29
   constexpr int YYN = 29;
   {_}
   break;}

  case 30:{  /* constant: NATURAL */
#define constant_natural 30
   constexpr int YYN = 30;
   {_}
   break;}

  case 31:{  /* constant: REAL */
#define constant_real 31
   constexpr int YYN = 31;
   {_}
   break;}

  case 32:{  /* strings: STRING */
#define strings_string 32
   constexpr int YYN = 32;
   {_}
   break;}

  case 33:{  /* strings: strings STRING */
#define strings_strings_string 33
   constexpr int YYN = 33;
   {_}
   break;}

  case 34:{  /* api: UNIT_SQUARE_MESH */
#define api_unit_square_mesh 34
   constexpr int YYN = 34;
   {_}
   break;}

  case 35:{  /* api: FUNCTION */
#define api_function 35
   constexpr int YYN = 35;
   {_}
   break;}

  case 36:{  /* api: FUNCTION_SPACE */
#define api_function_space 36
   constexpr int YYN = 36;
   {_}
   break;}

  case 37:{  /* api: EXPRESSION */
#define api_expression 37
   constexpr int YYN = 37;
   {_}
   break;}

  case 38:{  /* api: DIRICHLET_BC */
#define api_dirichlet_bc 38
   constexpr int YYN = 38;
   {_}
   break;}

  case 39:{  /* api: TRIAL_FUNCTION */
#define api_trial_function 39
   constexpr int YYN = 39;
   {_}
   break;}

  case 40:{  /* api: TEST_FUNCTION */
#define api_test_function 40
   constexpr int YYN = 40;
   {_}
   break;}

  case 41:{  /* api: CONSTANT_API */
#define api_constant_api 41
   constexpr int YYN = 41;
   {_}
   break;}

  case 42:{  /* primary_expr: IDENTIFIER */
#define primary_expr_identifier 42
   constexpr int YYN = 42;
   {_}
   break;}

  case 43:{  /* primary_expr: constant */
#define primary_expr_constant 43
   constexpr int YYN = 43;
   {_}
   break;}

  case 44:{  /* primary_expr: domain */
#define primary_expr_domain 44
   constexpr int YYN = 44;
   {_}
   break;}

  case 45:{  /* primary_expr: QUOTE */
#define primary_expr_quote 45
   constexpr int YYN = 45;
   {_}
   break;}

  case 46:{  /* primary_expr: strings */
#define primary_expr_strings 46
   constexpr int YYN = 46;
   {_}
   break;}

  case 47:{  /* primary_expr: LP expr RP */
#define primary_expr_lp_expr_rp 47
   constexpr int YYN = 47;
   {_}
   break;}

  case 48:{  /* primary_expr: LB expr RB */
#define primary_expr_lb_expr_rb 48
   constexpr int YYN = 48;
   {_}
   break;}

  case 49:{  /* primary_expr: LS coords RS */
#define primary_expr_ls_coords_rs 49
   constexpr int YYN = 49;
   {_}
   break;}

  case 50:{  /* primary_expr: api */
#define primary_expr_api 50
   constexpr int YYN = 50;
   {_}
   break;}

  case 51:{  /* pow_expr: postfix_expr POW constant */
#define pow_expr_postfix_expr_pow_constant 51
   constexpr int YYN = 51;
   {_}
   break;}

  case 52:{  /* pow_expr: postfix_expr POW IDENTIFIER */
#define pow_expr_postfix_expr_pow_identifier 52
   constexpr int YYN = 52;
   {_}
   break;}

  case 53:{  /* dot_expr: DOT_OP LP additive_expr COMA additive_expr RP */
#define dot_expr_dot_op_lp_additive_expr_coma_additive_expr_rp 53
   constexpr int YYN = 53;
   {_}
   break;}

  case 54:{  /* dot_expr: INNER_OP LP additive_expr COMA additive_expr RP */
#define dot_expr_inner_op_lp_additive_expr_coma_additive_expr_rp 54
   constexpr int YYN = 54;
   {_}
   break;}

  case 55:{  /* postfix_expr: primary_expr */
#define postfix_expr_primary_expr 55
   constexpr int YYN = 55;
   {_}
   break;}

  case 56:{  /* postfix_expr: pow_expr */
#define postfix_expr_pow_expr 56
   constexpr int YYN = 56;
   {_}
   break;}

  case 57:{  /* postfix_expr: postfix_expr LB expr RB */
#define postfix_expr_postfix_expr_lb_expr_rb 57
   constexpr int YYN = 57;
   {_}
   break;}

  case 58:{  /* postfix_expr: postfix_expr LP RP */
#define postfix_expr_postfix_expr_lp_rp 58
   constexpr int YYN = 58;
   {_}
   break;}

  case 59:{  /* postfix_expr: postfix_expr LP args_expr_list RP */
#define postfix_expr_postfix_expr_lp_args_expr_list_rp 59
   constexpr int YYN = 59;
   {_}
   break;}

  case 60:{  /* postfix_expr: postfix_expr DOT primary_expr */
#define postfix_expr_postfix_expr_dot_primary_expr 60
   constexpr int YYN = 60;
   {_}
   break;}

  case 61:{  /* postfix_expr: dot_expr */
#define postfix_expr_dot_expr 61
   constexpr int YYN = 61;
   {_}
   break;}

  case 62:{  /* postfix_expr: GRAD_OP LP additive_expr RP */
#define postfix_expr_grad_op_lp_additive_expr_rp 62
   constexpr int YYN = 62;
   {_}
   break;}

  case 63:{  /* postfix_expr: LHS LP IDENTIFIER RP */
#define postfix_expr_lhs_lp_identifier_rp 63
   constexpr int YYN = 63;
   {_}
   break;}

  case 64:{  /* postfix_expr: RHS LP IDENTIFIER RP */
#define postfix_expr_rhs_lp_identifier_rp 64
   constexpr int YYN = 64;
   {_}
   break;}

  case 65:{  /* unary_expr: postfix_expr */
#define unary_expr_postfix_expr 65
   constexpr int YYN = 65;
   {_}
   break;}

  case 66:{  /* unary_expr: unary_op cast_expr */
#define unary_expr_unary_op_cast_expr 66
   constexpr int YYN = 66;
   {_}
   break;}

  case 67:{  /* unary_op: MUL */
#define unary_op_mul 67
   constexpr int YYN = 67;
   {_}
   break;}

  case 68:{  /* unary_op: ADD */
#define unary_op_add 68
   constexpr int YYN = 68;
   {_}
   break;}

  case 69:{  /* unary_op: SUB */
#define unary_op_sub 69
   constexpr int YYN = 69;
   {_}
   break;}

  case 70:{  /* unary_op: TILDE */
#define unary_op_tilde 70
   constexpr int YYN = 70;
   {_}
   break;}

  case 71:{  /* unary_op: NOT */
#define unary_op_not 71
   constexpr int YYN = 71;
   {_}
   break;}

  case 72:{  /* cast_expr: unary_expr */
#define cast_expr_unary_expr 72
   constexpr int YYN = 72;
   {_}
   break;}

  case 73:{  /* multiplicative_expr: cast_expr */
#define multiplicative_expr_cast_expr 73
   constexpr int YYN = 73;
   {_}
   break;}

  case 74:{  /* multiplicative_expr: multiplicative_expr MUL cast_expr */
#define multiplicative_expr_multiplicative_expr_mul_cast_expr 74
   constexpr int YYN = 74;
   {_}
   break;}

  case 75:{  /* multiplicative_expr: multiplicative_expr DIV cast_expr */
#define multiplicative_expr_multiplicative_expr_div_cast_expr 75
   constexpr int YYN = 75;
   {_}
   break;}

  case 76:{  /* multiplicative_expr: multiplicative_expr MOD cast_expr */
#define multiplicative_expr_multiplicative_expr_mod_cast_expr 76
   constexpr int YYN = 76;
   {_}
   break;}

  case 77:{  /* additive_expr: multiplicative_expr */
#define additive_expr_multiplicative_expr 77
   constexpr int YYN = 77;
   {_}
   break;}

  case 78:{  /* additive_expr: additive_expr ADD multiplicative_expr */
#define additive_expr_additive_expr_add_multiplicative_expr 78
   constexpr int YYN = 78;
   {_}
   break;}

  case 79:{  /* additive_expr: additive_expr SUB multiplicative_expr */
#define additive_expr_additive_expr_sub_multiplicative_expr 79
   constexpr int YYN = 79;
   {_}
   break;}

  case 80:{  /* shift_expr: additive_expr */
#define shift_expr_additive_expr 80
   constexpr int YYN = 80;
   {_}
   break;}

  case 81:{  /* shift_expr: shift_expr LEFT_SHIFT additive_expr */
#define shift_expr_shift_expr_left_shift_additive_expr 81
   constexpr int YYN = 81;
   {_}
   break;}

  case 82:{  /* shift_expr: shift_expr RIGHT_SHIFT additive_expr */
#define shift_expr_shift_expr_right_shift_additive_expr 82
   constexpr int YYN = 82;
   {_}
   break;}

  case 83:{  /* relational_expr: shift_expr */
#define relational_expr_shift_expr 83
   constexpr int YYN = 83;
   {_}
   break;}

  case 84:{  /* relational_expr: relational_expr LT shift_expr */
#define relational_expr_relational_expr_lt_shift_expr 84
   constexpr int YYN = 84;
   {_}
   break;}

  case 85:{  /* relational_expr: relational_expr GT shift_expr */
#define relational_expr_relational_expr_gt_shift_expr 85
   constexpr int YYN = 85;
   {_}
   break;}

  case 86:{  /* relational_expr: relational_expr LT_EQ shift_expr */
#define relational_expr_relational_expr_lt_eq_shift_expr 86
   constexpr int YYN = 86;
   {_}
   break;}

  case 87:{  /* relational_expr: relational_expr GT_EQ shift_expr */
#define relational_expr_relational_expr_gt_eq_shift_expr 87
   constexpr int YYN = 87;
   {_}
   break;}

  case 88:{  /* equality_expr: relational_expr */
#define equality_expr_relational_expr 88
   constexpr int YYN = 88;
   {_}
   break;}

  case 89:{  /* equality_expr: equality_expr EQ_EQ relational_expr */
#define equality_expr_equality_expr_eq_eq_relational_expr 89
   constexpr int YYN = 89;
   {_}
   break;}

  case 90:{  /* equality_expr: equality_expr NOT_EQ relational_expr */
#define equality_expr_equality_expr_not_eq_relational_expr 90
   constexpr int YYN = 90;
   {_}
   break;}

  case 91:{  /* and_expr: equality_expr */
#define and_expr_equality_expr 91
   constexpr int YYN = 91;
   {_}
   break;}

  case 92:{  /* and_expr: and_expr AND equality_expr */
#define and_expr_and_expr_and_equality_expr 92
   constexpr int YYN = 92;
   {_}
   break;}

  case 93:{  /* exclusive_or_expr: and_expr */
#define exclusive_or_expr_and_expr 93
   constexpr int YYN = 93;
   {_}
   break;}

  case 94:{  /* exclusive_or_expr: exclusive_or_expr XOR and_expr */
#define exclusive_or_expr_exclusive_or_expr_xor_and_expr 94
   constexpr int YYN = 94;
   {_}
   break;}

  case 95:{  /* inclusive_or_expr: exclusive_or_expr */
#define inclusive_or_expr_exclusive_or_expr 95
   constexpr int YYN = 95;
   {_}
   break;}

  case 96:{  /* inclusive_or_expr: inclusive_or_expr OR exclusive_or_expr */
#define inclusive_or_expr_inclusive_or_expr_or_exclusive_or_expr 96
   constexpr int YYN = 96;
   {_}
   break;}

  case 97:{  /* logical_and_expr: inclusive_or_expr */
#define logical_and_expr_inclusive_or_expr 97
   constexpr int YYN = 97;
   {_}
   break;}

  case 98:{  /* logical_and_expr: logical_and_expr AND_AND inclusive_or_expr */
#define logical_and_expr_logical_and_expr_and_and_inclusive_or_expr 98
   constexpr int YYN = 98;
   {_}
   break;}

  case 99:{  /* logical_or_expr: logical_and_expr */
#define logical_or_expr_logical_and_expr 99
   constexpr int YYN = 99;
   {_}
   break;}

  case 100:{  /* logical_or_expr: logical_or_expr OR_OR logical_and_expr */
#define logical_or_expr_logical_or_expr_or_or_logical_and_expr 100
   constexpr int YYN = 100;
   {_}
   break;}

  case 101:{  /* conditional_expr: logical_or_expr */
#define conditional_expr_logical_or_expr 101
   constexpr int YYN = 101;
   {_}
   break;}

  case 102:{  /* conditional_expr: logical_or_expr QUESTION expr COLON conditional_expr */
#define conditional_expr_logical_or_expr_question_expr_colon_conditional_expr 102
   constexpr int YYN = 102;
   {_}
   break;}

  case 103:{  /* assign_expr: conditional_expr */
#define assign_expr_conditional_expr 103
   constexpr int YYN = 103;
   {_}
   break;}

  case 104:{  /* assign_expr: postfix_expr assign_op assign_expr */
#define assign_expr_postfix_expr_assign_op_assign_expr 104
   constexpr int YYN = 104;
   {_}
   break;}

  case 105:{  /* assign_op: EQ */
#define assign_op_eq 105
   constexpr int YYN = 105;
   {_}
   break;}

  case 106:{  /* assign_op: ADD_EQ */
#define assign_op_add_eq 106
   constexpr int YYN = 106;
   {_}
   break;}

  case 107:{  /* assign_op: SUB_EQ */
#define assign_op_sub_eq 107
   constexpr int YYN = 107;
   {_}
   break;}

  case 108:{  /* assign_op: MUL_EQ */
#define assign_op_mul_eq 108
   constexpr int YYN = 108;
   {_}
   break;}

  case 109:{  /* assign_op: DIV_EQ */
#define assign_op_div_eq 109
   constexpr int YYN = 109;
   {_}
   break;}

  case 110:{  /* assign_op: MOD_EQ */
#define assign_op_mod_eq 110
   constexpr int YYN = 110;
   {_}
   break;}

  case 111:{  /* assign_op: XOR_EQ */
#define assign_op_xor_eq 111
   constexpr int YYN = 111;
   {_}
   break;}

  case 112:{  /* assign_op: AND_EQ */
#define assign_op_and_eq 112
   constexpr int YYN = 112;
   {_}
   break;}

  case 113:{  /* assign_op: OR_EQ */
#define assign_op_or_eq 113
   constexpr int YYN = 113;
   {_}
   break;}

  case 114:{  /* assign_op: LEFT_EQ */
#define assign_op_left_eq 114
   constexpr int YYN = 114;
   {_}
   break;}

  case 115:{  /* assign_op: RIGHT_EQ */
#define assign_op_right_eq 115
   constexpr int YYN = 115;
   {_}
   break;}

  case 116:{  /* expr: assign_expr */
#define expr_assign_expr 116
   constexpr int YYN = 116;
   {_}
   break;}

  case 117:{  /* expr: expr COMA assign_expr */
#define expr_expr_coma_assign_expr 117
   constexpr int YYN = 117;
   {_}
   break;}

  case 118:{  /* args_expr_list: assign_expr */
#define args_expr_list_assign_expr 118
   constexpr int YYN = 118;
   {_}
   break;}

  case 119:{  /* args_expr_list: args_expr_list COMA assign_expr */
#define args_expr_list_args_expr_list_coma_assign_expr 119
   constexpr int YYN = 119;
   {_}
   break;}

  case 120:{  /* args_expr_list: args_expr_list COMA NL assign_expr */
#define args_expr_list_args_expr_list_coma_nl_assign_expr 120
   constexpr int YYN = 120;
   {_}
   break;}

  case 121:{  /* coord: LP constant COMA constant RP */
#define coord_lp_constant_coma_constant_rp 121
   constexpr int YYN = 121;
   {_}
   break;}

  case 122:{  /* coords: coord */
#define coords_coord 122
   constexpr int YYN = 122;
   {_}
   break;}

  case 123:{  /* coords: coords COLON coord */
#define coords_coords_colon_coord 123
   constexpr int YYN = 123;
   {_}
   break;}

  case 124:{  /* primary_math_expr: IDENTIFIER */
#define primary_math_expr_identifier 124
   constexpr int YYN = 124;
   {_}
   break;}

  case 125:{  /* primary_math_expr: constant */
#define primary_math_expr_constant 125
   constexpr int YYN = 125;
   {_}
   break;}

  case 126:{  /* primary_math_expr: domain */
#define primary_math_expr_domain 126
   constexpr int YYN = 126;
   {_}
   break;}

  case 127:{  /* primary_math_expr: QUOTE */
#define primary_math_expr_quote 127
   constexpr int YYN = 127;
   {_}
   break;}

  case 128:{  /* primary_math_expr: LP math_expr RP */
#define primary_math_expr_lp_math_expr_rp 128
   constexpr int YYN = 128;
   {_}
   break;}

  case 129:{  /* primary_math_expr: LB id_list RB */
#define primary_math_expr_lb_id_list_rb 129
   constexpr int YYN = 129;
   {_}
   break;}

  case 130:{  /* dot_math_expr: DOT_OP LP additive_expr COMA additive_expr RP */
#define dot_math_expr_dot_op_lp_additive_expr_coma_additive_expr_rp 130
   constexpr int YYN = 130;
   {_}
   break;}

  case 131:{  /* dot_math_expr: INNER_OP LP additive_expr COMA additive_expr RP */
#define dot_math_expr_inner_op_lp_additive_expr_coma_additive_expr_rp 131
   constexpr int YYN = 131;
   {_}
   break;}

  case 132:{  /* postfix_math_expr: primary_math_expr */
#define postfix_math_expr_primary_math_expr 132
   constexpr int YYN = 132;
   {_}
   break;}

  case 133:{  /* postfix_math_expr: postfix_math_expr LB math_expr RB */
#define postfix_math_expr_postfix_math_expr_lb_math_expr_rb 133
   constexpr int YYN = 133;
   {_}
   break;}

  case 134:{  /* postfix_math_expr: postfix_math_expr LP argument_math_expr_list RP */
#define postfix_math_expr_postfix_math_expr_lp_argument_math_expr_list_rp 134
   constexpr int YYN = 134;
   {_}
   break;}

  case 135:{  /* postfix_math_expr: postfix_math_expr DOT IDENTIFIER */
#define postfix_math_expr_postfix_math_expr_dot_identifier 135
   constexpr int YYN = 135;
   {_}
   break;}

  case 136:{  /* postfix_math_expr: postfix_math_expr INC_OP */
#define postfix_math_expr_postfix_math_expr_inc_op 136
   constexpr int YYN = 136;
   {_}
   break;}

  case 137:{  /* postfix_math_expr: postfix_math_expr DEC_OP */
#define postfix_math_expr_postfix_math_expr_dec_op 137
   constexpr int YYN = 137;
   {_}
   break;}

  case 138:{  /* postfix_math_expr: postfix_math_expr POW constant */
#define postfix_math_expr_postfix_math_expr_pow_constant 138
   constexpr int YYN = 138;
   {_}
   break;}

  case 139:{  /* postfix_math_expr: dot_math_expr */
#define postfix_math_expr_dot_math_expr 139
   constexpr int YYN = 139;
   {_}
   break;}

  case 140:{  /* postfix_math_expr: GRAD_OP LP additive_math_expr RP */
#define postfix_math_expr_grad_op_lp_additive_math_expr_rp 140
   constexpr int YYN = 140;
   {_}
   break;}

  case 141:{  /* argument_math_expr_list: assign_math_expr */
#define argument_math_expr_list_assign_math_expr 141
   constexpr int YYN = 141;
   {_}
   break;}

  case 142:{  /* argument_math_expr_list: argument_math_expr_list COMA assign_math_expr */
#define argument_math_expr_list_argument_math_expr_list_coma_assign_math_expr 142
   constexpr int YYN = 142;
   {_}
   break;}

  case 143:{  /* unary_math_expr: postfix_math_expr */
#define unary_math_expr_postfix_math_expr 143
   constexpr int YYN = 143;
   {_}
   break;}

  case 144:{  /* unary_math_expr: INC_OP unary_math_expr */
#define unary_math_expr_inc_op_unary_math_expr 144
   constexpr int YYN = 144;
   {_}
   break;}

  case 145:{  /* unary_math_expr: DEC_OP unary_math_expr */
#define unary_math_expr_dec_op_unary_math_expr 145
   constexpr int YYN = 145;
   {_}
   break;}

  case 146:{  /* unary_math_expr: MOD unary_math_expr */
#define unary_math_expr_mod_unary_math_expr 146
   constexpr int YYN = 146;
   {_}
   break;}

  case 147:{  /* unary_math_expr: unary_math_op unary_math_expr */
#define unary_math_expr_unary_math_op_unary_math_expr 147
   constexpr int YYN = 147;
   {_}
   break;}

  case 148:{  /* unary_math_op: MUL */
#define unary_math_op_mul 148
   constexpr int YYN = 148;
   {_}
   break;}

  case 149:{  /* unary_math_op: ADD */
#define unary_math_op_add 149
   constexpr int YYN = 149;
   {_}
   break;}

  case 150:{  /* unary_math_op: SUB */
#define unary_math_op_sub 150
   constexpr int YYN = 150;
   {_}
   break;}

  case 151:{  /* unary_math_op: AND */
#define unary_math_op_and 151
   constexpr int YYN = 151;
   {_}
   break;}

  case 152:{  /* unary_math_op: TILDE */
#define unary_math_op_tilde 152
   constexpr int YYN = 152;
   {_}
   break;}

  case 153:{  /* unary_math_op: NOT */
#define unary_math_op_not 153
   constexpr int YYN = 153;
   {_}
   break;}

  case 154:{  /* multiplicative_math_expr: unary_math_expr */
#define multiplicative_math_expr_unary_math_expr 154
   constexpr int YYN = 154;
   {_}
   break;}

  case 155:{  /* multiplicative_math_expr: multiplicative_math_expr MUL unary_math_expr */
#define multiplicative_math_expr_multiplicative_math_expr_mul_unary_math_expr 155
   constexpr int YYN = 155;
   {_}
   break;}

  case 156:{  /* multiplicative_math_expr: multiplicative_math_expr DIV unary_math_expr */
#define multiplicative_math_expr_multiplicative_math_expr_div_unary_math_expr 156
   constexpr int YYN = 156;
   {_}
   break;}

  case 157:{  /* multiplicative_math_expr: multiplicative_math_expr MOD unary_math_expr */
#define multiplicative_math_expr_multiplicative_math_expr_mod_unary_math_expr 157
   constexpr int YYN = 157;
   {_}
   break;}

  case 158:{  /* additive_math_expr: multiplicative_math_expr */
#define additive_math_expr_multiplicative_math_expr 158
   constexpr int YYN = 158;
   {_}
   break;}

  case 159:{  /* additive_math_expr: additive_math_expr ADD multiplicative_math_expr */
#define additive_math_expr_additive_math_expr_add_multiplicative_math_expr 159
   constexpr int YYN = 159;
   {_}
   break;}

  case 160:{  /* additive_math_expr: additive_math_expr SUB multiplicative_math_expr */
#define additive_math_expr_additive_math_expr_sub_multiplicative_math_expr 160
   constexpr int YYN = 160;
   {_}
   break;}

  case 161:{  /* shift_math_expr: additive_math_expr */
#define shift_math_expr_additive_math_expr 161
   constexpr int YYN = 161;
   {_}
   break;}

  case 162:{  /* shift_math_expr: shift_math_expr LEFT_SHIFT additive_math_expr */
#define shift_math_expr_shift_math_expr_left_shift_additive_math_expr 162
   constexpr int YYN = 162;
   {_}
   break;}

  case 163:{  /* shift_math_expr: shift_math_expr RIGHT_SHIFT additive_math_expr */
#define shift_math_expr_shift_math_expr_right_shift_additive_math_expr 163
   constexpr int YYN = 163;
   {_}
   break;}

  case 164:{  /* relational_math_expr: shift_math_expr */
#define relational_math_expr_shift_math_expr 164
   constexpr int YYN = 164;
   {_}
   break;}

  case 165:{  /* relational_math_expr: relational_math_expr LT shift_math_expr */
#define relational_math_expr_relational_math_expr_lt_shift_math_expr 165
   constexpr int YYN = 165;
   {_}
   break;}

  case 166:{  /* relational_math_expr: relational_math_expr GT shift_math_expr */
#define relational_math_expr_relational_math_expr_gt_shift_math_expr 166
   constexpr int YYN = 166;
   {_}
   break;}

  case 167:{  /* relational_math_expr: relational_math_expr LT_EQ shift_math_expr */
#define relational_math_expr_relational_math_expr_lt_eq_shift_math_expr 167
   constexpr int YYN = 167;
   {_}
   break;}

  case 168:{  /* relational_math_expr: relational_math_expr GT_EQ shift_math_expr */
#define relational_math_expr_relational_math_expr_gt_eq_shift_math_expr 168
   constexpr int YYN = 168;
   {_}
   break;}

  case 169:{  /* equality_math_expr: relational_math_expr */
#define equality_math_expr_relational_math_expr 169
   constexpr int YYN = 169;
   {_}
   break;}

  case 170:{  /* equality_math_expr: equality_math_expr EQ_EQ relational_math_expr */
#define equality_math_expr_equality_math_expr_eq_eq_relational_math_expr 170
   constexpr int YYN = 170;
   {_}
   break;}

  case 171:{  /* equality_math_expr: equality_math_expr NOT_EQ relational_math_expr */
#define equality_math_expr_equality_math_expr_not_eq_relational_math_expr 171
   constexpr int YYN = 171;
   {_}
   break;}

  case 172:{  /* and_math_expr: equality_math_expr */
#define and_math_expr_equality_math_expr 172
   constexpr int YYN = 172;
   {_}
   break;}

  case 173:{  /* and_math_expr: and_math_expr AND equality_math_expr */
#define and_math_expr_and_math_expr_and_equality_math_expr 173
   constexpr int YYN = 173;
   {_}
   break;}

  case 174:{  /* exclusive_or_math_expr: and_math_expr */
#define exclusive_or_math_expr_and_math_expr 174
   constexpr int YYN = 174;
   {_}
   break;}

  case 175:{  /* exclusive_or_math_expr: exclusive_or_math_expr XOR and_math_expr */
#define exclusive_or_math_expr_exclusive_or_math_expr_xor_and_math_expr 175
   constexpr int YYN = 175;
   {_}
   break;}

  case 176:{  /* inclusive_or_math_expr: exclusive_or_math_expr */
#define inclusive_or_math_expr_exclusive_or_math_expr 176
   constexpr int YYN = 176;
   {_}
   break;}

  case 177:{  /* inclusive_or_math_expr: inclusive_or_math_expr OR exclusive_or_math_expr */
#define inclusive_or_math_expr_inclusive_or_math_expr_or_exclusive_or_math_expr 177
   constexpr int YYN = 177;
   {_}
   break;}

  case 178:{  /* logical_and_math_expr: inclusive_or_math_expr */
#define logical_and_math_expr_inclusive_or_math_expr 178
   constexpr int YYN = 178;
   {_}
   break;}

  case 179:{  /* logical_and_math_expr: logical_and_math_expr AND_AND inclusive_or_math_expr */
#define logical_and_math_expr_logical_and_math_expr_and_and_inclusive_or_math_expr 179
   constexpr int YYN = 179;
   {_}
   break;}

  case 180:{  /* logical_or_math_expr: logical_and_math_expr */
#define logical_or_math_expr_logical_and_math_expr 180
   constexpr int YYN = 180;
   {_}
   break;}

  case 181:{  /* logical_or_math_expr: logical_or_math_expr OR_OR logical_and_math_expr */
#define logical_or_math_expr_logical_or_math_expr_or_or_logical_and_math_expr 181
   constexpr int YYN = 181;
   {_}
   break;}

  case 182:{  /* conditional_math_expr: logical_or_math_expr */
#define conditional_math_expr_logical_or_math_expr 182
   constexpr int YYN = 182;
   {_}
   break;}

  case 183:{  /* assign_math_expr: conditional_math_expr */
#define assign_math_expr_conditional_math_expr 183
   constexpr int YYN = 183;
   {_}
   break;}

  case 184:{  /* assign_math_expr: unary_math_expr assign_math_op assign_math_expr */
#define assign_math_expr_unary_math_expr_assign_math_op_assign_math_expr 184
   constexpr int YYN = 184;
   {_}
   break;}

  case 185:{  /* assign_math_op: EQ */
#define assign_math_op_eq 185
   constexpr int YYN = 185;
   {_}
   break;}

  case 186:{  /* assign_math_op: ADD_EQ */
#define assign_math_op_add_eq 186
   constexpr int YYN = 186;
   {_}
   break;}

  case 187:{  /* assign_math_op: SUB_EQ */
#define assign_math_op_sub_eq 187
   constexpr int YYN = 187;
   {_}
   break;}

  case 188:{  /* assign_math_op: MUL_EQ */
#define assign_math_op_mul_eq 188
   constexpr int YYN = 188;
   {_}
   break;}

  case 189:{  /* assign_math_op: DIV_EQ */
#define assign_math_op_div_eq 189
   constexpr int YYN = 189;
   {_}
   break;}

  case 190:{  /* assign_math_op: MOD_EQ */
#define assign_math_op_mod_eq 190
   constexpr int YYN = 190;
   {_}
   break;}

  case 191:{  /* assign_math_op: XOR_EQ */
#define assign_math_op_xor_eq 191
   constexpr int YYN = 191;
   {_}
   break;}

  case 192:{  /* assign_math_op: AND_EQ */
#define assign_math_op_and_eq 192
   constexpr int YYN = 192;
   {_}
   break;}

  case 193:{  /* assign_math_op: OR_EQ */
#define assign_math_op_or_eq 193
   constexpr int YYN = 193;
   {_}
   break;}

  case 194:{  /* assign_math_op: LEFT_EQ */
#define assign_math_op_left_eq 194
   constexpr int YYN = 194;
   {_}
   break;}

  case 195:{  /* assign_math_op: RIGHT_EQ */
#define assign_math_op_right_eq 195
   constexpr int YYN = 195;
   {_}
   break;}

  case 196:{  /* math_expr: assign_math_expr */
#define math_expr_assign_math_expr 196
   constexpr int YYN = 196;
   {_}
   break;}

  case 197:{  /* math_expr: math_expr COMA assign_math_expr */
#define math_expr_math_expr_coma_assign_math_expr 197
   constexpr int YYN = 197;
   {_}
   break;}



        default: break;
      }
    if (yychar_backup != yychar)
      YY_LAC_DISCARD ("yychar change");
  }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      {
        yypcontext_t yyctx
          = {yyssp, yyesa, &yyes, &yyes_capacity, yytoken};
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        if (yychar != YYEMPTY)
          YY_LAC_ESTABLISH;
        yysyntax_error_status = yysyntax_error (&yymsg_alloc, &yymsg, &yyctx);
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == -1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = YY_CAST (char *,
                             YYSTACK_ALLOC (YY_CAST (YYSIZE_T, yymsg_alloc)));
            if (yymsg)
              {
                yysyntax_error_status
                  = yysyntax_error (&yymsg_alloc, &yymsg, &yyctx);
                yymsgp = yymsg;
              }
            else
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = YYENOMEM;
              }
          }
        yyerror (root, yymsgp);
        if (yysyntax_error_status == YYENOMEM)
          goto yyexhaustedlab;
      }
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, root);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp, root);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  /* If the stack popping above didn't lose the initial context for the
     current lookahead token, the shift below will for sure.  */
  YY_LAC_DISCARD ("error recovery");

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;


#if 1
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (root, YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturn;
#endif


/*-------------------------------------------------------.
| yyreturn -- parsing is finished, clean up and return.  |
`-------------------------------------------------------*/
yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, root);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp, root);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  if (yyes != yyesa)
    YYSTACK_FREE (yyes);
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
  return yyresult;
}

 /////////////////////////////////////////////////////////////////////////////
// *INDENT-ON*
namespace yy
{
std::array<bool,yyruletype::yynrules> rules = {false};
}

#ifndef YYUNDEFTOK
#define YYUNDEFTOK YYSYMBOL_YYUNDEF
#endif
const int yy::undef() { return YYUNDEFTOK; }

const int yy::r1(int yyn) { return yyr1[yyn]; }

const int yy::r2(int yyn) { return yyr2[yyn]; }


const char* const yy::SymbolName(int sn)
{ return yysymbol_name(YY_CAST(yysymbol_kind_t, sn)); }

const int yy::ntokens(void) { return YYNTOKENS; }

signed char yy::Translate(int token_num) { return YYTRANSLATE(token_num);}

template<int YYN>
void yy::rhs(YYSTYPE *lhs, int yyn, YYSTYPE *yyvsp)
{
   assert(YYN == yyn);
   const int sn = yy::r1(yyn);
   const int nrhs = yy::r2(yyn);
   const char* const rule = yy::SymbolName(sn);
   //DBG("\n\033[33mYYN:%d yyn:%d sn:%d rule:%s", YYN, yyn, sn, rule);
   YYSTYPE parent = *lhs = astNewRule<YYN>(sn, rule);
   if (nrhs == 0) { return; }
   YYSTYPE n = yyvsp[(0 + 1) - (nrhs)];
   astAddChild(parent, n);
   for (int i = 1; i < nrhs; i++)
   {
      parent = n;
      n = yyvsp[(i + 1) - (nrhs)];
      astAddNext(parent, n);
   }
}

// *****************************************************************************
void dfs(Node *n, struct Middlend &ir)
{
   if (!n) { return; }
   bool updown = true;
   Node *extra = n;
   n->Apply(ir, updown, &extra); // down
   const int N = n->Number(); // N = SN | RN
   assert(N > 0);
   if (n->IsRule())
   {
      if (N > YYNRULES) { DBG("\n\033[31m[rule] N:%d/%d",N,YYNRULES); }
      assert(N <= YYNRULES);
   }
   if (n->IsToken())
   {
      if (N >= YYMAXUTOK) { DBG("\n\033[31m[token] N:%d/%d",N,YYMAXUTOK); }
      assert(N < YYMAXUTOK);
   }
   if (n->IsRule()) { yy::rules.at(N) = true; } // Set the state flags
   if (updown && n->children)
   {
      dfs(n->children, ir);
      if (extra!=n) { dfs(extra, ir); }
   }

   if (n->next) { dfs(n->next, ir); }
   if (n->IsRule()) { yy::rules.at(N) = false; } // Reset the state flags
   if (n->IsRule()) {n->Apply(ir, updown = false);} // up, only for rules
}

// *****************************************************************************
template<int RN>
Node* astNewRule(const int sn, const char *rule)
{
   return astAddNode(std::make_shared<Rule<RN>>(sn, rule));
}

// *****************************************************************************
template<int RN>
void Rule<RN>::Apply(struct Middlend &ir, bool &dfs, Node **extra)
{
   if (dfs && yyecho) { DBG("\033[35m[Apply] Rule:%d %s", RN, Name().c_str()); }
   ir.middlend<RN>(this, dfs, extra);
}

// *****************************************************************************
template<int RN> const int Rule<RN>::SymbolNumber() const { return yy::r1(RN); }
