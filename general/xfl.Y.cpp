// A Bison parser, made by GNU Bison 3.7.3.

// Skeleton implementation for Bison LALR(1) parsers in C++

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

// DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
// especially those whose name start with YY_ or yy_.  They are
// private implementation details that can be changed or removed.



// First part of user prologue.

#include "xfl.hpp"
#include "xfl.Y.hpp"
#include "xfc.hpp"
#ifndef XFL_C
YY_DECL;
#endif

#ifndef XFL_C
using symbol_t = yy::parser::symbol_kind_type;

template<int YYN> void rhs(Node**,                       // &yylhs.value
                           const int,                    // yyn
                           const symbol_t,               // yyr1n (sn)
                           const int,                    // yyr2n (nrhs, yylen)
                           const char*,                  // symbol_name(sn)
                           yy::parser::stack_type&);     // yystack
#define RHS {\
    const unsigned char sn_yyn = yyr1_[yyn];\
    const symbol_t sn = yy::parser::yytranslate_(sn_yyn);\
    const char *rule = yy::parser::symbol_name(sn);\
    rhs<YYN>(&yylhs.value, yyn, sn, yyr2_[yyn], rule, yystack_);}
#else
template<int> void rhs(Node**, int, Node**);
#define RHS rhs<YYN>(&yyval, yyn, yyvsp);
#endif

// %warning: %token order has to be sync'ed with the lexer's one
#ifndef XFL_C
#define START_PREFIX ufl.
#else
#define START_PREFIX *
#endif



#include "xfl.Y.hpp"


// Unqualified %code blocks.
 #include "xfc.hpp" 



#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> // FIXME: INFRINGES ON USER NAME SPACE.
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif


// Whether we are compiled with exception support.
#ifndef YY_EXCEPTIONS
# if defined __GNUC__ && !defined __EXCEPTIONS
#  define YY_EXCEPTIONS 0
# else
#  define YY_EXCEPTIONS 1
# endif
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K].location)
/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

# ifndef YYLLOC_DEFAULT
#  define YYLLOC_DEFAULT(Current, Rhs, N)                               \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).begin  = YYRHSLOC (Rhs, 1).begin;                   \
          (Current).end    = YYRHSLOC (Rhs, N).end;                     \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).begin = (Current).end = YYRHSLOC (Rhs, 0).end;      \
        }                                                               \
    while (false)
# endif


// Enable debugging if requested.
#if YYDEBUG

// A pseudo ostream that takes yydebug_ into account.
# define YYCDEBUG if (yydebug_) (*yycdebug_)

# define YY_SYMBOL_PRINT(Title, Symbol)         \
  do {                                          \
    if (yydebug_)                               \
    {                                           \
      *yycdebug_ << Title << ' ';               \
      yy_print_ (*yycdebug_, Symbol);           \
      *yycdebug_ << '\n';                       \
    }                                           \
  } while (false)

# define YY_REDUCE_PRINT(Rule)          \
  do {                                  \
    if (yydebug_)                       \
      yy_reduce_print_ (Rule);          \
  } while (false)

# define YY_STACK_PRINT()               \
  do {                                  \
    if (yydebug_)                       \
      yy_stack_print_ ();                \
  } while (false)

#else // !YYDEBUG

# define YYCDEBUG if (false) std::cerr
# define YY_SYMBOL_PRINT(Title, Symbol)  YYUSE (Symbol)
# define YY_REDUCE_PRINT(Rule)           static_cast<void> (0)
# define YY_STACK_PRINT()                static_cast<void> (0)

#endif // !YYDEBUG

#define yyerrok         (yyerrstatus_ = 0)
#define yyclearin       (yyla.clear ())

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYRECOVERING()  (!!yyerrstatus_)

namespace yy {

  /// Build a parser object.
  parser::parser (xfl &ufl_yyarg)
#if YYDEBUG
    : yydebug_ (false),
      yycdebug_ (&std::cerr),
#else
    :
#endif
      yy_lac_established_ (false),
      ufl (ufl_yyarg)
  {}

  parser::~parser ()
  {}

  parser::syntax_error::~syntax_error () YY_NOEXCEPT YY_NOTHROW
  {}

  /*---------------.
  | symbol kinds.  |
  `---------------*/

  // basic_symbol.
  template <typename Base>
  parser::basic_symbol<Base>::basic_symbol (const basic_symbol& that)
    : Base (that)
    , value (that.value)
    , location (that.location)
  {}


  /// Constructor for valueless symbols.
  template <typename Base>
  parser::basic_symbol<Base>::basic_symbol (typename Base::kind_type t, YY_MOVE_REF (location_type) l)
    : Base (t)
    , value ()
    , location (l)
  {}

  template <typename Base>
  parser::basic_symbol<Base>::basic_symbol (typename Base::kind_type t, YY_RVREF (semantic_type) v, YY_RVREF (location_type) l)
    : Base (t)
    , value (YY_MOVE (v))
    , location (YY_MOVE (l))
  {}

  template <typename Base>
  parser::symbol_kind_type
  parser::basic_symbol<Base>::type_get () const YY_NOEXCEPT
  {
    return this->kind ();
  }

  template <typename Base>
  bool
  parser::basic_symbol<Base>::empty () const YY_NOEXCEPT
  {
    return this->kind () == symbol_kind::S_YYEMPTY;
  }

  template <typename Base>
  void
  parser::basic_symbol<Base>::move (basic_symbol& s)
  {
    super_type::move (s);
    value = YY_MOVE (s.value);
    location = YY_MOVE (s.location);
  }

  // by_kind.
  parser::by_kind::by_kind ()
    : kind_ (symbol_kind::S_YYEMPTY)
  {}

#if 201103L <= YY_CPLUSPLUS
  parser::by_kind::by_kind (by_kind&& that)
    : kind_ (that.kind_)
  {
    that.clear ();
  }
#endif

  parser::by_kind::by_kind (const by_kind& that)
    : kind_ (that.kind_)
  {}

  parser::by_kind::by_kind (token_kind_type t)
    : kind_ (yytranslate_ (t))
  {}

  void
  parser::by_kind::clear ()
  {
    kind_ = symbol_kind::S_YYEMPTY;
  }

  void
  parser::by_kind::move (by_kind& that)
  {
    kind_ = that.kind_;
    that.clear ();
  }

  parser::symbol_kind_type
  parser::by_kind::kind () const YY_NOEXCEPT
  {
    return kind_;
  }

  parser::symbol_kind_type
  parser::by_kind::type_get () const YY_NOEXCEPT
  {
    return this->kind ();
  }


  // by_state.
  parser::by_state::by_state () YY_NOEXCEPT
    : state (empty_state)
  {}

  parser::by_state::by_state (const by_state& that) YY_NOEXCEPT
    : state (that.state)
  {}

  void
  parser::by_state::clear () YY_NOEXCEPT
  {
    state = empty_state;
  }

  void
  parser::by_state::move (by_state& that)
  {
    state = that.state;
    that.clear ();
  }

  parser::by_state::by_state (state_type s) YY_NOEXCEPT
    : state (s)
  {}

  parser::symbol_kind_type
  parser::by_state::kind () const YY_NOEXCEPT
  {
    if (state == empty_state)
      return symbol_kind::S_YYEMPTY;
    else
      return YY_CAST (symbol_kind_type, yystos_[+state]);
  }

  parser::stack_symbol_type::stack_symbol_type ()
  {}

  parser::stack_symbol_type::stack_symbol_type (YY_RVREF (stack_symbol_type) that)
    : super_type (YY_MOVE (that.state), YY_MOVE (that.value), YY_MOVE (that.location))
  {
#if 201103L <= YY_CPLUSPLUS
    // that is emptied.
    that.state = empty_state;
#endif
  }

  parser::stack_symbol_type::stack_symbol_type (state_type s, YY_MOVE_REF (symbol_type) that)
    : super_type (s, YY_MOVE (that.value), YY_MOVE (that.location))
  {
    // that is emptied.
    that.kind_ = symbol_kind::S_YYEMPTY;
  }

#if YY_CPLUSPLUS < 201103L
  parser::stack_symbol_type&
  parser::stack_symbol_type::operator= (const stack_symbol_type& that)
  {
    state = that.state;
    value = that.value;
    location = that.location;
    return *this;
  }

  parser::stack_symbol_type&
  parser::stack_symbol_type::operator= (stack_symbol_type& that)
  {
    state = that.state;
    value = that.value;
    location = that.location;
    // that is emptied.
    that.state = empty_state;
    return *this;
  }
#endif

  template <typename Base>
  void
  parser::yy_destroy_ (const char* yymsg, basic_symbol<Base>& yysym) const
  {
    if (yymsg)
      YY_SYMBOL_PRINT (yymsg, yysym);

    // User destructor.
    YYUSE (yysym.kind ());
  }

#if YYDEBUG
  template <typename Base>
  void
  parser::yy_print_ (std::ostream& yyo, const basic_symbol<Base>& yysym) const
  {
    std::ostream& yyoutput = yyo;
    YYUSE (yyoutput);
    if (yysym.empty ())
      yyo << "empty symbol";
    else
      {
        symbol_kind_type yykind = yysym.kind ();
        yyo << (yykind < YYNTOKENS ? "token" : "nterm")
            << ' ' << yysym.name () << " ("
            << yysym.location << ": ";
        YYUSE (yykind);
        yyo << ')';
      }
  }
#endif

  void
  parser::yypush_ (const char* m, YY_MOVE_REF (stack_symbol_type) sym)
  {
    if (m)
      YY_SYMBOL_PRINT (m, sym);
    yystack_.push (YY_MOVE (sym));
  }

  void
  parser::yypush_ (const char* m, state_type s, YY_MOVE_REF (symbol_type) sym)
  {
#if 201103L <= YY_CPLUSPLUS
    yypush_ (m, stack_symbol_type (s, std::move (sym)));
#else
    stack_symbol_type ss (s, sym);
    yypush_ (m, ss);
#endif
  }

  void
  parser::yypop_ (int n)
  {
    yystack_.pop (n);
  }

#if YYDEBUG
  std::ostream&
  parser::debug_stream () const
  {
    return *yycdebug_;
  }

  void
  parser::set_debug_stream (std::ostream& o)
  {
    yycdebug_ = &o;
  }


  parser::debug_level_type
  parser::debug_level () const
  {
    return yydebug_;
  }

  void
  parser::set_debug_level (debug_level_type l)
  {
    yydebug_ = l;
  }
#endif // YYDEBUG

  parser::state_type
  parser::yy_lr_goto_state_ (state_type yystate, int yysym)
  {
    int yyr = yypgoto_[yysym - YYNTOKENS] + yystate;
    if (0 <= yyr && yyr <= yylast_ && yycheck_[yyr] == yystate)
      return yytable_[yyr];
    else
      return yydefgoto_[yysym - YYNTOKENS];
  }

  bool
  parser::yy_pact_value_is_default_ (int yyvalue)
  {
    return yyvalue == yypact_ninf_;
  }

  bool
  parser::yy_table_value_is_error_ (int yyvalue)
  {
    return yyvalue == yytable_ninf_;
  }

  int
  parser::operator() ()
  {
    return parse ();
  }

  int
  parser::parse ()
  {
    int yyn;
    /// Length of the RHS of the rule being reduced.
    int yylen = 0;

    // Error handling.
    int yynerrs_ = 0;
    int yyerrstatus_ = 0;

    /// The lookahead symbol.
    symbol_type yyla;

    /// The locations where the error started and ended.
    stack_symbol_type yyerror_range[3];

    /// The return value of parse ().
    int yyresult;

    /// Discard the LAC context in case there still is one left from a
    /// previous invocation.
    yy_lac_discard_ ("init");

#if YY_EXCEPTIONS
    try
#endif // YY_EXCEPTIONS
      {
    YYCDEBUG << "Starting parse\n";


    /* Initialize the stack.  The initial state will be set in
       yynewstate, since the latter expects the semantical and the
       location values to have been already stored, initialize these
       stacks with a primary value.  */
    yystack_.clear ();
    yypush_ (YY_NULLPTR, 0, YY_MOVE (yyla));

  /*-----------------------------------------------.
  | yynewstate -- push a new symbol on the stack.  |
  `-----------------------------------------------*/
  yynewstate:
    YYCDEBUG << "Entering state " << int (yystack_[0].state) << '\n';
    YY_STACK_PRINT ();

    // Accept?
    if (yystack_[0].state == yyfinal_)
      YYACCEPT;

    goto yybackup;


  /*-----------.
  | yybackup.  |
  `-----------*/
  yybackup:
    // Try to take a decision without lookahead.
    yyn = yypact_[+yystack_[0].state];
    if (yy_pact_value_is_default_ (yyn))
      goto yydefault;

    // Read a lookahead token.
    if (yyla.empty ())
      {
        YYCDEBUG << "Reading a token\n";
#if YY_EXCEPTIONS
        try
#endif // YY_EXCEPTIONS
          {
            yyla.kind_ = yytranslate_ (yylex (&yyla.value, &yyla.location, ufl));
          }
#if YY_EXCEPTIONS
        catch (const syntax_error& yyexc)
          {
            YYCDEBUG << "Caught exception: " << yyexc.what() << '\n';
            error (yyexc);
            goto yyerrlab1;
          }
#endif // YY_EXCEPTIONS
      }
    YY_SYMBOL_PRINT ("Next token is", yyla);

    if (yyla.kind () == symbol_kind::S_YYerror)
    {
      // The scanner already issued an error message, process directly
      // to error recovery.  But do not keep the error token as
      // lookahead, it is too special and may lead us to an endless
      // loop in error recovery. */
      yyla.kind_ = symbol_kind::S_YYUNDEF;
      goto yyerrlab1;
    }

    /* If the proper action on seeing token YYLA.TYPE is to reduce or
       to detect an error, take that action.  */
    yyn += yyla.kind ();
    if (yyn < 0 || yylast_ < yyn || yycheck_[yyn] != yyla.kind ())
      {
        if (!yy_lac_establish_ (yyla.kind ()))
           goto yyerrlab;
        goto yydefault;
      }

    // Reduce or error.
    yyn = yytable_[yyn];
    if (yyn <= 0)
      {
        if (yy_table_value_is_error_ (yyn))
          goto yyerrlab;
        if (!yy_lac_establish_ (yyla.kind ()))
           goto yyerrlab;

        yyn = -yyn;
        goto yyreduce;
      }

    // Count tokens shifted since error; after three, turn off error status.
    if (yyerrstatus_)
      --yyerrstatus_;

    // Shift the lookahead token.
    yypush_ ("Shifting", state_type (yyn), YY_MOVE (yyla));
    yy_lac_discard_ ("shift");
    goto yynewstate;


  /*-----------------------------------------------------------.
  | yydefault -- do the default action for the current state.  |
  `-----------------------------------------------------------*/
  yydefault:
    yyn = yydefact_[+yystack_[0].state];
    if (yyn == 0)
      goto yyerrlab;
    goto yyreduce;


  /*-----------------------------.
  | yyreduce -- do a reduction.  |
  `-----------------------------*/
  yyreduce:
    yylen = yyr2_[yyn];
    {
      stack_symbol_type yylhs;
      yylhs.state = yy_lr_goto_state_ (yystack_[yylen].state, yyr1_[yyn]);
      /* If YYLEN is nonzero, implement the default value of the
         action: '$$ = $1'.  Otherwise, use the top of the stack.

         Otherwise, the following line sets YYLHS.VALUE to garbage.
         This behavior is undocumented and Bison users should not rely
         upon it.  */
      if (yylen)
        yylhs.value = yystack_[yylen - 1].value;
      else
        yylhs.value = yystack_[0].value;

      // Default location.
      {
        stack_type::slice range (yystack_, yylen);
        YYLLOC_DEFAULT (yylhs.location, range, yylen);
        yyerror_range[1].location = yylhs.location;
      }

      // Perform the reduction.
      YY_REDUCE_PRINT (yyn);
#if YY_EXCEPTIONS
      try
#endif // YY_EXCEPTIONS
        {
          switch (yyn)
            {
  case 2:{  // entry_point: statements
#define entry_point_statements 2
   constexpr int YYN = 2;
   { RHS START_PREFIX root = yylhs.value; } {RHS}
   break;}

  case 3:{  // statements: statement
#define statements_statement 3
   constexpr int YYN = 3;
   {} {RHS}
   break;}

  case 4:{  // statements: statements statement
#define statements_statements_statement 4
   constexpr int YYN = 4;
   {} {RHS}
   break;}

  case 5:{  // statement: NL
#define statement_nl 5
   constexpr int YYN = 5;
   {} {RHS}
   break;}

  case 6:{  // statement: decl NL
#define statement_decl_nl 6
   constexpr int YYN = 6;
   {} {RHS}
   break;}

  case 7:{  // decl: function
#define decl_function 7
   constexpr int YYN = 7;
   {} {RHS}
   break;}

  case 8:{  // decl: domain assign_op expr
#define decl_domain_assign_op_expr 8
   constexpr int YYN = 8;
   {} {RHS}
   break;}

  case 9:{  // decl: id_list assign_op expr
#define decl_id_list_assign_op_expr 9
   constexpr int YYN = 9;
   {} {RHS}
   break;}

  case 10:{  // decl: LP id_list RP assign_op expr
#define decl_lp_id_list_rp_assign_op_expr 10
   constexpr int YYN = 10;
   {} {RHS}
   break;}

  case 11:{  // decl: LB id_list RB assign_op expr
#define decl_lb_id_list_rb_assign_op_expr 11
   constexpr int YYN = 11;
   {} {RHS}
   break;}

  case 12:{  // decl: iteration_statement
#define decl_iteration_statement 12
   constexpr int YYN = 12;
   {} {RHS}
   break;}

  case 13:{  // decl: direct_declarator
#define decl_direct_declarator 13
   constexpr int YYN = 13;
   {} {RHS}
   break;}

  case 14:{  // postfix_id: IDENTIFIER
#define postfix_id_identifier 14
   constexpr int YYN = 14;
   {} {RHS}
   break;}

  case 15:{  // id_list: postfix_id
#define id_list_postfix_id 15
   constexpr int YYN = 15;
   {} {RHS}
   break;}

  case 16:{  // id_list: id_list COMA postfix_id
#define id_list_id_list_coma_postfix_id 16
   constexpr int YYN = 16;
   {} {RHS}
   break;}

  case 17:{  // function: DEF IDENTIFIER LP args_expr_list RP COLON def_statements RETURN math_expr
#define function_def_identifier_lp_args_expr_list_rp_colon_def_statements_return_math_expr 17
   constexpr int YYN = 17;
   {} {RHS}
   break;}

  case 18:{  // def_statements: def_statement
#define def_statements_def_statement 18
   constexpr int YYN = 18;
   {} {RHS}
   break;}

  case 19:{  // def_statements: def_statements def_statement
#define def_statements_def_statements_def_statement 19
   constexpr int YYN = 19;
   {} {RHS}
   break;}

  case 20:{  // def_statement: NL
#define def_statement_nl 20
   constexpr int YYN = 20;
   {} {RHS}
   break;}

  case 21:{  // def_statement: id_list assign_op expr NL
#define def_statement_id_list_assign_op_expr_nl 21
   constexpr int YYN = 21;
   {} {RHS}
   break;}

  case 22:{  // def_statement: LP id_list RP assign_op expr NL
#define def_statement_lp_id_list_rp_assign_op_expr_nl 22
   constexpr int YYN = 22;
   {} {RHS}
   break;}

  case 23:{  // direct_declarator: postfix_id
#define direct_declarator_postfix_id 23
   constexpr int YYN = 23;
   {} {RHS}
   break;}

  case 24:{  // direct_declarator: direct_declarator LP RP
#define direct_declarator_direct_declarator_lp_rp 24
   constexpr int YYN = 24;
   {} {RHS}
   break;}

  case 25:{  // direct_declarator: direct_declarator LP expr RP
#define direct_declarator_direct_declarator_lp_expr_rp 25
   constexpr int YYN = 25;
   {} {RHS}
   break;}

  case 26:{  // iteration_statement: FOR IDENTIFIER IN RANGE LP IDENTIFIER RP COLON NL expr
#define iteration_statement_for_identifier_in_range_lp_identifier_rp_colon_nl_expr 26
   constexpr int YYN = 26;
   {} {RHS}
   break;}

  case 27:{  // domain: DOM_DX
#define domain_dom_dx 27
   constexpr int YYN = 27;
   {} {RHS}
   break;}

  case 28:{  // domain: EXT_DS
#define domain_ext_ds 28
   constexpr int YYN = 28;
   {} {RHS}
   break;}

  case 29:{  // domain: INT_DS
#define domain_int_ds 29
   constexpr int YYN = 29;
   {} {RHS}
   break;}

  case 30:{  // constant: NATURAL
#define constant_natural 30
   constexpr int YYN = 30;
   {} {RHS}
   break;}

  case 31:{  // constant: REAL
#define constant_real 31
   constexpr int YYN = 31;
   {} {RHS}
   break;}

  case 32:{  // strings: STRING
#define strings_string 32
   constexpr int YYN = 32;
   {} {RHS}
   break;}

  case 33:{  // strings: strings STRING
#define strings_strings_string 33
   constexpr int YYN = 33;
   {} {RHS}
   break;}

  case 34:{  // api: UNIT_SQUARE_MESH
#define api_unit_square_mesh 34
   constexpr int YYN = 34;
   {} {RHS}
   break;}

  case 35:{  // api: FUNCTION
#define api_function 35
   constexpr int YYN = 35;
   {} {RHS}
   break;}

  case 36:{  // api: FUNCTION_SPACE
#define api_function_space 36
   constexpr int YYN = 36;
   {} {RHS}
   break;}

  case 37:{  // api: EXPRESSION
#define api_expression 37
   constexpr int YYN = 37;
   {} {RHS}
   break;}

  case 38:{  // api: DIRICHLET_BC
#define api_dirichlet_bc 38
   constexpr int YYN = 38;
   {} {RHS}
   break;}

  case 39:{  // api: TRIAL_FUNCTION
#define api_trial_function 39
   constexpr int YYN = 39;
   {} {RHS}
   break;}

  case 40:{  // api: TEST_FUNCTION
#define api_test_function 40
   constexpr int YYN = 40;
   {} {RHS}
   break;}

  case 41:{  // api: CONSTANT_API
#define api_constant_api 41
   constexpr int YYN = 41;
   {} {RHS}
   break;}

  case 42:{  // primary_expr: IDENTIFIER
#define primary_expr_identifier 42
   constexpr int YYN = 42;
   {} {RHS}
   break;}

  case 43:{  // primary_expr: constant
#define primary_expr_constant 43
   constexpr int YYN = 43;
   {} {RHS}
   break;}

  case 44:{  // primary_expr: domain
#define primary_expr_domain 44
   constexpr int YYN = 44;
   {} {RHS}
   break;}

  case 45:{  // primary_expr: QUOTE
#define primary_expr_quote 45
   constexpr int YYN = 45;
   {} {RHS}
   break;}

  case 46:{  // primary_expr: strings
#define primary_expr_strings 46
   constexpr int YYN = 46;
   {} {RHS}
   break;}

  case 47:{  // primary_expr: LP expr RP
#define primary_expr_lp_expr_rp 47
   constexpr int YYN = 47;
   {} {RHS}
   break;}

  case 48:{  // primary_expr: LB expr RB
#define primary_expr_lb_expr_rb 48
   constexpr int YYN = 48;
   {} {RHS}
   break;}

  case 49:{  // primary_expr: LS coords RS
#define primary_expr_ls_coords_rs 49
   constexpr int YYN = 49;
   {} {RHS}
   break;}

  case 50:{  // primary_expr: api
#define primary_expr_api 50
   constexpr int YYN = 50;
   {} {RHS}
   break;}

  case 51:{  // pow_expr: postfix_expr POW constant
#define pow_expr_postfix_expr_pow_constant 51
   constexpr int YYN = 51;
   {} {RHS}
   break;}

  case 52:{  // pow_expr: postfix_expr POW IDENTIFIER
#define pow_expr_postfix_expr_pow_identifier 52
   constexpr int YYN = 52;
   {} {RHS}
   break;}

  case 53:{  // dot_expr: DOT_OP LP additive_expr COMA additive_expr RP
#define dot_expr_dot_op_lp_additive_expr_coma_additive_expr_rp 53
   constexpr int YYN = 53;
   {} {RHS}
   break;}

  case 54:{  // dot_expr: INNER_OP LP additive_expr COMA additive_expr RP
#define dot_expr_inner_op_lp_additive_expr_coma_additive_expr_rp 54
   constexpr int YYN = 54;
   {} {RHS}
   break;}

  case 55:{  // postfix_expr: primary_expr
#define postfix_expr_primary_expr 55
   constexpr int YYN = 55;
   {} {RHS}
   break;}

  case 56:{  // postfix_expr: pow_expr
#define postfix_expr_pow_expr 56
   constexpr int YYN = 56;
   {} {RHS}
   break;}

  case 57:{  // postfix_expr: postfix_expr LB expr RB
#define postfix_expr_postfix_expr_lb_expr_rb 57
   constexpr int YYN = 57;
   {} {RHS}
   break;}

  case 58:{  // postfix_expr: postfix_expr LP RP
#define postfix_expr_postfix_expr_lp_rp 58
   constexpr int YYN = 58;
   {} {RHS}
   break;}

  case 59:{  // postfix_expr: postfix_expr LP args_expr_list RP
#define postfix_expr_postfix_expr_lp_args_expr_list_rp 59
   constexpr int YYN = 59;
   {} {RHS}
   break;}

  case 60:{  // postfix_expr: postfix_expr DOT primary_expr
#define postfix_expr_postfix_expr_dot_primary_expr 60
   constexpr int YYN = 60;
   {} {RHS}
   break;}

  case 61:{  // postfix_expr: dot_expr
#define postfix_expr_dot_expr 61
   constexpr int YYN = 61;
   {} {RHS}
   break;}

  case 62:{  // postfix_expr: GRAD_OP LP additive_expr RP
#define postfix_expr_grad_op_lp_additive_expr_rp 62
   constexpr int YYN = 62;
   {} {RHS}
   break;}

  case 63:{  // postfix_expr: LHS LP IDENTIFIER RP
#define postfix_expr_lhs_lp_identifier_rp 63
   constexpr int YYN = 63;
   {} {RHS}
   break;}

  case 64:{  // postfix_expr: RHS LP IDENTIFIER RP
#define postfix_expr_rhs_lp_identifier_rp 64
   constexpr int YYN = 64;
   {} {RHS}
   break;}

  case 65:{  // unary_expr: postfix_expr
#define unary_expr_postfix_expr 65
   constexpr int YYN = 65;
   {} {RHS}
   break;}

  case 66:{  // unary_expr: unary_op cast_expr
#define unary_expr_unary_op_cast_expr 66
   constexpr int YYN = 66;
   {} {RHS}
   break;}

  case 67:{  // unary_op: MUL
#define unary_op_mul 67
   constexpr int YYN = 67;
   {} {RHS}
   break;}

  case 68:{  // unary_op: ADD
#define unary_op_add 68
   constexpr int YYN = 68;
   {} {RHS}
   break;}

  case 69:{  // unary_op: SUB
#define unary_op_sub 69
   constexpr int YYN = 69;
   {} {RHS}
   break;}

  case 70:{  // unary_op: TILDE
#define unary_op_tilde 70
   constexpr int YYN = 70;
   {} {RHS}
   break;}

  case 71:{  // unary_op: NOT
#define unary_op_not 71
   constexpr int YYN = 71;
   {} {RHS}
   break;}

  case 72:{  // cast_expr: unary_expr
#define cast_expr_unary_expr 72
   constexpr int YYN = 72;
   {} {RHS}
   break;}

  case 73:{  // multiplicative_expr: cast_expr
#define multiplicative_expr_cast_expr 73
   constexpr int YYN = 73;
   {} {RHS}
   break;}

  case 74:{  // multiplicative_expr: multiplicative_expr MUL cast_expr
#define multiplicative_expr_multiplicative_expr_mul_cast_expr 74
   constexpr int YYN = 74;
   {} {RHS}
   break;}

  case 75:{  // multiplicative_expr: multiplicative_expr DIV cast_expr
#define multiplicative_expr_multiplicative_expr_div_cast_expr 75
   constexpr int YYN = 75;
   {} {RHS}
   break;}

  case 76:{  // multiplicative_expr: multiplicative_expr MOD cast_expr
#define multiplicative_expr_multiplicative_expr_mod_cast_expr 76
   constexpr int YYN = 76;
   {} {RHS}
   break;}

  case 77:{  // additive_expr: multiplicative_expr
#define additive_expr_multiplicative_expr 77
   constexpr int YYN = 77;
   {} {RHS}
   break;}

  case 78:{  // additive_expr: additive_expr ADD multiplicative_expr
#define additive_expr_additive_expr_add_multiplicative_expr 78
   constexpr int YYN = 78;
   {} {RHS}
   break;}

  case 79:{  // additive_expr: additive_expr SUB multiplicative_expr
#define additive_expr_additive_expr_sub_multiplicative_expr 79
   constexpr int YYN = 79;
   {} {RHS}
   break;}

  case 80:{  // shift_expr: additive_expr
#define shift_expr_additive_expr 80
   constexpr int YYN = 80;
   {} {RHS}
   break;}

  case 81:{  // shift_expr: shift_expr LEFT_SHIFT additive_expr
#define shift_expr_shift_expr_left_shift_additive_expr 81
   constexpr int YYN = 81;
   {} {RHS}
   break;}

  case 82:{  // shift_expr: shift_expr RIGHT_SHIFT additive_expr
#define shift_expr_shift_expr_right_shift_additive_expr 82
   constexpr int YYN = 82;
   {} {RHS}
   break;}

  case 83:{  // relational_expr: shift_expr
#define relational_expr_shift_expr 83
   constexpr int YYN = 83;
   {} {RHS}
   break;}

  case 84:{  // relational_expr: relational_expr LT shift_expr
#define relational_expr_relational_expr_lt_shift_expr 84
   constexpr int YYN = 84;
   {} {RHS}
   break;}

  case 85:{  // relational_expr: relational_expr GT shift_expr
#define relational_expr_relational_expr_gt_shift_expr 85
   constexpr int YYN = 85;
   {} {RHS}
   break;}

  case 86:{  // relational_expr: relational_expr LT_EQ shift_expr
#define relational_expr_relational_expr_lt_eq_shift_expr 86
   constexpr int YYN = 86;
   {} {RHS}
   break;}

  case 87:{  // relational_expr: relational_expr GT_EQ shift_expr
#define relational_expr_relational_expr_gt_eq_shift_expr 87
   constexpr int YYN = 87;
   {} {RHS}
   break;}

  case 88:{  // equality_expr: relational_expr
#define equality_expr_relational_expr 88
   constexpr int YYN = 88;
   {} {RHS}
   break;}

  case 89:{  // equality_expr: equality_expr EQ_EQ relational_expr
#define equality_expr_equality_expr_eq_eq_relational_expr 89
   constexpr int YYN = 89;
   {} {RHS}
   break;}

  case 90:{  // equality_expr: equality_expr NOT_EQ relational_expr
#define equality_expr_equality_expr_not_eq_relational_expr 90
   constexpr int YYN = 90;
   {} {RHS}
   break;}

  case 91:{  // and_expr: equality_expr
#define and_expr_equality_expr 91
   constexpr int YYN = 91;
   {} {RHS}
   break;}

  case 92:{  // and_expr: and_expr AND equality_expr
#define and_expr_and_expr_and_equality_expr 92
   constexpr int YYN = 92;
   {} {RHS}
   break;}

  case 93:{  // exclusive_or_expr: and_expr
#define exclusive_or_expr_and_expr 93
   constexpr int YYN = 93;
   {} {RHS}
   break;}

  case 94:{  // exclusive_or_expr: exclusive_or_expr XOR and_expr
#define exclusive_or_expr_exclusive_or_expr_xor_and_expr 94
   constexpr int YYN = 94;
   {} {RHS}
   break;}

  case 95:{  // inclusive_or_expr: exclusive_or_expr
#define inclusive_or_expr_exclusive_or_expr 95
   constexpr int YYN = 95;
   {} {RHS}
   break;}

  case 96:{  // inclusive_or_expr: inclusive_or_expr OR exclusive_or_expr
#define inclusive_or_expr_inclusive_or_expr_or_exclusive_or_expr 96
   constexpr int YYN = 96;
   {} {RHS}
   break;}

  case 97:{  // logical_and_expr: inclusive_or_expr
#define logical_and_expr_inclusive_or_expr 97
   constexpr int YYN = 97;
   {} {RHS}
   break;}

  case 98:{  // logical_and_expr: logical_and_expr AND_AND inclusive_or_expr
#define logical_and_expr_logical_and_expr_and_and_inclusive_or_expr 98
   constexpr int YYN = 98;
   {} {RHS}
   break;}

  case 99:{  // logical_or_expr: logical_and_expr
#define logical_or_expr_logical_and_expr 99
   constexpr int YYN = 99;
   {} {RHS}
   break;}

  case 100:{  // logical_or_expr: logical_or_expr OR_OR logical_and_expr
#define logical_or_expr_logical_or_expr_or_or_logical_and_expr 100
   constexpr int YYN = 100;
   {} {RHS}
   break;}

  case 101:{  // conditional_expr: logical_or_expr
#define conditional_expr_logical_or_expr 101
   constexpr int YYN = 101;
   {} {RHS}
   break;}

  case 102:{  // conditional_expr: logical_or_expr QUESTION expr COLON conditional_expr
#define conditional_expr_logical_or_expr_question_expr_colon_conditional_expr 102
   constexpr int YYN = 102;
   {} {RHS}
   break;}

  case 103:{  // assign_expr: conditional_expr
#define assign_expr_conditional_expr 103
   constexpr int YYN = 103;
   {} {RHS}
   break;}

  case 104:{  // assign_expr: postfix_expr assign_op assign_expr
#define assign_expr_postfix_expr_assign_op_assign_expr 104
   constexpr int YYN = 104;
   {} {RHS}
   break;}

  case 105:{  // assign_op: EQ
#define assign_op_eq 105
   constexpr int YYN = 105;
   {} {RHS}
   break;}

  case 106:{  // assign_op: ADD_EQ
#define assign_op_add_eq 106
   constexpr int YYN = 106;
   {} {RHS}
   break;}

  case 107:{  // assign_op: SUB_EQ
#define assign_op_sub_eq 107
   constexpr int YYN = 107;
   {} {RHS}
   break;}

  case 108:{  // assign_op: MUL_EQ
#define assign_op_mul_eq 108
   constexpr int YYN = 108;
   {} {RHS}
   break;}

  case 109:{  // assign_op: DIV_EQ
#define assign_op_div_eq 109
   constexpr int YYN = 109;
   {} {RHS}
   break;}

  case 110:{  // assign_op: MOD_EQ
#define assign_op_mod_eq 110
   constexpr int YYN = 110;
   {} {RHS}
   break;}

  case 111:{  // assign_op: XOR_EQ
#define assign_op_xor_eq 111
   constexpr int YYN = 111;
   {} {RHS}
   break;}

  case 112:{  // assign_op: AND_EQ
#define assign_op_and_eq 112
   constexpr int YYN = 112;
   {} {RHS}
   break;}

  case 113:{  // assign_op: OR_EQ
#define assign_op_or_eq 113
   constexpr int YYN = 113;
   {} {RHS}
   break;}

  case 114:{  // assign_op: LEFT_EQ
#define assign_op_left_eq 114
   constexpr int YYN = 114;
   {} {RHS}
   break;}

  case 115:{  // assign_op: RIGHT_EQ
#define assign_op_right_eq 115
   constexpr int YYN = 115;
   {} {RHS}
   break;}

  case 116:{  // expr: assign_expr
#define expr_assign_expr 116
   constexpr int YYN = 116;
   {} {RHS}
   break;}

  case 117:{  // expr: expr COMA assign_expr
#define expr_expr_coma_assign_expr 117
   constexpr int YYN = 117;
   {} {RHS}
   break;}

  case 118:{  // args_expr_list: assign_expr
#define args_expr_list_assign_expr 118
   constexpr int YYN = 118;
   {} {RHS}
   break;}

  case 119:{  // args_expr_list: args_expr_list COMA assign_expr
#define args_expr_list_args_expr_list_coma_assign_expr 119
   constexpr int YYN = 119;
   {} {RHS}
   break;}

  case 120:{  // args_expr_list: args_expr_list COMA NL assign_expr
#define args_expr_list_args_expr_list_coma_nl_assign_expr 120
   constexpr int YYN = 120;
   {} {RHS}
   break;}

  case 121:{  // coord: LP constant COMA constant RP
#define coord_lp_constant_coma_constant_rp 121
   constexpr int YYN = 121;
   {} {RHS}
   break;}

  case 122:{  // coords: coord
#define coords_coord 122
   constexpr int YYN = 122;
   {} {RHS}
   break;}

  case 123:{  // coords: coords COLON coord
#define coords_coords_colon_coord 123
   constexpr int YYN = 123;
   {} {RHS}
   break;}

  case 124:{  // primary_math_expr: IDENTIFIER
#define primary_math_expr_identifier 124
   constexpr int YYN = 124;
   {} {RHS}
   break;}

  case 125:{  // primary_math_expr: constant
#define primary_math_expr_constant 125
   constexpr int YYN = 125;
   {} {RHS}
   break;}

  case 126:{  // primary_math_expr: domain
#define primary_math_expr_domain 126
   constexpr int YYN = 126;
   {} {RHS}
   break;}

  case 127:{  // primary_math_expr: QUOTE
#define primary_math_expr_quote 127
   constexpr int YYN = 127;
   {} {RHS}
   break;}

  case 128:{  // primary_math_expr: LP math_expr RP
#define primary_math_expr_lp_math_expr_rp 128
   constexpr int YYN = 128;
   {} {RHS}
   break;}

  case 129:{  // primary_math_expr: LB id_list RB
#define primary_math_expr_lb_id_list_rb 129
   constexpr int YYN = 129;
   {} {RHS}
   break;}

  case 130:{  // dot_math_expr: DOT_OP LP additive_expr COMA additive_expr RP
#define dot_math_expr_dot_op_lp_additive_expr_coma_additive_expr_rp 130
   constexpr int YYN = 130;
   {} {RHS}
   break;}

  case 131:{  // dot_math_expr: INNER_OP LP additive_expr COMA additive_expr RP
#define dot_math_expr_inner_op_lp_additive_expr_coma_additive_expr_rp 131
   constexpr int YYN = 131;
   {} {RHS}
   break;}

  case 132:{  // postfix_math_expr: primary_math_expr
#define postfix_math_expr_primary_math_expr 132
   constexpr int YYN = 132;
   {} {RHS}
   break;}

  case 133:{  // postfix_math_expr: postfix_math_expr LB math_expr RB
#define postfix_math_expr_postfix_math_expr_lb_math_expr_rb 133
   constexpr int YYN = 133;
   {} {RHS}
   break;}

  case 134:{  // postfix_math_expr: postfix_math_expr LP argument_math_expr_list RP
#define postfix_math_expr_postfix_math_expr_lp_argument_math_expr_list_rp 134
   constexpr int YYN = 134;
   {} {RHS}
   break;}

  case 135:{  // postfix_math_expr: postfix_math_expr DOT IDENTIFIER
#define postfix_math_expr_postfix_math_expr_dot_identifier 135
   constexpr int YYN = 135;
   {} {RHS}
   break;}

  case 136:{  // postfix_math_expr: postfix_math_expr INC_OP
#define postfix_math_expr_postfix_math_expr_inc_op 136
   constexpr int YYN = 136;
   {} {RHS}
   break;}

  case 137:{  // postfix_math_expr: postfix_math_expr DEC_OP
#define postfix_math_expr_postfix_math_expr_dec_op 137
   constexpr int YYN = 137;
   {} {RHS}
   break;}

  case 138:{  // postfix_math_expr: postfix_math_expr POW constant
#define postfix_math_expr_postfix_math_expr_pow_constant 138
   constexpr int YYN = 138;
   {} {RHS}
   break;}

  case 139:{  // postfix_math_expr: dot_math_expr
#define postfix_math_expr_dot_math_expr 139
   constexpr int YYN = 139;
   {} {RHS}
   break;}

  case 140:{  // postfix_math_expr: GRAD_OP LP additive_math_expr RP
#define postfix_math_expr_grad_op_lp_additive_math_expr_rp 140
   constexpr int YYN = 140;
   {} {RHS}
   break;}

  case 141:{  // argument_math_expr_list: assign_math_expr
#define argument_math_expr_list_assign_math_expr 141
   constexpr int YYN = 141;
   {} {RHS}
   break;}

  case 142:{  // argument_math_expr_list: argument_math_expr_list COMA assign_math_expr
#define argument_math_expr_list_argument_math_expr_list_coma_assign_math_expr 142
   constexpr int YYN = 142;
   {} {RHS}
   break;}

  case 143:{  // unary_math_expr: postfix_math_expr
#define unary_math_expr_postfix_math_expr 143
   constexpr int YYN = 143;
   {} {RHS}
   break;}

  case 144:{  // unary_math_expr: INC_OP unary_math_expr
#define unary_math_expr_inc_op_unary_math_expr 144
   constexpr int YYN = 144;
   {} {RHS}
   break;}

  case 145:{  // unary_math_expr: DEC_OP unary_math_expr
#define unary_math_expr_dec_op_unary_math_expr 145
   constexpr int YYN = 145;
   {} {RHS}
   break;}

  case 146:{  // unary_math_expr: MOD unary_math_expr
#define unary_math_expr_mod_unary_math_expr 146
   constexpr int YYN = 146;
   {} {RHS}
   break;}

  case 147:{  // unary_math_expr: unary_math_op unary_math_expr
#define unary_math_expr_unary_math_op_unary_math_expr 147
   constexpr int YYN = 147;
   {} {RHS}
   break;}

  case 148:{  // unary_math_op: MUL
#define unary_math_op_mul 148
   constexpr int YYN = 148;
   {} {RHS}
   break;}

  case 149:{  // unary_math_op: ADD
#define unary_math_op_add 149
   constexpr int YYN = 149;
   {} {RHS}
   break;}

  case 150:{  // unary_math_op: SUB
#define unary_math_op_sub 150
   constexpr int YYN = 150;
   {} {RHS}
   break;}

  case 151:{  // unary_math_op: AND
#define unary_math_op_and 151
   constexpr int YYN = 151;
   {} {RHS}
   break;}

  case 152:{  // unary_math_op: TILDE
#define unary_math_op_tilde 152
   constexpr int YYN = 152;
   {} {RHS}
   break;}

  case 153:{  // unary_math_op: NOT
#define unary_math_op_not 153
   constexpr int YYN = 153;
   {} {RHS}
   break;}

  case 154:{  // multiplicative_math_expr: unary_math_expr
#define multiplicative_math_expr_unary_math_expr 154
   constexpr int YYN = 154;
   {} {RHS}
   break;}

  case 155:{  // multiplicative_math_expr: multiplicative_math_expr MUL unary_math_expr
#define multiplicative_math_expr_multiplicative_math_expr_mul_unary_math_expr 155
   constexpr int YYN = 155;
   {} {RHS}
   break;}

  case 156:{  // multiplicative_math_expr: multiplicative_math_expr DIV unary_math_expr
#define multiplicative_math_expr_multiplicative_math_expr_div_unary_math_expr 156
   constexpr int YYN = 156;
   {} {RHS}
   break;}

  case 157:{  // multiplicative_math_expr: multiplicative_math_expr MOD unary_math_expr
#define multiplicative_math_expr_multiplicative_math_expr_mod_unary_math_expr 157
   constexpr int YYN = 157;
   {} {RHS}
   break;}

  case 158:{  // additive_math_expr: multiplicative_math_expr
#define additive_math_expr_multiplicative_math_expr 158
   constexpr int YYN = 158;
   {} {RHS}
   break;}

  case 159:{  // additive_math_expr: additive_math_expr ADD multiplicative_math_expr
#define additive_math_expr_additive_math_expr_add_multiplicative_math_expr 159
   constexpr int YYN = 159;
   {} {RHS}
   break;}

  case 160:{  // additive_math_expr: additive_math_expr SUB multiplicative_math_expr
#define additive_math_expr_additive_math_expr_sub_multiplicative_math_expr 160
   constexpr int YYN = 160;
   {} {RHS}
   break;}

  case 161:{  // shift_math_expr: additive_math_expr
#define shift_math_expr_additive_math_expr 161
   constexpr int YYN = 161;
   {} {RHS}
   break;}

  case 162:{  // shift_math_expr: shift_math_expr LEFT_SHIFT additive_math_expr
#define shift_math_expr_shift_math_expr_left_shift_additive_math_expr 162
   constexpr int YYN = 162;
   {} {RHS}
   break;}

  case 163:{  // shift_math_expr: shift_math_expr RIGHT_SHIFT additive_math_expr
#define shift_math_expr_shift_math_expr_right_shift_additive_math_expr 163
   constexpr int YYN = 163;
   {} {RHS}
   break;}

  case 164:{  // relational_math_expr: shift_math_expr
#define relational_math_expr_shift_math_expr 164
   constexpr int YYN = 164;
   {} {RHS}
   break;}

  case 165:{  // relational_math_expr: relational_math_expr LT shift_math_expr
#define relational_math_expr_relational_math_expr_lt_shift_math_expr 165
   constexpr int YYN = 165;
   {} {RHS}
   break;}

  case 166:{  // relational_math_expr: relational_math_expr GT shift_math_expr
#define relational_math_expr_relational_math_expr_gt_shift_math_expr 166
   constexpr int YYN = 166;
   {} {RHS}
   break;}

  case 167:{  // relational_math_expr: relational_math_expr LT_EQ shift_math_expr
#define relational_math_expr_relational_math_expr_lt_eq_shift_math_expr 167
   constexpr int YYN = 167;
   {} {RHS}
   break;}

  case 168:{  // relational_math_expr: relational_math_expr GT_EQ shift_math_expr
#define relational_math_expr_relational_math_expr_gt_eq_shift_math_expr 168
   constexpr int YYN = 168;
   {} {RHS}
   break;}

  case 169:{  // equality_math_expr: relational_math_expr
#define equality_math_expr_relational_math_expr 169
   constexpr int YYN = 169;
   {} {RHS}
   break;}

  case 170:{  // equality_math_expr: equality_math_expr EQ_EQ relational_math_expr
#define equality_math_expr_equality_math_expr_eq_eq_relational_math_expr 170
   constexpr int YYN = 170;
   {} {RHS}
   break;}

  case 171:{  // equality_math_expr: equality_math_expr NOT_EQ relational_math_expr
#define equality_math_expr_equality_math_expr_not_eq_relational_math_expr 171
   constexpr int YYN = 171;
   {} {RHS}
   break;}

  case 172:{  // and_math_expr: equality_math_expr
#define and_math_expr_equality_math_expr 172
   constexpr int YYN = 172;
   {} {RHS}
   break;}

  case 173:{  // and_math_expr: and_math_expr AND equality_math_expr
#define and_math_expr_and_math_expr_and_equality_math_expr 173
   constexpr int YYN = 173;
   {} {RHS}
   break;}

  case 174:{  // exclusive_or_math_expr: and_math_expr
#define exclusive_or_math_expr_and_math_expr 174
   constexpr int YYN = 174;
   {} {RHS}
   break;}

  case 175:{  // exclusive_or_math_expr: exclusive_or_math_expr XOR and_math_expr
#define exclusive_or_math_expr_exclusive_or_math_expr_xor_and_math_expr 175
   constexpr int YYN = 175;
   {} {RHS}
   break;}

  case 176:{  // inclusive_or_math_expr: exclusive_or_math_expr
#define inclusive_or_math_expr_exclusive_or_math_expr 176
   constexpr int YYN = 176;
   {} {RHS}
   break;}

  case 177:{  // inclusive_or_math_expr: inclusive_or_math_expr OR exclusive_or_math_expr
#define inclusive_or_math_expr_inclusive_or_math_expr_or_exclusive_or_math_expr 177
   constexpr int YYN = 177;
   {} {RHS}
   break;}

  case 178:{  // logical_and_math_expr: inclusive_or_math_expr
#define logical_and_math_expr_inclusive_or_math_expr 178
   constexpr int YYN = 178;
   {} {RHS}
   break;}

  case 179:{  // logical_and_math_expr: logical_and_math_expr AND_AND inclusive_or_math_expr
#define logical_and_math_expr_logical_and_math_expr_and_and_inclusive_or_math_expr 179
   constexpr int YYN = 179;
   {} {RHS}
   break;}

  case 180:{  // logical_or_math_expr: logical_and_math_expr
#define logical_or_math_expr_logical_and_math_expr 180
   constexpr int YYN = 180;
   {} {RHS}
   break;}

  case 181:{  // logical_or_math_expr: logical_or_math_expr OR_OR logical_and_math_expr
#define logical_or_math_expr_logical_or_math_expr_or_or_logical_and_math_expr 181
   constexpr int YYN = 181;
   {} {RHS}
   break;}

  case 182:{  // conditional_math_expr: logical_or_math_expr
#define conditional_math_expr_logical_or_math_expr 182
   constexpr int YYN = 182;
   {} {RHS}
   break;}

  case 183:{  // assign_math_expr: conditional_math_expr
#define assign_math_expr_conditional_math_expr 183
   constexpr int YYN = 183;
   {} {RHS}
   break;}

  case 184:{  // assign_math_expr: unary_math_expr assign_math_op assign_math_expr
#define assign_math_expr_unary_math_expr_assign_math_op_assign_math_expr 184
   constexpr int YYN = 184;
   {} {RHS}
   break;}

  case 185:{  // assign_math_op: EQ
#define assign_math_op_eq 185
   constexpr int YYN = 185;
   {} {RHS}
   break;}

  case 186:{  // assign_math_op: ADD_EQ
#define assign_math_op_add_eq 186
   constexpr int YYN = 186;
   {} {RHS}
   break;}

  case 187:{  // assign_math_op: SUB_EQ
#define assign_math_op_sub_eq 187
   constexpr int YYN = 187;
   {} {RHS}
   break;}

  case 188:{  // assign_math_op: MUL_EQ
#define assign_math_op_mul_eq 188
   constexpr int YYN = 188;
   {} {RHS}
   break;}

  case 189:{  // assign_math_op: DIV_EQ
#define assign_math_op_div_eq 189
   constexpr int YYN = 189;
   {} {RHS}
   break;}

  case 190:{  // assign_math_op: MOD_EQ
#define assign_math_op_mod_eq 190
   constexpr int YYN = 190;
   {} {RHS}
   break;}

  case 191:{  // assign_math_op: XOR_EQ
#define assign_math_op_xor_eq 191
   constexpr int YYN = 191;
   {} {RHS}
   break;}

  case 192:{  // assign_math_op: AND_EQ
#define assign_math_op_and_eq 192
   constexpr int YYN = 192;
   {} {RHS}
   break;}

  case 193:{  // assign_math_op: OR_EQ
#define assign_math_op_or_eq 193
   constexpr int YYN = 193;
   {} {RHS}
   break;}

  case 194:{  // assign_math_op: LEFT_EQ
#define assign_math_op_left_eq 194
   constexpr int YYN = 194;
   {} {RHS}
   break;}

  case 195:{  // assign_math_op: RIGHT_EQ
#define assign_math_op_right_eq 195
   constexpr int YYN = 195;
   {} {RHS}
   break;}

  case 196:{  // math_expr: assign_math_expr
#define math_expr_assign_math_expr 196
   constexpr int YYN = 196;
   {} {RHS}
   break;}

  case 197:{  // math_expr: math_expr COMA assign_math_expr
#define math_expr_math_expr_coma_assign_math_expr 197
   constexpr int YYN = 197;
   {} {RHS}
   break;}



            default:
              break;
            }
        }
#if YY_EXCEPTIONS
      catch (const syntax_error& yyexc)
        {
          YYCDEBUG << "Caught exception: " << yyexc.what() << '\n';
          error (yyexc);
          YYERROR;
        }
#endif // YY_EXCEPTIONS
      YY_SYMBOL_PRINT ("-> $$ =", yylhs);
      yypop_ (yylen);
      yylen = 0;

      // Shift the result of the reduction.
      yypush_ (YY_NULLPTR, YY_MOVE (yylhs));
    }
    goto yynewstate;


  /*--------------------------------------.
  | yyerrlab -- here on detecting error.  |
  `--------------------------------------*/
  yyerrlab:
    // If not already recovering from an error, report this error.
    if (!yyerrstatus_)
      {
        ++yynerrs_;
        context yyctx (*this, yyla);
        std::string msg = yysyntax_error_ (yyctx);
        error (yyla.location, YY_MOVE (msg));
      }


    yyerror_range[1].location = yyla.location;
    if (yyerrstatus_ == 3)
      {
        /* If just tried and failed to reuse lookahead token after an
           error, discard it.  */

        // Return failure if at end of input.
        if (yyla.kind () == symbol_kind::S_YYEOF)
          YYABORT;
        else if (!yyla.empty ())
          {
            yy_destroy_ ("Error: discarding", yyla);
            yyla.clear ();
          }
      }

    // Else will try to reuse lookahead token after shifting the error token.
    goto yyerrlab1;


  /*---------------------------------------------------.
  | yyerrorlab -- error raised explicitly by YYERROR.  |
  `---------------------------------------------------*/
  yyerrorlab:
    /* Pacify compilers when the user code never invokes YYERROR and
       the label yyerrorlab therefore never appears in user code.  */
    if (false)
      YYERROR;

    /* Do not reclaim the symbols of the rule whose action triggered
       this YYERROR.  */
    yypop_ (yylen);
    yylen = 0;
    YY_STACK_PRINT ();
    goto yyerrlab1;


  /*-------------------------------------------------------------.
  | yyerrlab1 -- common code for both syntax error and YYERROR.  |
  `-------------------------------------------------------------*/
  yyerrlab1:
    yyerrstatus_ = 3;   // Each real token shifted decrements this.
    // Pop stack until we find a state that shifts the error token.
    for (;;)
      {
        yyn = yypact_[+yystack_[0].state];
        if (!yy_pact_value_is_default_ (yyn))
          {
            yyn += symbol_kind::S_YYerror;
            if (0 <= yyn && yyn <= yylast_
                && yycheck_[yyn] == symbol_kind::S_YYerror)
              {
                yyn = yytable_[yyn];
                if (0 < yyn)
                  break;
              }
          }

        // Pop the current state because it cannot handle the error token.
        if (yystack_.size () == 1)
          YYABORT;

        yyerror_range[1].location = yystack_[0].location;
        yy_destroy_ ("Error: popping", yystack_[0]);
        yypop_ ();
        YY_STACK_PRINT ();
      }
    {
      stack_symbol_type error_token;

      yyerror_range[2].location = yyla.location;
      YYLLOC_DEFAULT (error_token.location, yyerror_range, 2);

      // Shift the error token.
      yy_lac_discard_ ("error recovery");
      error_token.state = state_type (yyn);
      yypush_ ("Shifting", YY_MOVE (error_token));
    }
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


  /*-----------------------------------------------------.
  | yyreturn -- parsing is finished, return the result.  |
  `-----------------------------------------------------*/
  yyreturn:
    if (!yyla.empty ())
      yy_destroy_ ("Cleanup: discarding lookahead", yyla);

    /* Do not reclaim the symbols of the rule whose action triggered
       this YYABORT or YYACCEPT.  */
    yypop_ (yylen);
    YY_STACK_PRINT ();
    while (1 < yystack_.size ())
      {
        yy_destroy_ ("Cleanup: popping", yystack_[0]);
        yypop_ ();
      }

    return yyresult;
  }
#if YY_EXCEPTIONS
    catch (...)
      {
        YYCDEBUG << "Exception caught: cleaning lookahead and stack\n";
        // Do not try to display the values of the reclaimed symbols,
        // as their printers might throw an exception.
        if (!yyla.empty ())
          yy_destroy_ (YY_NULLPTR, yyla);

        while (1 < yystack_.size ())
          {
            yy_destroy_ (YY_NULLPTR, yystack_[0]);
            yypop_ ();
          }
        throw;
      }
#endif // YY_EXCEPTIONS
  }

  void
  parser::error (const syntax_error& yyexc)
  {
    error (yyexc.location, yyexc.what ());
  }

  const char *
  parser::symbol_name (symbol_kind_type yysymbol)
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



  // parser::context.
  parser::context::context (const parser& yyparser, const symbol_type& yyla)
    : yyparser_ (yyparser)
    , yyla_ (yyla)
  {}

  int
  parser::context::expected_tokens (symbol_kind_type yyarg[], int yyargn) const
  {
    // Actual number of expected tokens
    int yycount = 0;

#if YYDEBUG
    // Execute LAC once. We don't care if it is successful, we
    // only do it for the sake of debugging output.
    if (!yyparser_.yy_lac_established_)
      yyparser_.yy_lac_check_ (yyla_.kind ());
#endif

    for (int yyx = 0; yyx < YYNTOKENS; ++yyx)
      {
        symbol_kind_type yysym = YY_CAST (symbol_kind_type, yyx);
        if (yysym != symbol_kind::S_YYerror
            && yysym != symbol_kind::S_YYUNDEF
            && yyparser_.yy_lac_check_ (yysym))
          {
            if (!yyarg)
              ++yycount;
            else if (yycount == yyargn)
              return 0;
            else
              yyarg[yycount++] = yysym;
          }
      }
    if (yyarg && yycount == 0 && 0 < yyargn)
      yyarg[0] = symbol_kind::S_YYEMPTY;
    return yycount;
  }


  bool
  parser::yy_lac_check_ (symbol_kind_type yytoken) const
  {
    // Logically, the yylac_stack's lifetime is confined to this function.
    // Clear it, to get rid of potential left-overs from previous call.
    yylac_stack_.clear ();
    // Reduce until we encounter a shift and thereby accept the token.
#if YYDEBUG
    YYCDEBUG << "LAC: checking lookahead " << symbol_name (yytoken) << ':';
#endif
    std::ptrdiff_t lac_top = 0;
    while (true)
      {
        state_type top_state = (yylac_stack_.empty ()
                                ? yystack_[lac_top].state
                                : yylac_stack_.back ());
        int yyrule = yypact_[+top_state];
        if (yy_pact_value_is_default_ (yyrule)
            || (yyrule += yytoken) < 0 || yylast_ < yyrule
            || yycheck_[yyrule] != yytoken)
          {
            // Use the default action.
            yyrule = yydefact_[+top_state];
            if (yyrule == 0)
              {
                YYCDEBUG << " Err\n";
                return false;
              }
          }
        else
          {
            // Use the action from yytable.
            yyrule = yytable_[yyrule];
            if (yy_table_value_is_error_ (yyrule))
              {
                YYCDEBUG << " Err\n";
                return false;
              }
            if (0 < yyrule)
              {
                YYCDEBUG << " S" << yyrule << '\n';
                return true;
              }
            yyrule = -yyrule;
          }
        // By now we know we have to simulate a reduce.
        YYCDEBUG << " R" << yyrule - 1;
        // Pop the corresponding number of values from the stack.
        {
          std::ptrdiff_t yylen = yyr2_[yyrule];
          // First pop from the LAC stack as many tokens as possible.
          std::ptrdiff_t lac_size = std::ptrdiff_t (yylac_stack_.size ());
          if (yylen < lac_size)
            {
              yylac_stack_.resize (std::size_t (lac_size - yylen));
              yylen = 0;
            }
          else if (lac_size)
            {
              yylac_stack_.clear ();
              yylen -= lac_size;
            }
          // Only afterwards look at the main stack.
          // We simulate popping elements by incrementing lac_top.
          lac_top += yylen;
        }
        // Keep top_state in sync with the updated stack.
        top_state = (yylac_stack_.empty ()
                     ? yystack_[lac_top].state
                     : yylac_stack_.back ());
        // Push the resulting state of the reduction.
        state_type state = yy_lr_goto_state_ (top_state, yyr1_[yyrule]);
        YYCDEBUG << " G" << int (state);
        yylac_stack_.push_back (state);
      }
  }

  // Establish the initial context if no initial context currently exists.
  bool
  parser::yy_lac_establish_ (symbol_kind_type yytoken)
  {
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

       yy_lac_establish_ should be invoked when a reduction is about to be
       performed in an inconsistent state (which, for the purposes of LAC,
       includes consistent states that don't know they're consistent because
       their default reductions have been disabled).

       For parse.lac=full, the implementation of yy_lac_establish_ is as
       follows.  If no initial context is currently established for the
       current lookahead, then check if that lookahead can eventually be
       shifted if syntactic actions continue from the current context.  */
    if (!yy_lac_established_)
      {
#if YYDEBUG
        YYCDEBUG << "LAC: initial context established for "
                 << symbol_name (yytoken) << '\n';
#endif
        yy_lac_established_ = true;
        return yy_lac_check_ (yytoken);
      }
    return true;
  }

  // Discard any previous initial lookahead context.
  void
  parser::yy_lac_discard_ (const char* evt)
  {
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
    if (yy_lac_established_)
      {
        YYCDEBUG << "LAC: initial context discarded due to "
                 << evt << '\n';
        yy_lac_established_ = false;
      }
  }

  int
  parser::yy_syntax_error_arguments_ (const context& yyctx,
                                                 symbol_kind_type yyarg[], int yyargn) const
  {
    /* There are many possibilities here to consider:
       - If this state is a consistent state with a default action, then
         the only way this function was invoked is if the default action
         is an error action.  In that case, don't check for expected
         tokens because there are none.
       - The only way there can be no lookahead present (in yyla) is
         if this state is a consistent state with a default action.
         Thus, detecting the absence of a lookahead is sufficient to
         determine that there is no unexpected or expected token to
         report.  In that case, just report a simple "syntax error".
       - Don't assume there isn't a lookahead just because this state is
         a consistent state with a default action.  There might have
         been a previous inconsistent state, consistent state with a
         non-default action, or user semantic action that manipulated
         yyla.  (However, yyla is currently not documented for users.)
         In the first two cases, it might appear that the current syntax
         error should have been detected in the previous state when
         yy_lac_check was invoked.  However, at that time, there might
         have been a different syntax error that discarded a different
         initial context during error recovery, leaving behind the
         current lookahead.
    */

    if (!yyctx.lookahead ().empty ())
      {
        if (yyarg)
          yyarg[0] = yyctx.token ();
        int yyn = yyctx.expected_tokens (yyarg ? yyarg + 1 : yyarg, yyargn - 1);
        return yyn + 1;
      }
    return 0;
  }

  // Generate an error message.
  std::string
  parser::yysyntax_error_ (const context& yyctx) const
  {
    // Its maximum.
    enum { YYARGS_MAX = 5 };
    // Arguments of yyformat.
    symbol_kind_type yyarg[YYARGS_MAX];
    int yycount = yy_syntax_error_arguments_ (yyctx, yyarg, YYARGS_MAX);

    char const* yyformat = YY_NULLPTR;
    switch (yycount)
      {
#define YYCASE_(N, S)                         \
        case N:                               \
          yyformat = S;                       \
        break
      default: // Avoid compiler warnings.
        YYCASE_ (0, YY_("syntax error"));
        YYCASE_ (1, YY_("syntax error, unexpected %s"));
        YYCASE_ (2, YY_("syntax error, unexpected %s, expecting %s"));
        YYCASE_ (3, YY_("syntax error, unexpected %s, expecting %s or %s"));
        YYCASE_ (4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
        YYCASE_ (5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
      }

    std::string yyres;
    // Argument number.
    std::ptrdiff_t yyi = 0;
    for (char const* yyp = yyformat; *yyp; ++yyp)
      if (yyp[0] == '%' && yyp[1] == 's' && yyi < yycount)
        {
          yyres += symbol_name (yyarg[yyi++]);
          ++yyp;
        }
      else
        yyres += *yyp;
    return yyres;
  }


  const short parser::yypact_ninf_ = -264;

  const signed char parser::yytable_ninf_ = -24;

  const short
  parser::yypact_[] =
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

  const unsigned char
  parser::yydefact_[] =
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

  const short
  parser::yypgoto_[] =
  {
    -264,  -264,  -264,   394,  -264,    33,    20,  -264,  -264,   202,
    -264,  -264,     0,  -102,  -264,  -264,   287,  -264,  -264,    -5,
    -264,  -264,   -71,   254,  -103,   127,   248,   271,   274,   270,
     273,   275,  -264,   214,   -42,     3,    -2,   300,   262,  -264,
    -264,  -264,  -264,  -264,  -180,  -264,    95,  -216,   -31,    93,
     114,   150,   153,   116,   157,  -264,  -264,  -263,  -264,  -220
  };

  const short
  parser::yydefgoto_[] =
  {
      -1,    10,    11,    12,    13,    22,    15,    16,   204,   205,
      17,    18,    74,    75,    76,    77,    78,    79,    80,   122,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,    96,    40,    97,   102,   112,   113,
     240,   241,   242,   316,   311,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   286,   257
  };

  const short
  parser::yytable_[] =
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

  const short
  parser::yycheck_[] =
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

  const unsigned char
  parser::yystos_[] =
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

  const unsigned char
  parser::yyr1_[] =
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

  const signed char
  parser::yyr2_[] =
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




#if YYDEBUG
  const short
  parser::yyrline_[] =
  {
       0,   101,   101,   103,   103,   105,   105,   107,   108,   109,
     110,   111,   112,   113,   115,   119,   119,   123,   125,   125,
     126,   127,   128,   131,   132,   133,   136,   139,   139,   139,
     142,   142,   144,   144,   146,   147,   148,   149,   150,   151,
     152,   153,   155,   156,   157,   158,   159,   160,   161,   162,
     163,   165,   166,   168,   169,   171,   172,   173,   174,   175,
     176,   177,   178,   179,   180,   182,   182,   184,   184,   184,
     184,   184,   186,   188,   189,   190,   191,   193,   194,   195,
     197,   198,   199,   201,   202,   203,   204,   205,   207,   208,
     209,   211,   212,   214,   215,   217,   218,   220,   221,   223,
     224,   226,   227,   229,   230,   232,   233,   233,   233,   233,
     233,   234,   234,   234,   235,   235,   237,   238,   240,   241,
     242,   244,   246,   246,   251,   252,   253,   254,   255,   256,
     260,   261,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   275,   276,   279,   280,   281,   282,   283,   285,   285,
     285,   285,   285,   285,   287,   288,   289,   290,   292,   293,
     294,   296,   297,   298,   300,   301,   302,   303,   304,   306,
     307,   308,   310,   311,   313,   314,   316,   317,   319,   320,
     322,   323,   325,   327,   328,   330,   331,   331,   331,   331,
     331,   332,   332,   332,   333,   333,   335,   336
  };

  void
  parser::yy_stack_print_ () const
  {
    *yycdebug_ << "Stack now";
    for (stack_type::const_iterator
           i = yystack_.begin (),
           i_end = yystack_.end ();
         i != i_end; ++i)
      *yycdebug_ << ' ' << int (i->state);
    *yycdebug_ << '\n';
  }

  void
  parser::yy_reduce_print_ (int yyrule) const
  {
    int yylno = yyrline_[yyrule];
    int yynrhs = yyr2_[yyrule];
    // Print the symbols being reduced, and their result.
    *yycdebug_ << "Reducing stack by rule " << yyrule - 1
               << " (line " << yylno << "):\n";
    // The symbols being reduced.
    for (int yyi = 0; yyi < yynrhs; yyi++)
      YY_SYMBOL_PRINT ("   $" << yyi + 1 << " =",
                       yystack_[(yynrhs) - (yyi + 1)]);
  }
#endif // YYDEBUG

  parser::symbol_kind_type
  parser::yytranslate_ (int t)
  {
    return static_cast<symbol_kind_type> (t);
  }

} // yy

 /////////////////////////////////////////////////////////////////////////////
// *INDENT-ON*
namespace yy { std::array<bool, yyruletype::yynrules> rules = {false}; }

// *****************************************************************************
static Node* astAddChild(Node* root, Node* n)
{
   assert(n);
   assert(root);
   n->parent = root;
   if (!root->children) { return root->children = n; }
   Node* child = root->children;
   for (; child->next; child = child->next);
   return child->next = n;
}

// *****************************************************************************
static Node* astAddNext(Node* root, Node* n)
{
   assert(n);
   assert(root);
   n->parent = root->parent;
   if (!root->next) { return root->next = n; }
   Node* next = root;
   for (; next->next; next = next->next);
   return next->next = n;
}

#ifndef XFL_C
// *****************************************************************************
template<int RN>
Node* astNewRule(const int sn, const char *rule)
{
   return astAddNode(std::make_shared<Rule<RN>>(sn, rule));
}

// *****************************************************************************
template<int YYN> void rhs(Node **yylval,
                           const int yyn,
                           const yy::parser::symbol_kind_type yyr1n,
                           const int nrhs,
                           const char *rule,
                           yy::parser::stack_type &yystack)
{
   assert (nrhs);
   assert(YYN == yyn);
   const yy::parser::symbol_kind_type sn = yyr1n;
   Node *root = *yylval = astNewRule<YYN>(sn, rule);
   Node *n = yystack[nrhs-1].value;
   astAddChild(root, n);
   for (int i = 1; i < nrhs ; i++)
   {
      root = n;
      astAddNext(root, n = yystack[nrhs-1-i].value);
   }
}

// *****************************************************************************
void yy::dfs(Node *n, struct Middlend &ir)
{
   if (!n) { return; }
   bool updown = true;
   Node *extra = n;
   n->Apply(ir, updown, &extra); // down
   const int N = n->Number(); // N = SN | RN
   assert(N > 0);
   if (n->IsRule())
   {
      if (N > yynrules) { DBG("\n\033[31m[rule] N:%d/%d",N,yynrules); }
      assert(N <= yynrules);
   }
   if (n->IsToken())
   {
      constexpr int YYNTOKENS = parser::YYNTOKENS;
      if (N >= YYNTOKENS) { DBG("\n\033[31m[token] N:%d/%d",N,YYNTOKENS); }
      assert(N < YYNTOKENS);
   }
   if (n->IsRule()) { yy::rules.at(N) = true; } // Set the state flags
   if (updown && n->children)
   {
      dfs(n->children, ir);
      if (extra!=n) { dfs(extra, ir); }
   }
   if (n->next) { dfs(n->next, ir); }
   if (n->IsRule()) { yy::rules.at(N) = false; } // Reset the state flags
   if (n->IsRule()) {n->Apply(ir, updown = false, nullptr);} // up, only for rules
}

// *****************************************************************************
void yy::parser::error(const location_type& loc, const std::string& msg)
{
   std::cerr << loc << ": " << msg << '\n';
}

// *****************************************************************************
template<int RN>
void Rule<RN>::Apply(struct Middlend &ir, bool &dfs, Node **extra)
{
   ir.middlend<RN>(this, dfs, extra);
}
#else
#ifndef YYUNDEFTOK
#define YYUNDEFTOK YYSYMBOL_YYUNDEF
#endif
const int yy::undef() { return YYUNDEFTOK; }

const int yy::ntokens(void) { return YYNTOKENS; }

const int yy::r1(int yyn) { return yyr1[yyn]; }

const int yy::r2(int yyn) { return yyr2[yyn]; }

signed char yy::Translate(int token_num) { return YYTRANSLATE(token_num);}

const char* const yy::SymbolName(int sn)
{ return yysymbol_name(YY_CAST(yysymbol_kind_t, sn)); }

// *****************************************************************************
template<int RN>
Node* astNewRule(const int sn, const char *rule)
{
#ifdef OWN_VTABLE
   return astAddNode(std::make_shared<Node>(Rule<RN> {sn, rule}));
#else
   return astAddNode(std::make_shared<Rule<RN>>(sn, rule));
#endif
}

// *****************************************************************************
template<int YYN>
void rhs(YYSTYPE *lhs, int yyn, YYSTYPE *yyvsp)
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
void yy::dfs(Node *n, struct Middlend &ir)
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
   if (n->IsRule()) {n->Apply(ir, updown = false, nullptr);} // up, only for rules
}

// *****************************************************************************
template<int RN>
#ifdef OWN_VTABLE
void Rule<RN>::Apply(void *t, struct Middlend &ir, bool &dfs,
                     Node **extra)
{
   Rule<RN> *that = static_cast<Rule<RN>*>(t);
   if (dfs && yy::echo) { DBG("\033[35m[Apply] Rule:%d %s", RN, that->Name(t).c_str()); }
   ir.middlend<RN>(that, dfs, extra);
}
#else
void Rule<RN>::Apply(struct Middlend &ir, bool &dfs, Node **extra)
{
   if (dfs && yy::echo) { DBG("\033[35m[Apply] Rule:%d %s", RN, Name().c_str()); }
   ir.middlend<RN>(this, dfs, extra);
}
#endif
#endif
