// A Bison parser, made by GNU Bison 3.7.4.

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

#include "xfl.hpp"
#include "xfl.Y.hpp"
#include "xfc.hpp"
YY_DECL;

using symbol_t = yy::parser::symbol_kind_type;

template<int YYN> void rhs(xfl&,
                           Node**,                       // &yylhs.value
                           const int,                    // yyn
                           const symbol_t,               // yyr1n (sn)
                           const int,                    // yyr2n (nrhs, yylen)
                           const char*,                  // symbol_name(sn)
                           yy::parser::stack_type&);     // yystack
#define RHS {\
    const unsigned char sn_yyn = yyr1_[yyn];\
    const symbol_t sn = yy::parser::yytranslate_(sn_yyn);\
    const char *rule = yy::parser::symbol_name(sn);\
    rhs<YYN>(ufl, &yylhs.value, yyn, sn, yyr2_[yyn], rule, yystack_);}

// %warning: %token order has to be sync'ed with the lexer's one



#include "xfl.Y.hpp"




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

namespace yy
{

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
parser::basic_symbol<Base>::basic_symbol (typename Base::kind_type t,
                                          YY_MOVE_REF (location_type) l)
   : Base (t)
   , value ()
   , location (l)
{}

template <typename Base>
parser::basic_symbol<Base>::basic_symbol (typename Base::kind_type t,
                                          YY_RVREF (semantic_type) v, YY_RVREF (location_type) l)
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
   {
      return symbol_kind::S_YYEMPTY;
   }
   else
   {
      return YY_CAST (symbol_kind_type, yystos_[+state]);
   }
}

parser::stack_symbol_type::stack_symbol_type ()
{}

parser::stack_symbol_type::stack_symbol_type (YY_RVREF (stack_symbol_type) that)
   : super_type (YY_MOVE (that.state), YY_MOVE (that.value),
                 YY_MOVE (that.location))
{
#if 201103L <= YY_CPLUSPLUS
   // that is emptied.
   that.state = empty_state;
#endif
}

parser::stack_symbol_type::stack_symbol_type (state_type s,
                                              YY_MOVE_REF (symbol_type) that)
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
   {
      YY_SYMBOL_PRINT (yymsg, yysym);
   }

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
   {
      yyo << "empty symbol";
   }
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
   {
      YY_SYMBOL_PRINT (m, sym);
   }
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
   {
      return yytable_[yyr];
   }
   else
   {
      return yydefgoto_[yysym - YYNTOKENS];
   }
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
      {
         YYACCEPT;
      }

      goto yybackup;


      /*-----------.
      | yybackup.  |
      `-----------*/
   yybackup:
      // Try to take a decision without lookahead.
      yyn = yypact_[+yystack_[0].state];
      if (yy_pact_value_is_default_ (yyn))
      {
         goto yydefault;
      }

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
         {
            goto yyerrlab;
         }
         goto yydefault;
      }

      // Reduce or error.
      yyn = yytable_[yyn];
      if (yyn <= 0)
      {
         if (yy_table_value_is_error_ (yyn))
         {
            goto yyerrlab;
         }
         if (!yy_lac_establish_ (yyla.kind ()))
         {
            goto yyerrlab;
         }

         yyn = -yyn;
         goto yyreduce;
      }

      // Count tokens shifted since error; after three, turn off error status.
      if (yyerrstatus_)
      {
         --yyerrstatus_;
      }

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
      {
         goto yyerrlab;
      }
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
         {
            yylhs.value = yystack_[yylen - 1].value;
         }
         else
         {
            yylhs.value = yystack_[0].value;
         }

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
               case 2:   // entry_point: statements
               {
#define entry_point_statements 2
                  constexpr int YYN = 2;
                  { RHS ufl.root = yylhs.value; } {RHS}
                  break;
               }

               case 3:   // statements: statement
               {
#define statements_statement 3
                  constexpr int YYN = 3;
                  {} {RHS}
                  break;
               }

               case 4:   // statements: statements statement
               {
#define statements_statements_statement 4
                  constexpr int YYN = 4;
                  {} {RHS}
                  break;
               }

               case 5:   // statements: extra_status_rule
               {
#define statements_extra_status_rule 5
                  constexpr int YYN = 5;
                  {} {RHS}
                  break;
               }

               case 6:   // statement: NL
               {
#define statement_nl 6
                  constexpr int YYN = 6;
                  {} {RHS}
                  break;
               }

               case 7:   // statement: decl NL
               {
#define statement_decl_nl 7
                  constexpr int YYN = 7;
                  {} {RHS}
                  break;
               }

               case 8:   // decl: function
               {
#define decl_function 8
                  constexpr int YYN = 8;
                  {} {RHS}
                  break;
               }

               case 9:   // decl: domain assign_op expr
               {
#define decl_domain_assign_op_expr 9
                  constexpr int YYN = 9;
                  {} {RHS}
                  break;
               }

               case 10:   // decl: id_list assign_op expr
               {
#define decl_id_list_assign_op_expr 10
                  constexpr int YYN = 10;
                  {} {RHS}
                  break;
               }

               case 11:   // decl: LP id_list RP assign_op expr
               {
#define decl_lp_id_list_rp_assign_op_expr 11
                  constexpr int YYN = 11;
                  {} {RHS}
                  break;
               }

               case 12:   // decl: LB id_list RB assign_op expr
               {
#define decl_lb_id_list_rb_assign_op_expr 12
                  constexpr int YYN = 12;
                  {} {RHS}
                  break;
               }

               case 13:   // decl: if_statement
               {
#define decl_if_statement 13
                  constexpr int YYN = 13;
                  {} {RHS}
                  break;
               }

               case 14:   // decl: api_statement
               {
#define decl_api_statement 14
                  constexpr int YYN = 14;
                  {} {RHS}
                  break;
               }

               case 15:   // decl: iteration_statement
               {
#define decl_iteration_statement 15
                  constexpr int YYN = 15;
                  {} {RHS}
                  break;
               }

               case 16:   // decl: direct_declarator
               {
#define decl_direct_declarator 16
                  constexpr int YYN = 16;
                  {} {RHS}
                  break;
               }

               case 17:   // primary_id: IDENTIFIER
               {
#define primary_id_identifier 17
                  constexpr int YYN = 17;
                  {} {RHS}
                  break;
               }

               case 18:   // postfix_id: primary_id
               {
#define postfix_id_primary_id 18
                  constexpr int YYN = 18;
                  {} {RHS}
                  break;
               }

               case 19:   // postfix_id: postfix_id LP RP
               {
#define postfix_id_postfix_id_lp_rp 19
                  constexpr int YYN = 19;
                  {} {RHS}
                  break;
               }

               case 20:   // postfix_id: postfix_id LP expr RP
               {
#define postfix_id_postfix_id_lp_expr_rp 20
                  constexpr int YYN = 20;
                  {} {RHS}
                  break;
               }

               case 21:   // postfix_id: postfix_id DOT primary_id
               {
#define postfix_id_postfix_id_dot_primary_id 21
                  constexpr int YYN = 21;
                  {} {RHS}
                  break;
               }

               case 22:   // postfix_ids: postfix_id
               {
#define postfix_ids_postfix_id 22
                  constexpr int YYN = 22;
                  {} {RHS}
                  break;
               }

               case 23:   // postfix_ids: postfix_ids postfix_id
               {
#define postfix_ids_postfix_ids_postfix_id 23
                  constexpr int YYN = 23;
                  {} {RHS}
                  break;
               }

               case 24:   // id_list: postfix_ids
               {
#define id_list_postfix_ids 24
                  constexpr int YYN = 24;
                  {} {RHS}
                  break;
               }

               case 25:   // id_list: id_list COMA postfix_ids
               {
#define id_list_id_list_coma_postfix_ids 25
                  constexpr int YYN = 25;
                  {} {RHS}
                  break;
               }

               case 26:   // extra_status_rule: lhs
               {
#define extra_status_rule_lhs 26
                  constexpr int YYN = 26;
                  {} {RHS}
                  break;
               }

               case 27:   // extra_status_rule: var_xt
               {
#define extra_status_rule_var_xt 27
                  constexpr int YYN = 27;
                  {} {RHS}
                  break;
               }

               case 28:   // extra_status_rule: dom_xt
               {
#define extra_status_rule_dom_xt 28
                  constexpr int YYN = 28;
                  {} {RHS}
                  break;
               }

               case 29:   // extra_status_rule: expr_quote
               {
#define extra_status_rule_expr_quote 29
                  constexpr int YYN = 29;
                  {} {RHS}
                  break;
               }

               case 30:   // lhs: LHS
               {
#define lhs_lhs 30
                  constexpr int YYN = 30;
                  {} {RHS}
                  break;
               }

               case 31:   // var_xt: VAR_XT
               {
#define var_xt_var_xt 31
                  constexpr int YYN = 31;
                  {} {RHS}
                  break;
               }

               case 32:   // dom_xt: DOM_XT
               {
#define dom_xt_dom_xt 32
                  constexpr int YYN = 32;
                  {} {RHS}
                  break;
               }

               case 33:   // expr_quote: EXPR_QUOTE
               {
#define expr_quote_expr_quote 33
                  constexpr int YYN = 33;
                  {} {RHS}
                  break;
               }

               case 34:   // function: DEF IDENTIFIER LP args_expr_list RP COLON def_empty RETURN math_expr
               {
#define function_def_identifier_lp_args_expr_list_rp_colon_def_empty_return_math_expr 34
                  constexpr int YYN = 34;
                  {} {RHS}
                  break;
               }

               case 35:   // function: DEF IDENTIFIER LP args_expr_list RP COLON def_statements RETURN math_expr
               {
#define function_def_identifier_lp_args_expr_list_rp_colon_def_statements_return_math_expr 35
                  constexpr int YYN = 35;
                  {} {RHS}
                  break;
               }

               case 36:   // def_empty: %empty
               {
#define def_empty_empty 36
                  constexpr int YYN = 36;
                  {} {RHS}
                  break;
               }

               case 37:   // def_statements: def_statement
               {
#define def_statements_def_statement 37
                  constexpr int YYN = 37;
                  {} {RHS}
                  break;
               }

               case 38:   // def_statements: def_statements def_statement
               {
#define def_statements_def_statements_def_statement 38
                  constexpr int YYN = 38;
                  {} {RHS}
                  break;
               }

               case 39:   // def_statement: NL
               {
#define def_statement_nl 39
                  constexpr int YYN = 39;
                  {} {RHS}
                  break;
               }

               case 40:   // def_statement: id_list assign_op expr NL
               {
#define def_statement_id_list_assign_op_expr_nl 40
                  constexpr int YYN = 40;
                  {} {RHS}
                  break;
               }

               case 41:   // def_statement: LP id_list RP assign_op expr NL
               {
#define def_statement_lp_id_list_rp_assign_op_expr_nl 41
                  constexpr int YYN = 41;
                  {} {RHS}
                  break;
               }

               case 42:   // iteration_statement: FOR IDENTIFIER IN RANGE LP IDENTIFIER RP COLON NL expr
               {
#define iteration_statement_for_identifier_in_range_lp_identifier_rp_colon_nl_expr 42
                  constexpr int YYN = 42;
                  {} {RHS}
                  break;
               }

               case 43:   // iteration_statement: FOR IDENTIFIER IN RANGE LP IDENTIFIER RP COLON expr
               {
#define iteration_statement_for_identifier_in_range_lp_identifier_rp_colon_expr 43
                  constexpr int YYN = 43;
                  {} {RHS}
                  break;
               }

               case 44:   // if_statement: IF LP expr RP expr
               {
#define if_statement_if_lp_expr_rp_expr 44
                  constexpr int YYN = 44;
                  {} {RHS}
                  break;
               }

               case 45:   // api_statement: PLOT LP expr RP
               {
#define api_statement_plot_lp_expr_rp 45
                  constexpr int YYN = 45;
                  {} {RHS}
                  break;
               }

               case 46:   // api_statement: SAVE LP expr RP
               {
#define api_statement_save_lp_expr_rp 46
                  constexpr int YYN = 46;
                  {} {RHS}
                  break;
               }

               case 47:   // api_statement: SOLVE LP expr RP
               {
#define api_statement_solve_lp_expr_rp 47
                  constexpr int YYN = 47;
                  {} {RHS}
                  break;
               }

               case 48:   // api_statement: PROJECT LP expr RP
               {
#define api_statement_project_lp_expr_rp 48
                  constexpr int YYN = 48;
                  {} {RHS}
                  break;
               }

               case 49:   // direct_declarator: postfix_id
               {
#define direct_declarator_postfix_id 49
                  constexpr int YYN = 49;
                  {} {RHS}
                  break;
               }

               case 50:   // domain: DOM_DX
               {
#define domain_dom_dx 50
                  constexpr int YYN = 50;
                  {} {RHS}
                  break;
               }

               case 51:   // domain: EXT_DS
               {
#define domain_ext_ds 51
                  constexpr int YYN = 51;
                  {} {RHS}
                  break;
               }

               case 52:   // domain: INT_DS
               {
#define domain_int_ds 52
                  constexpr int YYN = 52;
                  {} {RHS}
                  break;
               }

               case 53:   // constant: NATURAL
               {
#define constant_natural 53
                  constexpr int YYN = 53;
                  {} {RHS}
                  break;
               }

               case 54:   // constant: REAL
               {
#define constant_real 54
                  constexpr int YYN = 54;
                  {} {RHS}
                  break;
               }

               case 55:   // strings: STRING
               {
#define strings_string 55
                  constexpr int YYN = 55;
                  {} {RHS}
                  break;
               }

               case 56:   // strings: strings STRING
               {
#define strings_strings_string 56
                  constexpr int YYN = 56;
                  {} {RHS}
                  break;
               }

               case 57:   // api: DEVICE
               {
#define api_device 57
                  constexpr int YYN = 57;
                  {} {RHS}
                  break;
               }

               case 58:   // api: MESH
               {
#define api_mesh 58
                  constexpr int YYN = 58;
                  {} {RHS}
                  break;
               }

               case 59:   // api: FINITE_ELEMENT
               {
#define api_finite_element 59
                  constexpr int YYN = 59;
                  {} {RHS}
                  break;
               }

               case 60:   // api: UNIT_SQUARE_MESH
               {
#define api_unit_square_mesh 60
                  constexpr int YYN = 60;
                  {} {RHS}
                  break;
               }

               case 61:   // api: UNIT_HEX_MESH
               {
#define api_unit_hex_mesh 61
                  constexpr int YYN = 61;
                  {} {RHS}
                  break;
               }

               case 62:   // api: FUNCTION
               {
#define api_function 62
                  constexpr int YYN = 62;
                  {} {RHS}
                  break;
               }

               case 63:   // api: FUNCTION_SPACE
               {
#define api_function_space 63
                  constexpr int YYN = 63;
                  {} {RHS}
                  break;
               }

               case 64:   // api: VECTOR_FUNCTION_SPACE
               {
#define api_vector_function_space 64
                  constexpr int YYN = 64;
                  {} {RHS}
                  break;
               }

               case 65:   // api: EXPRESSION
               {
#define api_expression 65
                  constexpr int YYN = 65;
                  {} {RHS}
                  break;
               }

               case 66:   // api: DIRICHLET_BC
               {
#define api_dirichlet_bc 66
                  constexpr int YYN = 66;
                  {} {RHS}
                  break;
               }

               case 67:   // api: TRIAL_FUNCTION
               {
#define api_trial_function 67
                  constexpr int YYN = 67;
                  {} {RHS}
                  break;
               }

               case 68:   // api: TEST_FUNCTION
               {
#define api_test_function 68
                  constexpr int YYN = 68;
                  {} {RHS}
                  break;
               }

               case 69:   // api: CONSTANT_API
               {
#define api_constant_api 69
                  constexpr int YYN = 69;
                  {} {RHS}
                  break;
               }

               case 70:   // api: api_statement
               {
#define api_api_statement 70
                  constexpr int YYN = 70;
                  {} {RHS}
                  break;
               }

               case 71:   // primary_expr: IDENTIFIER
               {
#define primary_expr_identifier 71
                  constexpr int YYN = 71;
                  {} {RHS}
                  break;
               }

               case 72:   // primary_expr: constant
               {
#define primary_expr_constant 72
                  constexpr int YYN = 72;
                  {} {RHS}
                  break;
               }

               case 73:   // primary_expr: domain
               {
#define primary_expr_domain 73
                  constexpr int YYN = 73;
                  {} {RHS}
                  break;
               }

               case 74:   // primary_expr: QUOTE
               {
#define primary_expr_quote 74
                  constexpr int YYN = 74;
                  {} {RHS}
                  break;
               }

               case 75:   // primary_expr: strings
               {
#define primary_expr_strings 75
                  constexpr int YYN = 75;
                  {} {RHS}
                  break;
               }

               case 76:   // primary_expr: LP expr RP
               {
#define primary_expr_lp_expr_rp 76
                  constexpr int YYN = 76;
                  {} {RHS}
                  break;
               }

               case 77:   // primary_expr: LB expr RB
               {
#define primary_expr_lb_expr_rb 77
                  constexpr int YYN = 77;
                  {} {RHS}
                  break;
               }

               case 78:   // primary_expr: LS coords RS
               {
#define primary_expr_ls_coords_rs 78
                  constexpr int YYN = 78;
                  {} {RHS}
                  break;
               }

               case 79:   // primary_expr: api
               {
#define primary_expr_api 79
                  constexpr int YYN = 79;
                  {} {RHS}
                  break;
               }

               case 80:   // pow_expr: postfix_expr POW constant
               {
#define pow_expr_postfix_expr_pow_constant 80
                  constexpr int YYN = 80;
                  {} {RHS}
                  break;
               }

               case 81:   // pow_expr: postfix_expr POW IDENTIFIER
               {
#define pow_expr_postfix_expr_pow_identifier 81
                  constexpr int YYN = 81;
                  {} {RHS}
                  break;
               }

               case 82:   // postfix_expr: primary_expr
               {
#define postfix_expr_primary_expr 82
                  constexpr int YYN = 82;
                  {} {RHS}
                  break;
               }

               case 83:   // postfix_expr: pow_expr
               {
#define postfix_expr_pow_expr 83
                  constexpr int YYN = 83;
                  {} {RHS}
                  break;
               }

               case 84:   // postfix_expr: INNER_OP LP additive_expr COMA additive_expr RP
               {
#define postfix_expr_inner_op_lp_additive_expr_coma_additive_expr_rp 84
                  constexpr int YYN = 84;
                  {} {RHS}
                  break;
               }

               case 85:   // postfix_expr: postfix_expr LB expr RB
               {
#define postfix_expr_postfix_expr_lb_expr_rb 85
                  constexpr int YYN = 85;
                  {} {RHS}
                  break;
               }

               case 86:   // postfix_expr: postfix_expr LP RP
               {
#define postfix_expr_postfix_expr_lp_rp 86
                  constexpr int YYN = 86;
                  {} {RHS}
                  break;
               }

               case 87:   // postfix_expr: postfix_expr LP args_expr_list RP
               {
#define postfix_expr_postfix_expr_lp_args_expr_list_rp 87
                  constexpr int YYN = 87;
                  {} {RHS}
                  break;
               }

               case 88:   // postfix_expr: postfix_expr DOT primary_expr
               {
#define postfix_expr_postfix_expr_dot_primary_expr 88
                  constexpr int YYN = 88;
                  {} {RHS}
                  break;
               }

               case 89:   // postfix_expr: GRAD_OP LP additive_expr RP
               {
#define postfix_expr_grad_op_lp_additive_expr_rp 89
                  constexpr int YYN = 89;
                  {} {RHS}
                  break;
               }

               case 90:   // postfix_expr: LHS LP IDENTIFIER RP
               {
#define postfix_expr_lhs_lp_identifier_rp 90
                  constexpr int YYN = 90;
                  {} {RHS}
                  break;
               }

               case 91:   // postfix_expr: RHS LP IDENTIFIER RP
               {
#define postfix_expr_rhs_lp_identifier_rp 91
                  constexpr int YYN = 91;
                  {} {RHS}
                  break;
               }

               case 92:   // unary_expr: postfix_expr
               {
#define unary_expr_postfix_expr 92
                  constexpr int YYN = 92;
                  {} {RHS}
                  break;
               }

               case 93:   // unary_expr: unary_op cast_expr
               {
#define unary_expr_unary_op_cast_expr 93
                  constexpr int YYN = 93;
                  {} {RHS}
                  break;
               }

               case 94:   // unary_op: MUL
               {
#define unary_op_mul 94
                  constexpr int YYN = 94;
                  {} {RHS}
                  break;
               }

               case 95:   // unary_op: ADD
               {
#define unary_op_add 95
                  constexpr int YYN = 95;
                  {} {RHS}
                  break;
               }

               case 96:   // unary_op: SUB
               {
#define unary_op_sub 96
                  constexpr int YYN = 96;
                  {} {RHS}
                  break;
               }

               case 97:   // unary_op: TILDE
               {
#define unary_op_tilde 97
                  constexpr int YYN = 97;
                  {} {RHS}
                  break;
               }

               case 98:   // unary_op: NOT
               {
#define unary_op_not 98
                  constexpr int YYN = 98;
                  {} {RHS}
                  break;
               }

               case 99:   // cast_expr: unary_expr
               {
#define cast_expr_unary_expr 99
                  constexpr int YYN = 99;
                  {} {RHS}
                  break;
               }

               case 100:   // multiplicative_expr: cast_expr
               {
#define multiplicative_expr_cast_expr 100
                  constexpr int YYN = 100;
                  {} {RHS}
                  break;
               }

               case 101:   // multiplicative_expr: multiplicative_expr MUL cast_expr
               {
#define multiplicative_expr_multiplicative_expr_mul_cast_expr 101
                  constexpr int YYN = 101;
                  {} {RHS}
                  break;
               }

               case 102:   // multiplicative_expr: multiplicative_expr DIV cast_expr
               {
#define multiplicative_expr_multiplicative_expr_div_cast_expr 102
                  constexpr int YYN = 102;
                  {} {RHS}
                  break;
               }

               case 103:   // multiplicative_expr: multiplicative_expr MOD cast_expr
               {
#define multiplicative_expr_multiplicative_expr_mod_cast_expr 103
                  constexpr int YYN = 103;
                  {} {RHS}
                  break;
               }

               case 104:   // dot_expr: multiplicative_expr
               {
#define dot_expr_multiplicative_expr 104
                  constexpr int YYN = 104;
                  {} {RHS}
                  break;
               }

               case 105:   // dot_expr: dot_expr DOT_OP multiplicative_expr
               {
#define dot_expr_dot_expr_dot_op_multiplicative_expr 105
                  constexpr int YYN = 105;
                  {} {RHS}
                  break;
               }

               case 106:   // additive_expr: dot_expr
               {
#define additive_expr_dot_expr 106
                  constexpr int YYN = 106;
                  {} {RHS}
                  break;
               }

               case 107:   // additive_expr: additive_expr ADD dot_expr
               {
#define additive_expr_additive_expr_add_dot_expr 107
                  constexpr int YYN = 107;
                  {} {RHS}
                  break;
               }

               case 108:   // additive_expr: additive_expr SUB dot_expr
               {
#define additive_expr_additive_expr_sub_dot_expr 108
                  constexpr int YYN = 108;
                  {} {RHS}
                  break;
               }

               case 109:   // shift_expr: additive_expr
               {
#define shift_expr_additive_expr 109
                  constexpr int YYN = 109;
                  {} {RHS}
                  break;
               }

               case 110:   // shift_expr: shift_expr LEFT_SHIFT additive_expr
               {
#define shift_expr_shift_expr_left_shift_additive_expr 110
                  constexpr int YYN = 110;
                  {} {RHS}
                  break;
               }

               case 111:   // shift_expr: shift_expr RIGHT_SHIFT additive_expr
               {
#define shift_expr_shift_expr_right_shift_additive_expr 111
                  constexpr int YYN = 111;
                  {} {RHS}
                  break;
               }

               case 112:   // relational_expr: shift_expr
               {
#define relational_expr_shift_expr 112
                  constexpr int YYN = 112;
                  {} {RHS}
                  break;
               }

               case 113:   // relational_expr: relational_expr LT shift_expr
               {
#define relational_expr_relational_expr_lt_shift_expr 113
                  constexpr int YYN = 113;
                  {} {RHS}
                  break;
               }

               case 114:   // relational_expr: relational_expr GT shift_expr
               {
#define relational_expr_relational_expr_gt_shift_expr 114
                  constexpr int YYN = 114;
                  {} {RHS}
                  break;
               }

               case 115:   // relational_expr: relational_expr LT_EQ shift_expr
               {
#define relational_expr_relational_expr_lt_eq_shift_expr 115
                  constexpr int YYN = 115;
                  {} {RHS}
                  break;
               }

               case 116:   // relational_expr: relational_expr GT_EQ shift_expr
               {
#define relational_expr_relational_expr_gt_eq_shift_expr 116
                  constexpr int YYN = 116;
                  {} {RHS}
                  break;
               }

               case 117:   // equality_expr: relational_expr
               {
#define equality_expr_relational_expr 117
                  constexpr int YYN = 117;
                  {} {RHS}
                  break;
               }

               case 118:   // equality_expr: equality_expr EQ_EQ relational_expr
               {
#define equality_expr_equality_expr_eq_eq_relational_expr 118
                  constexpr int YYN = 118;
                  {} {RHS}
                  break;
               }

               case 119:   // equality_expr: equality_expr NOT_EQ relational_expr
               {
#define equality_expr_equality_expr_not_eq_relational_expr 119
                  constexpr int YYN = 119;
                  {} {RHS}
                  break;
               }

               case 120:   // and_expr: equality_expr
               {
#define and_expr_equality_expr 120
                  constexpr int YYN = 120;
                  {} {RHS}
                  break;
               }

               case 121:   // and_expr: and_expr AND equality_expr
               {
#define and_expr_and_expr_and_equality_expr 121
                  constexpr int YYN = 121;
                  {} {RHS}
                  break;
               }

               case 122:   // exclusive_or_expr: and_expr
               {
#define exclusive_or_expr_and_expr 122
                  constexpr int YYN = 122;
                  {} {RHS}
                  break;
               }

               case 123:   // exclusive_or_expr: exclusive_or_expr XOR and_expr
               {
#define exclusive_or_expr_exclusive_or_expr_xor_and_expr 123
                  constexpr int YYN = 123;
                  {} {RHS}
                  break;
               }

               case 124:   // inclusive_or_expr: exclusive_or_expr
               {
#define inclusive_or_expr_exclusive_or_expr 124
                  constexpr int YYN = 124;
                  {} {RHS}
                  break;
               }

               case 125:   // inclusive_or_expr: inclusive_or_expr OR exclusive_or_expr
               {
#define inclusive_or_expr_inclusive_or_expr_or_exclusive_or_expr 125
                  constexpr int YYN = 125;
                  {} {RHS}
                  break;
               }

               case 126:   // logical_and_expr: inclusive_or_expr
               {
#define logical_and_expr_inclusive_or_expr 126
                  constexpr int YYN = 126;
                  {} {RHS}
                  break;
               }

               case 127:   // logical_and_expr: logical_and_expr AND_AND inclusive_or_expr
               {
#define logical_and_expr_logical_and_expr_and_and_inclusive_or_expr 127
                  constexpr int YYN = 127;
                  {} {RHS}
                  break;
               }

               case 128:   // logical_or_expr: logical_and_expr
               {
#define logical_or_expr_logical_and_expr 128
                  constexpr int YYN = 128;
                  {} {RHS}
                  break;
               }

               case 129:   // logical_or_expr: logical_or_expr OR_OR logical_and_expr
               {
#define logical_or_expr_logical_or_expr_or_or_logical_and_expr 129
                  constexpr int YYN = 129;
                  {} {RHS}
                  break;
               }

               case 130:   // conditional_expr: logical_or_expr
               {
#define conditional_expr_logical_or_expr 130
                  constexpr int YYN = 130;
                  {} {RHS}
                  break;
               }

               case 131:   // conditional_expr: logical_or_expr QUESTION expr COLON conditional_expr
               {
#define conditional_expr_logical_or_expr_question_expr_colon_conditional_expr 131
                  constexpr int YYN = 131;
                  {} {RHS}
                  break;
               }

               case 132:   // assign_expr: conditional_expr
               {
#define assign_expr_conditional_expr 132
                  constexpr int YYN = 132;
                  {} {RHS}
                  break;
               }

               case 133:   // assign_expr: postfix_expr assign_op assign_expr
               {
#define assign_expr_postfix_expr_assign_op_assign_expr 133
                  constexpr int YYN = 133;
                  {} {RHS}
                  break;
               }

               case 134:   // assign_op: EQ
               {
#define assign_op_eq 134
                  constexpr int YYN = 134;
                  {} {RHS}
                  break;
               }

               case 135:   // assign_op: ADD_EQ
               {
#define assign_op_add_eq 135
                  constexpr int YYN = 135;
                  {} {RHS}
                  break;
               }

               case 136:   // assign_op: SUB_EQ
               {
#define assign_op_sub_eq 136
                  constexpr int YYN = 136;
                  {} {RHS}
                  break;
               }

               case 137:   // assign_op: MUL_EQ
               {
#define assign_op_mul_eq 137
                  constexpr int YYN = 137;
                  {} {RHS}
                  break;
               }

               case 138:   // assign_op: DIV_EQ
               {
#define assign_op_div_eq 138
                  constexpr int YYN = 138;
                  {} {RHS}
                  break;
               }

               case 139:   // assign_op: MOD_EQ
               {
#define assign_op_mod_eq 139
                  constexpr int YYN = 139;
                  {} {RHS}
                  break;
               }

               case 140:   // assign_op: XOR_EQ
               {
#define assign_op_xor_eq 140
                  constexpr int YYN = 140;
                  {} {RHS}
                  break;
               }

               case 141:   // assign_op: AND_EQ
               {
#define assign_op_and_eq 141
                  constexpr int YYN = 141;
                  {} {RHS}
                  break;
               }

               case 142:   // assign_op: OR_EQ
               {
#define assign_op_or_eq 142
                  constexpr int YYN = 142;
                  {} {RHS}
                  break;
               }

               case 143:   // assign_op: LEFT_EQ
               {
#define assign_op_left_eq 143
                  constexpr int YYN = 143;
                  {} {RHS}
                  break;
               }

               case 144:   // assign_op: RIGHT_EQ
               {
#define assign_op_right_eq 144
                  constexpr int YYN = 144;
                  {} {RHS}
                  break;
               }

               case 145:   // expr: assign_expr
               {
#define expr_assign_expr 145
                  constexpr int YYN = 145;
                  {} {RHS}
                  break;
               }

               case 146:   // expr: expr COMA assign_expr
               {
#define expr_expr_coma_assign_expr 146
                  constexpr int YYN = 146;
                  {} {RHS}
                  break;
               }

               case 147:   // args_expr_list: assign_expr
               {
#define args_expr_list_assign_expr 147
                  constexpr int YYN = 147;
                  {} {RHS}
                  break;
               }

               case 148:   // args_expr_list: args_expr_list COMA assign_expr
               {
#define args_expr_list_args_expr_list_coma_assign_expr 148
                  constexpr int YYN = 148;
                  {} {RHS}
                  break;
               }

               case 149:   // args_expr_list: args_expr_list COMA NL assign_expr
               {
#define args_expr_list_args_expr_list_coma_nl_assign_expr 149
                  constexpr int YYN = 149;
                  {} {RHS}
                  break;
               }

               case 150:   // coord: LP constant COMA constant RP
               {
#define coord_lp_constant_coma_constant_rp 150
                  constexpr int YYN = 150;
                  {} {RHS}
                  break;
               }

               case 151:   // coords: coord
               {
#define coords_coord 151
                  constexpr int YYN = 151;
                  {} {RHS}
                  break;
               }

               case 152:   // coords: coords COLON coord
               {
#define coords_coords_colon_coord 152
                  constexpr int YYN = 152;
                  {} {RHS}
                  break;
               }

               case 153:   // primary_math_expr: IDENTIFIER
               {
#define primary_math_expr_identifier 153
                  constexpr int YYN = 153;
                  {} {RHS}
                  break;
               }

               case 154:   // primary_math_expr: constant
               {
#define primary_math_expr_constant 154
                  constexpr int YYN = 154;
                  {} {RHS}
                  break;
               }

               case 155:   // primary_math_expr: domain
               {
#define primary_math_expr_domain 155
                  constexpr int YYN = 155;
                  {} {RHS}
                  break;
               }

               case 156:   // primary_math_expr: QUOTE
               {
#define primary_math_expr_quote 156
                  constexpr int YYN = 156;
                  {} {RHS}
                  break;
               }

               case 157:   // primary_math_expr: LP math_expr RP
               {
#define primary_math_expr_lp_math_expr_rp 157
                  constexpr int YYN = 157;
                  {} {RHS}
                  break;
               }

               case 158:   // primary_math_expr: LB id_list RB
               {
#define primary_math_expr_lb_id_list_rb 158
                  constexpr int YYN = 158;
                  {} {RHS}
                  break;
               }

               case 159:   // dot_math_expr: INNER_OP LP additive_expr COMA additive_expr RP
               {
#define dot_math_expr_inner_op_lp_additive_expr_coma_additive_expr_rp 159
                  constexpr int YYN = 159;
                  {} {RHS}
                  break;
               }

               case 160:   // postfix_math_expr: primary_math_expr
               {
#define postfix_math_expr_primary_math_expr 160
                  constexpr int YYN = 160;
                  {} {RHS}
                  break;
               }

               case 161:   // postfix_math_expr: postfix_math_expr LB math_expr RB
               {
#define postfix_math_expr_postfix_math_expr_lb_math_expr_rb 161
                  constexpr int YYN = 161;
                  {} {RHS}
                  break;
               }

               case 162:   // postfix_math_expr: postfix_math_expr LP argument_math_expr_list RP
               {
#define postfix_math_expr_postfix_math_expr_lp_argument_math_expr_list_rp 162
                  constexpr int YYN = 162;
                  {} {RHS}
                  break;
               }

               case 163:   // postfix_math_expr: postfix_math_expr DOT IDENTIFIER
               {
#define postfix_math_expr_postfix_math_expr_dot_identifier 163
                  constexpr int YYN = 163;
                  {} {RHS}
                  break;
               }

               case 164:   // postfix_math_expr: postfix_math_expr INC_OP
               {
#define postfix_math_expr_postfix_math_expr_inc_op 164
                  constexpr int YYN = 164;
                  {} {RHS}
                  break;
               }

               case 165:   // postfix_math_expr: postfix_math_expr DEC_OP
               {
#define postfix_math_expr_postfix_math_expr_dec_op 165
                  constexpr int YYN = 165;
                  {} {RHS}
                  break;
               }

               case 166:   // postfix_math_expr: postfix_math_expr POW constant
               {
#define postfix_math_expr_postfix_math_expr_pow_constant 166
                  constexpr int YYN = 166;
                  {} {RHS}
                  break;
               }

               case 167:   // postfix_math_expr: dot_math_expr
               {
#define postfix_math_expr_dot_math_expr 167
                  constexpr int YYN = 167;
                  {} {RHS}
                  break;
               }

               case 168:   // postfix_math_expr: GRAD_OP LP additive_math_expr RP
               {
#define postfix_math_expr_grad_op_lp_additive_math_expr_rp 168
                  constexpr int YYN = 168;
                  {} {RHS}
                  break;
               }

               case 169:   // argument_math_expr_list: assign_math_expr
               {
#define argument_math_expr_list_assign_math_expr 169
                  constexpr int YYN = 169;
                  {} {RHS}
                  break;
               }

               case 170:   // argument_math_expr_list: argument_math_expr_list COMA assign_math_expr
               {
#define argument_math_expr_list_argument_math_expr_list_coma_assign_math_expr 170
                  constexpr int YYN = 170;
                  {} {RHS}
                  break;
               }

               case 171:   // unary_math_expr: postfix_math_expr
               {
#define unary_math_expr_postfix_math_expr 171
                  constexpr int YYN = 171;
                  {} {RHS}
                  break;
               }

               case 172:   // unary_math_expr: INC_OP unary_math_expr
               {
#define unary_math_expr_inc_op_unary_math_expr 172
                  constexpr int YYN = 172;
                  {} {RHS}
                  break;
               }

               case 173:   // unary_math_expr: DEC_OP unary_math_expr
               {
#define unary_math_expr_dec_op_unary_math_expr 173
                  constexpr int YYN = 173;
                  {} {RHS}
                  break;
               }

               case 174:   // unary_math_expr: MOD unary_math_expr
               {
#define unary_math_expr_mod_unary_math_expr 174
                  constexpr int YYN = 174;
                  {} {RHS}
                  break;
               }

               case 175:   // unary_math_expr: unary_math_op unary_math_expr
               {
#define unary_math_expr_unary_math_op_unary_math_expr 175
                  constexpr int YYN = 175;
                  {} {RHS}
                  break;
               }

               case 176:   // unary_math_op: MUL
               {
#define unary_math_op_mul 176
                  constexpr int YYN = 176;
                  {} {RHS}
                  break;
               }

               case 177:   // unary_math_op: ADD
               {
#define unary_math_op_add 177
                  constexpr int YYN = 177;
                  {} {RHS}
                  break;
               }

               case 178:   // unary_math_op: SUB
               {
#define unary_math_op_sub 178
                  constexpr int YYN = 178;
                  {} {RHS}
                  break;
               }

               case 179:   // unary_math_op: AND
               {
#define unary_math_op_and 179
                  constexpr int YYN = 179;
                  {} {RHS}
                  break;
               }

               case 180:   // unary_math_op: TILDE
               {
#define unary_math_op_tilde 180
                  constexpr int YYN = 180;
                  {} {RHS}
                  break;
               }

               case 181:   // unary_math_op: NOT
               {
#define unary_math_op_not 181
                  constexpr int YYN = 181;
                  {} {RHS}
                  break;
               }

               case 182:   // multiplicative_math_expr: unary_math_expr
               {
#define multiplicative_math_expr_unary_math_expr 182
                  constexpr int YYN = 182;
                  {} {RHS}
                  break;
               }

               case 183:   // multiplicative_math_expr: multiplicative_math_expr MUL unary_math_expr
               {
#define multiplicative_math_expr_multiplicative_math_expr_mul_unary_math_expr 183
                  constexpr int YYN = 183;
                  {} {RHS}
                  break;
               }

               case 184:   // multiplicative_math_expr: multiplicative_math_expr DIV unary_math_expr
               {
#define multiplicative_math_expr_multiplicative_math_expr_div_unary_math_expr 184
                  constexpr int YYN = 184;
                  {} {RHS}
                  break;
               }

               case 185:   // multiplicative_math_expr: multiplicative_math_expr MOD unary_math_expr
               {
#define multiplicative_math_expr_multiplicative_math_expr_mod_unary_math_expr 185
                  constexpr int YYN = 185;
                  {} {RHS}
                  break;
               }

               case 186:   // additive_math_expr: multiplicative_math_expr
               {
#define additive_math_expr_multiplicative_math_expr 186
                  constexpr int YYN = 186;
                  {} {RHS}
                  break;
               }

               case 187:   // additive_math_expr: additive_math_expr ADD multiplicative_math_expr
               {
#define additive_math_expr_additive_math_expr_add_multiplicative_math_expr 187
                  constexpr int YYN = 187;
                  {} {RHS}
                  break;
               }

               case 188:   // additive_math_expr: additive_math_expr SUB multiplicative_math_expr
               {
#define additive_math_expr_additive_math_expr_sub_multiplicative_math_expr 188
                  constexpr int YYN = 188;
                  {} {RHS}
                  break;
               }

               case 189:   // shift_math_expr: additive_math_expr
               {
#define shift_math_expr_additive_math_expr 189
                  constexpr int YYN = 189;
                  {} {RHS}
                  break;
               }

               case 190:   // shift_math_expr: shift_math_expr LEFT_SHIFT additive_math_expr
               {
#define shift_math_expr_shift_math_expr_left_shift_additive_math_expr 190
                  constexpr int YYN = 190;
                  {} {RHS}
                  break;
               }

               case 191:   // shift_math_expr: shift_math_expr RIGHT_SHIFT additive_math_expr
               {
#define shift_math_expr_shift_math_expr_right_shift_additive_math_expr 191
                  constexpr int YYN = 191;
                  {} {RHS}
                  break;
               }

               case 192:   // relational_math_expr: shift_math_expr
               {
#define relational_math_expr_shift_math_expr 192
                  constexpr int YYN = 192;
                  {} {RHS}
                  break;
               }

               case 193:   // relational_math_expr: relational_math_expr LT shift_math_expr
               {
#define relational_math_expr_relational_math_expr_lt_shift_math_expr 193
                  constexpr int YYN = 193;
                  {} {RHS}
                  break;
               }

               case 194:   // relational_math_expr: relational_math_expr GT shift_math_expr
               {
#define relational_math_expr_relational_math_expr_gt_shift_math_expr 194
                  constexpr int YYN = 194;
                  {} {RHS}
                  break;
               }

               case 195:   // relational_math_expr: relational_math_expr LT_EQ shift_math_expr
               {
#define relational_math_expr_relational_math_expr_lt_eq_shift_math_expr 195
                  constexpr int YYN = 195;
                  {} {RHS}
                  break;
               }

               case 196:   // relational_math_expr: relational_math_expr GT_EQ shift_math_expr
               {
#define relational_math_expr_relational_math_expr_gt_eq_shift_math_expr 196
                  constexpr int YYN = 196;
                  {} {RHS}
                  break;
               }

               case 197:   // equality_math_expr: relational_math_expr
               {
#define equality_math_expr_relational_math_expr 197
                  constexpr int YYN = 197;
                  {} {RHS}
                  break;
               }

               case 198:   // equality_math_expr: equality_math_expr EQ_EQ relational_math_expr
               {
#define equality_math_expr_equality_math_expr_eq_eq_relational_math_expr 198
                  constexpr int YYN = 198;
                  {} {RHS}
                  break;
               }

               case 199:   // equality_math_expr: equality_math_expr NOT_EQ relational_math_expr
               {
#define equality_math_expr_equality_math_expr_not_eq_relational_math_expr 199
                  constexpr int YYN = 199;
                  {} {RHS}
                  break;
               }

               case 200:   // and_math_expr: equality_math_expr
               {
#define and_math_expr_equality_math_expr 200
                  constexpr int YYN = 200;
                  {} {RHS}
                  break;
               }

               case 201:   // and_math_expr: and_math_expr AND equality_math_expr
               {
#define and_math_expr_and_math_expr_and_equality_math_expr 201
                  constexpr int YYN = 201;
                  {} {RHS}
                  break;
               }

               case 202:   // exclusive_or_math_expr: and_math_expr
               {
#define exclusive_or_math_expr_and_math_expr 202
                  constexpr int YYN = 202;
                  {} {RHS}
                  break;
               }

               case 203:   // exclusive_or_math_expr: exclusive_or_math_expr XOR and_math_expr
               {
#define exclusive_or_math_expr_exclusive_or_math_expr_xor_and_math_expr 203
                  constexpr int YYN = 203;
                  {} {RHS}
                  break;
               }

               case 204:   // inclusive_or_math_expr: exclusive_or_math_expr
               {
#define inclusive_or_math_expr_exclusive_or_math_expr 204
                  constexpr int YYN = 204;
                  {} {RHS}
                  break;
               }

               case 205:   // inclusive_or_math_expr: inclusive_or_math_expr OR exclusive_or_math_expr
               {
#define inclusive_or_math_expr_inclusive_or_math_expr_or_exclusive_or_math_expr 205
                  constexpr int YYN = 205;
                  {} {RHS}
                  break;
               }

               case 206:   // logical_and_math_expr: inclusive_or_math_expr
               {
#define logical_and_math_expr_inclusive_or_math_expr 206
                  constexpr int YYN = 206;
                  {} {RHS}
                  break;
               }

               case 207:   // logical_and_math_expr: logical_and_math_expr AND_AND inclusive_or_math_expr
               {
#define logical_and_math_expr_logical_and_math_expr_and_and_inclusive_or_math_expr 207
                  constexpr int YYN = 207;
                  {} {RHS}
                  break;
               }

               case 208:   // logical_or_math_expr: logical_and_math_expr
               {
#define logical_or_math_expr_logical_and_math_expr 208
                  constexpr int YYN = 208;
                  {} {RHS}
                  break;
               }

               case 209:   // logical_or_math_expr: logical_or_math_expr OR_OR logical_and_math_expr
               {
#define logical_or_math_expr_logical_or_math_expr_or_or_logical_and_math_expr 209
                  constexpr int YYN = 209;
                  {} {RHS}
                  break;
               }

               case 210:   // conditional_math_expr: logical_or_math_expr
               {
#define conditional_math_expr_logical_or_math_expr 210
                  constexpr int YYN = 210;
                  {} {RHS}
                  break;
               }

               case 211:   // assign_math_expr: conditional_math_expr
               {
#define assign_math_expr_conditional_math_expr 211
                  constexpr int YYN = 211;
                  {} {RHS}
                  break;
               }

               case 212:   // assign_math_expr: unary_math_expr assign_math_op assign_math_expr
               {
#define assign_math_expr_unary_math_expr_assign_math_op_assign_math_expr 212
                  constexpr int YYN = 212;
                  {} {RHS}
                  break;
               }

               case 213:   // assign_math_op: EQ
               {
#define assign_math_op_eq 213
                  constexpr int YYN = 213;
                  {} {RHS}
                  break;
               }

               case 214:   // assign_math_op: ADD_EQ
               {
#define assign_math_op_add_eq 214
                  constexpr int YYN = 214;
                  {} {RHS}
                  break;
               }

               case 215:   // assign_math_op: SUB_EQ
               {
#define assign_math_op_sub_eq 215
                  constexpr int YYN = 215;
                  {} {RHS}
                  break;
               }

               case 216:   // assign_math_op: MUL_EQ
               {
#define assign_math_op_mul_eq 216
                  constexpr int YYN = 216;
                  {} {RHS}
                  break;
               }

               case 217:   // assign_math_op: DIV_EQ
               {
#define assign_math_op_div_eq 217
                  constexpr int YYN = 217;
                  {} {RHS}
                  break;
               }

               case 218:   // assign_math_op: MOD_EQ
               {
#define assign_math_op_mod_eq 218
                  constexpr int YYN = 218;
                  {} {RHS}
                  break;
               }

               case 219:   // assign_math_op: XOR_EQ
               {
#define assign_math_op_xor_eq 219
                  constexpr int YYN = 219;
                  {} {RHS}
                  break;
               }

               case 220:   // assign_math_op: AND_EQ
               {
#define assign_math_op_and_eq 220
                  constexpr int YYN = 220;
                  {} {RHS}
                  break;
               }

               case 221:   // assign_math_op: OR_EQ
               {
#define assign_math_op_or_eq 221
                  constexpr int YYN = 221;
                  {} {RHS}
                  break;
               }

               case 222:   // assign_math_op: LEFT_EQ
               {
#define assign_math_op_left_eq 222
                  constexpr int YYN = 222;
                  {} {RHS}
                  break;
               }

               case 223:   // assign_math_op: RIGHT_EQ
               {
#define assign_math_op_right_eq 223
                  constexpr int YYN = 223;
                  {} {RHS}
                  break;
               }

               case 224:   // math_expr: assign_math_expr
               {
#define math_expr_assign_math_expr 224
                  constexpr int YYN = 224;
                  {} {RHS}
                  break;
               }

               case 225:   // math_expr: math_expr COMA assign_math_expr
               {
#define math_expr_math_expr_coma_assign_math_expr 225
                  constexpr int YYN = 225;
                  {} {RHS}
                  break;
               }



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
         {
            YYABORT;
         }
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
      {
         YYERROR;
      }

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
               {
                  break;
               }
            }
         }

         // Pop the current state because it cannot handle the error token.
         if (yystack_.size () == 1)
         {
            YYABORT;
         }

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
      {
         yy_destroy_ ("Cleanup: discarding lookahead", yyla);
      }

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
      {
         yy_destroy_ (YY_NULLPTR, yyla);
      }

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
      "FROM", "IMPORT", "RETURN", "PLOT", "SAVE", "SOLVE", "PROJECT", "STRING",
      "QUOTE", "IF", "FOR", "IN", "RANGE", "DOT_OP", "INNER_OP", "GRAD_OP",
      "LHS", "RHS", "DEVICE", "MESH", "FINITE_ELEMENT", "UNIT_SQUARE_MESH",
      "UNIT_HEX_MESH", "FUNCTION", "FUNCTION_SPACE", "VECTOR_FUNCTION_SPACE",
      "EXPRESSION", "DIRICHLET_BC", "TRIAL_FUNCTION", "TEST_FUNCTION",
      "CONSTANT_API", "OR_OR", "AND_AND", "DOM_DX", "EXT_DS", "INT_DS",
      "EQ_EQ", "ADD_EQ", "SUB_EQ", "MUL_EQ", "DIV_EQ", "MOD_EQ", "XOR_EQ",
      "AND_EQ", "OR_EQ", "LEFT_EQ", "RIGHT_EQ", "NATURAL", "REAL",
      "IDENTIFIER", "GT", "LT", "EQ", "ADD", "SUB", "MUL", "DIV", "POW", "LS",
      "RS", "LP", "RP", "LB", "RB", "COMA", "APOSTROPHE", "COLON", "DOT",
      "MOD", "TILDE", "LEFT_SHIFT", "RIGHT_SHIFT", "LT_EQ", "GT_EQ", "NOT_EQ",
      "AND", "XOR", "OR", "QUESTION", "NOT", "INC_OP", "DEC_OP", "VAR_XT",
      "DOM_XT", "EXPR_QUOTE", "EMPTY", "$accept", "entry_point", "statements",
      "statement", "decl", "primary_id", "postfix_id", "postfix_ids",
      "id_list", "extra_status_rule", "lhs", "var_xt", "dom_xt", "expr_quote",
      "function", "def_empty", "def_statements", "def_statement",
      "iteration_statement", "if_statement", "api_statement",
      "direct_declarator", "domain", "constant", "strings", "api",
      "primary_expr", "pow_expr", "postfix_expr", "unary_expr", "unary_op",
      "cast_expr", "multiplicative_expr", "dot_expr", "additive_expr",
      "shift_expr", "relational_expr", "equality_expr", "and_expr",
      "exclusive_or_expr", "inclusive_or_expr", "logical_and_expr",
      "logical_or_expr", "conditional_expr", "assign_expr", "assign_op",
      "expr", "args_expr_list", "coord", "coords", "primary_math_expr",
      "dot_math_expr", "postfix_math_expr", "argument_math_expr_list",
      "unary_math_expr", "unary_math_op", "multiplicative_math_expr",
      "additive_math_expr", "shift_math_expr", "relational_math_expr",
      "equality_math_expr", "and_math_expr", "exclusive_or_math_expr",
      "inclusive_or_math_expr", "logical_and_math_expr",
      "logical_or_math_expr", "conditional_math_expr", "assign_math_expr",
      "assign_math_op", "math_expr", YY_NULLPTR
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
   {
      yyparser_.yy_lac_check_ (yyla_.kind ());
   }
#endif

   for (int yyx = 0; yyx < YYNTOKENS; ++yyx)
   {
      symbol_kind_type yysym = YY_CAST (symbol_kind_type, yyx);
      if (yysym != symbol_kind::S_YYerror
          && yysym != symbol_kind::S_YYUNDEF
          && yyparser_.yy_lac_check_ (yysym))
      {
         if (!yyarg)
         {
            ++yycount;
         }
         else if (yycount == yyargn)
         {
            return 0;
         }
         else
         {
            yyarg[yycount++] = yysym;
         }
      }
   }
   if (yyarg && yycount == 0 && 0 < yyargn)
   {
      yyarg[0] = symbol_kind::S_YYEMPTY;
   }
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
      {
         yyarg[0] = yyctx.token ();
      }
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
      {
         yyres += *yyp;
      }
   return yyres;
}


const short parser::yypact_ninf_ = -297;

const signed char parser::yytable_ninf_ = -50;

const short
parser::yypact_[] =
{
   91,  -297,   -13,   -40,    -6,    16,    23,    60,    11,  -297,
   -297,  -297,  -297,  -297,    80,    80,  -297,  -297,  -297,   138,
   159,  -297,   136,  -297,    10,    80,   158,  -297,  -297,  -297,
   -297,  -297,  -297,  -297,  -297,  -297,  -297,   188,    85,   728,
   728,   728,   728,   728,   126,   -29,    51,    39,  -297,  -297,
   -297,   594,    80,   -29,  -297,  -297,  -297,  -297,  -297,  -297,
   -297,  -297,  -297,  -297,  -297,    80,   728,   728,   728,  -297,
   -297,    92,   101,   116,   128,  -297,  -297,  -297,  -297,  -297,
   -297,  -297,  -297,  -297,  -297,  -297,  -297,  -297,  -297,  -297,
   -297,  -297,  -297,  -297,   145,   728,   728,  -297,  -297,  -297,
   -297,  -297,   200,  -297,  -297,  -297,   308,  -297,   728,  -297,
   -21,   224,   164,   141,    77,    -8,   170,   174,   185,   223,
   -17,  -297,  -297,    58,    93,   122,   123,   152,   248,   188,
   188,  -297,   178,  -297,    80,   210,   210,  -297,   183,   728,
   728,   231,   233,   242,  -297,     3,   215,   309,  -297,   294,
   661,   728,   795,   728,   221,  -297,   728,   728,   728,   728,
   728,   728,   728,   728,   728,   728,   728,   728,   728,   728,
   728,   728,   728,   728,   728,   728,  -297,   728,  -297,  -297,
   -297,   728,   204,   728,   728,  -297,   218,   460,    20,    94,
   240,   253,   239,  -297,   145,  -297,  -297,  -297,  -297,  -297,
   235,   318,  -297,  -297,  -297,  -297,  -297,   -21,   224,   224,
   164,   164,   141,   141,   141,   141,    77,    77,    -8,   170,
   174,   185,   223,    21,  -297,   210,   267,   210,   210,    61,
   728,  -297,   728,  -297,  -297,  -297,   242,  -297,  -297,  -297,
   728,   260,  -297,    80,   158,   359,    44,  -297,  -297,   182,
   306,  -297,   310,   251,   728,   238,   238,  -297,  -297,  -297,
   527,   188,     8,  -297,   311,   314,  -297,  -297,  -297,  -297,
   238,    80,   238,  -297,  -297,  -297,   238,   238,  -297,  -297,
   -297,  -297,   110,   356,   238,    -3,   333,   319,   109,    -7,
   304,   307,   330,   379,   357,  -297,  -297,   327,   327,   728,
   210,   728,  -297,   728,   238,   298,   340,  -297,  -297,  -297,
   242,   238,   238,   368,  -297,  -297,  -297,  -297,  -297,  -297,
   -297,  -297,  -297,  -297,  -297,  -297,  -297,   238,  -297,   238,
   238,   238,   238,   238,   238,   238,   238,   238,   238,   238,
   238,   238,   238,   238,   238,   238,   238,   238,   210,    18,
   125,  -297,   214,  -297,  -297,  -297,   305,  -297,   342,  -297,
   -297,  -297,  -297,  -297,    -3,    -3,   333,   333,   319,   319,
   319,   319,   109,   109,    -7,   304,   307,   330,   379,  -297,
   -297,   728,  -297,  -297,   238,  -297,   241,  -297,  -297
};

const unsigned char
parser::yydefact_[] =
{
   0,     6,     0,     0,     0,     0,     0,     0,     0,    30,
   50,    51,    52,    17,     0,     0,    31,    32,    33,     0,
   2,     3,     0,    18,    22,    24,     0,     5,    26,    27,
   28,    29,     8,    15,    13,    14,    16,     0,     0,     0,
   0,     0,     0,     0,     0,    22,     0,     0,     1,     4,
   7,     0,     0,    23,   135,   136,   137,   138,   139,   140,
   141,   142,   143,   144,   134,     0,     0,     0,     0,    55,
   74,     0,     0,     0,     0,    57,    58,    59,    60,    61,
   62,    63,    64,    65,    66,    67,    68,    69,    53,    54,
   71,    95,    96,    94,     0,     0,     0,    97,    98,    70,
   73,    72,    75,    79,    82,    83,    92,    99,     0,   100,
   104,   106,   109,   112,   117,   120,   122,   124,   126,   128,
   130,   132,   145,     0,     0,     0,     0,     0,     0,     0,
   0,    19,     0,    21,    25,    10,     9,   147,     0,     0,
   0,     0,     0,     0,   151,     0,     0,     0,    56,     0,
   0,     0,     0,     0,    92,    93,     0,     0,     0,     0,
   0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   0,     0,     0,     0,     0,     0,    45,     0,    46,    47,
   48,     0,     0,     0,     0,    20,     0,     0,     0,     0,
   0,     0,     0,    78,     0,    76,    77,    81,    80,    86,
   0,     0,    88,   133,   101,   102,   103,   105,   107,   108,
   110,   111,   114,   113,   115,   116,   118,   119,   121,   123,
   125,   127,   129,     0,   146,    44,     0,    11,    12,    36,
   0,   148,     0,    89,    90,    91,     0,   152,    87,    85,
   0,     0,    39,     0,     0,     0,     0,    37,   149,     0,
   0,   131,     0,     0,     0,     0,     0,    38,    84,   150,
   0,     0,     0,   156,     0,     0,   153,   177,   178,   176,
   0,     0,     0,   180,   179,   181,     0,     0,   155,   154,
   160,   167,   171,   182,     0,   186,   189,   192,   197,   200,
   202,   204,   206,   208,   210,   211,   224,    34,    35,     0,
   43,     0,    40,     0,     0,     0,     0,   174,   172,   173,
   0,     0,     0,     0,   164,   165,   214,   215,   216,   217,
   218,   219,   220,   221,   222,   223,   213,     0,   175,     0,
   0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   0,     0,     0,     0,     0,     0,     0,     0,    42,     0,
   0,   182,     0,   157,   158,   166,     0,   169,     0,   163,
   212,   183,   184,   185,   187,   188,   190,   191,   194,   193,
   195,   196,   198,   199,   201,   203,   205,   207,   209,   225,
   41,     0,   168,   162,     0,   161,     0,   170,   159
};

const short
parser::yypgoto_[] =
{
   -297,  -297,  -297,   410,  -297,   380,    19,   366,     2,  -297,
   -297,  -297,  -297,  -297,  -297,  -297,  -297,   187,  -297,  -297,
   32,  -297,     0,   115,  -297,  -297,   282,  -297,   -38,  -297,
   -297,   -71,   276,   256,  -116,   151,   252,   266,   268,   265,
   270,   264,  -297,   201,   -57,   -31,   -33,   290,   269,  -297,
   -297,  -297,  -297,  -297,  -206,  -297,    90,  -279,    26,    88,
   124,   133,   121,   132,   134,  -297,  -297,  -296,  -297,  -230
};

const short
parser::yydefgoto_[] =
{
   -1,    19,    20,    21,    22,    23,    45,    25,    26,    27,
   28,    29,    30,    31,    32,   245,   246,   247,    33,    34,
   99,    36,   100,   101,   102,   103,   104,   105,   154,   107,
   108,   109,   110,   111,   112,   113,   114,   115,   116,   117,
   118,   119,   120,   121,   122,    66,   123,   138,   144,   145,
   280,   281,   282,   356,   351,   284,   285,   286,   287,   288,
   289,   290,   291,   292,   293,   294,   295,   296,   327,   297
};

const short
parser::yytable_[] =
{
   37,   106,   106,   106,   106,   106,    67,   124,   125,   126,
   127,   137,   302,   106,   -49,   357,    46,    47,   132,    24,
   37,   174,   380,   188,   189,   352,   298,    39,   106,   106,
   106,   360,    35,   135,   136,   168,   340,   155,    51,    24,
   305,   156,   157,    38,    53,    52,   210,   211,   242,   283,
   283,   379,    35,   256,   158,   366,   367,   106,   106,   329,
   330,    40,   146,   147,   283,   242,   307,    44,   175,   193,
   308,   309,   331,   169,   341,   153,   194,    51,   328,   177,
   160,   161,   358,    41,    52,   204,   205,   206,   387,   177,
   42,   232,   177,   137,   240,     1,   203,     2,   183,   184,
   13,     3,     4,     5,     6,   283,   283,     7,     8,   130,
   65,   243,   106,   106,     9,   106,   249,    13,   201,   129,
   224,   283,    65,   361,   362,   363,   176,    43,   243,   177,
   231,    10,    11,    12,   164,   165,    13,   106,    48,   106,
   50,   283,   223,   106,   128,   106,   106,    13,   225,   106,
   227,   228,    68,    53,   160,   161,   166,   167,    14,   139,
   15,   178,   233,     1,   177,     2,   336,   337,   140,     3,
   4,     5,     6,   248,   310,     7,     8,   311,   283,   312,
   16,    17,    18,   141,   313,   160,   161,   350,   338,   339,
   179,   180,   106,   177,   177,   142,   381,   314,   315,    10,
   11,    12,    54,    55,    56,    57,    58,    59,    60,    61,
   62,    63,   143,   254,   148,    13,   106,    64,   162,   163,
   181,   262,   106,   177,   160,   161,    14,   300,    15,    65,
   301,   244,    54,    55,    56,    57,    58,    59,    60,    61,
   62,    63,   160,   161,   159,   253,   185,    64,   244,   177,
   258,   186,   170,   263,   187,   278,   278,   171,   192,   264,
   265,   106,   173,   106,   198,   386,   348,   182,   349,   172,
   278,   226,   278,   306,   332,   333,   278,   278,    10,    11,
   12,   177,   382,   195,   278,   149,   177,   190,   150,   191,
   151,   229,    88,    89,   266,   152,    88,    89,   267,   268,
   269,   160,   161,   238,   278,   270,   187,   271,   234,   388,
   236,   278,   278,   272,   273,   212,   213,   214,   215,   261,
   274,   235,    65,   241,   275,   276,   277,   278,   252,   278,
   278,   278,   278,   278,   278,   278,   278,   278,   278,   278,
   278,   278,   278,   278,   278,   278,   278,   278,    88,    89,
   197,   250,    54,    55,    56,    57,    58,    59,    60,    61,
   62,    63,   368,   369,   370,   371,   353,    64,   255,   347,
   279,   279,   149,   383,   259,   150,   384,   151,   303,   196,
   177,   304,   152,   260,   278,   279,   342,   279,   239,   177,
   343,   279,   279,   332,   333,   346,   334,   335,   347,   279,
   316,   317,   318,   319,   320,   321,   322,   323,   324,   325,
   354,    65,   385,   347,   344,   326,   208,   209,   345,   279,
   216,   217,   364,   365,   359,   355,   279,   279,   372,   373,
   49,   134,   133,   257,   202,   207,   218,   220,   222,   219,
   200,   251,   279,   221,   279,   279,   279,   279,   279,   279,
   279,   279,   279,   279,   279,   279,   279,   279,   279,   279,
   279,   279,   279,   237,   230,   376,   374,     0,     0,     0,
   3,     4,     5,     6,    69,    70,   375,   377,     0,     0,
   378,    71,    72,    73,    74,    75,    76,    77,    78,    79,
   80,    81,    82,    83,    84,    85,    86,    87,     0,   279,
   10,    11,    12,     0,     0,     0,     0,     0,     0,     0,
   0,     0,     0,     0,    88,    89,    90,     0,     0,     0,
   91,    92,    93,     0,     0,    94,     0,    95,     0,    96,
   0,   299,     0,     0,     0,     0,    97,     3,     4,     5,
   6,    69,    70,     0,     0,     0,    98,     0,    71,    72,
   73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
   83,    84,    85,    86,    87,     0,     0,    10,    11,    12,
   0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   0,    88,    89,    90,     0,     0,     0,    91,    92,    93,
   0,     0,    94,     0,    95,     0,    96,     0,     0,     0,
   0,     0,     0,    97,     3,     4,     5,     6,    69,    70,
   0,     0,     0,    98,     0,    71,    72,    73,    74,    75,
   76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
   86,    87,     0,     0,    10,    11,    12,     0,     0,     0,
   0,     0,     0,     0,     0,     0,     0,     0,    88,    89,
   90,     0,     0,     0,    91,    92,    93,     0,     0,    94,
   0,    95,   131,    96,     0,     0,     0,     0,     0,     0,
   97,     3,     4,     5,     6,    69,    70,     0,     0,     0,
   98,     0,    71,    72,    73,    74,    75,    76,    77,    78,
   79,    80,    81,    82,    83,    84,    85,    86,    87,     0,
   0,    10,    11,    12,     0,     0,     0,     0,     0,     0,
   0,     0,     0,     0,     0,    88,    89,    90,     0,     0,
   0,    91,    92,    93,     0,     0,    94,     0,    95,   199,
   96,     0,     0,     0,     0,     0,     0,    97,     3,     4,
   5,     6,    69,    70,     0,     0,     0,    98,     0,    71,
   72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
   82,    83,    84,    85,    86,    87,     0,     0,    10,    11,
   12,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   0,     0,    88,    89,    90,     0,     0,     0,    91,    92,
   93,     0,     0,    94,     0,    95,     0,    96,     0,     0,
   0,     0,     0,     0,    97,     3,     4,     5,     6,    69,
   70,     0,     0,     0,    98,     0,     0,     0,     0,     0,
   75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
   85,    86,    87,     0,     0,    10,    11,    12,     0,     0,
   0,     0,     0,     0,     0,     0,     0,     0,     0,    88,
   89,    90,     0,     0,     0,     0,     0,     0,     0,     0,
   94,     0,    95,     0,    96
};

const short
parser::yycheck_[] =
{
   0,    39,    40,    41,    42,    43,    37,    40,    41,    42,
   43,    68,     4,    51,     4,   311,    14,    15,    51,     0,
   20,    38,     4,   139,   140,   304,   256,    67,    66,    67,
   68,   327,     0,    66,    67,    43,    43,   108,    67,    20,
   270,    62,    63,    56,    25,    74,   162,   163,     4,   255,
   256,   347,    20,     9,    75,   334,   335,    95,    96,    62,
   63,    67,    95,    96,   270,     4,   272,    56,    85,    66,
   276,   277,    75,    81,    81,   106,    73,    67,   284,    71,
   60,    61,   312,    67,    74,   156,   157,   158,   384,    71,
   67,    71,    71,   150,    73,     4,   153,     6,   129,   130,
   56,    10,    11,    12,    13,   311,   312,    16,    17,    70,
   71,    67,   150,   151,    23,   153,   232,    56,   151,    68,
   177,   327,    71,   329,   330,   331,    68,    67,    67,    71,
   187,    40,    41,    42,    57,    58,    56,   175,     0,   177,
   4,   347,   175,   181,    18,   183,   184,    56,   181,   187,
   183,   184,    67,   134,    60,    61,    79,    80,    67,    67,
   69,    68,    68,     4,    71,     6,    57,    58,    67,    10,
   11,    12,    13,   230,    64,    16,    17,    67,   384,    69,
   89,    90,    91,    67,    74,    60,    61,   303,    79,    80,
   68,    68,   230,    71,    71,    67,    71,    87,    88,    40,
   41,    42,    44,    45,    46,    47,    48,    49,    50,    51,
   52,    53,    67,   244,    14,    56,   254,    59,    77,    78,
   68,   254,   260,    71,    60,    61,    67,   260,    69,    71,
   261,   229,    44,    45,    46,    47,    48,    49,    50,    51,
   52,    53,    60,    61,    20,   243,    68,    59,   246,    71,
   68,    68,    82,    15,    71,   255,   256,    83,   143,    21,
   22,   299,    39,   301,   149,   381,   299,    19,   301,    84,
   270,    67,   272,   271,    60,    61,   276,   277,    40,    41,
   42,    71,    68,    68,   284,    64,    71,    56,    67,    56,
   69,    73,    54,    55,    56,    74,    54,    55,    60,    61,
   62,    60,    61,    68,   304,    67,    71,    69,    68,    68,
   71,   311,   312,    75,    76,   164,   165,   166,   167,    68,
   82,    68,    71,    56,    86,    87,    88,   327,    68,   329,
   330,   331,   332,   333,   334,   335,   336,   337,   338,   339,
   340,   341,   342,   343,   344,   345,   346,   347,    54,    55,
   56,   236,    44,    45,    46,    47,    48,    49,    50,    51,
   52,    53,   336,   337,   338,   339,    68,    59,     9,    71,
   255,   256,    64,    68,    68,    67,    71,    69,    67,    70,
   71,    67,    74,    73,   384,   270,    82,   272,    70,    71,
   83,   276,   277,    60,    61,    38,    77,    78,    71,   284,
   44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
   70,    71,    70,    71,    84,    59,   160,   161,    39,   304,
   168,   169,   332,   333,    56,   310,   311,   312,   340,   341,
   20,    65,    52,   246,   152,   159,   170,   172,   174,   171,
   150,   240,   327,   173,   329,   330,   331,   332,   333,   334,
   335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
   345,   346,   347,   194,     4,   344,   342,    -1,    -1,    -1,
   10,    11,    12,    13,    14,    15,   343,   345,    -1,    -1,
   346,    21,    22,    23,    24,    25,    26,    27,    28,    29,
   30,    31,    32,    33,    34,    35,    36,    37,    -1,   384,
   40,    41,    42,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   -1,    -1,    -1,    -1,    54,    55,    56,    -1,    -1,    -1,
   60,    61,    62,    -1,    -1,    65,    -1,    67,    -1,    69,
   -1,     4,    -1,    -1,    -1,    -1,    76,    10,    11,    12,
   13,    14,    15,    -1,    -1,    -1,    86,    -1,    21,    22,
   23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
   33,    34,    35,    36,    37,    -1,    -1,    40,    41,    42,
   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   -1,    54,    55,    56,    -1,    -1,    -1,    60,    61,    62,
   -1,    -1,    65,    -1,    67,    -1,    69,    -1,    -1,    -1,
   -1,    -1,    -1,    76,    10,    11,    12,    13,    14,    15,
   -1,    -1,    -1,    86,    -1,    21,    22,    23,    24,    25,
   26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
   36,    37,    -1,    -1,    40,    41,    42,    -1,    -1,    -1,
   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    54,    55,
   56,    -1,    -1,    -1,    60,    61,    62,    -1,    -1,    65,
   -1,    67,    68,    69,    -1,    -1,    -1,    -1,    -1,    -1,
   76,    10,    11,    12,    13,    14,    15,    -1,    -1,    -1,
   86,    -1,    21,    22,    23,    24,    25,    26,    27,    28,
   29,    30,    31,    32,    33,    34,    35,    36,    37,    -1,
   -1,    40,    41,    42,    -1,    -1,    -1,    -1,    -1,    -1,
   -1,    -1,    -1,    -1,    -1,    54,    55,    56,    -1,    -1,
   -1,    60,    61,    62,    -1,    -1,    65,    -1,    67,    68,
   69,    -1,    -1,    -1,    -1,    -1,    -1,    76,    10,    11,
   12,    13,    14,    15,    -1,    -1,    -1,    86,    -1,    21,
   22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
   32,    33,    34,    35,    36,    37,    -1,    -1,    40,    41,
   42,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   -1,    -1,    54,    55,    56,    -1,    -1,    -1,    60,    61,
   62,    -1,    -1,    65,    -1,    67,    -1,    69,    -1,    -1,
   -1,    -1,    -1,    -1,    76,    10,    11,    12,    13,    14,
   15,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,
   25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
   35,    36,    37,    -1,    -1,    40,    41,    42,    -1,    -1,
   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    54,
   55,    56,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   65,    -1,    67,    -1,    69
};

const unsigned char
parser::yystos_[] =
{
   0,     4,     6,    10,    11,    12,    13,    16,    17,    23,
   40,    41,    42,    56,    67,    69,    89,    90,    91,    94,
   95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
   105,   106,   107,   111,   112,   113,   114,   115,    56,    67,
   67,    67,    67,    67,    56,    99,   101,   101,     0,    96,
   4,    67,    74,    99,    44,    45,    46,    47,    48,    49,
   50,    51,    52,    53,    59,    71,   138,   138,    67,    14,
   15,    21,    22,    23,    24,    25,    26,    27,    28,    29,
   30,    31,    32,    33,    34,    35,    36,    37,    54,    55,
   56,    60,    61,    62,    65,    67,    69,    76,    86,   113,
   115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
   125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
   135,   136,   137,   139,   139,   139,   139,   139,    18,    68,
   70,    68,   139,    98,   100,   139,   139,   137,   140,    67,
   67,    67,    67,    67,   141,   142,   139,   139,    14,    64,
   67,    69,    74,   138,   121,   124,    62,    63,    75,    20,
   60,    61,    77,    78,    57,    58,    79,    80,    43,    81,
   82,    83,    84,    39,    38,    85,    68,    71,    68,    68,
   68,    68,    19,   138,   138,    68,    68,    71,   127,   127,
   56,    56,   116,    66,    73,    68,    70,    56,   116,    68,
   140,   139,   119,   137,   124,   124,   124,   125,   126,   126,
   127,   127,   128,   128,   128,   128,   129,   129,   130,   131,
   132,   133,   134,   139,   137,   139,    67,   139,   139,    73,
   4,   137,    71,    68,    68,    68,    71,   141,    68,    70,
   73,    56,     4,    67,   101,   108,   109,   110,   137,   127,
   116,   136,    68,   101,   138,     9,     9,   110,    68,    68,
   73,    68,   139,    15,    21,    22,    56,    60,    61,    62,
   67,    69,    75,    76,    82,    86,    87,    88,   115,   116,
   143,   144,   145,   147,   148,   149,   150,   151,   152,   153,
   154,   155,   156,   157,   158,   159,   160,   162,   162,     4,
   139,   138,     4,    67,    67,   162,   101,   147,   147,   147,
   64,    67,    69,    74,    87,    88,    44,    45,    46,    47,
   48,    49,    50,    51,    52,    53,    59,   161,   147,    62,
   63,    75,    60,    61,    77,    78,    57,    58,    79,    80,
   43,    81,    82,    83,    84,    39,    38,    71,   139,   139,
   127,   147,   150,    68,    70,   116,   146,   160,   162,    56,
   160,   147,   147,   147,   149,   149,   150,   150,   151,   151,
   151,   151,   152,   152,   153,   154,   155,   156,   157,   160,
   4,    71,    68,    68,    71,    70,   127,   160,    68
};

const unsigned char
parser::yyr1_[] =
{
   0,    93,    94,    95,    95,    95,    96,    96,    97,    97,
   97,    97,    97,    97,    97,    97,    97,    98,    99,    99,
   99,    99,   100,   100,   101,   101,   102,   102,   102,   102,
   103,   104,   105,   106,   107,   107,   108,   109,   109,   110,
   110,   110,   111,   111,   112,   113,   113,   113,   113,   114,
   115,   115,   115,   116,   116,   117,   117,   118,   118,   118,
   118,   118,   118,   118,   118,   118,   118,   118,   118,   118,
   118,   119,   119,   119,   119,   119,   119,   119,   119,   119,
   120,   120,   121,   121,   121,   121,   121,   121,   121,   121,
   121,   121,   122,   122,   123,   123,   123,   123,   123,   124,
   125,   125,   125,   125,   126,   126,   127,   127,   127,   128,
   128,   128,   129,   129,   129,   129,   129,   130,   130,   130,
   131,   131,   132,   132,   133,   133,   134,   134,   135,   135,
   136,   136,   137,   137,   138,   138,   138,   138,   138,   138,
   138,   138,   138,   138,   138,   139,   139,   140,   140,   140,
   141,   142,   142,   143,   143,   143,   143,   143,   143,   144,
   145,   145,   145,   145,   145,   145,   145,   145,   145,   146,
   146,   147,   147,   147,   147,   147,   148,   148,   148,   148,
   148,   148,   149,   149,   149,   149,   150,   150,   150,   151,
   151,   151,   152,   152,   152,   152,   152,   153,   153,   153,
   154,   154,   155,   155,   156,   156,   157,   157,   158,   158,
   159,   160,   160,   161,   161,   161,   161,   161,   161,   161,
   161,   161,   161,   161,   162,   162
};

const signed char
parser::yyr2_[] =
{
   0,     2,     1,     1,     2,     1,     1,     2,     1,     3,
   3,     5,     5,     1,     1,     1,     1,     1,     1,     3,
   4,     3,     1,     2,     1,     3,     1,     1,     1,     1,
   1,     1,     1,     1,     9,     9,     0,     1,     2,     1,
   4,     6,    10,     9,     5,     4,     4,     4,     4,     1,
   1,     1,     1,     1,     1,     1,     2,     1,     1,     1,
   1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
   1,     1,     1,     1,     1,     1,     3,     3,     3,     1,
   3,     3,     1,     1,     6,     4,     3,     4,     3,     4,
   4,     4,     1,     2,     1,     1,     1,     1,     1,     1,
   1,     3,     3,     3,     1,     3,     1,     3,     3,     1,
   3,     3,     1,     3,     3,     3,     3,     1,     3,     3,
   1,     3,     1,     3,     1,     3,     1,     3,     1,     3,
   1,     5,     1,     3,     1,     1,     1,     1,     1,     1,
   1,     1,     1,     1,     1,     1,     3,     1,     3,     4,
   5,     1,     3,     1,     1,     1,     1,     3,     3,     6,
   1,     4,     4,     3,     2,     2,     3,     1,     4,     1,
   3,     1,     2,     2,     2,     2,     1,     1,     1,     1,
   1,     1,     1,     3,     3,     3,     1,     3,     3,     1,
   3,     3,     1,     3,     3,     3,     3,     1,     3,     3,
   1,     3,     1,     3,     1,     3,     1,     3,     1,     3,
   1,     1,     3,     1,     1,     1,     1,     1,     1,     1,
   1,     1,     1,     1,     1,     3
};




#if YYDEBUG
const short
parser::yyrline_[] =
{
   0,    93,    93,    95,    95,    95,    97,    97,    99,   100,
   101,   102,   103,   104,   105,   106,   107,   109,   111,   112,
   113,   114,   116,   116,   118,   118,   121,   121,   121,   121,
   122,   123,   124,   125,   129,   130,   131,   132,   132,   133,
   134,   135,   139,   140,   143,   146,   147,   148,   149,   152,
   155,   155,   155,   158,   158,   160,   160,   162,   163,   164,
   165,   166,   167,   168,   169,   170,   171,   172,   173,   174,
   175,   177,   178,   179,   180,   181,   182,   183,   184,   185,
   187,   188,   190,   191,   192,   193,   194,   195,   196,   197,
   198,   199,   201,   201,   203,   203,   203,   203,   203,   205,
   207,   208,   209,   210,   212,   213,   215,   216,   217,   219,
   220,   221,   223,   224,   225,   226,   227,   229,   230,   231,
   233,   234,   236,   237,   239,   240,   242,   243,   245,   246,
   248,   249,   251,   252,   254,   255,   255,   255,   255,   255,
   256,   256,   256,   257,   257,   259,   260,   262,   263,   264,
   266,   268,   268,   273,   274,   275,   276,   277,   278,   280,
   283,   284,   285,   286,   287,   288,   289,   290,   291,   294,
   295,   298,   299,   300,   301,   302,   304,   304,   304,   304,
   304,   304,   306,   307,   308,   309,   311,   312,   313,   315,
   316,   317,   319,   320,   321,   322,   323,   325,   326,   327,
   329,   330,   332,   333,   335,   336,   338,   339,   341,   342,
   344,   346,   347,   349,   350,   350,   350,   350,   350,   351,
   351,   351,   352,   352,   354,   355
};

void
parser::yy_stack_print_ () const
{
   *yycdebug_ << "Stack now";
   for (stack_type::const_iterator
        i = yystack_.begin (),
        i_end = yystack_.end ();
        i != i_end; ++i)
   {
      *yycdebug_ << ' ' << int (i->state);
   }
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

// //////////////////////////////////////////////////////////////////////////
// *INDENT-ON*
#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 165
namespace yy { std::array<bool, yyruletype::yynrules> rules = {false}; }

// *****************************************************************************
static Node* astAddChild(Node* root, Node* n)
{
   assert(n);
   assert(root);
   n->root = root;
   if (!root->child) { return root->child = n; }
   Node* child = root->child;
   for (; child->next; child = child->next);
   return child->next = n;
}

// *****************************************************************************
static Node* astAddNext(Node* root, Node* n)
{
   assert(n);
   assert(root);
   n->root = root->root;
   if (!root->next) { return root->next = n; }
   Node* next = root;
   for (; next->next; next = next->next);
   return next->next = n;
}

// *****************************************************************************
template<int YYN>
void rhs(xfl &ufl, Node **yylval,  const int yyn,
         const symbol_t yyr1n, const int nrhs, const char *rule,
         yy::parser::stack_type &yystack)
{
   assert(YYN == yyn);
   Node *root = *yylval = ufl.astAddNode(std::make_shared<Rule>(yyn, rule));
   if (nrhs == 0) { return; } // %empty
   Node *n = yystack[nrhs-1].value;
   astAddChild(root, n);
   for (int i = 1; i < nrhs ; i++)
   { (root = n, astAddNext(root, n = yystack[nrhs-1-i].value)); }
}

// *****************************************************************************
void xfl::dfs(Node *n, Middlend &me)
{
   assert(n);
   if (!n) { return; }

   // Process the current node, setting up the dfs.down to true
   n->Accept(me);

   const int N = n->Number(); // N = SN (token) | RN (rule)
   assert(N > 0);

   // Sanity checks for rules
   if (n->IsRule())
   {
      constexpr int YYNRULES = yynrules;
      if (N > YYNRULES) { DBG("\n\033[31m[rule] N:%d/%d",N,YYNRULES); }
      assert(N <= YYNRULES);
   }

   // Sanity checks for tokens
   if (n->IsToken())
   {
      constexpr symbol_t YYNTOKENS = yy::parser::YYNTOKENS;
      if (N >= YYNTOKENS) { DBG("\n\033[31m[token] N:%d/%d",N,YYNTOKENS); }
      assert(N < YYNTOKENS);
   }

   // Set the state flags
   if (n->IsRule()) { yy::rules.at(N) = true; }

   // If n->dfs.down does not stop us from previous Accept, dfs down
   if (n->dfs.down && n->child)
   {
      dfs(n->child, me);
      // If me.ctx.extra has been set, re-run a dfs with it
      if (me.ufl.ctx.extra)
      {
         Node *extra = me.ufl.ctx.extra;
         me.ufl.ctx.extra = nullptr;
         dfs(extra, me);
      }
   }

   // Process the current node, setting up the dfs.down to false
   if (n->IsRule()) {n->Accept(me, false);} // up, only for rules

   // Reset the state flags
   if (n->IsRule()) { yy::rules.at(N) = false; }

   if (n->next) { dfs(n->next, me); }
}

// ****************************************************************************
bool xfl::HitRule(const int rule, Node *n)
{
   if (!n) { return false; }
   if (n->IsRule() && n->Number() == rule) { return true; }
   if (n->child) { if (HitRule(rule,n->child)) { return true; }}
   if (n->next) { if (HitRule(rule,n->next)) { return true; }}
   return false;
}

// ****************************************************************************
bool xfl::HitToken(const int tok, Node *n)
{
   if (!n) { return false; }
   if (n->IsToken() && n->Number() == tok) { return true; }
   if (n->child) { if (HitToken(tok,n->child)) { return true; }}
   if (n->next) { if (HitToken(tok,n->next)) { return true; }}
   return false;
}

// ****************************************************************************
bool xfl::OnlyToken(const int tok, Node *n)
{
   assert(n);
   if (n->IsToken() && n->Number() != tok) { return false; }
   if (n->child) { if (!OnlyToken(tok, n->child)) { return false; }}
   if (n->next) { if (!OnlyToken(tok, n->next)) { return false; }}
   return true;
}

// ****************************************************************************
Node *xfl::GetToken(const int tok, Node *n)
{
   assert(n);
   Node *m = nullptr;
   if (n->IsToken() && n->Number() == tok) { return n; }
   if (n->child) { if ((m=GetToken(tok, n->child))) { return m; }}
   if (n->next) { if ((m=GetToken(tok, n->next))) { return m; }}
   return m;
}

// *****************************************************************************
void yy::parser::error(const location_type&, const std::string& msg)
{
   std::cerr << (*ufl.loc) << ": " << msg << std::endl;
   abort();
}
