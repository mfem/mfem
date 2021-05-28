// A Bison parser, made by GNU Bison 3.7.5.

// Skeleton implementation for Bison LALR(1) parsers in C++

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

// DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
// especially those whose name start with YY_ or yy_.  They are
// private implementation details that can be changed or removed.



// First part of user prologue.

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

#include "xfl.hpp"
#include "xfl.Y.hpp"
#include "xfl_mid.hpp"
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
# define YY_SYMBOL_PRINT(Title, Symbol)  YY_USE (Symbol)
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
parser::by_kind::clear () YY_NOEXCEPT
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
   YY_USE (yysym.kind ());
}

#if YYDEBUG
template <typename Base>
void
parser::yy_print_ (std::ostream& yyo, const basic_symbol<Base>& yysym) const
{
   std::ostream& yyoutput = yyo;
   YY_USE (yyoutput);
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
      YY_USE (yykind);
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

               case 27:   // extra_status_rule: expr_quote
               {
#define extra_status_rule_expr_quote 27
                  constexpr int YYN = 27;
                  {} {RHS}
                  break;
               }

               case 28:   // extra_status_rule: transpose_xt
               {
#define extra_status_rule_transpose_xt 28
                  constexpr int YYN = 28;
                  {} {RHS}
                  break;
               }

               case 29:   // extra_status_rule: dot_xt
               {
#define extra_status_rule_dot_xt 29
                  constexpr int YYN = 29;
                  {} {RHS}
                  break;
               }

               case 30:   // extra_status_rule: eval_xt
               {
#define extra_status_rule_eval_xt 30
                  constexpr int YYN = 30;
                  {} {RHS}
                  break;
               }

               case 31:   // extra_status_rule: var_xt
               {
#define extra_status_rule_var_xt 31
                  constexpr int YYN = 31;
                  {} {RHS}
                  break;
               }

               case 32:   // extra_status_rule: dom_xt
               {
#define extra_status_rule_dom_xt 32
                  constexpr int YYN = 32;
                  {} {RHS}
                  break;
               }

               case 33:   // lhs: LHS
               {
#define lhs_lhs 33
                  constexpr int YYN = 33;
                  {} {RHS}
                  break;
               }

               case 34:   // dot_xt: DOT_XT
               {
#define dot_xt_dot_xt 34
                  constexpr int YYN = 34;
                  {} {RHS}
                  break;
               }

               case 35:   // eval_xt: EVAL_XT
               {
#define eval_xt_eval_xt 35
                  constexpr int YYN = 35;
                  {} {RHS}
                  break;
               }

               case 36:   // transpose_xt: TRANSPOSE_XT
               {
#define transpose_xt_transpose_xt 36
                  constexpr int YYN = 36;
                  {} {RHS}
                  break;
               }

               case 37:   // var_xt: VAR_XT
               {
#define var_xt_var_xt 37
                  constexpr int YYN = 37;
                  {} {RHS}
                  break;
               }

               case 38:   // dom_xt: DOM_XT
               {
#define dom_xt_dom_xt 38
                  constexpr int YYN = 38;
                  {} {RHS}
                  break;
               }

               case 39:   // expr_quote: EXPR_QUOTE
               {
#define expr_quote_expr_quote 39
                  constexpr int YYN = 39;
                  {} {RHS}
                  break;
               }

               case 40:   // function: DEF IDENTIFIER LP args_expr_list RP COLON def_empty RETURN math_expr
               {
#define function_def_identifier_lp_args_expr_list_rp_colon_def_empty_return_math_expr 40
                  constexpr int YYN = 40;
                  {} {RHS}
                  break;
               }

               case 41:   // function: DEF IDENTIFIER LP args_expr_list RP COLON def_statements RETURN math_expr
               {
#define function_def_identifier_lp_args_expr_list_rp_colon_def_statements_return_math_expr 41
                  constexpr int YYN = 41;
                  {} {RHS}
                  break;
               }

               case 42:   // def_empty: %empty
               {
#define def_empty_empty 42
                  constexpr int YYN = 42;
                  {} {RHS}
                  break;
               }

               case 43:   // def_statements: def_statement
               {
#define def_statements_def_statement 43
                  constexpr int YYN = 43;
                  {} {RHS}
                  break;
               }

               case 44:   // def_statements: def_statements def_statement
               {
#define def_statements_def_statements_def_statement 44
                  constexpr int YYN = 44;
                  {} {RHS}
                  break;
               }

               case 45:   // def_statement: NL
               {
#define def_statement_nl 45
                  constexpr int YYN = 45;
                  {} {RHS}
                  break;
               }

               case 46:   // def_statement: id_list assign_op expr NL
               {
#define def_statement_id_list_assign_op_expr_nl 46
                  constexpr int YYN = 46;
                  {} {RHS}
                  break;
               }

               case 47:   // def_statement: LP id_list RP assign_op expr NL
               {
#define def_statement_lp_id_list_rp_assign_op_expr_nl 47
                  constexpr int YYN = 47;
                  {} {RHS}
                  break;
               }

               case 48:   // iteration_statement: FOR IDENTIFIER IN RANGE LP IDENTIFIER RP COLON NL expr
               {
#define iteration_statement_for_identifier_in_range_lp_identifier_rp_colon_nl_expr 48
                  constexpr int YYN = 48;
                  {} {RHS}
                  break;
               }

               case 49:   // iteration_statement: FOR IDENTIFIER IN RANGE LP IDENTIFIER RP COLON expr
               {
#define iteration_statement_for_identifier_in_range_lp_identifier_rp_colon_expr 49
                  constexpr int YYN = 49;
                  {} {RHS}
                  break;
               }

               case 50:   // if_statement: IF LP expr RP expr
               {
#define if_statement_if_lp_expr_rp_expr 50
                  constexpr int YYN = 50;
                  {} {RHS}
                  break;
               }

               case 51:   // api_statement: PLOT LP expr RP
               {
#define api_statement_plot_lp_expr_rp 51
                  constexpr int YYN = 51;
                  {} {RHS}
                  break;
               }

               case 52:   // api_statement: SAVE LP expr RP
               {
#define api_statement_save_lp_expr_rp 52
                  constexpr int YYN = 52;
                  {} {RHS}
                  break;
               }

               case 53:   // api_statement: SOLVE LP expr RP
               {
#define api_statement_solve_lp_expr_rp 53
                  constexpr int YYN = 53;
                  {} {RHS}
                  break;
               }

               case 54:   // api_statement: PROJECT LP expr RP
               {
#define api_statement_project_lp_expr_rp 54
                  constexpr int YYN = 54;
                  {} {RHS}
                  break;
               }

               case 55:   // api_statement: BENCHMARK LP expr RP
               {
#define api_statement_benchmark_lp_expr_rp 55
                  constexpr int YYN = 55;
                  {} {RHS}
                  break;
               }

               case 56:   // direct_declarator: postfix_id
               {
#define direct_declarator_postfix_id 56
                  constexpr int YYN = 56;
                  {} {RHS}
                  break;
               }

               case 57:   // domain: DOM_DX
               {
#define domain_dom_dx 57
                  constexpr int YYN = 57;
                  {} {RHS}
                  break;
               }

               case 58:   // domain: EXT_DS
               {
#define domain_ext_ds 58
                  constexpr int YYN = 58;
                  {} {RHS}
                  break;
               }

               case 59:   // domain: INT_DS
               {
#define domain_int_ds 59
                  constexpr int YYN = 59;
                  {} {RHS}
                  break;
               }

               case 60:   // constant: NATURAL
               {
#define constant_natural 60
                  constexpr int YYN = 60;
                  {} {RHS}
                  break;
               }

               case 61:   // constant: REAL
               {
#define constant_real 61
                  constexpr int YYN = 61;
                  {} {RHS}
                  break;
               }

               case 62:   // constant: BOOL
               {
#define constant_bool 62
                  constexpr int YYN = 62;
                  {} {RHS}
                  break;
               }

               case 63:   // strings: STRING
               {
#define strings_string 63
                  constexpr int YYN = 63;
                  {} {RHS}
                  break;
               }

               case 64:   // strings: strings STRING
               {
#define strings_strings_string 64
                  constexpr int YYN = 64;
                  {} {RHS}
                  break;
               }

               case 65:   // id_n: conditional_expr
               {
#define id_n_conditional_expr 65
                  constexpr int YYN = 65;
                  {} {RHS}
                  break;
               }

               case 66:   // fes_args: IDENTIFIER COMA IDENTIFIER
               {
#define fes_args_identifier_coma_identifier 66
                  constexpr int YYN = 66;
                  {} {RHS}
                  break;
               }

               case 67:   // fes_args: IDENTIFIER COMA IDENTIFIER COMA id_n
               {
#define fes_args_identifier_coma_identifier_coma_id_n 67
                  constexpr int YYN = 67;
                  {} {RHS}
                  break;
               }

               case 68:   // fes_args: IDENTIFIER COMA QUOTE COMA id_n
               {
#define fes_args_identifier_coma_quote_coma_id_n 68
                  constexpr int YYN = 68;
                  {} {RHS}
                  break;
               }

               case 69:   // element_type: POINT
               {
#define element_type_point 69
                  constexpr int YYN = 69;
                  {} {RHS}
                  break;
               }

               case 70:   // element_type: TRIANGLE
               {
#define element_type_triangle 70
                  constexpr int YYN = 70;
                  {} {RHS}
                  break;
               }

               case 71:   // element_type: QUADRILATERAL
               {
#define element_type_quadrilateral 71
                  constexpr int YYN = 71;
                  {} {RHS}
                  break;
               }

               case 72:   // element_type: TETRAHEDRON
               {
#define element_type_tetrahedron 72
                  constexpr int YYN = 72;
                  {} {RHS}
                  break;
               }

               case 73:   // element_type: HEXAHEDRON
               {
#define element_type_hexahedron 73
                  constexpr int YYN = 73;
                  {} {RHS}
                  break;
               }

               case 74:   // element_type: WEDGE
               {
#define element_type_wedge 74
                  constexpr int YYN = 74;
                  {} {RHS}
                  break;
               }

               case 75:   // api: DEVICE
               {
#define api_device 75
                  constexpr int YYN = 75;
                  {} {RHS}
                  break;
               }

               case 76:   // api: MESH
               {
#define api_mesh 76
                  constexpr int YYN = 76;
                  {} {RHS}
                  break;
               }

               case 77:   // api: FUNCTION
               {
#define api_function 77
                  constexpr int YYN = 77;
                  {} {RHS}
                  break;
               }

               case 78:   // api: UNIT_HEX_MESH
               {
#define api_unit_hex_mesh 78
                  constexpr int YYN = 78;
                  {} {RHS}
                  break;
               }

               case 79:   // api: UNIT_SQUARE_MESH
               {
#define api_unit_square_mesh 79
                  constexpr int YYN = 79;
                  {} {RHS}
                  break;
               }

               case 80:   // api: FINITE_ELEMENT
               {
#define api_finite_element 80
                  constexpr int YYN = 80;
                  {} {RHS}
                  break;
               }

               case 81:   // api: FUNCTION_SPACE LP fes_args RP
               {
#define api_function_space_lp_fes_args_rp 81
                  constexpr int YYN = 81;
                  {} {RHS}
                  break;
               }

               case 82:   // api: VECTOR_FUNCTION_SPACE
               {
#define api_vector_function_space 82
                  constexpr int YYN = 82;
                  {} {RHS}
                  break;
               }

               case 83:   // api: EXPRESSION
               {
#define api_expression 83
                  constexpr int YYN = 83;
                  {} {RHS}
                  break;
               }

               case 84:   // api: DIRICHLET_BC
               {
#define api_dirichlet_bc 84
                  constexpr int YYN = 84;
                  {} {RHS}
                  break;
               }

               case 85:   // api: TRIAL_FUNCTION
               {
#define api_trial_function 85
                  constexpr int YYN = 85;
                  {} {RHS}
                  break;
               }

               case 86:   // api: TEST_FUNCTION
               {
#define api_test_function 86
                  constexpr int YYN = 86;
                  {} {RHS}
                  break;
               }

               case 87:   // api: CONSTANT_API
               {
#define api_constant_api 87
                  constexpr int YYN = 87;
                  {} {RHS}
                  break;
               }

               case 88:   // api: api_statement
               {
#define api_api_statement 88
                  constexpr int YYN = 88;
                  {} {RHS}
                  break;
               }

               case 89:   // api: element_type
               {
#define api_element_type 89
                  constexpr int YYN = 89;
                  {} {RHS}
                  break;
               }

               case 90:   // primary_expr: IDENTIFIER
               {
#define primary_expr_identifier 90
                  constexpr int YYN = 90;
                  {} {RHS}
                  break;
               }

               case 91:   // primary_expr: constant
               {
#define primary_expr_constant 91
                  constexpr int YYN = 91;
                  {} {RHS}
                  break;
               }

               case 92:   // primary_expr: domain
               {
#define primary_expr_domain 92
                  constexpr int YYN = 92;
                  {} {RHS}
                  break;
               }

               case 93:   // primary_expr: QUOTE
               {
#define primary_expr_quote 93
                  constexpr int YYN = 93;
                  {} {RHS}
                  break;
               }

               case 94:   // primary_expr: strings
               {
#define primary_expr_strings 94
                  constexpr int YYN = 94;
                  {} {RHS}
                  break;
               }

               case 95:   // primary_expr: LP expr RP
               {
#define primary_expr_lp_expr_rp 95
                  constexpr int YYN = 95;
                  {} {RHS}
                  break;
               }

               case 96:   // primary_expr: LB expr RB
               {
#define primary_expr_lb_expr_rb 96
                  constexpr int YYN = 96;
                  {} {RHS}
                  break;
               }

               case 97:   // primary_expr: LS coords RS
               {
#define primary_expr_ls_coords_rs 97
                  constexpr int YYN = 97;
                  {} {RHS}
                  break;
               }

               case 98:   // form_args: LP additive_expr RP
               {
#define form_args_lp_additive_expr_rp 98
                  constexpr int YYN = 98;
                  {} {RHS}
                  break;
               }

               case 99:   // grad_expr: GRAD_OP form_args
               {
#define grad_expr_grad_op_form_args 99
                  constexpr int YYN = 99;
                  {} {RHS}
                  break;
               }

               case 100:   // transpose_expr: TRANSPOSE_OP form_args
               {
#define transpose_expr_transpose_op_form_args 100
                  constexpr int YYN = 100;
                  {} {RHS}
                  break;
               }

               case 101:   // pow_expr: postfix_expr POW constant
               {
#define pow_expr_postfix_expr_pow_constant 101
                  constexpr int YYN = 101;
                  {} {RHS}
                  break;
               }

               case 102:   // pow_expr: postfix_expr POW IDENTIFIER
               {
#define pow_expr_postfix_expr_pow_identifier 102
                  constexpr int YYN = 102;
                  {} {RHS}
                  break;
               }

               case 103:   // postfix_expr: primary_expr
               {
#define postfix_expr_primary_expr 103
                  constexpr int YYN = 103;
                  {} {RHS}
                  break;
               }

               case 104:   // postfix_expr: pow_expr
               {
#define postfix_expr_pow_expr 104
                  constexpr int YYN = 104;
                  {} {RHS}
                  break;
               }

               case 105:   // postfix_expr: INNER_OP LP additive_expr COMA additive_expr RP
               {
#define postfix_expr_inner_op_lp_additive_expr_coma_additive_expr_rp 105
                  constexpr int YYN = 105;
                  {} {RHS}
                  break;
               }

               case 106:   // postfix_expr: postfix_expr LB expr RB
               {
#define postfix_expr_postfix_expr_lb_expr_rb 106
                  constexpr int YYN = 106;
                  {} {RHS}
                  break;
               }

               case 107:   // postfix_expr: postfix_expr LP RP
               {
#define postfix_expr_postfix_expr_lp_rp 107
                  constexpr int YYN = 107;
                  {} {RHS}
                  break;
               }

               case 108:   // postfix_expr: postfix_expr LP args_expr_list RP
               {
#define postfix_expr_postfix_expr_lp_args_expr_list_rp 108
                  constexpr int YYN = 108;
                  {} {RHS}
                  break;
               }

               case 109:   // postfix_expr: postfix_expr DOT primary_expr
               {
#define postfix_expr_postfix_expr_dot_primary_expr 109
                  constexpr int YYN = 109;
                  {} {RHS}
                  break;
               }

               case 110:   // postfix_expr: grad_expr
               {
#define postfix_expr_grad_expr 110
                  constexpr int YYN = 110;
                  {} {RHS}
                  break;
               }

               case 111:   // postfix_expr: transpose_expr
               {
#define postfix_expr_transpose_expr 111
                  constexpr int YYN = 111;
                  {} {RHS}
                  break;
               }

               case 112:   // postfix_expr: LHS LP IDENTIFIER RP
               {
#define postfix_expr_lhs_lp_identifier_rp 112
                  constexpr int YYN = 112;
                  {} {RHS}
                  break;
               }

               case 113:   // postfix_expr: RHS LP IDENTIFIER RP
               {
#define postfix_expr_rhs_lp_identifier_rp 113
                  constexpr int YYN = 113;
                  {} {RHS}
                  break;
               }

               case 114:   // postfix_expr: api
               {
#define postfix_expr_api 114
                  constexpr int YYN = 114;
                  {} {RHS}
                  break;
               }

               case 115:   // unary_expr: postfix_expr
               {
#define unary_expr_postfix_expr 115
                  constexpr int YYN = 115;
                  {} {RHS}
                  break;
               }

               case 116:   // unary_expr: unary_op cast_expr
               {
#define unary_expr_unary_op_cast_expr 116
                  constexpr int YYN = 116;
                  {} {RHS}
                  break;
               }

               case 117:   // unary_op: MUL
               {
#define unary_op_mul 117
                  constexpr int YYN = 117;
                  {} {RHS}
                  break;
               }

               case 118:   // unary_op: ADD
               {
#define unary_op_add 118
                  constexpr int YYN = 118;
                  {} {RHS}
                  break;
               }

               case 119:   // unary_op: SUB
               {
#define unary_op_sub 119
                  constexpr int YYN = 119;
                  {} {RHS}
                  break;
               }

               case 120:   // unary_op: TILDE
               {
#define unary_op_tilde 120
                  constexpr int YYN = 120;
                  {} {RHS}
                  break;
               }

               case 121:   // unary_op: NOT
               {
#define unary_op_not 121
                  constexpr int YYN = 121;
                  {} {RHS}
                  break;
               }

               case 122:   // cast_expr: unary_expr
               {
#define cast_expr_unary_expr 122
                  constexpr int YYN = 122;
                  {} {RHS}
                  break;
               }

               case 123:   // multiplicative_expr: cast_expr
               {
#define multiplicative_expr_cast_expr 123
                  constexpr int YYN = 123;
                  {} {RHS}
                  break;
               }

               case 124:   // multiplicative_expr: multiplicative_expr MUL cast_expr
               {
#define multiplicative_expr_multiplicative_expr_mul_cast_expr 124
                  constexpr int YYN = 124;
                  {} {RHS}
                  break;
               }

               case 125:   // multiplicative_expr: multiplicative_expr DIV cast_expr
               {
#define multiplicative_expr_multiplicative_expr_div_cast_expr 125
                  constexpr int YYN = 125;
                  {} {RHS}
                  break;
               }

               case 126:   // multiplicative_expr: multiplicative_expr MOD cast_expr
               {
#define multiplicative_expr_multiplicative_expr_mod_cast_expr 126
                  constexpr int YYN = 126;
                  {} {RHS}
                  break;
               }

               case 127:   // dot_expr: multiplicative_expr
               {
#define dot_expr_multiplicative_expr 127
                  constexpr int YYN = 127;
                  {} {RHS}
                  break;
               }

               case 128:   // dot_expr: dot_expr DOT_OP multiplicative_expr
               {
#define dot_expr_dot_expr_dot_op_multiplicative_expr 128
                  constexpr int YYN = 128;
                  {} {RHS}
                  break;
               }

               case 129:   // additive_expr: dot_expr
               {
#define additive_expr_dot_expr 129
                  constexpr int YYN = 129;
                  {} {RHS}
                  break;
               }

               case 130:   // additive_expr: additive_expr ADD dot_expr
               {
#define additive_expr_additive_expr_add_dot_expr 130
                  constexpr int YYN = 130;
                  {} {RHS}
                  break;
               }

               case 131:   // additive_expr: additive_expr SUB dot_expr
               {
#define additive_expr_additive_expr_sub_dot_expr 131
                  constexpr int YYN = 131;
                  {} {RHS}
                  break;
               }

               case 132:   // shift_expr: additive_expr
               {
#define shift_expr_additive_expr 132
                  constexpr int YYN = 132;
                  {} {RHS}
                  break;
               }

               case 133:   // shift_expr: shift_expr LEFT_SHIFT additive_expr
               {
#define shift_expr_shift_expr_left_shift_additive_expr 133
                  constexpr int YYN = 133;
                  {} {RHS}
                  break;
               }

               case 134:   // shift_expr: shift_expr RIGHT_SHIFT additive_expr
               {
#define shift_expr_shift_expr_right_shift_additive_expr 134
                  constexpr int YYN = 134;
                  {} {RHS}
                  break;
               }

               case 135:   // relational_expr: shift_expr
               {
#define relational_expr_shift_expr 135
                  constexpr int YYN = 135;
                  {} {RHS}
                  break;
               }

               case 136:   // relational_expr: relational_expr LT shift_expr
               {
#define relational_expr_relational_expr_lt_shift_expr 136
                  constexpr int YYN = 136;
                  {} {RHS}
                  break;
               }

               case 137:   // relational_expr: relational_expr GT shift_expr
               {
#define relational_expr_relational_expr_gt_shift_expr 137
                  constexpr int YYN = 137;
                  {} {RHS}
                  break;
               }

               case 138:   // relational_expr: relational_expr LT_EQ shift_expr
               {
#define relational_expr_relational_expr_lt_eq_shift_expr 138
                  constexpr int YYN = 138;
                  {} {RHS}
                  break;
               }

               case 139:   // relational_expr: relational_expr GT_EQ shift_expr
               {
#define relational_expr_relational_expr_gt_eq_shift_expr 139
                  constexpr int YYN = 139;
                  {} {RHS}
                  break;
               }

               case 140:   // equality_expr: relational_expr
               {
#define equality_expr_relational_expr 140
                  constexpr int YYN = 140;
                  {} {RHS}
                  break;
               }

               case 141:   // equality_expr: equality_expr EQ_EQ relational_expr
               {
#define equality_expr_equality_expr_eq_eq_relational_expr 141
                  constexpr int YYN = 141;
                  {} {RHS}
                  break;
               }

               case 142:   // equality_expr: equality_expr NOT_EQ relational_expr
               {
#define equality_expr_equality_expr_not_eq_relational_expr 142
                  constexpr int YYN = 142;
                  {} {RHS}
                  break;
               }

               case 143:   // and_expr: equality_expr
               {
#define and_expr_equality_expr 143
                  constexpr int YYN = 143;
                  {} {RHS}
                  break;
               }

               case 144:   // and_expr: and_expr AND equality_expr
               {
#define and_expr_and_expr_and_equality_expr 144
                  constexpr int YYN = 144;
                  {} {RHS}
                  break;
               }

               case 145:   // exclusive_or_expr: and_expr
               {
#define exclusive_or_expr_and_expr 145
                  constexpr int YYN = 145;
                  {} {RHS}
                  break;
               }

               case 146:   // exclusive_or_expr: exclusive_or_expr XOR and_expr
               {
#define exclusive_or_expr_exclusive_or_expr_xor_and_expr 146
                  constexpr int YYN = 146;
                  {} {RHS}
                  break;
               }

               case 147:   // inclusive_or_expr: exclusive_or_expr
               {
#define inclusive_or_expr_exclusive_or_expr 147
                  constexpr int YYN = 147;
                  {} {RHS}
                  break;
               }

               case 148:   // inclusive_or_expr: inclusive_or_expr OR exclusive_or_expr
               {
#define inclusive_or_expr_inclusive_or_expr_or_exclusive_or_expr 148
                  constexpr int YYN = 148;
                  {} {RHS}
                  break;
               }

               case 149:   // logical_and_expr: inclusive_or_expr
               {
#define logical_and_expr_inclusive_or_expr 149
                  constexpr int YYN = 149;
                  {} {RHS}
                  break;
               }

               case 150:   // logical_and_expr: logical_and_expr AND_AND inclusive_or_expr
               {
#define logical_and_expr_logical_and_expr_and_and_inclusive_or_expr 150
                  constexpr int YYN = 150;
                  {} {RHS}
                  break;
               }

               case 151:   // logical_or_expr: logical_and_expr
               {
#define logical_or_expr_logical_and_expr 151
                  constexpr int YYN = 151;
                  {} {RHS}
                  break;
               }

               case 152:   // logical_or_expr: logical_or_expr OR_OR logical_and_expr
               {
#define logical_or_expr_logical_or_expr_or_or_logical_and_expr 152
                  constexpr int YYN = 152;
                  {} {RHS}
                  break;
               }

               case 153:   // conditional_expr: logical_or_expr
               {
#define conditional_expr_logical_or_expr 153
                  constexpr int YYN = 153;
                  {} {RHS}
                  break;
               }

               case 154:   // conditional_expr: logical_or_expr QUESTION expr COLON conditional_expr
               {
#define conditional_expr_logical_or_expr_question_expr_colon_conditional_expr 154
                  constexpr int YYN = 154;
                  {} {RHS}
                  break;
               }

               case 155:   // assign_expr: conditional_expr
               {
#define assign_expr_conditional_expr 155
                  constexpr int YYN = 155;
                  {} {RHS}
                  break;
               }

               case 156:   // assign_expr: postfix_expr assign_op assign_expr
               {
#define assign_expr_postfix_expr_assign_op_assign_expr 156
                  constexpr int YYN = 156;
                  {} {RHS}
                  break;
               }

               case 157:   // assign_op: EQ
               {
#define assign_op_eq 157
                  constexpr int YYN = 157;
                  {} {RHS}
                  break;
               }

               case 158:   // assign_op: ADD_EQ
               {
#define assign_op_add_eq 158
                  constexpr int YYN = 158;
                  {} {RHS}
                  break;
               }

               case 159:   // assign_op: SUB_EQ
               {
#define assign_op_sub_eq 159
                  constexpr int YYN = 159;
                  {} {RHS}
                  break;
               }

               case 160:   // assign_op: MUL_EQ
               {
#define assign_op_mul_eq 160
                  constexpr int YYN = 160;
                  {} {RHS}
                  break;
               }

               case 161:   // assign_op: DIV_EQ
               {
#define assign_op_div_eq 161
                  constexpr int YYN = 161;
                  {} {RHS}
                  break;
               }

               case 162:   // assign_op: MOD_EQ
               {
#define assign_op_mod_eq 162
                  constexpr int YYN = 162;
                  {} {RHS}
                  break;
               }

               case 163:   // assign_op: XOR_EQ
               {
#define assign_op_xor_eq 163
                  constexpr int YYN = 163;
                  {} {RHS}
                  break;
               }

               case 164:   // assign_op: AND_EQ
               {
#define assign_op_and_eq 164
                  constexpr int YYN = 164;
                  {} {RHS}
                  break;
               }

               case 165:   // assign_op: OR_EQ
               {
#define assign_op_or_eq 165
                  constexpr int YYN = 165;
                  {} {RHS}
                  break;
               }

               case 166:   // assign_op: LEFT_EQ
               {
#define assign_op_left_eq 166
                  constexpr int YYN = 166;
                  {} {RHS}
                  break;
               }

               case 167:   // assign_op: RIGHT_EQ
               {
#define assign_op_right_eq 167
                  constexpr int YYN = 167;
                  {} {RHS}
                  break;
               }

               case 168:   // expr: assign_expr
               {
#define expr_assign_expr 168
                  constexpr int YYN = 168;
                  {} {RHS}
                  break;
               }

               case 169:   // expr: expr COMA assign_expr
               {
#define expr_expr_coma_assign_expr 169
                  constexpr int YYN = 169;
                  {} {RHS}
                  break;
               }

               case 170:   // args_expr_list: assign_expr
               {
#define args_expr_list_assign_expr 170
                  constexpr int YYN = 170;
                  {} {RHS}
                  break;
               }

               case 171:   // args_expr_list: args_expr_list COMA assign_expr
               {
#define args_expr_list_args_expr_list_coma_assign_expr 171
                  constexpr int YYN = 171;
                  {} {RHS}
                  break;
               }

               case 172:   // args_expr_list: args_expr_list COMA NL assign_expr
               {
#define args_expr_list_args_expr_list_coma_nl_assign_expr 172
                  constexpr int YYN = 172;
                  {} {RHS}
                  break;
               }

               case 173:   // coord: LP constant COMA constant RP
               {
#define coord_lp_constant_coma_constant_rp 173
                  constexpr int YYN = 173;
                  {} {RHS}
                  break;
               }

               case 174:   // coords: coord
               {
#define coords_coord 174
                  constexpr int YYN = 174;
                  {} {RHS}
                  break;
               }

               case 175:   // coords: coords COLON coord
               {
#define coords_coords_colon_coord 175
                  constexpr int YYN = 175;
                  {} {RHS}
                  break;
               }

               case 176:   // primary_math_expr: IDENTIFIER
               {
#define primary_math_expr_identifier 176
                  constexpr int YYN = 176;
                  {} {RHS}
                  break;
               }

               case 177:   // primary_math_expr: constant
               {
#define primary_math_expr_constant 177
                  constexpr int YYN = 177;
                  {} {RHS}
                  break;
               }

               case 178:   // primary_math_expr: domain
               {
#define primary_math_expr_domain 178
                  constexpr int YYN = 178;
                  {} {RHS}
                  break;
               }

               case 179:   // primary_math_expr: QUOTE
               {
#define primary_math_expr_quote 179
                  constexpr int YYN = 179;
                  {} {RHS}
                  break;
               }

               case 180:   // primary_math_expr: LP math_expr RP
               {
#define primary_math_expr_lp_math_expr_rp 180
                  constexpr int YYN = 180;
                  {} {RHS}
                  break;
               }

               case 181:   // primary_math_expr: LB id_list RB
               {
#define primary_math_expr_lb_id_list_rb 181
                  constexpr int YYN = 181;
                  {} {RHS}
                  break;
               }

               case 182:   // dot_math_expr: INNER_OP LP additive_expr COMA additive_expr RP
               {
#define dot_math_expr_inner_op_lp_additive_expr_coma_additive_expr_rp 182
                  constexpr int YYN = 182;
                  {} {RHS}
                  break;
               }

               case 183:   // postfix_math_expr: primary_math_expr
               {
#define postfix_math_expr_primary_math_expr 183
                  constexpr int YYN = 183;
                  {} {RHS}
                  break;
               }

               case 184:   // postfix_math_expr: postfix_math_expr LB math_expr RB
               {
#define postfix_math_expr_postfix_math_expr_lb_math_expr_rb 184
                  constexpr int YYN = 184;
                  {} {RHS}
                  break;
               }

               case 185:   // postfix_math_expr: postfix_math_expr LP argument_math_expr_list RP
               {
#define postfix_math_expr_postfix_math_expr_lp_argument_math_expr_list_rp 185
                  constexpr int YYN = 185;
                  {} {RHS}
                  break;
               }

               case 186:   // postfix_math_expr: postfix_math_expr DOT IDENTIFIER
               {
#define postfix_math_expr_postfix_math_expr_dot_identifier 186
                  constexpr int YYN = 186;
                  {} {RHS}
                  break;
               }

               case 187:   // postfix_math_expr: postfix_math_expr INC_OP
               {
#define postfix_math_expr_postfix_math_expr_inc_op 187
                  constexpr int YYN = 187;
                  {} {RHS}
                  break;
               }

               case 188:   // postfix_math_expr: postfix_math_expr DEC_OP
               {
#define postfix_math_expr_postfix_math_expr_dec_op 188
                  constexpr int YYN = 188;
                  {} {RHS}
                  break;
               }

               case 189:   // postfix_math_expr: postfix_math_expr POW constant
               {
#define postfix_math_expr_postfix_math_expr_pow_constant 189
                  constexpr int YYN = 189;
                  {} {RHS}
                  break;
               }

               case 190:   // postfix_math_expr: dot_math_expr
               {
#define postfix_math_expr_dot_math_expr 190
                  constexpr int YYN = 190;
                  {} {RHS}
                  break;
               }

               case 191:   // postfix_math_expr: GRAD_OP LP additive_math_expr RP
               {
#define postfix_math_expr_grad_op_lp_additive_math_expr_rp 191
                  constexpr int YYN = 191;
                  {} {RHS}
                  break;
               }

               case 192:   // argument_math_expr_list: assign_math_expr
               {
#define argument_math_expr_list_assign_math_expr 192
                  constexpr int YYN = 192;
                  {} {RHS}
                  break;
               }

               case 193:   // argument_math_expr_list: argument_math_expr_list COMA assign_math_expr
               {
#define argument_math_expr_list_argument_math_expr_list_coma_assign_math_expr 193
                  constexpr int YYN = 193;
                  {} {RHS}
                  break;
               }

               case 194:   // unary_math_expr: postfix_math_expr
               {
#define unary_math_expr_postfix_math_expr 194
                  constexpr int YYN = 194;
                  {} {RHS}
                  break;
               }

               case 195:   // unary_math_expr: INC_OP unary_math_expr
               {
#define unary_math_expr_inc_op_unary_math_expr 195
                  constexpr int YYN = 195;
                  {} {RHS}
                  break;
               }

               case 196:   // unary_math_expr: DEC_OP unary_math_expr
               {
#define unary_math_expr_dec_op_unary_math_expr 196
                  constexpr int YYN = 196;
                  {} {RHS}
                  break;
               }

               case 197:   // unary_math_expr: MOD unary_math_expr
               {
#define unary_math_expr_mod_unary_math_expr 197
                  constexpr int YYN = 197;
                  {} {RHS}
                  break;
               }

               case 198:   // unary_math_expr: unary_math_op unary_math_expr
               {
#define unary_math_expr_unary_math_op_unary_math_expr 198
                  constexpr int YYN = 198;
                  {} {RHS}
                  break;
               }

               case 199:   // unary_math_op: MUL
               {
#define unary_math_op_mul 199
                  constexpr int YYN = 199;
                  {} {RHS}
                  break;
               }

               case 200:   // unary_math_op: ADD
               {
#define unary_math_op_add 200
                  constexpr int YYN = 200;
                  {} {RHS}
                  break;
               }

               case 201:   // unary_math_op: SUB
               {
#define unary_math_op_sub 201
                  constexpr int YYN = 201;
                  {} {RHS}
                  break;
               }

               case 202:   // unary_math_op: AND
               {
#define unary_math_op_and 202
                  constexpr int YYN = 202;
                  {} {RHS}
                  break;
               }

               case 203:   // unary_math_op: TILDE
               {
#define unary_math_op_tilde 203
                  constexpr int YYN = 203;
                  {} {RHS}
                  break;
               }

               case 204:   // unary_math_op: NOT
               {
#define unary_math_op_not 204
                  constexpr int YYN = 204;
                  {} {RHS}
                  break;
               }

               case 205:   // multiplicative_math_expr: unary_math_expr
               {
#define multiplicative_math_expr_unary_math_expr 205
                  constexpr int YYN = 205;
                  {} {RHS}
                  break;
               }

               case 206:   // multiplicative_math_expr: multiplicative_math_expr MUL unary_math_expr
               {
#define multiplicative_math_expr_multiplicative_math_expr_mul_unary_math_expr 206
                  constexpr int YYN = 206;
                  {} {RHS}
                  break;
               }

               case 207:   // multiplicative_math_expr: multiplicative_math_expr DIV unary_math_expr
               {
#define multiplicative_math_expr_multiplicative_math_expr_div_unary_math_expr 207
                  constexpr int YYN = 207;
                  {} {RHS}
                  break;
               }

               case 208:   // multiplicative_math_expr: multiplicative_math_expr MOD unary_math_expr
               {
#define multiplicative_math_expr_multiplicative_math_expr_mod_unary_math_expr 208
                  constexpr int YYN = 208;
                  {} {RHS}
                  break;
               }

               case 209:   // additive_math_expr: multiplicative_math_expr
               {
#define additive_math_expr_multiplicative_math_expr 209
                  constexpr int YYN = 209;
                  {} {RHS}
                  break;
               }

               case 210:   // additive_math_expr: additive_math_expr ADD multiplicative_math_expr
               {
#define additive_math_expr_additive_math_expr_add_multiplicative_math_expr 210
                  constexpr int YYN = 210;
                  {} {RHS}
                  break;
               }

               case 211:   // additive_math_expr: additive_math_expr SUB multiplicative_math_expr
               {
#define additive_math_expr_additive_math_expr_sub_multiplicative_math_expr 211
                  constexpr int YYN = 211;
                  {} {RHS}
                  break;
               }

               case 212:   // shift_math_expr: additive_math_expr
               {
#define shift_math_expr_additive_math_expr 212
                  constexpr int YYN = 212;
                  {} {RHS}
                  break;
               }

               case 213:   // shift_math_expr: shift_math_expr LEFT_SHIFT additive_math_expr
               {
#define shift_math_expr_shift_math_expr_left_shift_additive_math_expr 213
                  constexpr int YYN = 213;
                  {} {RHS}
                  break;
               }

               case 214:   // shift_math_expr: shift_math_expr RIGHT_SHIFT additive_math_expr
               {
#define shift_math_expr_shift_math_expr_right_shift_additive_math_expr 214
                  constexpr int YYN = 214;
                  {} {RHS}
                  break;
               }

               case 215:   // relational_math_expr: shift_math_expr
               {
#define relational_math_expr_shift_math_expr 215
                  constexpr int YYN = 215;
                  {} {RHS}
                  break;
               }

               case 216:   // relational_math_expr: relational_math_expr LT shift_math_expr
               {
#define relational_math_expr_relational_math_expr_lt_shift_math_expr 216
                  constexpr int YYN = 216;
                  {} {RHS}
                  break;
               }

               case 217:   // relational_math_expr: relational_math_expr GT shift_math_expr
               {
#define relational_math_expr_relational_math_expr_gt_shift_math_expr 217
                  constexpr int YYN = 217;
                  {} {RHS}
                  break;
               }

               case 218:   // relational_math_expr: relational_math_expr LT_EQ shift_math_expr
               {
#define relational_math_expr_relational_math_expr_lt_eq_shift_math_expr 218
                  constexpr int YYN = 218;
                  {} {RHS}
                  break;
               }

               case 219:   // relational_math_expr: relational_math_expr GT_EQ shift_math_expr
               {
#define relational_math_expr_relational_math_expr_gt_eq_shift_math_expr 219
                  constexpr int YYN = 219;
                  {} {RHS}
                  break;
               }

               case 220:   // equality_math_expr: relational_math_expr
               {
#define equality_math_expr_relational_math_expr 220
                  constexpr int YYN = 220;
                  {} {RHS}
                  break;
               }

               case 221:   // equality_math_expr: equality_math_expr EQ_EQ relational_math_expr
               {
#define equality_math_expr_equality_math_expr_eq_eq_relational_math_expr 221
                  constexpr int YYN = 221;
                  {} {RHS}
                  break;
               }

               case 222:   // equality_math_expr: equality_math_expr NOT_EQ relational_math_expr
               {
#define equality_math_expr_equality_math_expr_not_eq_relational_math_expr 222
                  constexpr int YYN = 222;
                  {} {RHS}
                  break;
               }

               case 223:   // and_math_expr: equality_math_expr
               {
#define and_math_expr_equality_math_expr 223
                  constexpr int YYN = 223;
                  {} {RHS}
                  break;
               }

               case 224:   // and_math_expr: and_math_expr AND equality_math_expr
               {
#define and_math_expr_and_math_expr_and_equality_math_expr 224
                  constexpr int YYN = 224;
                  {} {RHS}
                  break;
               }

               case 225:   // exclusive_or_math_expr: and_math_expr
               {
#define exclusive_or_math_expr_and_math_expr 225
                  constexpr int YYN = 225;
                  {} {RHS}
                  break;
               }

               case 226:   // exclusive_or_math_expr: exclusive_or_math_expr XOR and_math_expr
               {
#define exclusive_or_math_expr_exclusive_or_math_expr_xor_and_math_expr 226
                  constexpr int YYN = 226;
                  {} {RHS}
                  break;
               }

               case 227:   // inclusive_or_math_expr: exclusive_or_math_expr
               {
#define inclusive_or_math_expr_exclusive_or_math_expr 227
                  constexpr int YYN = 227;
                  {} {RHS}
                  break;
               }

               case 228:   // inclusive_or_math_expr: inclusive_or_math_expr OR exclusive_or_math_expr
               {
#define inclusive_or_math_expr_inclusive_or_math_expr_or_exclusive_or_math_expr 228
                  constexpr int YYN = 228;
                  {} {RHS}
                  break;
               }

               case 229:   // logical_and_math_expr: inclusive_or_math_expr
               {
#define logical_and_math_expr_inclusive_or_math_expr 229
                  constexpr int YYN = 229;
                  {} {RHS}
                  break;
               }

               case 230:   // logical_and_math_expr: logical_and_math_expr AND_AND inclusive_or_math_expr
               {
#define logical_and_math_expr_logical_and_math_expr_and_and_inclusive_or_math_expr 230
                  constexpr int YYN = 230;
                  {} {RHS}
                  break;
               }

               case 231:   // logical_or_math_expr: logical_and_math_expr
               {
#define logical_or_math_expr_logical_and_math_expr 231
                  constexpr int YYN = 231;
                  {} {RHS}
                  break;
               }

               case 232:   // logical_or_math_expr: logical_or_math_expr OR_OR logical_and_math_expr
               {
#define logical_or_math_expr_logical_or_math_expr_or_or_logical_and_math_expr 232
                  constexpr int YYN = 232;
                  {} {RHS}
                  break;
               }

               case 233:   // conditional_math_expr: logical_or_math_expr
               {
#define conditional_math_expr_logical_or_math_expr 233
                  constexpr int YYN = 233;
                  {} {RHS}
                  break;
               }

               case 234:   // assign_math_expr: conditional_math_expr
               {
#define assign_math_expr_conditional_math_expr 234
                  constexpr int YYN = 234;
                  {} {RHS}
                  break;
               }

               case 235:   // assign_math_expr: unary_math_expr assign_math_op assign_math_expr
               {
#define assign_math_expr_unary_math_expr_assign_math_op_assign_math_expr 235
                  constexpr int YYN = 235;
                  {} {RHS}
                  break;
               }

               case 236:   // assign_math_op: EQ
               {
#define assign_math_op_eq 236
                  constexpr int YYN = 236;
                  {} {RHS}
                  break;
               }

               case 237:   // assign_math_op: ADD_EQ
               {
#define assign_math_op_add_eq 237
                  constexpr int YYN = 237;
                  {} {RHS}
                  break;
               }

               case 238:   // assign_math_op: SUB_EQ
               {
#define assign_math_op_sub_eq 238
                  constexpr int YYN = 238;
                  {} {RHS}
                  break;
               }

               case 239:   // assign_math_op: MUL_EQ
               {
#define assign_math_op_mul_eq 239
                  constexpr int YYN = 239;
                  {} {RHS}
                  break;
               }

               case 240:   // assign_math_op: DIV_EQ
               {
#define assign_math_op_div_eq 240
                  constexpr int YYN = 240;
                  {} {RHS}
                  break;
               }

               case 241:   // assign_math_op: MOD_EQ
               {
#define assign_math_op_mod_eq 241
                  constexpr int YYN = 241;
                  {} {RHS}
                  break;
               }

               case 242:   // assign_math_op: XOR_EQ
               {
#define assign_math_op_xor_eq 242
                  constexpr int YYN = 242;
                  {} {RHS}
                  break;
               }

               case 243:   // assign_math_op: AND_EQ
               {
#define assign_math_op_and_eq 243
                  constexpr int YYN = 243;
                  {} {RHS}
                  break;
               }

               case 244:   // assign_math_op: OR_EQ
               {
#define assign_math_op_or_eq 244
                  constexpr int YYN = 244;
                  {} {RHS}
                  break;
               }

               case 245:   // assign_math_op: LEFT_EQ
               {
#define assign_math_op_left_eq 245
                  constexpr int YYN = 245;
                  {} {RHS}
                  break;
               }

               case 246:   // assign_math_op: RIGHT_EQ
               {
#define assign_math_op_right_eq 246
                  constexpr int YYN = 246;
                  {} {RHS}
                  break;
               }

               case 247:   // math_expr: assign_math_expr
               {
#define math_expr_assign_math_expr 247
                  constexpr int YYN = 247;
                  {} {RHS}
                  break;
               }

               case 248:   // math_expr: math_expr COMA assign_math_expr
               {
#define math_expr_math_expr_coma_assign_math_expr 248
                  constexpr int YYN = 248;
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
      "FROM", "IMPORT", "RETURN", "PLOT", "BENCHMARK", "SAVE", "SOLVE",
      "PROJECT", "STRING", "QUOTE", "IF", "FOR", "IN", "RANGE", "DOT_OP",
      "INNER_OP", "GRAD_OP", "TRANSPOSE_OP", "LHS", "RHS", "DEVICE", "MESH",
      "FINITE_ELEMENT", "UNIT_SQUARE_MESH", "UNIT_HEX_MESH", "FUNCTION",
      "FUNCTION_SPACE", "VECTOR_FUNCTION_SPACE", "EXPRESSION", "DIRICHLET_BC",
      "TRIAL_FUNCTION", "TEST_FUNCTION", "CONSTANT_API", "POINT", "SEGMENT",
      "TRIANGLE", "QUADRILATERAL", "TETRAHEDRON", "HEXAHEDRON", "WEDGE",
      "OR_OR", "AND_AND", "DOM_DX", "EXT_DS", "INT_DS", "EQ_EQ", "ADD_EQ",
      "SUB_EQ", "MUL_EQ", "DIV_EQ", "MOD_EQ", "XOR_EQ", "AND_EQ", "OR_EQ",
      "LEFT_EQ", "RIGHT_EQ", "NATURAL", "REAL", "BOOL", "IDENTIFIER", "GT",
      "LT", "EQ", "ADD", "SUB", "MUL", "DIV", "POW", "LS", "RS", "LP", "RP",
      "LB", "RB", "COMA", "APOSTROPHE", "COLON", "DOT", "MOD", "TILDE",
      "LEFT_SHIFT", "RIGHT_SHIFT", "LT_EQ", "GT_EQ", "NOT_EQ", "AND", "XOR",
      "OR", "QUESTION", "NOT", "INC_OP", "DEC_OP", "TRANSPOSE_XT", "DOT_XT",
      "EVAL_XT", "GRAD_XT", "VAR_XT", "DOM_XT", "EXPR_QUOTE", "EMPTY",
      "$accept", "entry_point", "statements", "statement", "decl",
      "primary_id", "postfix_id", "postfix_ids", "id_list",
      "extra_status_rule", "lhs", "dot_xt", "eval_xt", "transpose_xt",
      "var_xt", "dom_xt", "expr_quote", "function", "def_empty",
      "def_statements", "def_statement", "iteration_statement", "if_statement",
      "api_statement", "direct_declarator", "domain", "constant", "strings",
      "id_n", "fes_args", "element_type", "api", "primary_expr", "form_args",
      "grad_expr", "transpose_expr", "pow_expr", "postfix_expr", "unary_expr",
      "unary_op", "cast_expr", "multiplicative_expr", "dot_expr",
      "additive_expr", "shift_expr", "relational_expr", "equality_expr",
      "and_expr", "exclusive_or_expr", "inclusive_or_expr", "logical_and_expr",
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


const short parser::yypact_ninf_ = -312;

const signed char parser::yytable_ninf_ = -57;

const short
parser::yypact_[] =
{
   249,  -312,     9,   -37,    14,    46,    58,    69,    73,    23,
   -312,  -312,  -312,  -312,  -312,    42,    42,  -312,  -312,  -312,
   -312,  -312,  -312,    95,   437,  -312,   135,  -312,    17,    42,
   276,  -312,  -312,  -312,  -312,  -312,  -312,  -312,  -312,  -312,
   -312,  -312,  -312,  -312,   333,    75,   805,   805,   805,   805,
   805,   805,   140,     8,   -26,   212,  -312,  -312,  -312,   651,
   42,     8,  -312,  -312,  -312,  -312,  -312,  -312,  -312,  -312,
   -312,  -312,  -312,    42,   805,   805,   805,  -312,  -312,    77,
   84,    84,   120,   127,  -312,  -312,  -312,  -312,  -312,  -312,
   163,  -312,  -312,  -312,  -312,  -312,  -312,  -312,  -312,  -312,
   -312,  -312,  -312,  -312,  -312,  -312,  -312,  -312,  -312,  -312,
   171,   805,   805,  -312,  -312,  -312,  -312,  -312,   126,  -312,
   -312,  -312,  -312,  -312,  -312,   172,  -312,   805,  -312,    64,
   159,   142,   234,    13,    -5,   125,   129,   156,   206,   -21,
   -312,  -312,    12,    36,    49,    97,   103,   128,   238,   333,
   333,  -312,   143,  -312,    42,   187,   187,  -312,   164,   805,
   805,  -312,  -312,   204,   222,   231,   254,  -312,    29,   194,
   232,  -312,   214,   728,   805,   843,   805,   121,  -312,   805,
   805,   805,   805,   805,   805,   805,   805,   805,   805,   805,
   805,   805,   805,   805,   805,   805,   805,   805,   805,  -312,
   805,  -312,  -312,  -312,  -312,   805,   218,   805,   805,  -312,
   237,   497,   -20,    86,   223,   247,   246,   266,   302,  -312,
   171,  -312,  -312,  -312,  -312,  -312,   209,   243,  -312,  -312,
   -312,  -312,  -312,    64,   159,   159,   142,   142,   234,   234,
   234,   234,    13,    13,    -5,   125,   129,   156,   206,   -27,
   -312,   187,   285,   187,   187,    38,   805,  -312,   805,  -312,
   -312,  -312,    16,  -312,   254,  -312,  -312,  -312,   805,   322,
   -312,    42,   276,   395,    32,  -312,  -312,   112,   328,   329,
   334,  -312,   330,   225,   805,   122,   122,  -312,  -312,   805,
   805,  -312,   574,   333,    18,  -312,   337,   339,  -312,  -312,
   -312,  -312,   122,    42,   122,  -312,  -312,  -312,   122,   122,
   -312,  -312,  -312,  -312,   -14,   367,   122,   104,   270,   255,
   80,    -4,   319,   324,   336,   370,   384,  -312,  -312,   351,
   351,  -312,  -312,  -312,   805,   187,   805,  -312,   805,   122,
   229,   275,  -312,  -312,  -312,   254,   122,   122,   368,  -312,
   -312,  -312,  -312,  -312,  -312,  -312,  -312,  -312,  -312,  -312,
   -312,  -312,   122,  -312,   122,   122,   122,   122,   122,   122,
   122,   122,   122,   122,   122,   122,   122,   122,   122,   122,
   122,   122,   122,   187,    19,    62,  -312,   132,  -312,  -312,
   -312,   233,  -312,   278,  -312,  -312,  -312,  -312,  -312,   104,
   104,   270,   270,   255,   255,   255,   255,    80,    80,    -4,
   319,   324,   336,   370,  -312,  -312,   805,  -312,  -312,   122,
   -312,   145,  -312,  -312
};

const unsigned char
parser::yydefact_[] =
{
   0,     6,     0,     0,     0,     0,     0,     0,     0,     0,
   33,    57,    58,    59,    17,     0,     0,    36,    34,    35,
   37,    38,    39,     0,     2,     3,     0,    18,    22,    24,
   0,     5,    26,    29,    30,    28,    31,    32,    27,     8,
   15,    13,    14,    16,     0,     0,     0,     0,     0,     0,
   0,     0,     0,    22,     0,     0,     1,     4,     7,     0,
   0,    23,   158,   159,   160,   161,   162,   163,   164,   165,
   166,   167,   157,     0,     0,     0,     0,    63,    93,     0,
   0,     0,     0,     0,    75,    76,    80,    79,    78,    77,
   0,    82,    83,    84,    85,    86,    87,    69,    70,    71,
   72,    73,    74,    60,    61,    62,    90,   118,   119,   117,
   0,     0,     0,   120,   121,    88,    92,    91,    94,    89,
   114,   103,   110,   111,   104,   115,   122,     0,   123,   127,
   129,   132,   135,   140,   143,   145,   147,   149,   151,   153,
   155,   168,     0,     0,     0,     0,     0,     0,     0,     0,
   0,    19,     0,    21,    25,    10,     9,   170,     0,     0,
   0,    99,   100,     0,     0,     0,     0,   174,     0,     0,
   0,    64,     0,     0,     0,     0,     0,   115,   116,     0,
   0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   0,     0,     0,     0,     0,     0,     0,     0,     0,    51,
   0,    55,    52,    53,    54,     0,     0,     0,     0,    20,
   0,     0,     0,     0,     0,     0,     0,     0,     0,    97,
   0,    95,    96,   102,   101,   107,     0,     0,   109,   156,
   124,   125,   126,   128,   130,   131,   133,   134,   137,   136,
   138,   139,   141,   142,   144,   146,   148,   150,   152,     0,
   169,    50,     0,    11,    12,    42,     0,   171,     0,    98,
   112,   113,     0,    81,     0,   175,   108,   106,     0,     0,
   45,     0,     0,     0,     0,    43,   172,     0,     0,    66,
   0,   154,     0,     0,     0,     0,     0,    44,   105,     0,
   0,   173,     0,     0,     0,   179,     0,     0,   176,   200,
   201,   199,     0,     0,     0,   203,   202,   204,     0,     0,
   178,   177,   183,   190,   194,   205,     0,   209,   212,   215,
   220,   223,   225,   227,   229,   231,   233,   234,   247,    40,
   41,    68,    65,    67,     0,    49,     0,    46,     0,     0,
   0,     0,   197,   195,   196,     0,     0,     0,     0,   187,
   188,   237,   238,   239,   240,   241,   242,   243,   244,   245,
   246,   236,     0,   198,     0,     0,     0,     0,     0,     0,
   0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   0,     0,     0,    48,     0,     0,   205,     0,   180,   181,
   189,     0,   192,     0,   186,   235,   206,   207,   208,   210,
   211,   213,   214,   217,   216,   218,   219,   221,   222,   224,
   226,   228,   230,   232,   248,    47,     0,   191,   185,     0,
   184,     0,   193,   182
};

const short
parser::yypgoto_[] =
{
   -312,  -312,  -312,   409,  -312,   375,    20,   364,     2,  -312,
   -312,  -312,  -312,  -312,  -312,  -312,  -312,  -312,  -312,  -312,
   165,  -312,  -312,    33,  -312,     0,    99,  -312,   150,  -312,
   -312,  -312,   267,   371,  -312,  -312,  -312,   -45,  -312,  -312,
   -102,   271,   177,  -147,    48,   205,   263,   264,   262,   286,
   265,  -312,  -171,   -60,   -29,   -40,   287,   239,  -312,  -312,
   -312,  -312,  -312,  -240,  -312,    31,  -311,   -90,    30,   106,
   107,   105,   109,   110,  -312,  -312,  -309,  -312,  -259
};

const short
parser::yydefgoto_[] =
{
   0,    23,    24,    25,    26,    27,    53,    29,    30,    31,
   32,    33,    34,    35,    36,    37,    38,    39,   273,   274,
   275,    40,    41,   115,    43,   116,   117,   118,   331,   217,
   119,   120,   121,   161,   122,   123,   124,   177,   126,   127,
   128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
   138,   139,   140,   141,    74,   142,   158,   167,   168,   312,
   313,   314,   391,   386,   316,   317,   318,   319,   320,   321,
   322,   323,   324,   325,   326,   327,   328,   362,   329
};

const short
parser::yytable_[] =
{
   44,   125,   125,   125,   125,   125,   125,   143,   144,   145,
   146,   147,   212,   213,   125,    75,   157,    54,    55,   152,
   28,   -56,   337,   415,    44,   178,   197,   330,   387,   125,
   125,   125,   278,    42,   155,   156,   270,   392,   236,   237,
   46,   286,   270,   340,    28,   315,   315,   191,   375,    61,
   183,   184,   149,   395,   200,    73,   268,    42,   401,   402,
   345,   258,   315,   346,   342,   347,   125,   125,   343,   344,
   348,   169,   170,   414,   198,    45,   363,   230,   231,   232,
   187,   188,   279,   349,   350,    59,   192,   376,   393,    52,
   199,    47,    60,   200,    59,    56,   176,   281,    14,   200,
   200,    60,   189,   190,    14,   219,   315,   315,    14,   271,
   422,   277,   220,   157,   201,   271,   229,   200,   332,   332,
   207,   208,   315,    48,   396,   397,   398,   202,   125,   125,
   200,   125,   183,   184,   227,    49,   179,   180,   295,    58,
   250,   171,   315,   416,   296,   297,    50,   371,   372,   181,
   51,   257,    76,   125,   159,   125,   183,   184,   249,   148,
   125,   160,   125,   125,   259,   251,   125,   253,   254,   373,
   374,    11,    12,    13,    61,   203,   364,   365,   200,   315,
   182,   204,   183,   184,   200,   103,   104,   105,   298,   366,
   288,   385,   299,   300,   301,   172,   276,   163,   173,   302,
   174,   303,   367,   368,   164,   175,   205,   304,   305,   200,
   417,   125,   183,   184,   306,   183,   184,   193,   307,   308,
   309,   209,   194,   423,   200,    62,    63,    64,    65,    66,
   67,    68,    69,    70,    71,   238,   239,   240,   241,   125,
   165,    72,   210,   284,   294,   211,   172,   125,   166,   173,
   195,   174,   335,     1,   196,     2,   175,   272,   206,     3,
   4,     5,     6,     7,   336,   218,     8,     9,   200,   421,
   214,   224,   221,   283,    10,   200,   272,   103,   104,   105,
   223,   403,   404,   405,   406,   310,   310,   266,   215,   125,
   211,   125,   150,    73,   383,   252,   384,   216,    11,    12,
   13,   260,   310,   293,   310,   341,    73,   388,   310,   310,
   382,   418,   222,   200,   419,    14,   310,   103,   104,   105,
   255,   185,   186,   267,   200,   261,    15,   262,    16,    62,
   63,    64,    65,    66,    67,    68,    69,    70,    71,   310,
   367,   368,   369,   370,   263,    72,   310,   310,    17,    18,
   19,   269,    20,    21,    22,   389,    73,    73,   420,   382,
   234,   235,   310,   280,   310,   310,   310,   310,   310,   310,
   310,   310,   310,   310,   310,   310,   310,   310,   310,   310,
   310,   310,   310,   264,   311,   311,    62,    63,    64,    65,
   66,    67,    68,    69,    70,    71,   242,   243,   399,   400,
   282,   311,    72,   311,   285,   407,   408,   311,   311,   289,
   290,   377,   291,   292,   338,   311,   339,   378,   380,   310,
   351,   352,   353,   354,   355,   356,   357,   358,   359,   360,
   379,   381,   382,    57,   394,   153,   361,   154,   311,   287,
   333,     1,   228,     2,   390,   311,   311,     3,     4,     5,
   6,     7,   162,   233,     8,     9,   244,   246,   245,   265,
   226,   311,   248,   311,   311,   311,   311,   311,   311,   311,
   311,   311,   311,   311,   311,   311,   311,   311,   311,   311,
   311,   311,   247,   409,   411,   410,    11,    12,    13,   412,
   0,   413,     0,     0,     0,     0,     0,     0,     0,     0,
   0,   256,     0,    14,     0,     0,     0,     3,     4,     5,
   6,     7,    77,    78,    15,     0,    16,     0,   311,    79,
   80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
   90,    91,    92,    93,    94,    95,    96,    97,     0,    98,
   99,   100,   101,   102,     0,     0,    11,    12,    13,     0,
   0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   103,   104,   105,   106,     0,     0,     0,   107,   108,   109,
   0,     0,   110,     0,   111,     0,   112,     0,   334,     0,
   0,     0,     0,   113,     3,     4,     5,     6,     7,    77,
   78,     0,     0,   114,     0,     0,    79,    80,    81,    82,
   83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
   93,    94,    95,    96,    97,     0,    98,    99,   100,   101,
   102,     0,     0,    11,    12,    13,     0,     0,     0,     0,
   0,     0,     0,     0,     0,     0,     0,   103,   104,   105,
   106,     0,     0,     0,   107,   108,   109,     0,     0,   110,
   0,   111,     0,   112,     0,     0,     0,     0,     0,     0,
   113,     3,     4,     5,     6,     7,    77,    78,     0,     0,
   114,     0,     0,    79,    80,    81,    82,    83,    84,    85,
   86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
   96,    97,     0,    98,    99,   100,   101,   102,     0,     0,
   11,    12,    13,     0,     0,     0,     0,     0,     0,     0,
   0,     0,     0,     0,   103,   104,   105,   106,     0,     0,
   0,   107,   108,   109,     0,     0,   110,     0,   111,   151,
   112,     0,     0,     0,     0,     0,     0,   113,     3,     4,
   5,     6,     7,    77,    78,     0,     0,   114,     0,     0,
   79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
   89,    90,    91,    92,    93,    94,    95,    96,    97,     0,
   98,    99,   100,   101,   102,     0,     0,    11,    12,    13,
   0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   0,   103,   104,   105,   106,     0,     0,     0,   107,   108,
   109,     0,     0,   110,     0,   111,   225,   112,     0,     0,
   0,     0,     0,     0,   113,     3,     4,     5,     6,     7,
   77,    78,     0,     0,   114,     0,     0,    79,    80,    81,
   82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
   92,    93,    94,    95,    96,    97,     0,    98,    99,   100,
   101,   102,     0,     0,    11,    12,    13,     0,    77,    78,
   0,     0,     0,     0,     0,     0,     0,     0,   103,   104,
   105,   106,     0,     0,     0,   107,   108,   109,     0,     0,
   110,     0,   111,     0,   112,     0,     0,     0,     0,     0,
   0,   113,    11,    12,    13,     0,     0,     0,     0,     0,
   0,   114,     0,     0,     0,     0,   103,   104,   105,   106,
   0,     0,     0,     0,     0,     0,     0,     0,   110,     0,
   111,     0,   112
};

const short
parser::yycheck_[] =
{
   0,    46,    47,    48,    49,    50,    51,    47,    48,    49,
   50,    51,   159,   160,    59,    44,    76,    15,    16,    59,
   0,     4,     4,     4,    24,   127,    47,   286,   339,    74,
   75,    76,    16,     0,    74,    75,     4,   346,   185,   186,
   77,     9,     4,   302,    24,   285,   286,    52,    52,    29,
   70,    71,    78,   362,    81,    81,    83,    24,   369,   370,
   74,    81,   302,    77,   304,    79,   111,   112,   308,   309,
   84,   111,   112,   382,    95,    66,   316,   179,   180,   181,
   67,    68,    66,    97,    98,    77,    91,    91,   347,    66,
   78,    77,    84,    81,    77,     0,   125,   268,    66,    81,
   81,    84,    89,    90,    66,    76,   346,   347,    66,    77,
   419,   258,    83,   173,    78,    77,   176,    81,   289,   290,
   149,   150,   362,    77,   364,   365,   366,    78,   173,   174,
   81,   176,    70,    71,   174,    77,    72,    73,    16,     4,
   200,    15,   382,    81,    22,    23,    77,    67,    68,    85,
   77,   211,    77,   198,    77,   200,    70,    71,   198,    19,
   205,    77,   207,   208,    78,   205,   211,   207,   208,    89,
   90,    49,    50,    51,   154,    78,    72,    73,    81,   419,
   21,    78,    70,    71,    81,    63,    64,    65,    66,    85,
   78,   338,    70,    71,    72,    74,   256,    77,    77,    77,
   79,    79,    70,    71,    77,    84,    78,    85,    86,    81,
   78,   256,    70,    71,    92,    70,    71,    92,    96,    97,
   98,    78,    93,    78,    81,    53,    54,    55,    56,    57,
   58,    59,    60,    61,    62,   187,   188,   189,   190,   284,
   77,    69,    78,   272,   284,    81,    74,   292,    77,    77,
   94,    79,   292,     4,    48,     6,    84,   255,    20,    10,
   11,    12,    13,    14,   293,   166,    17,    18,    81,   416,
   66,   172,    78,   271,    25,    81,   274,    63,    64,    65,
   66,   371,   372,   373,   374,   285,   286,    78,    66,   334,
   81,   336,    80,    81,   334,    77,   336,    66,    49,    50,
   51,    78,   302,    78,   304,   303,    81,    78,   308,   309,
   81,    78,    80,    81,    81,    66,   316,    63,    64,    65,
   83,    87,    88,    80,    81,    78,    77,    81,    79,    53,
   54,    55,    56,    57,    58,    59,    60,    61,    62,   339,
   70,    71,    87,    88,    78,    69,   346,   347,    99,   100,
   101,    66,   103,   104,   105,    80,    81,    81,    80,    81,
   183,   184,   362,   264,   364,   365,   366,   367,   368,   369,
   370,   371,   372,   373,   374,   375,   376,   377,   378,   379,
   380,   381,   382,    81,   285,   286,    53,    54,    55,    56,
   57,    58,    59,    60,    61,    62,   191,   192,   367,   368,
   78,   302,    69,   304,     9,   375,   376,   308,   309,    81,
   81,    92,    78,    83,    77,   316,    77,    93,    48,   419,
   53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
   94,    47,    81,    24,    66,    60,    69,    73,   339,   274,
   290,     4,   175,     6,   345,   346,   347,    10,    11,    12,
   13,    14,    81,   182,    17,    18,   193,   195,   194,   220,
   173,   362,   197,   364,   365,   366,   367,   368,   369,   370,
   371,   372,   373,   374,   375,   376,   377,   378,   379,   380,
   381,   382,   196,   377,   379,   378,    49,    50,    51,   380,
   -1,   381,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   -1,     4,    -1,    66,    -1,    -1,    -1,    10,    11,    12,
   13,    14,    15,    16,    77,    -1,    79,    -1,   419,    22,
   23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
   33,    34,    35,    36,    37,    38,    39,    40,    -1,    42,
   43,    44,    45,    46,    -1,    -1,    49,    50,    51,    -1,
   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   63,    64,    65,    66,    -1,    -1,    -1,    70,    71,    72,
   -1,    -1,    75,    -1,    77,    -1,    79,    -1,     4,    -1,
   -1,    -1,    -1,    86,    10,    11,    12,    13,    14,    15,
   16,    -1,    -1,    96,    -1,    -1,    22,    23,    24,    25,
   26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
   36,    37,    38,    39,    40,    -1,    42,    43,    44,    45,
   46,    -1,    -1,    49,    50,    51,    -1,    -1,    -1,    -1,
   -1,    -1,    -1,    -1,    -1,    -1,    -1,    63,    64,    65,
   66,    -1,    -1,    -1,    70,    71,    72,    -1,    -1,    75,
   -1,    77,    -1,    79,    -1,    -1,    -1,    -1,    -1,    -1,
   86,    10,    11,    12,    13,    14,    15,    16,    -1,    -1,
   96,    -1,    -1,    22,    23,    24,    25,    26,    27,    28,
   29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
   39,    40,    -1,    42,    43,    44,    45,    46,    -1,    -1,
   49,    50,    51,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   -1,    -1,    -1,    -1,    63,    64,    65,    66,    -1,    -1,
   -1,    70,    71,    72,    -1,    -1,    75,    -1,    77,    78,
   79,    -1,    -1,    -1,    -1,    -1,    -1,    86,    10,    11,
   12,    13,    14,    15,    16,    -1,    -1,    96,    -1,    -1,
   22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
   32,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
   42,    43,    44,    45,    46,    -1,    -1,    49,    50,    51,
   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   -1,    63,    64,    65,    66,    -1,    -1,    -1,    70,    71,
   72,    -1,    -1,    75,    -1,    77,    78,    79,    -1,    -1,
   -1,    -1,    -1,    -1,    86,    10,    11,    12,    13,    14,
   15,    16,    -1,    -1,    96,    -1,    -1,    22,    23,    24,
   25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
   35,    36,    37,    38,    39,    40,    -1,    42,    43,    44,
   45,    46,    -1,    -1,    49,    50,    51,    -1,    15,    16,
   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    63,    64,
   65,    66,    -1,    -1,    -1,    70,    71,    72,    -1,    -1,
   75,    -1,    77,    -1,    79,    -1,    -1,    -1,    -1,    -1,
   -1,    86,    49,    50,    51,    -1,    -1,    -1,    -1,    -1,
   -1,    96,    -1,    -1,    -1,    -1,    63,    64,    65,    66,
   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    75,    -1,
   77,    -1,    79
};

const unsigned char
parser::yystos_[] =
{
   0,     4,     6,    10,    11,    12,    13,    14,    17,    18,
   25,    49,    50,    51,    66,    77,    79,    99,   100,   101,
   103,   104,   105,   108,   109,   110,   111,   112,   113,   114,
   115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
   128,   129,   130,   131,   132,    66,    77,    77,    77,    77,
   77,    77,    66,   113,   115,   115,     0,   110,     4,    77,
   84,   113,    53,    54,    55,    56,    57,    58,    59,    60,
   61,    62,    69,    81,   161,   161,    77,    15,    16,    22,
   23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
   33,    34,    35,    36,    37,    38,    39,    40,    42,    43,
   44,    45,    46,    63,    64,    65,    66,    70,    71,    72,
   75,    77,    79,    86,    96,   130,   132,   133,   134,   137,
   138,   139,   141,   142,   143,   144,   145,   146,   147,   148,
   149,   150,   151,   152,   153,   154,   155,   156,   157,   158,
   159,   160,   162,   162,   162,   162,   162,   162,    19,    78,
   80,    78,   162,   112,   114,   162,   162,   160,   163,    77,
   77,   140,   140,    77,    77,    77,    77,   164,   165,   162,
   162,    15,    74,    77,    79,    84,   161,   144,   147,    72,
   73,    85,    21,    70,    71,    87,    88,    67,    68,    89,
   90,    52,    91,    92,    93,    94,    48,    47,    95,    78,
   81,    78,    78,    78,    78,    78,    20,   161,   161,    78,
   78,    81,   150,   150,    66,    66,    66,   136,   133,    76,
   83,    78,    80,    66,   133,    78,   163,   162,   139,   160,
   147,   147,   147,   148,   149,   149,   150,   150,   151,   151,
   151,   151,   152,   152,   153,   154,   155,   156,   157,   162,
   160,   162,    77,   162,   162,    83,     4,   160,    81,    78,
   78,    78,    81,    78,    81,   164,    78,    80,    83,    66,
   4,    77,   115,   125,   126,   127,   160,   150,    16,    66,
   133,   159,    78,   115,   161,     9,     9,   127,    78,    81,
   81,    78,    83,    78,   162,    16,    22,    23,    66,    70,
   71,    72,    77,    79,    85,    86,    92,    96,    97,    98,
   132,   133,   166,   167,   168,   170,   171,   172,   173,   174,
   175,   176,   177,   178,   179,   180,   181,   182,   183,   185,
   185,   135,   159,   135,     4,   162,   161,     4,    77,    77,
   185,   115,   170,   170,   170,    74,    77,    79,    84,    97,
   98,    53,    54,    55,    56,    57,    58,    59,    60,    61,
   62,    69,   184,   170,    72,    73,    85,    70,    71,    87,
   88,    67,    68,    89,    90,    52,    91,    92,    93,    94,
   48,    47,    81,   162,   162,   150,   170,   173,    78,    80,
   133,   169,   183,   185,    66,   183,   170,   170,   170,   172,
   172,   173,   173,   174,   174,   174,   174,   175,   175,   176,
   177,   178,   179,   180,   183,     4,    81,    78,    78,    81,
   80,   150,   183,    78
};

const unsigned char
parser::yyr1_[] =
{
   0,   107,   108,   109,   109,   109,   110,   110,   111,   111,
   111,   111,   111,   111,   111,   111,   111,   112,   113,   113,
   113,   113,   114,   114,   115,   115,   116,   116,   116,   116,
   116,   116,   116,   117,   118,   119,   120,   121,   122,   123,
   124,   124,   125,   126,   126,   127,   127,   127,   128,   128,
   129,   130,   130,   130,   130,   130,   131,   132,   132,   132,
   133,   133,   133,   134,   134,   135,   136,   136,   136,   137,
   137,   137,   137,   137,   137,   138,   138,   138,   138,   138,
   138,   138,   138,   138,   138,   138,   138,   138,   138,   138,
   139,   139,   139,   139,   139,   139,   139,   139,   140,   141,
   142,   143,   143,   144,   144,   144,   144,   144,   144,   144,
   144,   144,   144,   144,   144,   145,   145,   146,   146,   146,
   146,   146,   147,   148,   148,   148,   148,   149,   149,   150,
   150,   150,   151,   151,   151,   152,   152,   152,   152,   152,
   153,   153,   153,   154,   154,   155,   155,   156,   156,   157,
   157,   158,   158,   159,   159,   160,   160,   161,   161,   161,
   161,   161,   161,   161,   161,   161,   161,   161,   162,   162,
   163,   163,   163,   164,   165,   165,   166,   166,   166,   166,
   166,   166,   167,   168,   168,   168,   168,   168,   168,   168,
   168,   168,   169,   169,   170,   170,   170,   170,   170,   171,
   171,   171,   171,   171,   171,   172,   172,   172,   172,   173,
   173,   173,   174,   174,   174,   175,   175,   175,   175,   175,
   176,   176,   176,   177,   177,   178,   178,   179,   179,   180,
   180,   181,   181,   182,   183,   183,   184,   184,   184,   184,
   184,   184,   184,   184,   184,   184,   184,   185,   185
};

const signed char
parser::yyr2_[] =
{
   0,     2,     1,     1,     2,     1,     1,     2,     1,     3,
   3,     5,     5,     1,     1,     1,     1,     1,     1,     3,
   4,     3,     1,     2,     1,     3,     1,     1,     1,     1,
   1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
   9,     9,     0,     1,     2,     1,     4,     6,    10,     9,
   5,     4,     4,     4,     4,     4,     1,     1,     1,     1,
   1,     1,     1,     1,     2,     1,     3,     5,     5,     1,
   1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
   1,     4,     1,     1,     1,     1,     1,     1,     1,     1,
   1,     1,     1,     1,     1,     3,     3,     3,     3,     2,
   2,     3,     3,     1,     1,     6,     4,     3,     4,     3,
   1,     1,     4,     4,     1,     1,     2,     1,     1,     1,
   1,     1,     1,     1,     3,     3,     3,     1,     3,     1,
   3,     3,     1,     3,     3,     1,     3,     3,     3,     3,
   1,     3,     3,     1,     3,     1,     3,     1,     3,     1,
   3,     1,     3,     1,     5,     1,     3,     1,     1,     1,
   1,     1,     1,     1,     1,     1,     1,     1,     1,     3,
   1,     3,     4,     5,     1,     3,     1,     1,     1,     1,
   3,     3,     6,     1,     4,     4,     3,     2,     2,     3,
   1,     4,     1,     3,     1,     2,     2,     2,     2,     1,
   1,     1,     1,     1,     1,     1,     3,     3,     3,     1,
   3,     3,     1,     3,     3,     1,     3,     3,     3,     3,
   1,     3,     3,     1,     3,     1,     3,     1,     3,     1,
   3,     1,     3,     1,     1,     3,     1,     1,     1,     1,
   1,     1,     1,     1,     1,     1,     1,     1,     3
};




#if YYDEBUG
const short
parser::yyrline_[] =
{
   0,    95,    95,    97,    97,    97,    99,    99,   101,   102,
   103,   104,   105,   106,   107,   108,   109,   111,   113,   114,
   115,   116,   118,   118,   120,   120,   123,   123,   124,   124,
   124,   125,   125,   126,   127,   128,   130,   131,   132,   133,
   137,   138,   139,   140,   140,   141,   142,   143,   147,   148,
   151,   154,   155,   156,   157,   158,   161,   164,   164,   164,
   167,   167,   167,   169,   169,   172,   173,   174,   175,   177,
   177,   177,   178,   178,   178,   181,   182,   183,   184,   185,
   186,   187,   188,   189,   190,   191,   192,   193,   194,   195,
   197,   198,   199,   200,   201,   202,   203,   204,   206,   208,
   210,   212,   212,   214,   215,   216,   217,   218,   219,   220,
   221,   222,   223,   224,   225,   227,   227,   229,   229,   229,
   229,   229,   231,   233,   234,   235,   236,   238,   239,   241,
   242,   243,   245,   246,   247,   249,   250,   251,   252,   253,
   255,   256,   257,   259,   260,   262,   263,   265,   266,   268,
   269,   271,   272,   274,   275,   277,   278,   280,   281,   281,
   281,   281,   281,   282,   282,   282,   283,   283,   285,   286,
   288,   289,   290,   292,   294,   294,   299,   300,   301,   302,
   303,   304,   306,   309,   310,   311,   312,   313,   314,   315,
   316,   317,   320,   321,   324,   325,   326,   327,   328,   330,
   330,   330,   330,   330,   330,   332,   333,   334,   335,   337,
   338,   339,   341,   342,   343,   345,   346,   347,   348,   349,
   351,   352,   353,   355,   356,   358,   359,   361,   362,   364,
   365,   367,   368,   370,   372,   373,   375,   376,   376,   376,
   376,   376,   377,   377,   377,   378,   378,   380,   381
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
// clang-format on

#undef MFEM_DEBUG_COLOR
#define MFEM_DEBUG_COLOR 165
namespace yy
{
std::array<bool, yyruletype::yynrules> rules = {false};
}

// *****************************************************************************
static Node *astAddChild(Node *root, Node *n)
{
   assert(n);
   assert(root);
   n->root = root;
   if (!root->child)
   {
      return root->child = n;
   }
   Node *child = root->child;
   for (; child->next; child = child->next)
      ;
   return child->next = n;
}

// *****************************************************************************
static Node *astAddNext(Node *root, Node *n)
{
   assert(n);
   assert(root);
   n->root = root->root;
   if (!root->next)
   {
      return root->next = n;
   }
   Node *next = root;
   for (; next->next; next = next->next)
      ;
   return next->next = n;
}

// ****************************************************************************
Node *xfl::NewRule(const int YN, const char *text) const
{
   return NewNode(std::make_shared<Rule>(YN, text));
}

// *****************************************************************************
template <int YYN>
void rhs(xfl &ufl, Node **yylval, const int yyn, const symbol_t yyr1n,
         const int nrhs, const char *rule, yy::parser::stack_type &yystack)
{
   assert(YYN == yyn);
   Node *root = *yylval = ufl.NewRule(yyn, rule);
   if (nrhs == 0)
   {
      return;
   }  // %empty
   Node *n = yystack[nrhs - 1].value;
   astAddChild(root, n);
   for (int i = 1; i < nrhs; i++)
   {
      (root = n, astAddNext(root, n = yystack[nrhs - 1 - i].value));
   }
}

// *****************************************************************************
void xfl::DfsPreOrder(Node *n, Middlend &me) const
{
   assert(n);
   if (!n)
   {
      return;
   }

   // Process the current node, setting up the dfs.down to true
   n->Accept(me);

   const int N = n->Number();  // N = SN (token) | RN (rule)
   assert(N > 0);

   // Sanity checks for rules
   if (n->IsRule())
   {
      constexpr int YYNRULES = yynrules;
      if (N > YYNRULES)
      {
         dbg("N:%d/%d", N, YYNRULES);
      }
      assert(N <= YYNRULES);
   }

   // Sanity checks for tokens
   if (n->IsToken())
   {
      constexpr symbol_t YYNTOKENS = yy::parser::YYNTOKENS;
      if (N >= YYNTOKENS)
      {
         dbg("N:%d/%d", N, YYNTOKENS);
      }
      assert(N < YYNTOKENS);
   }

   // Set the state flags
   if (n->IsRule())
   {
      yy::rules.at(N) = true;
   }

   // If n->dfs.down does not stop us from previous Accept, dfs down
   if (n->dfs.down && n->child)
   {
      DfsPreOrder(n->child, me);
      // If me.ctx.extra has been set, re-run a dfs with it
      if (me.ufl.ctx.extra)
      {
         Node *extra = me.ufl.ctx.extra;
         me.ufl.ctx.extra = nullptr;
         DfsPreOrder(extra, me);
      }
   }

   // Process the current node, setting up the dfs.down to false
   if (n->IsRule())
   {
      n->Accept(me, false);
   }  // up, only for rules

   // Reset the state flags
   if (n->IsRule())
   {
      yy::rules.at(N) = false;
   }

   if (n->next)
   {
      DfsPreOrder(n->next, me);
   }
}

// *****************************************************************************
void xfl::DfsInOrder(Node *n, Middlend &me) const
{
   if (!n)
   {
      return;
   }
   if (n->child)
   {
      DfsInOrder(n->child, me);
   }
   n->Accept(me);
   if (n->next)
   {
      DfsInOrder(n->next, me);
   }
}

// *****************************************************************************
void xfl::DfsPostOrder(Node *n, Middlend &me) const
{
   if (!n)
   {
      return;
   }
   if (n->child)
   {
      DfsPostOrder(n->child, me);
   }
   if (n->next)
   {
      DfsPostOrder(n->next, me);
   }
   n->Accept(me);
}

// ****************************************************************************
bool xfl::HitRule(const int rule, Node *n) const
{
   if (!n)
   {
      return false;
   }
   if (n->IsRule() && n->Number() == rule)
   {
      return true;
   }
   if (n->child)
   {
      if (HitRule(rule, n->child))
      {
         return true;
      }
   }
   if (n->next)
   {
      if (HitRule(rule, n->next))
      {
         return true;
      }
   }
   return false;
}

// ****************************************************************************
bool xfl::HitToken(const int tok, const Node *n) const
{
   if (!n)
   {
      return false;
   }
   if (n->IsToken() && n->Number() == tok)
   {
      return true;
   }
   if (n->child)
   {
      if (HitToken(tok, n->child))
      {
         return true;
      }
   }
   if (n->next)
   {
      if (HitToken(tok, n->next))
      {
         return true;
      }
   }
   return false;
}

// ****************************************************************************
bool xfl::OnlyToken(const int tok, Node *n)
{
   assert(n);
   if (n->IsToken() && n->Number() != tok)
   {
      return false;
   }
   if (n->child)
   {
      if (!OnlyToken(tok, n->child))
      {
         return false;
      }
   }
   if (n->next)
   {
      if (!OnlyToken(tok, n->next))
      {
         return false;
      }
   }
   return true;
}

// ****************************************************************************
const Node *xfl::GetToken(const int tok, const Node *n) const
{
   assert(n);
   Node const *m = nullptr;
   if (n->IsToken() && n->Number() == tok)
   {
      return n;
   }
   if (n->child && (m = GetToken(tok, n->child)))
   {
      return m;
   }
   if (n->next && (m = GetToken(tok, n->next)))
   {
      return m;
   }
   return m;
}

// ****************************************************************************
const Node *xfl::GetTokenArg(const int tok, const Node *n) const
{
   assert(n);
   // dbg("%s",n->Name().c_str());
   Node const *m = nullptr;
   if (n->IsToken() && n->Number() == tok)   /*dbg("1");*/
   {
      return n;
   }
   if (n->IsToken() && n->Number() == TOK::COMA)   /*dbg("2");*/
   {
      return n;
   }
   if (n->child && (m = GetTokenArg(tok, n->child)))
   {
      // dbg("3");
      if (m->Number() == tok)
      {
         return m;
      }
      // dbg("4");
      return nullptr;
   }
   if (n->next && (m = GetTokenArg(tok, n->next)))
   {
      // dbg("5");
      if (m->Number() == tok)
      {
         return m;
      }
      // dbg("6");
      return nullptr;
   }
   // dbg("7");
   assert(!m);
   return nullptr;
}

// ****************************************************************************
const Node *xfl::NextToken(const Node *n) const
{
   assert(n);
   Node const *m = nullptr;
   if (n->IsToken())
   {
      return n;
   }
   if (n->child && (m = NextToken(n->child)))
   {
      return m;
   }
   return nullptr;
}

// ****************************************************************************
bool xfl::CondEval(const Node *n, std::ostream &out) const
{
   mfem::internal::EvalCondExpr eval(*const_cast<xfl *>(this), out);
   DfsPostOrder(const_cast<Node *>(n), eval);
   return ctx.eval;
}

// ****************************************************************************
const Node *xfl::NextTokenEval(const Node *n, std::ostream &out) const
{
   assert(n);
   Node const *m = nullptr;
   Node const *child = n->child;
   const int ternary =
      conditional_expr_logical_or_expr_question_expr_colon_conditional_expr;
   if (n->IsRule() && n->n == ternary)
   {
      // dbg("\033[31m[ternary]");
      const bool eval = CondEval(n->child->child, out);
      child = eval ? n->child->next->next : n->child->next->next->next->next;
   }
   if (n->IsToken())
   {
      return n;
   }
   if (child && (m = NextTokenEval(child, out)))
   {
      return m;
   }
   return nullptr;
}

// ****************************************************************************
const Node *xfl::AssignOpNextToken(const int TOK_OP, const int TOK_GET,
                                   const Node *n, std::ostream &out) const
{
   assert(n);
   // Get the next node with the given op token
   const Node *op = GetToken(TOK_OP, n);
   if (!op)
   {
      return nullptr;
   }
   // point to the right tree
   assert(op->root);
   assert(op->root->next);
   Node *expr = op->root->next;
   assert(expr->n == expr_assign_expr);
   // find the next token, evaluate ternary ops
   const Node *m = NextTokenEval(expr->child, out);
   if (m->Number() == TOK_GET)
   {
      return m;
   }
   return nullptr;
}

// ****************************************************************************
const Node *xfl::UpNextToken(const int TOK, const Node *n) const
{
   while (n->root && !(n->next && n->next->Number() == TOK))
   {
      n = n->root;
   }
   if (n->next && n->next->Number() == TOK)
   {
      return n->next;
   }
   return nullptr;
}

// *****************************************************************************
void yy::parser::error(const location_type &, const std::string &msg)
{
   std::cerr << (*ufl.loc) << ": " << msg << std::endl;
   abort();
}
