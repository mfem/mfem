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

               case 3:   // extra_status_rules: lhs
               {
#define extra_status_rules_lhs 3
                  constexpr int YYN = 3;
                  {} {RHS}
                  break;
               }

               case 4:   // lhs: LHS
               {
#define lhs_lhs 4
                  constexpr int YYN = 4;
                  {} {RHS}
                  break;
               }

               case 5:   // statements: statement
               {
#define statements_statement 5
                  constexpr int YYN = 5;
                  {} {RHS}
                  break;
               }

               case 6:   // statements: statements statement
               {
#define statements_statements_statement 6
                  constexpr int YYN = 6;
                  {} {RHS}
                  break;
               }

               case 7:   // statements: extra_status_rules
               {
#define statements_extra_status_rules 7
                  constexpr int YYN = 7;
                  {} {RHS}
                  break;
               }

               case 8:   // statement: NL
               {
#define statement_nl 8
                  constexpr int YYN = 8;
                  {} {RHS}
                  break;
               }

               case 9:   // statement: decl NL
               {
#define statement_decl_nl 9
                  constexpr int YYN = 9;
                  {} {RHS}
                  break;
               }

               case 10:   // decl: function
               {
#define decl_function 10
                  constexpr int YYN = 10;
                  {} {RHS}
                  break;
               }

               case 11:   // decl: domain assign_op expr
               {
#define decl_domain_assign_op_expr 11
                  constexpr int YYN = 11;
                  {} {RHS}
                  break;
               }

               case 12:   // decl: id_list assign_op expr
               {
#define decl_id_list_assign_op_expr 12
                  constexpr int YYN = 12;
                  {} {RHS}
                  break;
               }

               case 13:   // decl: LP id_list RP assign_op expr
               {
#define decl_lp_id_list_rp_assign_op_expr 13
                  constexpr int YYN = 13;
                  {} {RHS}
                  break;
               }

               case 14:   // decl: LB id_list RB assign_op expr
               {
#define decl_lb_id_list_rb_assign_op_expr 14
                  constexpr int YYN = 14;
                  {} {RHS}
                  break;
               }

               case 15:   // decl: if_statement
               {
#define decl_if_statement 15
                  constexpr int YYN = 15;
                  {} {RHS}
                  break;
               }

               case 16:   // decl: api_statement
               {
#define decl_api_statement 16
                  constexpr int YYN = 16;
                  {} {RHS}
                  break;
               }

               case 17:   // decl: iteration_statement
               {
#define decl_iteration_statement 17
                  constexpr int YYN = 17;
                  {} {RHS}
                  break;
               }

               case 18:   // decl: direct_declarator
               {
#define decl_direct_declarator 18
                  constexpr int YYN = 18;
                  {} {RHS}
                  break;
               }

               case 19:   // primary_id: IDENTIFIER
               {
#define primary_id_identifier 19
                  constexpr int YYN = 19;
                  {} {RHS}
                  break;
               }

               case 20:   // postfix_id: primary_id
               {
#define postfix_id_primary_id 20
                  constexpr int YYN = 20;
                  {} {RHS}
                  break;
               }

               case 21:   // postfix_id: postfix_id LP RP
               {
#define postfix_id_postfix_id_lp_rp 21
                  constexpr int YYN = 21;
                  {} {RHS}
                  break;
               }

               case 22:   // postfix_id: postfix_id LP expr RP
               {
#define postfix_id_postfix_id_lp_expr_rp 22
                  constexpr int YYN = 22;
                  {} {RHS}
                  break;
               }

               case 23:   // postfix_id: postfix_id DOT primary_id
               {
#define postfix_id_postfix_id_dot_primary_id 23
                  constexpr int YYN = 23;
                  {} {RHS}
                  break;
               }

               case 24:   // postfix_ids: postfix_id
               {
#define postfix_ids_postfix_id 24
                  constexpr int YYN = 24;
                  {} {RHS}
                  break;
               }

               case 25:   // postfix_ids: postfix_ids postfix_id
               {
#define postfix_ids_postfix_ids_postfix_id 25
                  constexpr int YYN = 25;
                  {} {RHS}
                  break;
               }

               case 26:   // id_list: postfix_ids
               {
#define id_list_postfix_ids 26
                  constexpr int YYN = 26;
                  {} {RHS}
                  break;
               }

               case 27:   // id_list: id_list COMA postfix_ids
               {
#define id_list_id_list_coma_postfix_ids 27
                  constexpr int YYN = 27;
                  {} {RHS}
                  break;
               }

               case 28:   // function: DEF IDENTIFIER LP args_expr_list RP COLON def_empty RETURN math_expr
               {
#define function_def_identifier_lp_args_expr_list_rp_colon_def_empty_return_math_expr 28
                  constexpr int YYN = 28;
                  {} {RHS}
                  break;
               }

               case 29:   // function: DEF IDENTIFIER LP args_expr_list RP COLON def_statements RETURN math_expr
               {
#define function_def_identifier_lp_args_expr_list_rp_colon_def_statements_return_math_expr 29
                  constexpr int YYN = 29;
                  {} {RHS}
                  break;
               }

               case 30:   // def_empty: %empty
               {
#define def_empty_empty 30
                  constexpr int YYN = 30;
                  {} {RHS}
                  break;
               }

               case 31:   // def_statements: def_statement
               {
#define def_statements_def_statement 31
                  constexpr int YYN = 31;
                  {} {RHS}
                  break;
               }

               case 32:   // def_statements: def_statements def_statement
               {
#define def_statements_def_statements_def_statement 32
                  constexpr int YYN = 32;
                  {} {RHS}
                  break;
               }

               case 33:   // def_statement: NL
               {
#define def_statement_nl 33
                  constexpr int YYN = 33;
                  {} {RHS}
                  break;
               }

               case 34:   // def_statement: id_list assign_op expr NL
               {
#define def_statement_id_list_assign_op_expr_nl 34
                  constexpr int YYN = 34;
                  {} {RHS}
                  break;
               }

               case 35:   // def_statement: LP id_list RP assign_op expr NL
               {
#define def_statement_lp_id_list_rp_assign_op_expr_nl 35
                  constexpr int YYN = 35;
                  {} {RHS}
                  break;
               }

               case 36:   // iteration_statement: FOR IDENTIFIER IN RANGE LP IDENTIFIER RP COLON NL expr
               {
#define iteration_statement_for_identifier_in_range_lp_identifier_rp_colon_nl_expr 36
                  constexpr int YYN = 36;
                  {} {RHS}
                  break;
               }

               case 37:   // iteration_statement: FOR IDENTIFIER IN RANGE LP IDENTIFIER RP COLON expr
               {
#define iteration_statement_for_identifier_in_range_lp_identifier_rp_colon_expr 37
                  constexpr int YYN = 37;
                  {} {RHS}
                  break;
               }

               case 38:   // if_statement: IF LP expr RP expr
               {
#define if_statement_if_lp_expr_rp_expr 38
                  constexpr int YYN = 38;
                  {} {RHS}
                  break;
               }

               case 39:   // api_statement: PLOT LP expr RP
               {
#define api_statement_plot_lp_expr_rp 39
                  constexpr int YYN = 39;
                  {} {RHS}
                  break;
               }

               case 40:   // api_statement: SAVE LP expr RP
               {
#define api_statement_save_lp_expr_rp 40
                  constexpr int YYN = 40;
                  {} {RHS}
                  break;
               }

               case 41:   // api_statement: SOLVE LP expr RP
               {
#define api_statement_solve_lp_expr_rp 41
                  constexpr int YYN = 41;
                  {} {RHS}
                  break;
               }

               case 42:   // api_statement: PROJECT LP expr RP
               {
#define api_statement_project_lp_expr_rp 42
                  constexpr int YYN = 42;
                  {} {RHS}
                  break;
               }

               case 43:   // direct_declarator: postfix_id
               {
#define direct_declarator_postfix_id 43
                  constexpr int YYN = 43;
                  {} {RHS}
                  break;
               }

               case 44:   // domain: DOM_DX
               {
#define domain_dom_dx 44
                  constexpr int YYN = 44;
                  {} {RHS}
                  break;
               }

               case 45:   // domain: EXT_DS
               {
#define domain_ext_ds 45
                  constexpr int YYN = 45;
                  {} {RHS}
                  break;
               }

               case 46:   // domain: INT_DS
               {
#define domain_int_ds 46
                  constexpr int YYN = 46;
                  {} {RHS}
                  break;
               }

               case 47:   // constant: NATURAL
               {
#define constant_natural 47
                  constexpr int YYN = 47;
                  {} {RHS}
                  break;
               }

               case 48:   // constant: REAL
               {
#define constant_real 48
                  constexpr int YYN = 48;
                  {} {RHS}
                  break;
               }

               case 49:   // strings: STRING
               {
#define strings_string 49
                  constexpr int YYN = 49;
                  {} {RHS}
                  break;
               }

               case 50:   // strings: strings STRING
               {
#define strings_strings_string 50
                  constexpr int YYN = 50;
                  {} {RHS}
                  break;
               }

               case 51:   // api: DEVICE
               {
#define api_device 51
                  constexpr int YYN = 51;
                  {} {RHS}
                  break;
               }

               case 52:   // api: MESH
               {
#define api_mesh 52
                  constexpr int YYN = 52;
                  {} {RHS}
                  break;
               }

               case 53:   // api: FINITE_ELEMENT
               {
#define api_finite_element 53
                  constexpr int YYN = 53;
                  {} {RHS}
                  break;
               }

               case 54:   // api: UNIT_SQUARE_MESH
               {
#define api_unit_square_mesh 54
                  constexpr int YYN = 54;
                  {} {RHS}
                  break;
               }

               case 55:   // api: UNIT_HEX_MESH
               {
#define api_unit_hex_mesh 55
                  constexpr int YYN = 55;
                  {} {RHS}
                  break;
               }

               case 56:   // api: FUNCTION
               {
#define api_function 56
                  constexpr int YYN = 56;
                  {} {RHS}
                  break;
               }

               case 57:   // api: FUNCTION_SPACE
               {
#define api_function_space 57
                  constexpr int YYN = 57;
                  {} {RHS}
                  break;
               }

               case 58:   // api: VECTOR_FUNCTION_SPACE
               {
#define api_vector_function_space 58
                  constexpr int YYN = 58;
                  {} {RHS}
                  break;
               }

               case 59:   // api: EXPRESSION
               {
#define api_expression 59
                  constexpr int YYN = 59;
                  {} {RHS}
                  break;
               }

               case 60:   // api: DIRICHLET_BC
               {
#define api_dirichlet_bc 60
                  constexpr int YYN = 60;
                  {} {RHS}
                  break;
               }

               case 61:   // api: TRIAL_FUNCTION
               {
#define api_trial_function 61
                  constexpr int YYN = 61;
                  {} {RHS}
                  break;
               }

               case 62:   // api: TEST_FUNCTION
               {
#define api_test_function 62
                  constexpr int YYN = 62;
                  {} {RHS}
                  break;
               }

               case 63:   // api: CONSTANT_API
               {
#define api_constant_api 63
                  constexpr int YYN = 63;
                  {} {RHS}
                  break;
               }

               case 64:   // api: api_statement
               {
#define api_api_statement 64
                  constexpr int YYN = 64;
                  {} {RHS}
                  break;
               }

               case 65:   // primary_expr: IDENTIFIER
               {
#define primary_expr_identifier 65
                  constexpr int YYN = 65;
                  {} {RHS}
                  break;
               }

               case 66:   // primary_expr: constant
               {
#define primary_expr_constant 66
                  constexpr int YYN = 66;
                  {} {RHS}
                  break;
               }

               case 67:   // primary_expr: domain
               {
#define primary_expr_domain 67
                  constexpr int YYN = 67;
                  {} {RHS}
                  break;
               }

               case 68:   // primary_expr: QUOTE
               {
#define primary_expr_quote 68
                  constexpr int YYN = 68;
                  {} {RHS}
                  break;
               }

               case 69:   // primary_expr: strings
               {
#define primary_expr_strings 69
                  constexpr int YYN = 69;
                  {} {RHS}
                  break;
               }

               case 70:   // primary_expr: LP expr RP
               {
#define primary_expr_lp_expr_rp 70
                  constexpr int YYN = 70;
                  {} {RHS}
                  break;
               }

               case 71:   // primary_expr: LB expr RB
               {
#define primary_expr_lb_expr_rb 71
                  constexpr int YYN = 71;
                  {} {RHS}
                  break;
               }

               case 72:   // primary_expr: LS coords RS
               {
#define primary_expr_ls_coords_rs 72
                  constexpr int YYN = 72;
                  {} {RHS}
                  break;
               }

               case 73:   // primary_expr: api
               {
#define primary_expr_api 73
                  constexpr int YYN = 73;
                  {} {RHS}
                  break;
               }

               case 74:   // pow_expr: postfix_expr POW constant
               {
#define pow_expr_postfix_expr_pow_constant 74
                  constexpr int YYN = 74;
                  {} {RHS}
                  break;
               }

               case 75:   // pow_expr: postfix_expr POW IDENTIFIER
               {
#define pow_expr_postfix_expr_pow_identifier 75
                  constexpr int YYN = 75;
                  {} {RHS}
                  break;
               }

               case 76:   // dot_expr: DOT_OP LP additive_expr COMA additive_expr RP
               {
#define dot_expr_dot_op_lp_additive_expr_coma_additive_expr_rp 76
                  constexpr int YYN = 76;
                  {} {RHS}
                  break;
               }

               case 77:   // dot_expr: INNER_OP LP additive_expr COMA additive_expr RP
               {
#define dot_expr_inner_op_lp_additive_expr_coma_additive_expr_rp 77
                  constexpr int YYN = 77;
                  {} {RHS}
                  break;
               }

               case 78:   // postfix_expr: primary_expr
               {
#define postfix_expr_primary_expr 78
                  constexpr int YYN = 78;
                  {} {RHS}
                  break;
               }

               case 79:   // postfix_expr: pow_expr
               {
#define postfix_expr_pow_expr 79
                  constexpr int YYN = 79;
                  {} {RHS}
                  break;
               }

               case 80:   // postfix_expr: postfix_expr LB expr RB
               {
#define postfix_expr_postfix_expr_lb_expr_rb 80
                  constexpr int YYN = 80;
                  {} {RHS}
                  break;
               }

               case 81:   // postfix_expr: postfix_expr LP RP
               {
#define postfix_expr_postfix_expr_lp_rp 81
                  constexpr int YYN = 81;
                  {} {RHS}
                  break;
               }

               case 82:   // postfix_expr: postfix_expr LP args_expr_list RP
               {
#define postfix_expr_postfix_expr_lp_args_expr_list_rp 82
                  constexpr int YYN = 82;
                  {} {RHS}
                  break;
               }

               case 83:   // postfix_expr: postfix_expr DOT primary_expr
               {
#define postfix_expr_postfix_expr_dot_primary_expr 83
                  constexpr int YYN = 83;
                  {} {RHS}
                  break;
               }

               case 84:   // postfix_expr: dot_expr
               {
#define postfix_expr_dot_expr 84
                  constexpr int YYN = 84;
                  {} {RHS}
                  break;
               }

               case 85:   // postfix_expr: GRAD_OP LP additive_expr RP
               {
#define postfix_expr_grad_op_lp_additive_expr_rp 85
                  constexpr int YYN = 85;
                  {} {RHS}
                  break;
               }

               case 86:   // postfix_expr: LHS LP IDENTIFIER RP
               {
#define postfix_expr_lhs_lp_identifier_rp 86
                  constexpr int YYN = 86;
                  {} {RHS}
                  break;
               }

               case 87:   // postfix_expr: RHS LP IDENTIFIER RP
               {
#define postfix_expr_rhs_lp_identifier_rp 87
                  constexpr int YYN = 87;
                  {} {RHS}
                  break;
               }

               case 88:   // unary_expr: postfix_expr
               {
#define unary_expr_postfix_expr 88
                  constexpr int YYN = 88;
                  {} {RHS}
                  break;
               }

               case 89:   // unary_expr: unary_op cast_expr
               {
#define unary_expr_unary_op_cast_expr 89
                  constexpr int YYN = 89;
                  {} {RHS}
                  break;
               }

               case 90:   // unary_op: MUL
               {
#define unary_op_mul 90
                  constexpr int YYN = 90;
                  {} {RHS}
                  break;
               }

               case 91:   // unary_op: ADD
               {
#define unary_op_add 91
                  constexpr int YYN = 91;
                  {} {RHS}
                  break;
               }

               case 92:   // unary_op: SUB
               {
#define unary_op_sub 92
                  constexpr int YYN = 92;
                  {} {RHS}
                  break;
               }

               case 93:   // unary_op: TILDE
               {
#define unary_op_tilde 93
                  constexpr int YYN = 93;
                  {} {RHS}
                  break;
               }

               case 94:   // unary_op: NOT
               {
#define unary_op_not 94
                  constexpr int YYN = 94;
                  {} {RHS}
                  break;
               }

               case 95:   // cast_expr: unary_expr
               {
#define cast_expr_unary_expr 95
                  constexpr int YYN = 95;
                  {} {RHS}
                  break;
               }

               case 96:   // multiplicative_expr: cast_expr
               {
#define multiplicative_expr_cast_expr 96
                  constexpr int YYN = 96;
                  {} {RHS}
                  break;
               }

               case 97:   // multiplicative_expr: multiplicative_expr MUL cast_expr
               {
#define multiplicative_expr_multiplicative_expr_mul_cast_expr 97
                  constexpr int YYN = 97;
                  {} {RHS}
                  break;
               }

               case 98:   // multiplicative_expr: multiplicative_expr DIV cast_expr
               {
#define multiplicative_expr_multiplicative_expr_div_cast_expr 98
                  constexpr int YYN = 98;
                  {} {RHS}
                  break;
               }

               case 99:   // multiplicative_expr: multiplicative_expr MOD cast_expr
               {
#define multiplicative_expr_multiplicative_expr_mod_cast_expr 99
                  constexpr int YYN = 99;
                  {} {RHS}
                  break;
               }

               case 100:   // additive_expr: multiplicative_expr
               {
#define additive_expr_multiplicative_expr 100
                  constexpr int YYN = 100;
                  {} {RHS}
                  break;
               }

               case 101:   // additive_expr: additive_expr ADD multiplicative_expr
               {
#define additive_expr_additive_expr_add_multiplicative_expr 101
                  constexpr int YYN = 101;
                  {} {RHS}
                  break;
               }

               case 102:   // additive_expr: additive_expr SUB multiplicative_expr
               {
#define additive_expr_additive_expr_sub_multiplicative_expr 102
                  constexpr int YYN = 102;
                  {} {RHS}
                  break;
               }

               case 103:   // shift_expr: additive_expr
               {
#define shift_expr_additive_expr 103
                  constexpr int YYN = 103;
                  {} {RHS}
                  break;
               }

               case 104:   // shift_expr: shift_expr LEFT_SHIFT additive_expr
               {
#define shift_expr_shift_expr_left_shift_additive_expr 104
                  constexpr int YYN = 104;
                  {} {RHS}
                  break;
               }

               case 105:   // shift_expr: shift_expr RIGHT_SHIFT additive_expr
               {
#define shift_expr_shift_expr_right_shift_additive_expr 105
                  constexpr int YYN = 105;
                  {} {RHS}
                  break;
               }

               case 106:   // relational_expr: shift_expr
               {
#define relational_expr_shift_expr 106
                  constexpr int YYN = 106;
                  {} {RHS}
                  break;
               }

               case 107:   // relational_expr: relational_expr LT shift_expr
               {
#define relational_expr_relational_expr_lt_shift_expr 107
                  constexpr int YYN = 107;
                  {} {RHS}
                  break;
               }

               case 108:   // relational_expr: relational_expr GT shift_expr
               {
#define relational_expr_relational_expr_gt_shift_expr 108
                  constexpr int YYN = 108;
                  {} {RHS}
                  break;
               }

               case 109:   // relational_expr: relational_expr LT_EQ shift_expr
               {
#define relational_expr_relational_expr_lt_eq_shift_expr 109
                  constexpr int YYN = 109;
                  {} {RHS}
                  break;
               }

               case 110:   // relational_expr: relational_expr GT_EQ shift_expr
               {
#define relational_expr_relational_expr_gt_eq_shift_expr 110
                  constexpr int YYN = 110;
                  {} {RHS}
                  break;
               }

               case 111:   // equality_expr: relational_expr
               {
#define equality_expr_relational_expr 111
                  constexpr int YYN = 111;
                  {} {RHS}
                  break;
               }

               case 112:   // equality_expr: equality_expr EQ_EQ relational_expr
               {
#define equality_expr_equality_expr_eq_eq_relational_expr 112
                  constexpr int YYN = 112;
                  {} {RHS}
                  break;
               }

               case 113:   // equality_expr: equality_expr NOT_EQ relational_expr
               {
#define equality_expr_equality_expr_not_eq_relational_expr 113
                  constexpr int YYN = 113;
                  {} {RHS}
                  break;
               }

               case 114:   // and_expr: equality_expr
               {
#define and_expr_equality_expr 114
                  constexpr int YYN = 114;
                  {} {RHS}
                  break;
               }

               case 115:   // and_expr: and_expr AND equality_expr
               {
#define and_expr_and_expr_and_equality_expr 115
                  constexpr int YYN = 115;
                  {} {RHS}
                  break;
               }

               case 116:   // exclusive_or_expr: and_expr
               {
#define exclusive_or_expr_and_expr 116
                  constexpr int YYN = 116;
                  {} {RHS}
                  break;
               }

               case 117:   // exclusive_or_expr: exclusive_or_expr XOR and_expr
               {
#define exclusive_or_expr_exclusive_or_expr_xor_and_expr 117
                  constexpr int YYN = 117;
                  {} {RHS}
                  break;
               }

               case 118:   // inclusive_or_expr: exclusive_or_expr
               {
#define inclusive_or_expr_exclusive_or_expr 118
                  constexpr int YYN = 118;
                  {} {RHS}
                  break;
               }

               case 119:   // inclusive_or_expr: inclusive_or_expr OR exclusive_or_expr
               {
#define inclusive_or_expr_inclusive_or_expr_or_exclusive_or_expr 119
                  constexpr int YYN = 119;
                  {} {RHS}
                  break;
               }

               case 120:   // logical_and_expr: inclusive_or_expr
               {
#define logical_and_expr_inclusive_or_expr 120
                  constexpr int YYN = 120;
                  {} {RHS}
                  break;
               }

               case 121:   // logical_and_expr: logical_and_expr AND_AND inclusive_or_expr
               {
#define logical_and_expr_logical_and_expr_and_and_inclusive_or_expr 121
                  constexpr int YYN = 121;
                  {} {RHS}
                  break;
               }

               case 122:   // logical_or_expr: logical_and_expr
               {
#define logical_or_expr_logical_and_expr 122
                  constexpr int YYN = 122;
                  {} {RHS}
                  break;
               }

               case 123:   // logical_or_expr: logical_or_expr OR_OR logical_and_expr
               {
#define logical_or_expr_logical_or_expr_or_or_logical_and_expr 123
                  constexpr int YYN = 123;
                  {} {RHS}
                  break;
               }

               case 124:   // conditional_expr: logical_or_expr
               {
#define conditional_expr_logical_or_expr 124
                  constexpr int YYN = 124;
                  {} {RHS}
                  break;
               }

               case 125:   // conditional_expr: logical_or_expr QUESTION expr COLON conditional_expr
               {
#define conditional_expr_logical_or_expr_question_expr_colon_conditional_expr 125
                  constexpr int YYN = 125;
                  {} {RHS}
                  break;
               }

               case 126:   // assign_expr: conditional_expr
               {
#define assign_expr_conditional_expr 126
                  constexpr int YYN = 126;
                  {} {RHS}
                  break;
               }

               case 127:   // assign_expr: postfix_expr assign_op assign_expr
               {
#define assign_expr_postfix_expr_assign_op_assign_expr 127
                  constexpr int YYN = 127;
                  {} {RHS}
                  break;
               }

               case 128:   // assign_op: EQ
               {
#define assign_op_eq 128
                  constexpr int YYN = 128;
                  {} {RHS}
                  break;
               }

               case 129:   // assign_op: ADD_EQ
               {
#define assign_op_add_eq 129
                  constexpr int YYN = 129;
                  {} {RHS}
                  break;
               }

               case 130:   // assign_op: SUB_EQ
               {
#define assign_op_sub_eq 130
                  constexpr int YYN = 130;
                  {} {RHS}
                  break;
               }

               case 131:   // assign_op: MUL_EQ
               {
#define assign_op_mul_eq 131
                  constexpr int YYN = 131;
                  {} {RHS}
                  break;
               }

               case 132:   // assign_op: DIV_EQ
               {
#define assign_op_div_eq 132
                  constexpr int YYN = 132;
                  {} {RHS}
                  break;
               }

               case 133:   // assign_op: MOD_EQ
               {
#define assign_op_mod_eq 133
                  constexpr int YYN = 133;
                  {} {RHS}
                  break;
               }

               case 134:   // assign_op: XOR_EQ
               {
#define assign_op_xor_eq 134
                  constexpr int YYN = 134;
                  {} {RHS}
                  break;
               }

               case 135:   // assign_op: AND_EQ
               {
#define assign_op_and_eq 135
                  constexpr int YYN = 135;
                  {} {RHS}
                  break;
               }

               case 136:   // assign_op: OR_EQ
               {
#define assign_op_or_eq 136
                  constexpr int YYN = 136;
                  {} {RHS}
                  break;
               }

               case 137:   // assign_op: LEFT_EQ
               {
#define assign_op_left_eq 137
                  constexpr int YYN = 137;
                  {} {RHS}
                  break;
               }

               case 138:   // assign_op: RIGHT_EQ
               {
#define assign_op_right_eq 138
                  constexpr int YYN = 138;
                  {} {RHS}
                  break;
               }

               case 139:   // expr: assign_expr
               {
#define expr_assign_expr 139
                  constexpr int YYN = 139;
                  {} {RHS}
                  break;
               }

               case 140:   // expr: expr COMA assign_expr
               {
#define expr_expr_coma_assign_expr 140
                  constexpr int YYN = 140;
                  {} {RHS}
                  break;
               }

               case 141:   // args_expr_list: assign_expr
               {
#define args_expr_list_assign_expr 141
                  constexpr int YYN = 141;
                  {} {RHS}
                  break;
               }

               case 142:   // args_expr_list: args_expr_list COMA assign_expr
               {
#define args_expr_list_args_expr_list_coma_assign_expr 142
                  constexpr int YYN = 142;
                  {} {RHS}
                  break;
               }

               case 143:   // args_expr_list: args_expr_list COMA NL assign_expr
               {
#define args_expr_list_args_expr_list_coma_nl_assign_expr 143
                  constexpr int YYN = 143;
                  {} {RHS}
                  break;
               }

               case 144:   // coord: LP constant COMA constant RP
               {
#define coord_lp_constant_coma_constant_rp 144
                  constexpr int YYN = 144;
                  {} {RHS}
                  break;
               }

               case 145:   // coords: coord
               {
#define coords_coord 145
                  constexpr int YYN = 145;
                  {} {RHS}
                  break;
               }

               case 146:   // coords: coords COLON coord
               {
#define coords_coords_colon_coord 146
                  constexpr int YYN = 146;
                  {} {RHS}
                  break;
               }

               case 147:   // primary_math_expr: IDENTIFIER
               {
#define primary_math_expr_identifier 147
                  constexpr int YYN = 147;
                  {} {RHS}
                  break;
               }

               case 148:   // primary_math_expr: constant
               {
#define primary_math_expr_constant 148
                  constexpr int YYN = 148;
                  {} {RHS}
                  break;
               }

               case 149:   // primary_math_expr: domain
               {
#define primary_math_expr_domain 149
                  constexpr int YYN = 149;
                  {} {RHS}
                  break;
               }

               case 150:   // primary_math_expr: QUOTE
               {
#define primary_math_expr_quote 150
                  constexpr int YYN = 150;
                  {} {RHS}
                  break;
               }

               case 151:   // primary_math_expr: LP math_expr RP
               {
#define primary_math_expr_lp_math_expr_rp 151
                  constexpr int YYN = 151;
                  {} {RHS}
                  break;
               }

               case 152:   // primary_math_expr: LB id_list RB
               {
#define primary_math_expr_lb_id_list_rb 152
                  constexpr int YYN = 152;
                  {} {RHS}
                  break;
               }

               case 153:   // dot_math_expr: DOT_OP LP additive_expr COMA additive_expr RP
               {
#define dot_math_expr_dot_op_lp_additive_expr_coma_additive_expr_rp 153
                  constexpr int YYN = 153;
                  {} {RHS}
                  break;
               }

               case 154:   // dot_math_expr: INNER_OP LP additive_expr COMA additive_expr RP
               {
#define dot_math_expr_inner_op_lp_additive_expr_coma_additive_expr_rp 154
                  constexpr int YYN = 154;
                  {} {RHS}
                  break;
               }

               case 155:   // postfix_math_expr: primary_math_expr
               {
#define postfix_math_expr_primary_math_expr 155
                  constexpr int YYN = 155;
                  {} {RHS}
                  break;
               }

               case 156:   // postfix_math_expr: postfix_math_expr LB math_expr RB
               {
#define postfix_math_expr_postfix_math_expr_lb_math_expr_rb 156
                  constexpr int YYN = 156;
                  {} {RHS}
                  break;
               }

               case 157:   // postfix_math_expr: postfix_math_expr LP argument_math_expr_list RP
               {
#define postfix_math_expr_postfix_math_expr_lp_argument_math_expr_list_rp 157
                  constexpr int YYN = 157;
                  {} {RHS}
                  break;
               }

               case 158:   // postfix_math_expr: postfix_math_expr DOT IDENTIFIER
               {
#define postfix_math_expr_postfix_math_expr_dot_identifier 158
                  constexpr int YYN = 158;
                  {} {RHS}
                  break;
               }

               case 159:   // postfix_math_expr: postfix_math_expr INC_OP
               {
#define postfix_math_expr_postfix_math_expr_inc_op 159
                  constexpr int YYN = 159;
                  {} {RHS}
                  break;
               }

               case 160:   // postfix_math_expr: postfix_math_expr DEC_OP
               {
#define postfix_math_expr_postfix_math_expr_dec_op 160
                  constexpr int YYN = 160;
                  {} {RHS}
                  break;
               }

               case 161:   // postfix_math_expr: postfix_math_expr POW constant
               {
#define postfix_math_expr_postfix_math_expr_pow_constant 161
                  constexpr int YYN = 161;
                  {} {RHS}
                  break;
               }

               case 162:   // postfix_math_expr: dot_math_expr
               {
#define postfix_math_expr_dot_math_expr 162
                  constexpr int YYN = 162;
                  {} {RHS}
                  break;
               }

               case 163:   // postfix_math_expr: GRAD_OP LP additive_math_expr RP
               {
#define postfix_math_expr_grad_op_lp_additive_math_expr_rp 163
                  constexpr int YYN = 163;
                  {} {RHS}
                  break;
               }

               case 164:   // argument_math_expr_list: assign_math_expr
               {
#define argument_math_expr_list_assign_math_expr 164
                  constexpr int YYN = 164;
                  {} {RHS}
                  break;
               }

               case 165:   // argument_math_expr_list: argument_math_expr_list COMA assign_math_expr
               {
#define argument_math_expr_list_argument_math_expr_list_coma_assign_math_expr 165
                  constexpr int YYN = 165;
                  {} {RHS}
                  break;
               }

               case 166:   // unary_math_expr: postfix_math_expr
               {
#define unary_math_expr_postfix_math_expr 166
                  constexpr int YYN = 166;
                  {} {RHS}
                  break;
               }

               case 167:   // unary_math_expr: INC_OP unary_math_expr
               {
#define unary_math_expr_inc_op_unary_math_expr 167
                  constexpr int YYN = 167;
                  {} {RHS}
                  break;
               }

               case 168:   // unary_math_expr: DEC_OP unary_math_expr
               {
#define unary_math_expr_dec_op_unary_math_expr 168
                  constexpr int YYN = 168;
                  {} {RHS}
                  break;
               }

               case 169:   // unary_math_expr: MOD unary_math_expr
               {
#define unary_math_expr_mod_unary_math_expr 169
                  constexpr int YYN = 169;
                  {} {RHS}
                  break;
               }

               case 170:   // unary_math_expr: unary_math_op unary_math_expr
               {
#define unary_math_expr_unary_math_op_unary_math_expr 170
                  constexpr int YYN = 170;
                  {} {RHS}
                  break;
               }

               case 171:   // unary_math_op: MUL
               {
#define unary_math_op_mul 171
                  constexpr int YYN = 171;
                  {} {RHS}
                  break;
               }

               case 172:   // unary_math_op: ADD
               {
#define unary_math_op_add 172
                  constexpr int YYN = 172;
                  {} {RHS}
                  break;
               }

               case 173:   // unary_math_op: SUB
               {
#define unary_math_op_sub 173
                  constexpr int YYN = 173;
                  {} {RHS}
                  break;
               }

               case 174:   // unary_math_op: AND
               {
#define unary_math_op_and 174
                  constexpr int YYN = 174;
                  {} {RHS}
                  break;
               }

               case 175:   // unary_math_op: TILDE
               {
#define unary_math_op_tilde 175
                  constexpr int YYN = 175;
                  {} {RHS}
                  break;
               }

               case 176:   // unary_math_op: NOT
               {
#define unary_math_op_not 176
                  constexpr int YYN = 176;
                  {} {RHS}
                  break;
               }

               case 177:   // multiplicative_math_expr: unary_math_expr
               {
#define multiplicative_math_expr_unary_math_expr 177
                  constexpr int YYN = 177;
                  {} {RHS}
                  break;
               }

               case 178:   // multiplicative_math_expr: multiplicative_math_expr MUL unary_math_expr
               {
#define multiplicative_math_expr_multiplicative_math_expr_mul_unary_math_expr 178
                  constexpr int YYN = 178;
                  {} {RHS}
                  break;
               }

               case 179:   // multiplicative_math_expr: multiplicative_math_expr DIV unary_math_expr
               {
#define multiplicative_math_expr_multiplicative_math_expr_div_unary_math_expr 179
                  constexpr int YYN = 179;
                  {} {RHS}
                  break;
               }

               case 180:   // multiplicative_math_expr: multiplicative_math_expr MOD unary_math_expr
               {
#define multiplicative_math_expr_multiplicative_math_expr_mod_unary_math_expr 180
                  constexpr int YYN = 180;
                  {} {RHS}
                  break;
               }

               case 181:   // additive_math_expr: multiplicative_math_expr
               {
#define additive_math_expr_multiplicative_math_expr 181
                  constexpr int YYN = 181;
                  {} {RHS}
                  break;
               }

               case 182:   // additive_math_expr: additive_math_expr ADD multiplicative_math_expr
               {
#define additive_math_expr_additive_math_expr_add_multiplicative_math_expr 182
                  constexpr int YYN = 182;
                  {} {RHS}
                  break;
               }

               case 183:   // additive_math_expr: additive_math_expr SUB multiplicative_math_expr
               {
#define additive_math_expr_additive_math_expr_sub_multiplicative_math_expr 183
                  constexpr int YYN = 183;
                  {} {RHS}
                  break;
               }

               case 184:   // shift_math_expr: additive_math_expr
               {
#define shift_math_expr_additive_math_expr 184
                  constexpr int YYN = 184;
                  {} {RHS}
                  break;
               }

               case 185:   // shift_math_expr: shift_math_expr LEFT_SHIFT additive_math_expr
               {
#define shift_math_expr_shift_math_expr_left_shift_additive_math_expr 185
                  constexpr int YYN = 185;
                  {} {RHS}
                  break;
               }

               case 186:   // shift_math_expr: shift_math_expr RIGHT_SHIFT additive_math_expr
               {
#define shift_math_expr_shift_math_expr_right_shift_additive_math_expr 186
                  constexpr int YYN = 186;
                  {} {RHS}
                  break;
               }

               case 187:   // relational_math_expr: shift_math_expr
               {
#define relational_math_expr_shift_math_expr 187
                  constexpr int YYN = 187;
                  {} {RHS}
                  break;
               }

               case 188:   // relational_math_expr: relational_math_expr LT shift_math_expr
               {
#define relational_math_expr_relational_math_expr_lt_shift_math_expr 188
                  constexpr int YYN = 188;
                  {} {RHS}
                  break;
               }

               case 189:   // relational_math_expr: relational_math_expr GT shift_math_expr
               {
#define relational_math_expr_relational_math_expr_gt_shift_math_expr 189
                  constexpr int YYN = 189;
                  {} {RHS}
                  break;
               }

               case 190:   // relational_math_expr: relational_math_expr LT_EQ shift_math_expr
               {
#define relational_math_expr_relational_math_expr_lt_eq_shift_math_expr 190
                  constexpr int YYN = 190;
                  {} {RHS}
                  break;
               }

               case 191:   // relational_math_expr: relational_math_expr GT_EQ shift_math_expr
               {
#define relational_math_expr_relational_math_expr_gt_eq_shift_math_expr 191
                  constexpr int YYN = 191;
                  {} {RHS}
                  break;
               }

               case 192:   // equality_math_expr: relational_math_expr
               {
#define equality_math_expr_relational_math_expr 192
                  constexpr int YYN = 192;
                  {} {RHS}
                  break;
               }

               case 193:   // equality_math_expr: equality_math_expr EQ_EQ relational_math_expr
               {
#define equality_math_expr_equality_math_expr_eq_eq_relational_math_expr 193
                  constexpr int YYN = 193;
                  {} {RHS}
                  break;
               }

               case 194:   // equality_math_expr: equality_math_expr NOT_EQ relational_math_expr
               {
#define equality_math_expr_equality_math_expr_not_eq_relational_math_expr 194
                  constexpr int YYN = 194;
                  {} {RHS}
                  break;
               }

               case 195:   // and_math_expr: equality_math_expr
               {
#define and_math_expr_equality_math_expr 195
                  constexpr int YYN = 195;
                  {} {RHS}
                  break;
               }

               case 196:   // and_math_expr: and_math_expr AND equality_math_expr
               {
#define and_math_expr_and_math_expr_and_equality_math_expr 196
                  constexpr int YYN = 196;
                  {} {RHS}
                  break;
               }

               case 197:   // exclusive_or_math_expr: and_math_expr
               {
#define exclusive_or_math_expr_and_math_expr 197
                  constexpr int YYN = 197;
                  {} {RHS}
                  break;
               }

               case 198:   // exclusive_or_math_expr: exclusive_or_math_expr XOR and_math_expr
               {
#define exclusive_or_math_expr_exclusive_or_math_expr_xor_and_math_expr 198
                  constexpr int YYN = 198;
                  {} {RHS}
                  break;
               }

               case 199:   // inclusive_or_math_expr: exclusive_or_math_expr
               {
#define inclusive_or_math_expr_exclusive_or_math_expr 199
                  constexpr int YYN = 199;
                  {} {RHS}
                  break;
               }

               case 200:   // inclusive_or_math_expr: inclusive_or_math_expr OR exclusive_or_math_expr
               {
#define inclusive_or_math_expr_inclusive_or_math_expr_or_exclusive_or_math_expr 200
                  constexpr int YYN = 200;
                  {} {RHS}
                  break;
               }

               case 201:   // logical_and_math_expr: inclusive_or_math_expr
               {
#define logical_and_math_expr_inclusive_or_math_expr 201
                  constexpr int YYN = 201;
                  {} {RHS}
                  break;
               }

               case 202:   // logical_and_math_expr: logical_and_math_expr AND_AND inclusive_or_math_expr
               {
#define logical_and_math_expr_logical_and_math_expr_and_and_inclusive_or_math_expr 202
                  constexpr int YYN = 202;
                  {} {RHS}
                  break;
               }

               case 203:   // logical_or_math_expr: logical_and_math_expr
               {
#define logical_or_math_expr_logical_and_math_expr 203
                  constexpr int YYN = 203;
                  {} {RHS}
                  break;
               }

               case 204:   // logical_or_math_expr: logical_or_math_expr OR_OR logical_and_math_expr
               {
#define logical_or_math_expr_logical_or_math_expr_or_or_logical_and_math_expr 204
                  constexpr int YYN = 204;
                  {} {RHS}
                  break;
               }

               case 205:   // conditional_math_expr: logical_or_math_expr
               {
#define conditional_math_expr_logical_or_math_expr 205
                  constexpr int YYN = 205;
                  {} {RHS}
                  break;
               }

               case 206:   // assign_math_expr: conditional_math_expr
               {
#define assign_math_expr_conditional_math_expr 206
                  constexpr int YYN = 206;
                  {} {RHS}
                  break;
               }

               case 207:   // assign_math_expr: unary_math_expr assign_math_op assign_math_expr
               {
#define assign_math_expr_unary_math_expr_assign_math_op_assign_math_expr 207
                  constexpr int YYN = 207;
                  {} {RHS}
                  break;
               }

               case 208:   // assign_math_op: EQ
               {
#define assign_math_op_eq 208
                  constexpr int YYN = 208;
                  {} {RHS}
                  break;
               }

               case 209:   // assign_math_op: ADD_EQ
               {
#define assign_math_op_add_eq 209
                  constexpr int YYN = 209;
                  {} {RHS}
                  break;
               }

               case 210:   // assign_math_op: SUB_EQ
               {
#define assign_math_op_sub_eq 210
                  constexpr int YYN = 210;
                  {} {RHS}
                  break;
               }

               case 211:   // assign_math_op: MUL_EQ
               {
#define assign_math_op_mul_eq 211
                  constexpr int YYN = 211;
                  {} {RHS}
                  break;
               }

               case 212:   // assign_math_op: DIV_EQ
               {
#define assign_math_op_div_eq 212
                  constexpr int YYN = 212;
                  {} {RHS}
                  break;
               }

               case 213:   // assign_math_op: MOD_EQ
               {
#define assign_math_op_mod_eq 213
                  constexpr int YYN = 213;
                  {} {RHS}
                  break;
               }

               case 214:   // assign_math_op: XOR_EQ
               {
#define assign_math_op_xor_eq 214
                  constexpr int YYN = 214;
                  {} {RHS}
                  break;
               }

               case 215:   // assign_math_op: AND_EQ
               {
#define assign_math_op_and_eq 215
                  constexpr int YYN = 215;
                  {} {RHS}
                  break;
               }

               case 216:   // assign_math_op: OR_EQ
               {
#define assign_math_op_or_eq 216
                  constexpr int YYN = 216;
                  {} {RHS}
                  break;
               }

               case 217:   // assign_math_op: LEFT_EQ
               {
#define assign_math_op_left_eq 217
                  constexpr int YYN = 217;
                  {} {RHS}
                  break;
               }

               case 218:   // assign_math_op: RIGHT_EQ
               {
#define assign_math_op_right_eq 218
                  constexpr int YYN = 218;
                  {} {RHS}
                  break;
               }

               case 219:   // math_expr: assign_math_expr
               {
#define math_expr_assign_math_expr 219
                  constexpr int YYN = 219;
                  {} {RHS}
                  break;
               }

               case 220:   // math_expr: math_expr COMA assign_math_expr
               {
#define math_expr_math_expr_coma_assign_math_expr 220
                  constexpr int YYN = 220;
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
      "AND", "XOR", "OR", "QUESTION", "NOT", "INC_OP", "DEC_OP",
      "DOM_DX_PREFIX", "DOM_DX_POSTFIX", "EMPTY", "$accept", "entry_point",
      "extra_status_rules", "lhs", "statements", "statement", "decl",
      "primary_id", "postfix_id", "postfix_ids", "id_list", "function",
      "def_empty", "def_statements", "def_statement", "iteration_statement",
      "if_statement", "api_statement", "direct_declarator", "domain",
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


const short parser::yypact_ninf_ = -291;

const signed char parser::yytable_ninf_ = -44;

const short
parser::yypact_[] =
{
   347,  -291,    23,   -22,   -16,    14,    27,    33,    57,  -291,
   -291,  -291,  -291,  -291,    87,    87,    56,  -291,  -291,   815,
   -291,   137,  -291,     8,    87,   156,  -291,  -291,  -291,  -291,
   -291,   240,    81,   711,   711,   711,   711,   711,   134,    -1,
   68,   -10,  -291,  -291,  -291,   577,    87,    -1,  -291,  -291,
   -291,  -291,  -291,  -291,  -291,  -291,  -291,  -291,  -291,    87,
   711,   711,   711,  -291,  -291,   108,   130,   157,   173,   182,
   -291,  -291,  -291,  -291,  -291,  -291,  -291,  -291,  -291,  -291,
   -291,  -291,  -291,  -291,  -291,  -291,  -291,  -291,  -291,   188,
   711,   711,  -291,  -291,  -291,  -291,  -291,   230,  -291,  -291,
   -291,  -291,   120,  -291,   711,  -291,   -27,    58,   152,    75,
   -5,   179,   180,   194,   241,   -11,  -291,  -291,    74,    85,
   90,   115,   117,   263,   240,   240,  -291,   124,  -291,    87,
   227,   227,  -291,   128,   711,   711,   711,   267,   270,   251,
   -291,    39,   146,   237,  -291,    69,   644,   711,   833,   711,
   169,  -291,   711,   711,   711,   711,   711,   711,   711,   711,
   711,   711,   711,   711,   711,   711,   711,   711,   711,   711,
   711,  -291,   711,  -291,  -291,  -291,   711,   283,   711,   711,
   -291,   279,   443,    -6,     7,   150,   288,   293,   291,  -291,
   188,  -291,  -291,  -291,  -291,  -291,   154,   239,  -291,  -291,
   -291,  -291,  -291,   -27,   -27,    58,    58,   152,   152,   152,
   152,    75,    75,    -5,   179,   180,   194,   241,    64,  -291,
   227,   310,   227,   227,    21,   711,  -291,   711,   711,  -291,
   -291,  -291,   251,  -291,  -291,  -291,   711,   305,  -291,    87,
   156,   365,    43,  -291,  -291,   160,   186,   307,  -291,   303,
   177,   711,   748,   748,  -291,  -291,  -291,  -291,   510,   240,
   18,  -291,   311,   313,   314,  -291,  -291,  -291,  -291,   748,
   87,   748,  -291,  -291,  -291,   748,   748,  -291,  -291,  -291,
   -291,    42,   346,   748,   -13,   253,   238,   133,     3,   295,
   299,   300,   344,   363,  -291,  -291,   331,   331,   711,   227,
   711,  -291,   711,   711,   748,   202,   247,  -291,  -291,  -291,
   251,   748,   748,   329,  -291,  -291,  -291,  -291,  -291,  -291,
   -291,  -291,  -291,  -291,  -291,  -291,  -291,   748,  -291,   748,
   748,   748,   748,   748,   748,   748,   748,   748,   748,   748,
   748,   748,   748,   748,   748,   748,   748,   748,   227,    19,
   9,    26,  -291,   190,  -291,  -291,  -291,   206,  -291,   249,
   -291,  -291,  -291,  -291,  -291,   -13,   -13,   253,   253,   238,
   238,   238,   238,   133,   133,     3,   295,   299,   300,   344,
   -291,  -291,   711,   711,  -291,  -291,   748,  -291,   199,   204,
   -291,  -291,  -291
};

const unsigned char
parser::yydefact_[] =
{
   0,     8,     0,     0,     0,     0,     0,     0,     0,     4,
   44,    45,    46,    19,     0,     0,     0,     7,     3,     2,
   5,     0,    20,    24,    26,     0,    10,    17,    15,    16,
   18,     0,     0,     0,     0,     0,     0,     0,     0,    24,
   0,     0,     1,     6,     9,     0,     0,    25,   129,   130,
   131,   132,   133,   134,   135,   136,   137,   138,   128,     0,
   0,     0,     0,    49,    68,     0,     0,     0,     0,     0,
   51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
   61,    62,    63,    47,    48,    65,    91,    92,    90,     0,
   0,     0,    93,    94,    64,    67,    66,    69,    73,    78,
   79,    84,    88,    95,     0,    96,   100,   103,   106,   111,
   114,   116,   118,   120,   122,   124,   126,   139,     0,     0,
   0,     0,     0,     0,     0,     0,    21,     0,    23,    27,
   12,    11,   141,     0,     0,     0,     0,     0,     0,     0,
   145,     0,     0,     0,    50,     0,     0,     0,     0,     0,
   88,    89,     0,     0,     0,     0,     0,     0,     0,     0,
   0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   0,    39,     0,    40,    41,    42,     0,     0,     0,     0,
   22,     0,     0,     0,     0,     0,     0,     0,     0,    72,
   0,    70,    71,    75,    74,    81,     0,     0,    83,   127,
   97,    98,    99,   101,   102,   104,   105,   108,   107,   109,
   110,   112,   113,   115,   117,   119,   121,   123,     0,   140,
   38,     0,    13,    14,    30,     0,   142,     0,     0,    85,
   86,    87,     0,   146,    82,    80,     0,     0,    33,     0,
   0,     0,     0,    31,   143,     0,     0,     0,   125,     0,
   0,     0,     0,     0,    32,    76,    77,   144,     0,     0,
   0,   150,     0,     0,     0,   147,   172,   173,   171,     0,
   0,     0,   175,   174,   176,     0,     0,   149,   148,   155,
   162,   166,   177,     0,   181,   184,   187,   192,   195,   197,
   199,   201,   203,   205,   206,   219,    28,    29,     0,    37,
   0,    34,     0,     0,     0,     0,     0,   169,   167,   168,
   0,     0,     0,     0,   159,   160,   209,   210,   211,   212,
   213,   214,   215,   216,   217,   218,   208,     0,   170,     0,
   0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   0,     0,     0,     0,     0,     0,     0,     0,    36,     0,
   0,     0,   177,     0,   151,   152,   161,     0,   164,     0,
   158,   207,   178,   179,   180,   182,   183,   185,   186,   189,
   188,   190,   191,   193,   194,   196,   198,   200,   202,   204,
   220,    35,     0,     0,   163,   157,     0,   156,     0,     0,
   165,   153,   154
};

const short
parser::yypgoto_[] =
{
   -291,  -291,  -291,  -291,  -291,   385,  -291,   364,    20,   350,
   -8,  -291,  -291,  -291,   170,  -291,  -291,    34,  -291,     0,
   96,  -291,  -291,   265,  -291,  -291,   -32,  -291,  -291,   -61,
   166,  -126,   135,   161,   246,   252,   248,   254,   250,  -291,
   181,   -51,   -17,   -19,   274,   231,  -291,  -291,  -291,  -291,
   -291,  -149,  -291,    22,  -271,   -36,    28,    82,   101,   102,
   100,   103,  -291,  -291,  -290,  -291,  -229
};

const short
parser::yydefgoto_[] =
{
   -1,    16,    17,    18,    19,    20,    21,    22,    39,    24,
   25,    26,   241,   242,   243,    27,    28,    94,    30,    95,
   96,    97,    98,    99,   100,   101,   150,   103,   104,   105,
   106,   107,   108,   109,   110,   111,   112,   113,   114,   115,
   116,   117,    60,   118,   133,   140,   141,   279,   280,   281,
   357,   352,   283,   284,   285,   286,   287,   288,   289,   290,
   291,   292,   293,   294,   295,   327,   296
};

const short
parser::yytable_[] =
{
   31,   102,   102,   102,   102,   102,    40,    41,   183,   184,
   185,   132,   -43,   102,    61,   119,   120,   121,   122,    31,
   23,   358,   301,   381,   297,   238,   127,   169,   102,   102,
   102,   205,   206,   353,    29,   152,   153,   361,   163,    23,
   305,   130,   131,   151,    47,    33,   340,   238,   154,   329,
   330,    34,   253,    29,   155,   156,    42,   380,   102,   102,
   125,    59,   331,   367,   368,   227,    45,   155,   156,   155,
   156,   142,   143,    46,   170,    45,   164,    13,   228,    32,
   382,    35,    46,   359,   341,   149,   155,   156,   239,   172,
   172,   200,   201,   202,    36,   132,   390,   383,   199,    13,
   37,   245,   246,   282,   282,   189,   310,   178,   179,   311,
   239,   312,   190,    38,   102,   102,   313,   102,   155,   156,
   282,   219,   307,    83,    84,   193,   308,   309,   197,   314,
   315,   226,   159,   160,   328,   172,   124,   236,   102,    59,
   102,    44,   171,    13,   102,   172,   102,   102,    62,    47,
   102,   218,   123,   173,   161,   162,   172,   220,   174,   222,
   223,   172,   282,   282,    48,    49,    50,    51,    52,    53,
   54,    55,    56,    57,   244,   134,   350,   351,   282,    58,
   362,   363,   364,   175,   145,   176,   172,   146,   172,   147,
   336,   337,   180,   102,   148,   172,   181,   135,   282,   182,
   48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
   155,   156,   338,   339,   191,    58,   240,   172,   229,   102,
   155,   156,   234,   251,   136,   182,   102,    59,   255,   157,
   158,   250,   260,   145,   240,   188,   146,   282,   147,   299,
   137,   194,   300,   148,   144,   259,   155,   156,    59,   138,
   332,   333,   277,   277,   256,   139,   388,   389,   384,   155,
   156,   165,   306,   166,   155,   156,   102,   391,   102,   277,
   354,   277,   392,   347,   385,   277,   277,   386,   167,   348,
   168,   349,   177,   277,    48,    49,    50,    51,    52,    53,
   54,    55,    56,    57,   207,   208,   209,   210,   172,    58,
   369,   370,   371,   372,   277,    83,    84,   192,   172,   235,
   172,   277,   277,   332,   333,   334,   335,   355,    59,   387,
   347,   203,   204,   186,   211,   212,   187,   277,   247,   277,
   277,   277,   277,   277,   277,   277,   277,   277,   277,   277,
   277,   277,   277,   277,   277,   277,   277,   277,   278,   278,
   221,     1,   224,     2,   365,   366,   230,     3,     4,     5,
   6,   231,   232,     7,     8,   278,   237,   278,   373,   374,
   9,   278,   278,   249,   252,   257,   258,   342,   302,   278,
   303,   304,   343,   345,   344,   360,   277,    10,    11,    12,
   316,   317,   318,   319,   320,   321,   322,   323,   324,   325,
   278,   346,   347,    13,    43,   326,   356,   278,   278,   129,
   128,   213,   254,   198,    14,   215,    15,   248,   214,   217,
   196,   233,   216,   278,   375,   278,   278,   278,   278,   278,
   278,   278,   278,   278,   278,   278,   278,   278,   278,   278,
   278,   278,   278,   278,   376,   378,   377,   225,     0,   379,
   0,     0,     0,     3,     4,     5,     6,    63,    64,     0,
   0,     0,     0,    65,    66,    67,    68,    69,    70,    71,
   72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
   82,     0,   278,    10,    11,    12,     0,     0,     0,     0,
   0,     0,     0,     0,     0,     0,     0,    83,    84,    85,
   0,     0,     0,    86,    87,    88,     0,     0,    89,     0,
   90,     0,    91,     0,   298,     0,     0,     0,     0,    92,
   3,     4,     5,     6,    63,    64,     0,     0,     0,    93,
   65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
   75,    76,    77,    78,    79,    80,    81,    82,     0,     0,
   10,    11,    12,     0,     0,     0,     0,     0,     0,     0,
   0,     0,     0,     0,    83,    84,    85,     0,     0,     0,
   86,    87,    88,     0,     0,    89,     0,    90,     0,    91,
   0,     0,     0,     0,     0,     0,    92,     3,     4,     5,
   6,    63,    64,     0,     0,     0,    93,    65,    66,    67,
   68,    69,    70,    71,    72,    73,    74,    75,    76,    77,
   78,    79,    80,    81,    82,     0,     0,    10,    11,    12,
   0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   0,    83,    84,    85,     0,     0,     0,    86,    87,    88,
   0,     0,    89,     0,    90,   126,    91,     0,     0,     0,
   0,     0,     0,    92,     3,     4,     5,     6,    63,    64,
   0,     0,     0,    93,    65,    66,    67,    68,    69,    70,
   71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
   81,    82,     0,     0,    10,    11,    12,     0,     0,     0,
   0,     0,     0,     0,     0,     0,     0,     0,    83,    84,
   85,     0,     0,     0,    86,    87,    88,     0,     0,    89,
   0,    90,   195,    91,     0,     0,     0,     0,     0,     0,
   92,     3,     4,     5,     6,    63,    64,     0,     0,     0,
   93,    65,    66,    67,    68,    69,    70,    71,    72,    73,
   74,    75,    76,    77,    78,    79,    80,    81,    82,     0,
   0,    10,    11,    12,     0,     0,     0,     0,     0,     0,
   0,     0,     0,   261,     0,    83,    84,    85,   262,   263,
   264,    86,    87,    88,     0,     0,    89,     0,    90,     0,
   91,     0,     0,     0,     0,     0,     0,    92,    10,    11,
   12,     0,     0,     0,     0,     0,     0,    93,     0,     0,
   0,     0,    83,    84,   265,     0,     0,     0,   266,   267,
   268,     0,     0,     0,     0,   269,     0,   270,     0,     1,
   0,     2,     0,   271,   272,     3,     4,     5,     6,     0,
   273,     7,     8,     0,   274,   275,   276,     0,     0,     0,
   0,     0,     0,     3,     4,     5,     6,    63,    64,     0,
   0,     0,     0,     0,     0,    10,    11,    12,    70,    71,
   72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
   82,    13,     0,    10,    11,    12,     0,     0,     0,     0,
   0,     0,    14,     0,    15,     0,     0,    83,    84,    85,
   0,     0,     0,     0,     0,     0,     0,     0,    89,     0,
   90,     0,    91
};

const short
parser::yycheck_[] =
{
   0,    33,    34,    35,    36,    37,    14,    15,   134,   135,
   136,    62,     4,    45,    31,    34,    35,    36,    37,    19,
   0,   311,     4,     4,   253,     4,    45,    38,    60,    61,
   62,   157,   158,   304,     0,    62,    63,   327,    43,    19,
   269,    60,    61,   104,    24,    67,    43,     4,    75,    62,
   63,    67,     9,    19,    60,    61,     0,   347,    90,    91,
   70,    71,    75,   334,   335,    71,    67,    60,    61,    60,
   61,    90,    91,    74,    85,    67,    81,    56,    71,    56,
   71,    67,    74,   312,    81,   102,    60,    61,    67,    71,
   71,   152,   153,   154,    67,   146,   386,    71,   149,    56,
   67,   227,   228,   252,   253,    66,    64,   124,   125,    67,
   67,    69,    73,    56,   146,   147,    74,   149,    60,    61,
   269,   172,   271,    54,    55,    56,   275,   276,   147,    87,
   88,   182,    57,    58,   283,    71,    68,    73,   170,    71,
   172,     4,    68,    56,   176,    71,   178,   179,    67,   129,
   182,   170,    18,    68,    79,    80,    71,   176,    68,   178,
   179,    71,   311,   312,    44,    45,    46,    47,    48,    49,
   50,    51,    52,    53,   225,    67,   302,   303,   327,    59,
   329,   330,   331,    68,    64,    68,    71,    67,    71,    69,
   57,    58,    68,   225,    74,    71,    68,    67,   347,    71,
   44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
   60,    61,    79,    80,    68,    59,   224,    71,    68,   251,
   60,    61,    68,   240,    67,    71,   258,    71,    68,    77,
   78,   239,   251,    64,   242,   139,    67,   386,    69,   258,
   67,   145,   259,    74,    14,    68,    60,    61,    71,    67,
   60,    61,   252,   253,    68,    67,   382,   383,    68,    60,
   61,    82,   270,    83,    60,    61,   298,    68,   300,   269,
   68,   271,    68,    71,    68,   275,   276,    71,    84,   298,
   39,   300,    19,   283,    44,    45,    46,    47,    48,    49,
   50,    51,    52,    53,   159,   160,   161,   162,    71,    59,
   336,   337,   338,   339,   304,    54,    55,    70,    71,    70,
   71,   311,   312,    60,    61,    77,    78,    70,    71,    70,
   71,   155,   156,    56,   163,   164,    56,   327,   232,   329,
   330,   331,   332,   333,   334,   335,   336,   337,   338,   339,
   340,   341,   342,   343,   344,   345,   346,   347,   252,   253,
   67,     4,    73,     6,   332,   333,    68,    10,    11,    12,
   13,    68,    71,    16,    17,   269,    56,   271,   340,   341,
   23,   275,   276,    68,     9,    68,    73,    82,    67,   283,
   67,    67,    83,    39,    84,    56,   386,    40,    41,    42,
   44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
   304,    38,    71,    56,    19,    59,   310,   311,   312,    59,
   46,   165,   242,   148,    67,   167,    69,   236,   166,   169,
   146,   190,   168,   327,   342,   329,   330,   331,   332,   333,
   334,   335,   336,   337,   338,   339,   340,   341,   342,   343,
   344,   345,   346,   347,   343,   345,   344,     4,    -1,   346,
   -1,    -1,    -1,    10,    11,    12,    13,    14,    15,    -1,
   -1,    -1,    -1,    20,    21,    22,    23,    24,    25,    26,
   27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
   37,    -1,   386,    40,    41,    42,    -1,    -1,    -1,    -1,
   -1,    -1,    -1,    -1,    -1,    -1,    -1,    54,    55,    56,
   -1,    -1,    -1,    60,    61,    62,    -1,    -1,    65,    -1,
   67,    -1,    69,    -1,     4,    -1,    -1,    -1,    -1,    76,
   10,    11,    12,    13,    14,    15,    -1,    -1,    -1,    86,
   20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
   30,    31,    32,    33,    34,    35,    36,    37,    -1,    -1,
   40,    41,    42,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   -1,    -1,    -1,    -1,    54,    55,    56,    -1,    -1,    -1,
   60,    61,    62,    -1,    -1,    65,    -1,    67,    -1,    69,
   -1,    -1,    -1,    -1,    -1,    -1,    76,    10,    11,    12,
   13,    14,    15,    -1,    -1,    -1,    86,    20,    21,    22,
   23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
   33,    34,    35,    36,    37,    -1,    -1,    40,    41,    42,
   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   -1,    54,    55,    56,    -1,    -1,    -1,    60,    61,    62,
   -1,    -1,    65,    -1,    67,    68,    69,    -1,    -1,    -1,
   -1,    -1,    -1,    76,    10,    11,    12,    13,    14,    15,
   -1,    -1,    -1,    86,    20,    21,    22,    23,    24,    25,
   26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
   36,    37,    -1,    -1,    40,    41,    42,    -1,    -1,    -1,
   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    54,    55,
   56,    -1,    -1,    -1,    60,    61,    62,    -1,    -1,    65,
   -1,    67,    68,    69,    -1,    -1,    -1,    -1,    -1,    -1,
   76,    10,    11,    12,    13,    14,    15,    -1,    -1,    -1,
   86,    20,    21,    22,    23,    24,    25,    26,    27,    28,
   29,    30,    31,    32,    33,    34,    35,    36,    37,    -1,
   -1,    40,    41,    42,    -1,    -1,    -1,    -1,    -1,    -1,
   -1,    -1,    -1,    15,    -1,    54,    55,    56,    20,    21,
   22,    60,    61,    62,    -1,    -1,    65,    -1,    67,    -1,
   69,    -1,    -1,    -1,    -1,    -1,    -1,    76,    40,    41,
   42,    -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,
   -1,    -1,    54,    55,    56,    -1,    -1,    -1,    60,    61,
   62,    -1,    -1,    -1,    -1,    67,    -1,    69,    -1,     4,
   -1,     6,    -1,    75,    76,    10,    11,    12,    13,    -1,
   82,    16,    17,    -1,    86,    87,    88,    -1,    -1,    -1,
   -1,    -1,    -1,    10,    11,    12,    13,    14,    15,    -1,
   -1,    -1,    -1,    -1,    -1,    40,    41,    42,    25,    26,
   27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
   37,    56,    -1,    40,    41,    42,    -1,    -1,    -1,    -1,
   -1,    -1,    67,    -1,    69,    -1,    -1,    54,    55,    56,
   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    65,    -1,
   67,    -1,    69
};

const unsigned char
parser::yystos_[] =
{
   0,     4,     6,    10,    11,    12,    13,    16,    17,    23,
   40,    41,    42,    56,    67,    69,    93,    94,    95,    96,
   97,    98,    99,   100,   101,   102,   103,   107,   108,   109,
   110,   111,    56,    67,    67,    67,    67,    67,    56,   100,
   102,   102,     0,    97,     4,    67,    74,   100,    44,    45,
   46,    47,    48,    49,    50,    51,    52,    53,    59,    71,
   134,   134,    67,    14,    15,    20,    21,    22,    23,    24,
   25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
   35,    36,    37,    54,    55,    56,    60,    61,    62,    65,
   67,    69,    76,    86,   109,   111,   112,   113,   114,   115,
   116,   117,   118,   119,   120,   121,   122,   123,   124,   125,
   126,   127,   128,   129,   130,   131,   132,   133,   135,   135,
   135,   135,   135,    18,    68,    70,    68,   135,    99,   101,
   135,   135,   133,   136,    67,    67,    67,    67,    67,    67,
   137,   138,   135,   135,    14,    64,    67,    69,    74,   134,
   118,   121,    62,    63,    75,    60,    61,    77,    78,    57,
   58,    79,    80,    43,    81,    82,    83,    84,    39,    38,
   85,    68,    71,    68,    68,    68,    68,    19,   134,   134,
   68,    68,    71,   123,   123,   123,    56,    56,   112,    66,
   73,    68,    70,    56,   112,    68,   136,   135,   115,   133,
   121,   121,   121,   122,   122,   123,   123,   124,   124,   124,
   124,   125,   125,   126,   127,   128,   129,   130,   135,   133,
   135,    67,   135,   135,    73,     4,   133,    71,    71,    68,
   68,    68,    71,   137,    68,    70,    73,    56,     4,    67,
   102,   104,   105,   106,   133,   123,   123,   112,   132,    68,
   102,   134,     9,     9,   106,    68,    68,    68,    73,    68,
   135,    15,    20,    21,    22,    56,    60,    61,    62,    67,
   69,    75,    76,    82,    86,    87,    88,   111,   112,   139,
   140,   141,   143,   144,   145,   146,   147,   148,   149,   150,
   151,   152,   153,   154,   155,   156,   158,   158,     4,   135,
   134,     4,    67,    67,    67,   158,   102,   143,   143,   143,
   64,    67,    69,    74,    87,    88,    44,    45,    46,    47,
   48,    49,    50,    51,    52,    53,    59,   157,   143,    62,
   63,    75,    60,    61,    77,    78,    57,    58,    79,    80,
   43,    81,    82,    83,    84,    39,    38,    71,   135,   135,
   123,   123,   143,   146,    68,    70,   112,   142,   156,   158,
   56,   156,   143,   143,   143,   145,   145,   146,   146,   147,
   147,   147,   147,   148,   148,   149,   150,   151,   152,   153,
   156,     4,    71,    71,    68,    68,    71,    70,   123,   123,
   156,    68,    68
};

const unsigned char
parser::yyr1_[] =
{
   0,    92,    93,    94,    95,    96,    96,    96,    97,    97,
   98,    98,    98,    98,    98,    98,    98,    98,    98,    99,
   100,   100,   100,   100,   101,   101,   102,   102,   103,   103,
   104,   105,   105,   106,   106,   106,   107,   107,   108,   109,
   109,   109,   109,   110,   111,   111,   111,   112,   112,   113,
   113,   114,   114,   114,   114,   114,   114,   114,   114,   114,
   114,   114,   114,   114,   114,   115,   115,   115,   115,   115,
   115,   115,   115,   115,   116,   116,   117,   117,   118,   118,
   118,   118,   118,   118,   118,   118,   118,   118,   119,   119,
   120,   120,   120,   120,   120,   121,   122,   122,   122,   122,
   123,   123,   123,   124,   124,   124,   125,   125,   125,   125,
   125,   126,   126,   126,   127,   127,   128,   128,   129,   129,
   130,   130,   131,   131,   132,   132,   133,   133,   134,   134,
   134,   134,   134,   134,   134,   134,   134,   134,   134,   135,
   135,   136,   136,   136,   137,   138,   138,   139,   139,   139,
   139,   139,   139,   140,   140,   141,   141,   141,   141,   141,
   141,   141,   141,   141,   142,   142,   143,   143,   143,   143,
   143,   144,   144,   144,   144,   144,   144,   145,   145,   145,
   145,   146,   146,   146,   147,   147,   147,   148,   148,   148,
   148,   148,   149,   149,   149,   150,   150,   151,   151,   152,
   152,   153,   153,   154,   154,   155,   156,   156,   157,   157,
   157,   157,   157,   157,   157,   157,   157,   157,   157,   158,
   158
};

const signed char
parser::yyr2_[] =
{
   0,     2,     1,     1,     1,     1,     2,     1,     1,     2,
   1,     3,     3,     5,     5,     1,     1,     1,     1,     1,
   1,     3,     4,     3,     1,     2,     1,     3,     9,     9,
   0,     1,     2,     1,     4,     6,    10,     9,     5,     4,
   4,     4,     4,     1,     1,     1,     1,     1,     1,     1,
   2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
   1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
   3,     3,     3,     1,     3,     3,     6,     6,     1,     1,
   4,     3,     4,     3,     1,     4,     4,     4,     1,     2,
   1,     1,     1,     1,     1,     1,     1,     3,     3,     3,
   1,     3,     3,     1,     3,     3,     1,     3,     3,     3,
   3,     1,     3,     3,     1,     3,     1,     3,     1,     3,
   1,     3,     1,     3,     1,     5,     1,     3,     1,     1,
   1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
   3,     1,     3,     4,     5,     1,     3,     1,     1,     1,
   1,     3,     3,     6,     6,     1,     4,     4,     3,     2,
   2,     3,     1,     4,     1,     3,     1,     2,     2,     2,
   2,     1,     1,     1,     1,     1,     1,     1,     3,     3,
   3,     1,     3,     3,     1,     3,     3,     1,     3,     3,
   3,     3,     1,     3,     3,     1,     3,     1,     3,     1,
   3,     1,     3,     1,     3,     1,     1,     3,     1,     1,
   1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
   3
};




#if YYDEBUG
const short
parser::yyrline_[] =
{
   0,    93,    93,    96,    97,    99,    99,    99,   101,   101,
   103,   104,   105,   106,   107,   108,   109,   110,   111,   113,
   115,   116,   117,   118,   120,   120,   122,   122,   126,   127,
   128,   129,   129,   130,   131,   132,   136,   137,   140,   143,
   144,   145,   146,   149,   152,   152,   152,   155,   155,   157,
   157,   159,   160,   161,   162,   163,   164,   165,   166,   167,
   168,   169,   170,   171,   172,   174,   175,   176,   177,   178,
   179,   180,   181,   182,   184,   185,   187,   188,   190,   191,
   192,   193,   194,   195,   196,   197,   198,   199,   201,   201,
   203,   203,   203,   203,   203,   205,   207,   208,   209,   210,
   212,   213,   214,   216,   217,   218,   220,   221,   222,   223,
   224,   226,   227,   228,   230,   231,   233,   234,   236,   237,
   239,   240,   242,   243,   245,   246,   248,   249,   251,   252,
   252,   252,   252,   252,   253,   253,   253,   254,   254,   256,
   257,   259,   260,   261,   263,   265,   265,   270,   271,   272,
   273,   274,   275,   278,   279,   282,   283,   284,   285,   286,
   287,   288,   289,   290,   293,   294,   297,   298,   299,   300,
   301,   303,   303,   303,   303,   303,   303,   305,   306,   307,
   308,   310,   311,   312,   314,   315,   316,   318,   319,   320,
   321,   322,   324,   325,   326,   328,   329,   331,   332,   334,
   335,   337,   338,   340,   341,   343,   345,   346,   348,   349,
   349,   349,   349,   349,   350,   350,   350,   351,   351,   353,
   354
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
namespace yy { std::array<bool, yyruletype::yynrules> rules = {false}; }
std::array<Node*,8> nodes = {nullptr};

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
   if (!n) { return; }
   n->Accept(me);
   const int N = n->Number(); // N = SN | RN
   assert(N > 0);
   if (n->IsRule())
   {
      if (N > yynrules) { DBG("\n\033[31m[rule] N:%d/%d",N,yynrules); }
      assert(N <= yynrules);
   }
   if (n->IsToken())
   {
      constexpr int YYNTOKENS = yy::parser::YYNTOKENS;
      if (N >= YYNTOKENS) { DBG("\n\033[31m[token] N:%d/%d",N,YYNTOKENS); }
      assert(N < YYNTOKENS);
   }
   if (n->IsRule()) { yy::rules.at(N) = true; } // Set the state flags
   // If n->dfs.down does not stop us from previous Accept, dfs down
   if (n->dfs.down && n->child)
   {
      dfs(n->child, me);
      // If dfs.n has changed, re-run a dfs with it
      if (n->dfs.n != n) { dfs(n->dfs.n, me); }
   }
   if (n->next) { dfs(n->next, me); }
   if (n->IsRule()) { yy::rules.at(N) = false; } // Reset the state flags
   if (n->IsRule()) {n->Accept(me, false);} // up, only for rules
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

// *****************************************************************************
void yy::parser::error(const location_type&, const std::string& msg)
{
   std::cerr << (*ufl.loc) << ": " << msg << std::endl;
   abort();
}
