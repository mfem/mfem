#ifndef MFEM_XFL_HPP
#define MFEM_XFL_HPP

#include <list>
#include <array>
#include <string>
#include <memory>
#include <iostream>

#include <cassert>
#include <cstring> // strcat
#include <getopt.h>

// *****************************************************************************
#define DBG(...) { printf("\033[32m");printf(__VA_ARGS__);printf("\033[m");}

// *****************************************************************************
struct Node
{
   bool keep;
   Node *next, *children, *parent;
   int sn, id, nb_siblings;
   Node(const int);
   virtual ~Node();
   virtual void Apply(struct Middlend&, bool&, Node** = nullptr) = 0;
   virtual const int Number() const = 0;
   virtual const int SymbolNumber() const = 0;
   virtual const std::string Name() const = 0;
   virtual const bool IsRule() const = 0;
   virtual const bool IsToken() const = 0;
};

// *****************************************************************************
int yylex(void);
void yyerror(Node**, char const *message);
extern int yydebug;
extern bool yyecho;

// *****************************************************************************
template<int RN> class Rule : public Node
{
   std::string rule;
public:
   Rule(const int sym_num, const char *rule): Node(sym_num), rule(rule) {}
   void Apply(struct Middlend&, bool&, Node** = nullptr);
   const int Number() const { return RN; }
   const int SymbolNumber() const;
   const std::string Name() const { return rule; }
   const bool IsRule() const { return true; }
   const bool IsToken() const { return false; }
};

// *****************************************************************************
template<int TK = 0> class Token : public Node
{
   std::string name, text;
public:
   Token(const int sn, const char *name, const char *text):
      Node(sn), name(name), text(text) {}
   void Apply(struct Middlend&, bool&, Node** = nullptr);
   const int Number() const { return TK; }
   const int SymbolNumber() const { return Number(); }
   const std::string Name() const { return name; }
   const bool IsRule() const { return false; }
   const bool IsToken() const { return true; }
   const std::string Text() const { return text; }
};

// *****************************************************************************
namespace yy
{

const int undef();

// yyn is the number of a rule to reduce with
const int ntokens();

// YYR1[yyn] -- Symbol number of symbol that rule YYN derives
const int r1(int yyn);

// YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.
const int r2(int yyn);

signed char Translate(int);

const char* const SymbolName(int);

inline bool is_token(const int sn) { return sn < yy::ntokens(); }
inline bool is_token(const Node *n) { return is_token(n->sn); }

inline bool is_rule(const int sn) { return sn >= yy::ntokens(); }
inline bool is_rule(const Node *n) { return is_rule(n->sn); }

template<int> void rhs(Node**, int, Node**);

} // yy

// *****************************************************************************
template<typename D> struct Backend
{
   std::ostream &out;

   Backend(std::ostream &out): out(out) {}
   D const& that() const { return static_cast<const D&>(*this); }

   template<int TK>
   void token(std::string text) const { that().template token<TK>(text); }

   template<int RN>
   void rule(bool &down, Node **extra) const { that().template rule<RN>(down, extra); }
};

// *****************************************************************************
struct CPU: Backend<CPU>
{
   CPU(std::ostream &out): Backend<CPU>(out) {}

   template<int TK> void token(std::string text) const
   {
      out << " " << text; // output the token
      const int sn = yy::Translate(TK);
      const char *const name = yy::SymbolName(sn);
      if (yyecho) { DBG("\033[m => \033[37mDefault %s\n", name); }
   }

   template<int RN> void rule(bool&, Node**) const
   {
      const int sn = yy::r1(RN);
      const char *const rule = yy::SymbolName(sn);
      if (yyecho) { DBG("\033[37m => %s\n",rule); }
   }
};

// *****************************************************************************
struct GPU: Backend<GPU>
{
   GPU(std::ostream &out): Backend<GPU>(out) {}
   template<int> void token(std::string) const { assert(false); }
   template<int> void rule(bool&, Node**) const { assert(false); }
};

// *****************************************************************************
struct Middlend
{
   Backend<CPU> *cpu;
   Backend<GPU> *gpu;

   explicit Middlend(Backend<CPU> *dev): cpu(dev), gpu(nullptr) {}
   explicit Middlend(Backend<GPU> *dev): cpu(nullptr), gpu(dev) {}

   template<int SN> void middlend(Token<SN> *t) const
   {
      if (cpu) { cpu->template token<SN>(t->Text()); }
      if (gpu) { gpu->template token<SN>(t->Text()); }
   }

   template<int RN> void middlend(Rule<RN>*, bool &dfs, Node **extra) const
   {
      if (cpu) { cpu->template rule<RN>(dfs, extra); }
      if (gpu) { gpu->template rule<RN>(dfs, extra); }
   }
};

// *****************************************************************************
struct cpu: Middlend
{
   CPU dev;
   cpu(std::ostream &out): Middlend(&dev), dev(out) {}
};

struct gpu: Middlend
{
   GPU dev;
   gpu(std::ostream &out): Middlend(&dev), dev(out) {}
};

// *****************************************************************************
void dfs(Node*, struct Middlend&);

// *****************************************************************************
using Node_ptr = std::shared_ptr<Node>;
Node *astAddNode(Node_ptr);

template<int T> Node* astNewToken(const int, const char*, const char*);
template<int T> Node* astNewRule(const int, const char*);
Node* astAddChild(Node*, Node*);
Node* astAddNext(Node*, Node*);

// *****************************************************************************
typedef enum { OPTION_HELP = 0x445ECB9D } XFL_OPTION;

#endif // MFEM_XFL_HPP
