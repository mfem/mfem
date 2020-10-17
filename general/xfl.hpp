#ifndef MFEM_XFL_HPP
#define MFEM_XFL_HPP

#include <list>
#include <array>
#include <string>
#include <memory>
#include <iostream>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cassert>
#include <cstring>

// *****************************************************************************
#define DBG(...) { printf("\033[32m");printf(__VA_ARGS__);printf("\033[m");fflush(0);}

// *****************************************************************************
#ifndef XFL_C
// *****************************************************************************
struct Node
{
   bool keep {false};
   int sn, id{0}, nb_siblings{0};
   Node *next {nullptr}, *children {nullptr}, *parent {nullptr};
   Node(const int sn): sn(sn) {}
   virtual ~Node() {}
   virtual void Apply(struct Middlend&, bool&, Node** = nullptr) = 0;
   virtual const int Number() const = 0;
   virtual const int SymbolNumber() = 0;
   virtual const std::string Name() const = 0;
   virtual const bool IsRule() const = 0;
   virtual const bool IsToken() const = 0;
};

// *****************************************************************************
template<int RN> class Rule : public Node
{
   std::string rule;
public:
   Rule(const int sn, const char *rule): Node(sn), rule(rule) {}
   void Apply(struct Middlend&, bool&, Node** = nullptr);
   const int Number() const { return RN; }
   const bool IsRule() const { return true; }
   const bool IsToken() const { return false; }
   const int SymbolNumber() { return sn; }
   const std::string Name() const { return rule; }
};

// *****************************************************************************
template<int TK> class Token : public Node
{
   std::string token;
public:
   Token(const int sn, const char *token): Node(sn), token(token) {}
   void Apply(struct Middlend&, bool&, Node** = nullptr);
   const int Number() const { return TK; }
   const bool IsRule() const { return false; }
   const bool IsToken() const { return true; }
   const int SymbolNumber(); // YYTOKENSHIFT(TK)
   const std::string Name() const { return token; }
};

// *****************************************************************************
class xfl
{
public:
   Node *root;
   bool trace_parsing;
   bool trace_scanning;
   std::string i_filename, o_filename;
public:
   xfl(): root(nullptr), trace_parsing(false), trace_scanning(false) {}
   void ll_open();
   void ll_close();
   int yy_parse(const std::string&);
   Node **Root() { return &root;}
};

// *****************************************************************************
namespace yy
{
extern int debug;
extern bool echo;
void dfs(Node*, struct Middlend&);
} // yy

// *****************************************************************************
Node *astAddNode(std::shared_ptr<Node>);
void yyerror(Node**, char const *message);
#else
// *****************************************************************************
namespace yy
{

extern int debug;
extern bool echo;

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

inline bool is_rule(const int sn) { return sn >= yy::ntokens(); }

} // yy

#ifdef OWN_VTABLE
// *****************************************************************************
struct Node;
struct Middlend;

// *****************************************************************************
struct Node
{
   bool keep;
   Node *next, *children, *parent;
   int sn, id, nb_siblings;

   void *t;
   const bool (*IsRule_p)();
   const bool (*IsToken_p)();
   const int (*Number_p)();
   const int (*SymbolNumber_p)();
   const std::string (*Name_p)(void*);
   void (*Apply_ptr)(void*, struct Middlend&, bool&, Node**);
   void (*Delete_p)(void*);

   template<class T> Node(T const &t):
      keep(false), next(nullptr), children(nullptr), parent(nullptr),
      id(0), nb_siblings(0),
      t(new T(t)),
      IsRule_p(&t.IsRule),
      IsToken_p(&t.IsToken),
      Number_p(&t.Number),
      SymbolNumber_p(&t.SymbolNumber),
      Name_p(&t.Name),
      Apply_ptr(&t.Apply),
      Delete_p(&t.Delete)
   {
      sn = SymbolNumber();
      //const bool (*IsRule)() noexcept = &t.IsRule;
   }
   inline const bool IsRule() { return IsRule_p(); }
   inline const bool IsToken() const { return IsToken_p(); }
   inline const int Number() { return Number_p(); }
   inline const int SymbolNumber() { return SymbolNumber_p(); }
   inline const std::string Name() { return Name_p(t); }
   inline void Apply(struct Middlend &ir, bool &dfs, Node* *extra)
   { Apply_ptr(t, ir, dfs, extra); }
   ~Node() { Delete_p(t); }
};

// *****************************************************************************
template<int RN> struct Rule
{
   int sn;
   std::string rule;
   Rule(const int sn, const char *rule): sn(sn), rule(rule) { assert(sn == SymbolNumber()); }
   /////////////////////////////////////////////////////////////////////////////
   static inline const int Number() { return RN; }
   static inline const bool IsRule() { return true; }
   static inline const bool IsToken() { return false; }
   static inline const int SymbolNumber() { return yy::r1(RN); }
   static inline const std::string Name(void *t)
   { return static_cast<Rule<RN>*>(t)->rule; }
   static inline void Apply(void*, struct Middlend&, bool&, Node**);
   static void Delete(void *t) { delete static_cast<Rule<RN>*>(t); }
};

// *****************************************************************************
template<int TK> struct Token
{
   int sn;
   std::string name, text;
   Token(const int sn, const char *name, const char *text):
      sn(sn), name(name), text(text) { assert(sn == SymbolNumber()); }
   const std::string Text() const { return text; }
   /////////////////////////////////////////////////////////////////////////////
   static inline const int Number() { return TK; }
   static inline const bool IsRule() { return false; }
   static inline const bool IsToken() { return true; }
   static inline const int SymbolNumber();
   static inline const std::string Name(void *t)
   { return static_cast<Token<TK>*>(t)->name; }
   static inline void Apply(void*, struct Middlend&, bool&, Node**);
   static void Delete(void *t) { delete static_cast<Token<TK>*>(t); }
};

#else //////////////////////////////////////////////////////////////////////////
// *****************************************************************************
struct Node
{
   bool keep;
   Node *next, *children, *parent;
   int sn, id, nb_siblings;
   Node(const int sn): keep(false),
      next(nullptr), children(nullptr), parent(nullptr),
      sn(sn), id(0), nb_siblings(0) {}
   virtual ~Node() {}
   virtual void Apply(struct Middlend&, bool&, Node** = nullptr) = 0;
   virtual const int Number() const = 0;
   virtual const int SymbolNumber() = 0;
   virtual const std::string Name() const = 0;
   virtual const bool IsRule() const = 0;
   virtual const bool IsToken() const = 0;
};

// *****************************************************************************
template<int RN> class Rule : public Node
{
   std::string rule;
public:
   Rule(const int sn, const char *rule): Node(sn), rule(rule) {}
   void Apply(struct Middlend&, bool&, Node** = nullptr);
   const int Number() const { return RN; }
   const bool IsRule() const { return true; }
   const bool IsToken() const { return false; }
   const int SymbolNumber() { return yy::r1(RN); }
   const std::string Name() const { return rule; }
};

// *****************************************************************************
template<int TK> class Token : public Node
{
   std::string name, text;
public:
   Token(const int sn, const char *name, const char *text):
      Node(sn), name(name), text(text) {}
   void Apply(struct Middlend&, bool&, Node** = nullptr);
   const int Number() const { return TK; }
   const bool IsRule() const { return false; }
   const bool IsToken() const { return true; }
   const int SymbolNumber();
   const std::string Name() const { return name; }
   const std::string Text() const { return text; }
};
#endif

// *****************************************************************************
namespace yy
{

void dfs(Node*, struct Middlend&);

} // yy

// *****************************************************************************
int yylex(void);
void yyerror(Node**, char const *message);

// *****************************************************************************
Node *astAddNode(std::shared_ptr<Node>);

// *****************************************************************************
typedef enum { OPTION_HELP = 0x445ECB9D } XFL_OPTION;
#endif

#endif // MFEM_XFL_HPP
