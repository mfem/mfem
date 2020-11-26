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
#ifndef MFEM_XFL_HPP
#define MFEM_XFL_HPP

#include <list>
#include <array>
#include <string>
#include <memory>

// *****************************************************************************
template <typename T, typename... Types>
struct Atom : public Atom<T>, public Atom<Types...>
{
   using Atom<T>::Accept;
   using Atom<Types...>::Accept;
};
template <typename T> struct Atom<T>
{ virtual void Accept(T&, const bool = true) = 0; };

template <typename T, typename... Types>
struct Middlends : public Middlends<T>, public Middlends<Types...>
{
   using Middlends<T>::Visit;
   using Middlends<Types...>::Visit;
};
template <typename T> struct Middlends<T> { virtual void Visit(T&) = 0; };

// *****************************************************************************
class xfl;
class Debug;
struct Node;
struct Rule;
struct Token;
namespace yy { class location; }

// *****************************************************************************
struct Middlend : public Middlends<Rule, Token>
{
   xfl &ufl;
   Middlend(xfl &ufl) : ufl(ufl) { }
};

// *****************************************************************************
struct Node: public Atom<Middlend>
{
   int n;
   std::string name;
   bool keep {false};
   int id {0}, nnext {0};
   struct {bool down; Node *n;} dfs {true, nullptr};
   Node *next {nullptr}, *child {nullptr}, *root {nullptr};
   Node(int n, const char *name): n(n), name(name) {}
   const int Number() const { return n; }
   const std::string Name() const { return name; }
   const char *CStr() const { return name.c_str(); }
   virtual void Accept(Middlend&, const bool down = true) = 0;
   virtual const bool IsRule() const = 0;
   virtual const bool IsToken() const = 0;
   virtual ~Node() {}
};

// *****************************************************************************
struct Rule : public Node
{
   Rule(int rn, const char *name): Node(rn, name) {}
   void Accept(Middlend &me, const bool down = true)
   { dfs.down = down; dfs.n = this; me.Visit(*this); }
   const bool IsRule() const { return true; }
   const bool IsToken() const { return false; }
};

// *****************************************************************************
struct Token : public Node
{
   Token(int tk, const char *name): Node(tk, name) {}
   void Accept(Middlend &me, const bool down = true) { me.Visit(*this); }
   const bool IsRule() const { return false; }
   const bool IsToken() const { return true; }
};

// *****************************************************************************
class xfl
{
   using Node_sptr = std::shared_ptr<Node>;
public:
   Node *root;
   yy::location *loc;
   bool yy_debug;
   bool ll_debug;
   std::string &input, &output;
   struct {int debug; bool echo;} ll;
public:
   xfl(bool yy_debug, bool ll_debug, std::string &input, std::string &output);
   ~xfl();
   int open();
   int close();
   int parse(const std::string&, std::ostream&);
   int morph(std::ostream&);
   int code(std::ostream&);
   Node* &Root() { return root;}
public:
   Node *astAddNode(Node_sptr);
   void dfs(Node*, Middlend&);
   bool HitRule(const int, Node*);
   bool HitToken(const int, Node*);
   bool OnlyToken(const int, Node*);
};

// *****************************************************************************
#define DBG(...) { printf("\033[33m");  \
                   printf(__VA_ARGS__); \
                   printf(" \n\033[m"); \
                   fflush(0); }

#endif // MFEM_XFL_HPP
