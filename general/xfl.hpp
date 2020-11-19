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
#include <iostream>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cassert>
#include <cstring>

// *****************************************************************************
#define DBG(...) {\
    printf("\033[32m"); printf(__VA_ARGS__); printf("\033[m"); fflush(0); }

namespace mfem { namespace internal { struct Middlend; } }

// *****************************************************************************
struct Node
{
   int n;
   std::string name;
   bool keep {false};
   int id {0}, nnext {0};
   Node *next {nullptr}, *child {nullptr}, *root {nullptr};
   Node(int n, int sn, const char *name): n(n), name(name) {}
   const int Number() const { return n; }
   const std::string Name() const { return name; }
   virtual ~Node() {}
   virtual void Apply(mfem::internal::Middlend&, bool&, Node** = nullptr) = 0;
   virtual const bool IsRule() const = 0;
   virtual const bool IsToken() const = 0;
};

// *****************************************************************************
template<int RN> struct Rule : public Node
{
   Rule(int rn, int sn, const char *name): Node(rn, sn, name) {}
   void Apply(mfem::internal::Middlend&, bool&, Node** = nullptr);
   const bool IsRule() const { return true; }
   const bool IsToken() const { return false; }
};

// *****************************************************************************
template<int TK> struct Token : public Node
{
   Token(int tk, int sn, const char *name): Node(tk, sn, name) {}
   void Apply(mfem::internal::Middlend&, bool&, Node** = nullptr);
   const bool IsRule() const { return false; }
   const bool IsToken() const { return true; }
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
   Node* &Root() { return root;}
};

// *****************************************************************************
namespace yy
{
extern int debug;
extern bool echo;
} // yy

// *****************************************************************************
Node *astAddNode(std::shared_ptr<Node>);
void yyerror(Node**, char const*);

// *****************************************************************************
void dfs(Node*, mfem::internal::Middlend&);
bool HitRule(const int, Node*);
bool HitToken(const int, Node*);
bool OnlyToken(const int, Node*);

#endif // MFEM_XFL_HPP
