// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include <string>
#include <cctype>
#include <stack>
#include <string>
#include <memory>
#include <thread>

#define DBG_COLOR ::debug::kMagenta
#include "debug.hpp"

#include "error.hpp"

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
// #define MTK_PRIVATE_IMPLEMENTATION
// #define CA_PRIVATE_IMPLEMENTATION
#include "metal.h"

#include "metal.hpp"

//// ///////////////////////////////////////////////////////////////////////////
namespace NS
{
static NS::Error *error = nullptr;

inline void Check(const bool test)
{
   if (test) { return; }
   __builtin_printf("\033[31m%s\033[m\n",
                    error->localizedDescription()->utf8String());
   fflush(nullptr);
   assert(false);
}

} // namespace NS

//// ///////////////////////////////////////////////////////////////////////////
namespace mfem
{

namespace metal
{

/// ///////////////////////////////////////////////////////////////////////////
template <typename T, typename... Types>
struct Base : public Base<T>, public Base<Types...>
{
   using Base<T>::Accept;
   using Base<Types...>::Accept;
};
template <typename T> struct Base<T>
{
   virtual void Accept(T &) = 0;
   virtual bool IsOps() { return false; };
};

template <typename T, typename... Types>
struct Visitors : public Visitors<T>, public Visitors<Types...>
{
   using Visitors<T>::Visit;
   using Visitors<Types...>::Visit;
};
template <typename T> struct Visitors<T>
{
   virtual void Visit(T &) = 0;
};

struct Node;

struct Rule;
struct Token;
struct Const;
struct Scalar;

struct Visitor : public Visitors<Rule, Token, Const, Scalar> { };

/// ///////////////////////////////////////////////////////////////////////////
struct Node : public Base<Visitor>
{
   const char c;
   std::shared_ptr<Node> left, right;
   Node(const char c) : c(c), left(nullptr), right(nullptr) { }

   void Accept(Visitor &) override = 0;
   virtual ~Node() {}
};

/// ///////////////////////////////////////////////////////////////////////////
struct Rule : public Node
{
   Rule(const char c) : Node(c) {}
   void Accept(Visitor &me) override { me.Visit(*this); }
   bool IsOps() override { return true;}
};

struct Token : public Node
{
   Token(const char c) : Node(c) {}
   void Accept(Visitor &me) override { me.Visit(*this); }
};

struct Const : public Node
{
   Const(const char c) : Node(c) {}
   void Accept(Visitor &me) override { me.Visit(*this); }
};

struct Scalar : public Node
{
   Scalar(const char c) : Node(c) {}
   void Accept(Visitor &me) override { me.Visit(*this); }
};

using node_t = std::shared_ptr<Node>;

/// ///////////////////////////////////////////////////////////////////////////
[[maybe_unused]] static void Doze()
{
   std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

/// ///////////////////////////////////////////////////////////////////////////
// https://en.wikipedia.org/wiki/Shunting_yard_algorithm
static auto ShuntingYard(const std::string& input)
{
   std::string infix(input);

   auto priority = [](const char c) -> int
   {
      if (c == '=') { return 1; }
      if (c == '*' || c == '/') { return 3; }
      if (c == '+' || c == '-') { return 2; }
      return 0;
   };

   auto isOperator = [](const char c) -> bool
   {
      return (c == '+' || c == '-' || c == '*' || c == '/' || c == '^' || c == '=');
   };

   char op_eq = 0;
   std::stack<char> ops;
   std::stack<node_t> ast;

   auto NewRule = [&ast](char c) { ast.push(std::make_shared<Rule>(c)); };
   auto NewToken = [&ast](char c) { ast.push(std::make_shared<Token>(c)); };
   auto NewConst = [&ast](char c) { ast.push(std::make_shared<Const>(c)); };
   auto NewScalar = [&ast](char c) { ast.push(std::make_shared<Scalar>(c)); };

   auto NewOp = [&ast, &op_eq, NewRule,NewToken](char c)
   {
      // dbg("\033[33m[NewOp] c: {}", c), Doze();
      if (c == '=')
      {
         const auto top = ast.top(); ast.pop();
         NewToken('z'), ast.push(top);
         if (op_eq == 0) { op_eq = '='; }
      }
      if (ast.size() == 0) { op_eq = c; return; }
      assert(ast.size() > 1);
      const auto right = ast.top(); ast.pop();
      const auto left = ast.top(); ast.pop();
      NewRule(c);
      ast.top()->right = right, ast.top()->left = left;
   };

   for (char c : infix)
   {
      // dbg("new '{}'", c), Doze();
      if (isdigit(c)) { NewConst(c); }
      else if (std::isspace(c)) { continue; }
      else if (isalpha(c) && c >= 'x') { NewToken(c); }
      else if (isalpha(c) && c < 'x') { NewScalar(c); }
      else if (isOperator(c))
      {
         while (!ops.empty() && priority(ops.top()) >= priority(c))
         {
            NewOp(ops.top()), ops.pop();
         }
         ops.push(c);
      }
      else if (c == '(') { ops.push(c); }
      else if (c == ')')
      {
         while (!ops.empty() && ops.top() != '(') { NewOp(ops.top()), ops.pop(); }
         if (!ops.empty() && ops.top() == '(') {  ops.pop(); }
      }
      else
      {
         MFEM_ABORT("unsupported character");
      }
   }

   while (!ops.empty()) { NewOp(ops.top()), ops.pop(); }

   return std::make_tuple(ast.top(), op_eq);
}

/// ///////////////////////////////////////////////////////////////////////////
[[maybe_unused]] static void DfsPreOrder(const node_t &n, Visitor& m)
{
   if (!n) { return; }
   n->Accept(m);
   if (n->left) { DfsPreOrder(n->left, m); }
   if (n->right) { DfsPreOrder(n->right, m); }
}

/// ///////////////////////////////////////////////////////////////////////////
static void DfsInOrder(const node_t &n, Visitor& m)
{
   if (!n) { return; }
   if (n->left) { DfsInOrder(n->left, m); }
   n->Accept(m);
   if (n->right) { DfsInOrder(n->right, m); }
}

/// ///////////////////////////////////////////////////////////////////////////
[[maybe_unused]] static void DfsPostOrder(const node_t &n, Visitor& m)
{
   if (!n) { return; }
   if (n->left) { DfsPostOrder(n->left, m); }
   if (n->right) { DfsPostOrder(n->right, m); }
   n->Accept(m);
}

/// ///////////////////////////////////////////////////////////////////////////
void printAST(const node_t& node, int depth = 0)
{
   if (!node) { return; }

   for (int i = 0; i < depth; ++i) { std::cout << "  "; }
   std::cout << node->c << std::endl;

   printAST(node->left, depth + 1);
   printAST(node->right, depth + 1);
}

/// ///////////////////////////////////////////////////////////////////////////
struct KernelDump : public Visitor
{
   void Visit(Rule &n) override { dbg("\033[33mrule {}",n.c); }
   void Visit(Token &n) override { dbg("\033[33mtoken {}",n.c); }
   void Visit(Const &n) override { dbg("\033[33mconst {}",n.c); }
   void Visit(Scalar &n) override { dbg("\033[33mscalar {}",n.c); }
};

/// ///////////////////////////////////////////////////////////////////////////
struct KernelSignature : public Visitor
{
   const std::string name;
   std::ostringstream oss;

   KernelSignature(std::string name): name(name)
   {
      oss << "\n[[kernel]] void " << name << "(\n";
   }

   void Visit(Rule &) override { /* nothing to do */}

   void Visit(Token &n) override
   {
      if (n.c == 'z') { return; }
      dbg("token {}", n.c);
      oss << "\tdevice const float* " << n.c << ",\n";
   }

   void Visit(Scalar &n) override
   {
      dbg("const {}",n.c);
      oss << "\tconstant float& " << n.c << ",\n";
   }

   void Visit(Const &) override { /* nothing to do */ }

   std::string operator()()
   {
      oss << "\tdevice float* z,\n";
      oss << "\tconst uint i [[thread_position_in_grid]]){\n";
      return oss.str();
   }
};

/// ///////////////////////////////////////////////////////////////////////////
struct KernelBody : public Visitor
{
   const char op_eq;
   std::stack<std::string> stack;

   KernelBody(const char op_eq): op_eq(op_eq) { dbg("op_eq: {}", op_eq); }

   void Visit(Rule &n) override
   {
      dbg("Rule {}", n.c);
      assert(stack.size() > 1);
      std::string op1 = stack.top();
      stack.pop();
      std::string op2 = stack.top();
      stack.pop();

      std::string op;
      op = op2 + " ";
      if (n.c == '=' && op_eq != '=') { op += op_eq; }
      std::string eq(1,n.c);
      op += eq + " " + op1;

      if (n.c != '=') { stack.push("(" + op + ")"); }
      else { stack.push(op); }
   }

   void Visit(Token &n) override { stack.push(std::string(1, n.c) + "[i]"); }

   void Visit(Scalar &n) override { stack.emplace(1, n.c); }

   void Visit(Const &n) override { stack.emplace(1, n.c); }

   std::string operator()()
   {

      return std::string(op_eq == 0 ? "z[i] = ":"") + stack.top() + ";\n}";
   }
};

/// ///////////////////////////////////////////////////////////////////////////
std::string KernelOps(const char *name, const char *ops)
{
   dbg("Kernel '{}': \033[33m{}", name, ops);

   std::string infix {ops};
   auto [ast, op_eq] = ShuntingYard(infix);
   printAST(ast);

   KernelDump kd;
   // dbg("\033[32mPreOrder");
   // DfsPreOrder(ast, kd);

   // dbg("\033[32mInOrder");
   // DfsInOrder(ast, kd);

   dbg("\033[32mPostOrder");
   DfsPostOrder(ast, kd);


   KernelSignature ker_signature(name);
   KernelBody ker_body(op_eq);

   DfsInOrder(ast, ker_signature);
   DfsPostOrder(ast, ker_body);

   std::ostringstream oss;
   oss << ker_signature() << ker_body();

   std::string kernel = oss.str();
   dbg("Kernel: {}", kernel);
   // dbg("\033[31mEXIT"); std::exit(0);

   return kernel;
}

/// ////////////////////////////////////////////////////////////////////////////
setup_t KernelSetup(const char* name, const char *src)
{
   static auto *device = MTL::CreateSystemDefaultDevice();
#ifdef MFEM_USE_METAL_JIT
   const MTL::CompileOptions *options = nullptr;
   auto kernel_str = NS::String::string(src, NS::UTF8StringEncoding);
   auto library = device->newLibrary(kernel_str, options, &NS::error);
   NS::Check(library);
#else // MFEM_USE_METAL_JIT
   constexpr auto path = MFEM_SOURCE_DIR "/build/mfem.mlb";
   // constexpr auto path = MFEM_INSTALL_DIR "/mfem.mlb";
   const static auto filepath = NS::String::string(path, NS::ASCIIStringEncoding);
   static auto library = device->newLibrary(filepath, &NS::error);
   NS::Check(library);
#endif // MFEM_USE_METAL_JIT

   // function
   auto function = library->newFunction(
                      NS::String::string(name, NS::UTF8StringEncoding));
   assert(function);

   // kernel
   auto kernel = device->newComputePipelineState(function, &NS::error);
   assert(kernel);

   // queue
   auto Q = device->newCommandQueue();
   assert(Q);

   // commands
   auto commands = Q->commandBuffer();
   assert(commands);

   // encoder
   auto encoder = commands->computeCommandEncoder();
   assert(encoder);

   // enqueue the kernel
   encoder->setComputePipelineState(kernel);

   const auto MaxThreadsPerGroup = kernel->maxTotalThreadsPerThreadgroup();

   return std::make_tuple(device, commands, encoder, MaxThreadsPerGroup);
}

} // namespace metal

} // namespace mfem