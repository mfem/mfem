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
#include <fstream>

#include "xfl.hpp"
#include "xfl.Y.hpp"
#include "xfc.hpp"

// ****************************************************************************
extern yy::location xfl_location; // defined in xfl.ll

// ****************************************************************************
// * tr \" to \'
// ****************************************************************************
static const char* strKillQuote(const std::string &str)
{
   char *p = const_cast<char*>(str.c_str());
   for (; *p != 0; p++) if (*p == '\"') { *p = '\''; }
   return str.c_str();
}

// *****************************************************************************
static bool skip(const Node *n, const bool simplify)
{
   if (!simplify) { return false; }
   if (n->IsToken()) { return false; }
   if (n->nnext == 0) { return false; }
   if (n->keep) { return false; }
   const bool has_child = n->child != nullptr; // %empty
   const bool is_child_a_rule = has_child ? n->child->IsRule() : false;
   return is_child_a_rule && n->child->nnext == 1;
}

// *****************************************************************************
static int astTreeSaveNodes(FILE* fTreeOutput, Node *n,
                            const bool simplify, int id)
{
   for (; n not_eq NULL; n = n->next)
   {
      if (n->IsRule())
      {
         if (!skip(n, simplify))
            fprintf(fTreeOutput,
                    "\n\tNode_%d [label=\"%s\" color=\"#%s\"]",
                    n->id = id++,
                    strKillQuote(n->Name()),
                    skip(n, true) ? "FFDDCC" : "CCDDCC");
      }
      else
      {
         fprintf(fTreeOutput,
                 "\n\tNode_%d [label=\"%s\" color=\"#CCCCDD\"]",
                 n->id = id++,
                 strKillQuote(n->Name()));
      }
      id = astTreeSaveNodes(fTreeOutput, n->child, simplify, id);
   }
   return id;
}

// *****************************************************************************
static int astTreeSaveEdges(FILE* fTreeOutput,
                            const Node *n,
                            const Node *father,
                            const bool simplify)
{
   for (; n; astTreeSaveEdges(fTreeOutput, n->child, n, simplify),
        n = n->next)
   {
      if (skip(n, simplify)) { continue; }
      const Node *from = father;
      while (skip(from, simplify)) { from = from->root; }
      fprintf(fTreeOutput, "\n\tNode_%d -> Node_%d;", from->id, n->id);
   }
   return 0;
}

// *****************************************************************************
static void setNNext(const Node *n)
{
   int nnext = 0;
   for (Node *c = n->child; c; c = c->next) { nnext += 1; }
   for (Node *c = n->child; c; c = c->next) { c->nnext = nnext; }
}

// *****************************************************************************
static void astNNext(Node *n)
{
   if (!n) { return; }
   if (n->root && n->nnext == 0) { setNNext(n->root); }
   (astNNext(n->child), astNNext(n->next));
}

// *****************************************************************************
// * astTreeSave
// *****************************************************************************
int astTreeSave(const char* file_name, Node *root, const bool simplify)
{
   FILE *file;
   char fName[FILENAME_MAX];
   // ***************************************************************************
   astNNext(root);
   // ***************************************************************************
   sprintf(fName, "%s.dot", file_name);
   // Saving tree file
   if ((file = fopen(fName, "w")) == 0)
   {
      return -1 | printf("[astTreeSave] fopen ERROR");
   }
   fprintf(file,
           "digraph {\nordering=out;\n\tNode [style = filled, shape = circle];");

   astTreeSaveNodes(file, root, simplify, 0);
   if (astTreeSaveEdges(file, root->child, root, simplify) not_eq 0)
   {
      return -1 | printf("[astTreeSave] ERROR");
   }
   fprintf(file, "\n}\n");
   fclose(file);
   return 0;
}

// ****************************************************************************
Node *xfl::astAddNode(Node_sptr n_ptr)
{
   assert(n_ptr);
   static std::list<Node_sptr> node_list_to_be_destructed;
   node_list_to_be_destructed.push_back(n_ptr);
   return n_ptr.get();
}

// *****************************************************************************
xfl::xfl(bool yy_debug, bool ll_debug, std::string &input, std::string &output):
   root(nullptr), loc(new yy::location(&input)),
   yy_debug(yy_debug), ll_debug(ll_debug),
   input(input), output(output) { }

// *****************************************************************************
int xfl::parse(const std::string &f, std::ostream &out)
{
   yy::parser parser(*this);
   parser.set_debug_level(yy_debug);
   const int result = parser();
   assert(root);
   return result;
}

// *****************************************************************************
xfl::~xfl() { delete loc; }

// *****************************************************************************
int xfl::morph(std::ostream &out)
{
   // First transformation to add prefix/postfix to DOM_DX
   mfem::internal::DomDx dx(*this, out);
   dfs(root, dx);
   return EXIT_SUCCESS;
}

// *****************************************************************************
int xfl::code(std::ostream &out)
{
   // Then code generation
   mfem::internal::Code me(*this, out);
   dfs(root, me);
   return EXIT_SUCCESS;
}

// *****************************************************************************
namespace version
{

constexpr double gratio(int n) { return n <= 1 ? 1.0 : 1.0 + 1.0/gratio(n-1); }

template <typename T = long double>
constexpr T gdrt(T x = 1.0, T eps = 1.e-15L)
{
   return x > 2 ? 0 :
          (x*(x-1.0L) >= (1.0L-eps)) && (x*(x-1.0L) <= (1.0L+eps)) ? x :
          version::gdrt(1.L+1.L/x, eps);
}

} // cst namespace

// *****************************************************************************
// 1.6180339887498948482045868343656381L
#define XFL_N 1
#define XFL_pre -1.0
#define XFL_ver version::gdrt() XFL_pre
#define XFL_str(n) #n
#define XFL_man(n) "[1;36m[XFL] version %." XFL_str(n) "Lf[m\n"

// *****************************************************************************
int main (int argc, char *argv[])
{
   bool ast = false;
   bool yy_debug = false;
   bool ll_debug = false;
   std::string input, output;

   if (argc == 1) { exit(~0 | fprintf(stderr, XFL_man(XFL_N), XFL_ver)); }

   for (int i = 1; i < argc; ++i)
   {
      if (argv[i] == std::string ("-p")) { yy_debug = true; }
      else if (argv[i] == std::string ("-s")) { ll_debug = true; }
      else if (argv[i] == std::string ("-i")) { input.assign(argv[++i]); }
      else if (argv[i] == std::string ("-o")) { output.assign(argv[++i]); }
      else if (argv[i] == std::string ("-t")) { ast = true; }
      else { break; }
   }
   const std::string &last_argv = argv[argc-1];

   const bool is_i_file = !input.empty();
   const bool is_o_file = !output.empty();
   // Make sure either both or none are set
   assert (!(is_i_file ^ is_o_file));

   std::ifstream i_file(input.c_str(), std::ios::in | std::ios::binary);
   if (is_i_file) { assert(!i_file.fail() && i_file.is_open()); }
   else { input = last_argv; }

   xfl ufl(yy_debug, ll_debug, input, output);

   constexpr auto os = std::ios::out | std::ios::binary | std::ios::trunc;
   std::ofstream o_file(output.c_str(), os);
   if (is_o_file) { assert(o_file.is_open()); }

   std::ostream &out = is_o_file ? o_file : std::cout;

   if (ufl.open() != 0) { return EXIT_FAILURE; }

   if (ufl.parse(last_argv, out) != 0) { return EXIT_FAILURE; }

   if (ufl.close() != 0) { return EXIT_FAILURE; }

   if (ufl.morph(out) != 0 ) { return EXIT_FAILURE; }

   if (ufl.code(out) != 0 ) { return EXIT_FAILURE; }

   if (ast) { astTreeSave("ast", ufl.root, false); }

   return EXIT_SUCCESS;
}
