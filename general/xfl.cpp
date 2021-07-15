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
#include <fstream>
#include <algorithm>

#include "xfl.hpp"
#include "xfl.Y.hpp"
#include "xfl_mid.hpp"
#include "xfl_ker.hpp"

// *****************************************************************************
extern "C" const __attribute__((aligned(4))) unsigned char xfl_mfem_hpp[];
extern "C" const __attribute__((aligned(4))) unsigned char *xfl_mfem_hpp_end;
extern "C" const unsigned int xfl_mfem_hpp_size;

// *****************************************************************************
template <typename T = long double>
constexpr T XFL_version(T x = 1.0, T eps = 1.e-15L)
{
   return x > 2 ? 0 :
          (x*(x-1.0L) >= (1.0L-eps)) && (x*(x-1.0L) <= (1.0L+eps)) ? x :
          XFL_version(1.L+1.L/x, eps);
}
#define XFL_N 1
#define XFL_alpha -1.0
#define XFL_ver XFL_version() XFL_alpha
#define XFL_str(n) #n
#define XFL_man(n) "[1;36m[XFL] version %." XFL_str(n) "Lf[m\n"

// *****************************************************************************
int AstSave(const char*, Node*, const int);

// *****************************************************************************
int main(int argc, char *argv[])
{
   int ast_debug = 2;     // [0-2] debug modes
   bool ast_save = false;
   bool yy_debug = false;
   bool ll_debug = false;
   bool ceed_benchmark = false;
   std::string input, output;

   if (argc == 1) { exit(~0 | fprintf(stderr, XFL_man(XFL_N), XFL_ver)); }

   for (int i = 1; i < argc; ++i)
   {
      if (argv[i] == std::string ("-p")) { yy_debug = true; }
      else if (argv[i] == std::string ("-s")) { ll_debug = true; }
      else if (argv[i] == std::string ("-i")) { input.assign(argv[++i]); }
      else if (argv[i] == std::string ("-o")) { output.assign(argv[++i]); }
      else if (argv[i] == std::string ("-t")) { ast_save = true; }
      else if (argv[i] == std::string ("-b")) { ceed_benchmark = true; }
      else if (argv[i] == std::string ("-n")) { ast_debug = atoi(argv[++i]); }
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

   xfl ufl(yy_debug, ll_debug, input, output, ceed_benchmark);

   constexpr auto os = std::ios::out | std::ios::binary | std::ios::trunc;
   std::ofstream o_file(output.c_str(), os);
   if (is_o_file) { assert(o_file.is_open()); }

   std::ostream &out = is_o_file ? o_file : std::cout;

   if (ufl.open() != 0) { return EXIT_FAILURE; }

   if (ufl.parse() != 0) { return EXIT_FAILURE; }

   if (ufl.close() != 0) { return EXIT_FAILURE; }

   if (ufl.code(out) != 0 ) { return EXIT_FAILURE; }

   if (ast_save) { AstSave("ast", ufl.root, ast_debug); }

   return EXIT_SUCCESS;
}

// *****************************************************************************
static bool Simplify(const Node *n, const int debug = 2)
{
   if (debug == 0) { return false; } // keep all rules if requested
   if (n->IsToken() && n->n == TOK::NL) { return true; } // remove NL
   if (n->child && n->child->root != n) { return false; } // keep links
   if (n->IsRule() && n->n == extra_status_rule_transpose_xt) { return false; } // keep transpose XT
   if (n->IsRule() && n->n == extra_status_rule_eval_xt) { return false; } // keep eval XT
   if (n->IsRule() && n->n == grad_expr_grad_op_form_args) { return false; } // keep grad XT
   if (n->IsRule() && n->n == extra_status_rule_dot_xt) { return false; } // keep dot XT
   if (n->IsRule() && n->n == statement_nl) { return true; } // remove NL
   if (n->IsRule() && n->n == statements_statement) { return true; } // remove end of statement list
   if (n->IsRule() && n->n == statements_statements_statement) { return true; } // remove statements list
   if (n->IsToken()) { return false; } // keep all tokens
   if (debug == 1 && n->nnext > 1) { return false; } // keep when siblings
   if (n->nnext == 0) { assert(!n->root); return false; } // keep entry_point
   const bool has_child = n->child != nullptr; // %empty
   const bool is_child_a_rule = has_child ? n->child->IsRule() : false;
   return is_child_a_rule && n->child->nnext == 1;
}

// *****************************************************************************
static void astTreeAddNewNode(FILE *file, Node *n, int &id)
{
   const bool link = n->child && n->child->root != n;
   constexpr char const *GRAY = "FFAAAA"; // link
   constexpr char const *BLUE = "CCCCDD"; // token
   constexpr char const *GREEN = "CCDDCC"; // rule with children
   constexpr char const *ORANGE = "FFDDCC"; // rule with one child
   const char *color =
      link ? GRAY : n->IsRule() ? Simplify(n) ? ORANGE : GREEN : BLUE;
   char const *NODE = "\n\tNode_%d [label=\"%s\" color=\"#%s\"]";
   std::replace(n->name.begin(), n->name.end(), '\"', '\'');
   fprintf(file, NODE, n->id = id++, n->Name().c_str(), color);
}

// *****************************************************************************
static int astTreeSaveNodes(FILE* file, Node *n, const int debug, int id)
{
   for (; n; n = n->next)
   {
      if (!Simplify(n, debug))
      {
         astTreeAddNewNode(file, n, id);
         // if our child is not our's, continue with next
         if (n->child && n->child->root != n) { continue; }
      }
      id = astTreeSaveNodes(file, n->child, debug, id);
   }
   return id;
}

// *****************************************************************************
static void astTreeSaveEdges(FILE* file, const Node *n,
                             const Node *root, const int debug)
{
   for (; n; n = n->next)
   {
      if (!Simplify(n, debug))
      {
         const Node *from = root;
         while (Simplify(from, debug)) { from = from->root; }
         constexpr char const * EDGE = "\n\tNode_%d -> Node_%d [%s];";
         fprintf(file, EDGE, from->id, n->id, "style=solid");
         if (n->child && n->child->root != n)
         {
            const Node *to = n->child;
            while (Simplify(to, debug)) { to = to->child; }
            fprintf(file, EDGE, n->id, to->id, "style=dashed");
            continue;
         }
      }
      astTreeSaveEdges(file, n->child, n, debug);
   }
}

// *****************************************************************************
static void setNNext(const Node *n)
{
   int nnext = 0;
   for (Node *c = n->child; c; c = c->next) { nnext += 1; }
   for (Node *c = n->child; c; c = c->next) { c->nnext = nnext; }
}

// *****************************************************************************
static void dfsSetNNext(const Node *n)
{
   if (!n) { return; }
   if (n->root && n->nnext == 0) { setNNext(n->root); }
   (dfsSetNNext(n->child), dfsSetNNext(n->next));
}

// *****************************************************************************
int AstSave(const char* filename, Node *root, const int debug)
{
   FILE *file;
   char fname[FILENAME_MAX];
   // Computes 'nnext' of each node
   dfsSetNNext(root);
   sprintf(fname, "%s.dot", filename);
   // Saving tree file
   if ((file = fopen(fname, "w")) == 0) { return EXIT_FAILURE; }
   constexpr char const *DIGRAPH = "digraph {\n\tordering=out;";
   constexpr char const *NODE_STYLE = "\n\tNode [style=filled, shape=circle];";
   fprintf(file, "%s%s", DIGRAPH, NODE_STYLE);
   // Save all nodes as Rules or Tokens, filling `id` by the way
   astTreeSaveNodes(file, root, debug, /*id=*/ 0);
   // Save all edges
   astTreeSaveEdges(file, root->child, root, debug);
   fprintf(file, "\n}\n");
   fclose(file);
   return EXIT_SUCCESS;
}

// ****************************************************************************
Node *xfl::NewNode(Node_sptr n) const
{
   assert(n);
   static std::list<Node_sptr> node_list_to_be_destructed;
   node_list_to_be_destructed.push_back(n);
   return n.get();
}

// ****************************************************************************
void xfl::InsertNode(Node *root, Node *child)
{
   root->root = child->root;
   root->child = child;

   child->root->child = root;
   child->root = root;
}

// *****************************************************************************
xfl::xfl(bool yy_debug, bool ll_debug,
         std::string &input, std::string &output,
         bool ceed_benchmark):
   root(nullptr),
   loc(new yy::location(&input)),
   yy_debug(yy_debug), ll_debug(ll_debug), ceed_benchmark(ceed_benchmark),
   input(input), output(output) { }

// *****************************************************************************
int xfl::parse(void)
{
   yy::parser parser(*this);
   parser.set_debug_level(yy_debug);
   const int result = parser();
   assert(root);
   assert(root->child);
   root->child->root = root;
   return result;
}

// *****************************************************************************
xfl::~xfl() { delete loc; }


// *****************************************************************************
int xfl::code(std::ostream &out)
{
   // Transformation which turns forms inner to PA mult rules
   mfem::internal::DotPA pa(*this, out);
   DfsPreOrder(root, pa);

   // Transformation to add the extra_status_rule_dom_dx rule
   mfem::internal::DomDx dx(*this, out);
   DfsPreOrder(root, dx);

   // Then code generation
   std::ostringstream main;
   mfem::internal::Code mc(*this, main, ceed_benchmark);
   DfsPreOrder(root, mc);

   // Static kernels
   std::ostringstream kernels;
   mfem::internal::StaticKernels sk(*this, kernels);
   DfsPreOrder(root, sk);

   out << xfl_mfem_hpp << std::endl;
   out << kernels.str() << std:: endl;
   out << main.str() << std:: endl;

   return EXIT_SUCCESS;
}
