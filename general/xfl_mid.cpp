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
#include <algorithm>
#include <array>
#include <cctype>
#include <iostream>
#include <sstream>
#include <cstdlib> // getenv

using std::ostream;
using std::string;

#define MFEM_DEBUG_COLOR 118

#include "../mfem.hpp"
#include "xfl.Y.hpp"
// xfc/xfk headers use xfl.Y.hpp
#include "xfl_mid.hpp"
#include "xfl_ker.hpp"

#ifndef MFEM_USE_MPI
#define ParMesh Mesh
#define GetParMesh GetMesh
#define ParFiniteElementSpace FiniteElementSpace
#endif

// *****************************************************************************
#define QUOTE(...) #__VA_ARGS__
#define RAW(...) R"delimiter(#__VA_ARGS__)delimiter"
#define UTF8(...) u8"#__VA_ARGS__"

// *****************************************************************************
namespace yy
{
extern std::array<bool, yyruletype::yynrules> rules;
}

namespace mfem
{

namespace internal
{

/** @cond */  // Doxygen warning: documented symbol was not declared or defined

// *****************************************************************************
/// AST code generation
// *****************************************************************************
#define QUOTE(...) #__VA_ARGS__
static void ceed_benchmark_defines(std::ostringstream &out)
{
   // Parameters set on the compilation command line of the CEED benchmarks
   out << "#ifndef GEOM\n#define GEOM Geometry::CUBE\n#endif\n\n"
       << "#ifndef MESH_P\n#define MESH_P 1\n#endif\n\n"
       << "#ifndef SOL_P\n#define SOL_P 5\n#endif\n\n"
       << "#ifndef IR_ORDER\n#define IR_ORDER 0\n#endif\n\n"
       << "#ifndef IR_TYPE\n#define IR_TYPE 0\n#endif\n\n"
       << "#ifndef PROBLEM\n#define PROBLEM 0\n#endif\n\n"
       << "#ifndef VDIM\n#define VDIM 1\n#endif\n";
}

static void ceed_benchmark_options(std::ostringstream &out)
{
   // CEED benchmark options for output grep
   out << "\n\t"
       << QUOTE(
          assert(VDIM==1); // Scalar
          assert(MESH_P==1);
          assert(IR_TYPE==0);
          assert(IR_ORDER==0);
          assert(PROBLEM==0); // Diffusion
          assert(GEOM==Geometry::CUBE);
          const char *mesh_file = "../../data/hex-01x01x01.mesh";
          int ser_ref_levels = 4;
          int par_ref_levels = 0;
          Array<int> nxyz;
          int order = SOL_P;
          const char *basis_type = "G";
          bool static_cond = false;
          const char *pc = "none";
          bool perf = true;
          bool matrix_free = true;
          int max_iter = 50;
          bool visualization = false;
          OptionsParser args(argc, argv);
          args.AddOption(&mesh_file, "-m", "--mesh",
                         "Mesh file to use.");
          args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                         "Number of times to refine the mesh uniformly in serial.");
          args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                         "Number of times to refine the mesh uniformly in parallel.");
          args.AddOption(&nxyz, "-c", "--cartesian-partitioning",
                         "Use Cartesian partitioning.");
          args.AddOption(&order, "-o", "--order",
                         "Finite element order (polynomial degree) or -1 for"
                         " isoparametric space.");
          args.AddOption(&basis_type, "-b", "--basis-type",
                         "Basis: G - Gauss-Lobatto, P - Positive, U - Uniform");
          args.AddOption(&perf, "-perf", "--hpc-version", "-std", "--standard-version",
                         "Enable high-performance, tensor-based, assembly/evaluation.");
          args.AddOption(&matrix_free, "-mf", "--matrix-free", "-asm", "--assembly",
                         "Use matrix-free evaluation or efficient matrix assembly in "
                         "the high-performance version.");
          args.AddOption(&pc, "-pc", "--preconditioner",
                         "Preconditioner: lor - low-order-refined (matrix-free) AMG, "
                         "ho - high-order (assembled) AMG, none.");
          args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                         "--no-static-condensation", "Enable static condensation.");
          args.AddOption(&max_iter, "-mi", "--max-iter",
                         "Maximum number of iterations.");
          args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                         "--no-visualization",
                         "Enable or disable GLVis visualization.");
          args.Parse();
   if (!args.Good()) { if (myid == 0) { args.PrintUsage(std::cout); } return 1; }
   if (myid == 0) { args.PrintOptions(std::cout); }
   assert(SOL_P == order); // Make sure order is in sync with SOL_P

   ParMesh *pmesh = nullptr;
   {
      Mesh *mesh = new Mesh(mesh_file, 1, 1);
      int dim = mesh->Dimension();
      {
         int ref_levels = (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
         ref_levels = (ser_ref_levels != -1) ? ser_ref_levels : ref_levels;
         for (int l = 0; l < ref_levels; l++)
         {
            if (myid == 0)
            {
               std::cout << "Serial refinement: level " << l << " -> level " << l+1
                         << " ..." << std::flush;
            }
            mesh->UniformRefinement();
            MPI_Barrier(MPI_COMM_WORLD);
            if (myid == 0)
            {
               std::cout << " done." << std::endl;
            }
         }
      }

      MFEM_VERIFY(nxyz.Size() == 0 || nxyz.Size() == mesh->SpaceDimension(),
                  "Expected " << mesh->SpaceDimension() << " integers with the "
                  "option --cartesian-partitioning.");
      int *partitioning = nxyz.Size() ? mesh->CartesianPartitioning(nxyz) : NULL;
      NewParMesh(pmesh, mesh, partitioning);
      delete [] partitioning;
      {
         for (int l = 0; l < par_ref_levels; l++)
         {
            if (myid == 0)
            {
               std::cout << "Parallel refinement: level " << l << " -> level " << l+1
                         << " ..." << std::flush;
            }
            pmesh->UniformRefinement();
            MPI_Barrier(MPI_COMM_WORLD);
            if (myid == 0)
            {
               std::cout << " done." << std::endl;
            }
         }
      }
      pmesh->PrintInfo(std::cout);
   }
       ) << "\n\n";
}
void Code::entry_point_statements_d(Rule *) const
{
   dbg();
   if (ceed_benchmark) { ceed_benchmark_defines(out); }
   out << "\nint main(int argc, char* argv[]){\n";
   out << "\tint status = 0;\n";
   out << "\tint num_procs = 1, myid = 0;\n";
   out << "\tMPI_Init(&argc, &argv);\n";
   out << "\tMPI_Comm_size(MPI_COMM_WORLD, &num_procs);\n";
   out << "\tMPI_Comm_rank(MPI_COMM_WORLD, &myid);\n";
   if (ceed_benchmark) { ceed_benchmark_options(out); }
}
void Code::entry_point_statements_u(Rule *) const
{
   // Make sure p is in sync with SOL_P
   if (ceed_benchmark) { out << "\tassert(SOL_P == p);\n";}
   out << "\tMPI_Finalize();" << std::endl;
   out << "\treturn status;\n}" << std::endl;
}

// *****************************************************************************
void Code::decl_domain_assign_op_expr_d(Rule *) const {}
void Code::decl_domain_assign_op_expr_u(Rule *) const { out << ";\n"; }

// *****************************************************************************
static bool KernelCheck(xfl &ufl, Node *n)
{
   assert(n->child);
   assert(n->child->next);
   assert(n->child->next->IsRule());
   assert(n->Number() == decl_id_list_assign_op_expr);

   bool known = false;
   const Node *id = ufl.GetToken(TOK::IDENTIFIER, n->child);
   assert(id);
   const std::string &name = id->Name();
   for (auto &p : ufl.ctx.var)
   {
      if (name == p.second.name)
      {
         known = true;
         break;
      }
   }
   Node *expr = n->child->next->next;
   assert(expr->Number() == expr_assign_expr);
   const bool only_real = ufl.OnlyToken(TOK::REAL, expr);
   if (known && only_real && ufl.HitRule(primary_expr_constant, expr))
   {
      return true;
   }
   return false;
}

// *****************************************************************************
void Code::decl_id_list_assign_op_expr_d(Rule *n) const
{
   // dbg("\033[31mdecl_id_list_assign_op_expr_t");
   const bool kernel = KernelCheck(ufl, n);
   out << "\t";
   ufl.ctx.tmp.type = -1;

   // instead of probing, could pick the right one directly ?

   if (ufl.AssignOpNextToken(TOK::EQ, TOK::BOOL, n, out))
   {
      out << "const bool ";
      ufl.ctx.tmp.type = TOK::STRING;
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::STRING, n, out))
   {
      out << "const char *";
      ufl.ctx.tmp.type = TOK::STRING;
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::DEVICE, n, out))
   {
      out << "auto &&";
      ufl.ctx.tmp.type = TOK::DEVICE;
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::QUADRILATERAL, n, out))
   {
      out << "const int ";
      ufl.ctx.tmp.type = TOK::QUADRILATERAL;
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::HEXAHEDRON, n, out))
   {
      out << "const int ";
      ufl.ctx.tmp.type = TOK::HEXAHEDRON;
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::TRIANGLE, n, out))
   {
      out << "const int ";
      ufl.ctx.tmp.type = TOK::TRIANGLE;
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::TEST_FUNCTION, n, out))
   {
      out << "xfl::TestFunction ";
      ufl.ctx.tmp.type = TOK::TEST_FUNCTION;
      const Node *id = ufl.GetToken(TOK::IDENTIFIER, n);
      assert(id);
      const std::string &id_name = id->Name();
      const Node *lp = ufl.GetToken(TOK::LP, n);
      assert(lp);
      const Node *fes_id = ufl.GetToken(TOK::IDENTIFIER, lp->next);
      assert(fes_id);
      const std::string &fes_name = fes_id->Name();
      dbg("\033[31mTEST_FUNCTION: %s(%s)", id_name.c_str(), fes_name.c_str());
      ufl.ctx.tmp.fes = fes_id->Name();
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::TRIAL_FUNCTION, n, out))
   {
      out << "xfl::TrialFunction ";
      ufl.ctx.tmp.type = TOK::TRIAL_FUNCTION;
      const Node *id = ufl.GetToken(TOK::IDENTIFIER, n);
      assert(id);
      const std::string &id_name = id->Name();
      const Node *lp = ufl.GetToken(TOK::LP, n);
      assert(lp);
      const Node *fes_id = ufl.GetToken(TOK::IDENTIFIER, lp->next);
      assert(fes_id);
      const std::string &fes_name = fes_id->Name();
      dbg("\033[31mTRIAL_FUNCTION: %s(%s)", id_name.c_str(), fes_name.c_str());
      ufl.ctx.tmp.fes = fes_id->Name();
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::FUNCTION, n, out))
   {
      out << "xfl::Function ";
      ufl.ctx.tmp.type = TOK::FUNCTION;
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::CONSTANT_API, n, out))
   {
      out << "xfl::Constant ";
      ufl.ctx.tmp.type = TOK::CONSTANT_API;
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::EXPRESSION, n, out))
   {
      out << "xfl::Expression ";
      ufl.ctx.tmp.type = TOK::EXPRESSION;
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::MESH, n, out))
   {
      out << "auto &";
      ufl.ctx.tmp.type = TOK::MESH;
      dbg("Reading mesh...");
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::UNIT_SQUARE_MESH, n, out))
   {
      out << "auto &";
      const Node *id = ufl.GetToken(TOK::IDENTIFIER, n);
      const char *msh_name = id->Name().c_str();
      dbg("msh_name:%s", msh_name);
      ufl.ctx.tmp.type = TOK::UNIT_SQUARE_MESH;
      void *msh;
      {
         //assert(false);
         constexpr int N = 16;
         Element::Type quad = Element::Type::QUADRILATERAL;
         const bool generate_edges = false, sfc_ordering = true;
         const double sx = 1.0, sy = 1.0;
         msh = new mfem::Mesh(N, N, quad, generate_edges, sx, sy, sfc_ordering);
      }
      auto res = ufl.ctx.msh.emplace(msh_name, msh);
      if (res.second == false)
      {
         std::cerr << "MSH already present: " << res.first->first << std::endl;
         std::abort();
      }
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::UNIT_HEX_MESH, n, out))
   {
      out << "auto &";
      ufl.ctx.tmp.type = TOK::UNIT_HEX_MESH;
      const Node *id = ufl.GetToken(TOK::IDENTIFIER, n);
      const char *msh_name = id->Name().c_str();
      dbg("\033[35m[MESH]msh_name:%s", msh_name);
      ufl.ctx.tmp.type = TOK::UNIT_SQUARE_MESH;
      // assert(false);
      void *msh;
      {
         assert(false);
         constexpr int N = 16;
         Element::Type hex = Element::Type::HEXAHEDRON;
         const bool generate_edges = false, sfc_ordering = true;
         const double sx = 1.0, sy = 1.0, sz = 1.0;
         msh = new mfem::Mesh(N, N, N, hex, generate_edges, sx, sy, sz,
                              sfc_ordering);
      }
      auto res = ufl.ctx.msh.emplace(msh_name, msh);
      if (res.second == false)
      {
         std::cerr << "MSH already present: " << res.first->first << std::endl;
         std::abort();
      }
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::DIRICHLET_BC, n, out))
   {
      out << "const Array<int> ";
      ufl.ctx.tmp.type = TOK::DIRICHLET_BC;
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::FUNCTION_SPACE, n, out))
   {
      out << "mfem::ParFiniteElementSpace *";
      ufl.ctx.tmp.type = TOK::FUNCTION_SPACE;

      const Node *lp = ufl.GetToken(TOK::LP, n);
      assert(lp);

      const Node *mesh_id = ufl.GetToken(TOK::IDENTIFIER, lp->next);
      assert(mesh_id);
      const std::string &mesh_name = mesh_id->Name();
      dbg("\033[35mFUNCTION_SPACE: mesh: %s", mesh_name.c_str());
      void *msh = nullptr;
      if (ufl.ctx.msh.find(mesh_name) != ufl.ctx.msh.end())
      {
         msh = ufl.ctx.msh.at(mesh_name);
      }

      Node const *next_coma = ufl.UpNextToken(TOK::COMA, mesh_id);
      assert(next_coma);

      std::string fec_name{};  // the FEC of this FES

      Node const *next_token = ufl.NextToken(next_coma->next);
      assert(next_token);

      if (next_token->Number() != TOK::IDENTIFIER)  // inlined FEC
      {
         // family
         std::string family = next_token->Name();
         std::replace(family.begin(), family.end(), '\'', '\"');
         dbg("\033[31mFUNCTION_SPACE: inline family %s", family.c_str());
         assert(family == "\"P\"");
         std::string family_name = "Lagrange";

         // Should get the type from the mesh
         std::string type_name = "quadrilateral";

         Node const *next_coma = ufl.UpNextToken(TOK::COMA, next_token);
         assert(next_coma);

         Node const *next_token = ufl.NextToken(next_coma->next);
         assert(next_token);

         // Get the order
         int order = -1;
         if (next_token->Number() == TOK::IDENTIFIER)
         {
            const std::string &name = next_token->Name();
            if (ufl.ctx.N.find(name) == ufl.ctx.N.end())
            {
               std::cerr << "Unkown variable '" << name << "': " << strerror(errno)
                         << std::endl;
               std::abort();
            }
            order = ufl.ctx.N.at(name);
            dbg("\033[31mFUNCTION_SPACE: variable %s=%d", name.c_str(), order);
         }
         else if (next_token->Number() == TOK::NATURAL)
         {
            order = std::atoi(next_token->Name().c_str());
         }
         else
         {
            assert(false);
         }
         assert(order >= 0);

         // add new FEC and push to local variable
         dbg("%s:%d", ufl.loc->end.filename->c_str(), ufl.loc->end.line);
         fec_name = "fec_inlined_";
         fec_name += std::to_string(ufl.loc->end.line);
         dbg("Adding FEC: [%s,%s,%s,%d]", fec_name.c_str(), family_name.c_str(),
             type_name.c_str(), order);
         const int dim = type_name == "quadrilateral" ? 2
                         : type_name == "hexahedron"  ? 3
                         : 0;
         assert(dim == 2 || dim == 3);

         void *fec = new H1_FECollection(order, dim);
         assert(fec);
         auto res = ufl.ctx.fec.emplace(
                       fec_name,
                       xfl::fec{fec_name, family_name, type_name, order, dim, fec});
         assert(res.second);
         // assert(false);
      }
      else if (next_token->Number() == TOK::IDENTIFIER)    // fec variable
      {
         if (ufl.ctx.fec.find(next_token->Name()) == ufl.ctx.fec.end())
         {
            // should handle the mixed cases like TH = P2 * P1
            fec_name = "unhandled_mixed";
         }
         else
         {
            fec_name = next_token->Name();
         }
         // there can be the sdim as extra argument - not handled yet
      }
      else
      {
         assert(false);
      }

      // Adding this FES to the map
      const Node *id = ufl.GetToken(TOK::IDENTIFIER, n);
      assert(id);
      dbg("\033[33mAdding new FES '%s:[%s,%s]'", id->Name(), mesh_name, fec_name);

      void *mfem_fec = nullptr;
      if (ufl.ctx.fec.find(fec_name) != ufl.ctx.fec.end())
      {
         xfl::fec &fec = ufl.ctx.fec.at(fec_name);
         mfem_fec = fec.fec;
      }

      void *fes =
         (msh && mfem_fec)
         ? new ParFiniteElementSpace(
            static_cast<mfem::ParMesh *>(msh),
            static_cast<mfem::FiniteElementCollection *>(mfem_fec))
         : nullptr;
      auto res = ufl.ctx.fes.emplace(
                    id->Name(),
                    xfl::fes{id->Name(), mesh_name, fec_name, msh, fes});  // MFEM objects
      if (res.second == false)
      {
         std::cerr << "FES already present: " << res.first->first << std::endl;
         std::abort();
      }
      // assert(false);
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::FINITE_ELEMENT, n, out))
   {
      dbg("\033[31mFINITE_ELEMENT");
      out << "FiniteElementCollection *";
      ufl.ctx.tmp.type = TOK::FINITE_ELEMENT;

      const Node *fe = ufl.GetToken(TOK::FINITE_ELEMENT, n);
      assert(fe);
      assert(n->child);
      assert(n->child->next);
      assert(n->child->next->IsRule());
      Node *expr = n->child->next->next;
      // could check for LP
      const Node *family = ufl.GetToken(TOK::STRING, expr);
      const Node *quote = ufl.GetToken(TOK::QUOTE, expr);
      assert(family || quote);
      const std::string &family_name = family ? family->Name() : quote->Name();
      dbg("\033[31mfamily: %s", family_name.c_str());

      const Node *first_coma = ufl.GetToken(TOK::COMA, expr);
      assert(first_coma);

      const Node *type_id = ufl.GetTokenArg(TOK::IDENTIFIER, first_coma->next);
      dbg("type_id:%p (%s)", type_id, type_id ? type_id->Name().c_str() : "");
      const Node *type_str = ufl.GetToken(TOK::STRING, first_coma->next);
      const Node *type_tri = ufl.GetToken(TOK::TRIANGLE, first_coma->next);
      const Node *type_quad = ufl.GetToken(TOK::QUADRILATERAL, first_coma->next);
      const Node *type_tetra = ufl.GetToken(TOK::TETRAHEDRON, first_coma->next);
      const Node *type_hexa = ufl.GetToken(TOK::HEXAHEDRON, first_coma->next);
      assert(type_id || type_str || type_tri || type_quad || type_tetra ||
             type_hexa);
      std::string type_name = type_id      ? type_id->Name()
                              : type_str   ? type_str->Name()
                              : type_tri   ? type_tri->Name()
                              : type_tetra ? type_tetra->Name()
                              : type_hexa  ? type_hexa->Name()
                              : type_quad->Name();
      Node const *type_n = type_id      ? type_id
                           : type_str   ? type_str
                           : type_tri   ? type_tri
                           : type_tetra ? type_tetra
                           : type_hexa  ? type_hexa
                           : type_quad;
      dbg("\033[31mtype name: %s", type_n->Name().c_str());
      if (type_name == "triangle")
      {
         dbg("\033[31mTRIANGLES");
      }
      else if (type_name == "tetrahedron")
      {
         dbg("\033[31mTETRAS");
      }
      else if (type_name == "quadrilateral")
      {
         dbg("\033[31mQUADS");
      }
      else if (type_name == "hexahedron")
      {
         dbg("\033[31mHEXAS");
      }
      else if (ufl.ctx.var.find(type_name) != ufl.ctx.var.end())
      {
         auto &var = ufl.ctx.var.at(type_name);
         dbg("\033[31mKnown variable: %s:%d", var.name, var.type);
         // assert(var.type == TOK::HEXAHEDRON || var.type == TOK::QUADRILATERAL);
         if (var.type == TOK::HEXAHEDRON)
         {
            type_name = "hexahedron";
         }
         if (var.type == TOK::QUADRILATERAL)
         {
            type_name = "quadrilateral";
         }
      }
      else if (type_id)
      {
      }  // tetrahedron, ...
      else if (type_str)
      {
      }  // "triangle", ...
      else
      {
         assert(false);
      }

      Node const *next_coma = ufl.UpNextToken(TOK::COMA, type_n);
      assert(next_coma);

      int order = -1;
      Node const *natural = ufl.GetToken(TOK::NATURAL, next_coma->next);
      if (natural)
      {
         dbg("\033[31mNATURAL:%s", natural->Name().c_str());
         order = std::atoi(natural->Name().c_str());
         assert(order >= 0);
      }

      Node const *variable = ufl.GetToken(TOK::IDENTIFIER, next_coma->next);
      if (variable)
      {
         const std::string &name = variable->Name();
         dbg("\033[31mvariable:%s", name.c_str());
         if (ufl.ctx.N.find(name) == ufl.ctx.N.end())
         {
            std::cerr << "Unkown variable '" << name << "': " << strerror(errno)
                      << std::endl;
            std::abort();
         }
         order = ufl.ctx.N.at(name);
      }
      assert(order >= 0);
      // add this FiniteElementCollection to the context map
      const Node *id = ufl.GetToken(TOK::IDENTIFIER, n);
      assert(id);
      dbg("\033[33mAdding FEC: %s:[%s,%s,%d]", id->Name(), family_name, type_name,
          order);
      const int dim = type_name == "quadrilateral" ? 2
                      : type_name == "hexahedron"  ? 3
                      : 0;
      void *fec = nullptr;
      if (family_name == "\"Lagrange\"")
      {
         fec = new H1_FECollection(order, dim);
         assert(fec);
      }
      auto res = ufl.ctx.fec.emplace(
                    id->Name(),
                    xfl::fec{id->Name(), family_name, type_name, order, dim, fec});
      if (res.second == false)
      {
         std::cerr << "FEC already present: " << res.first->first << std::endl;
         std::abort();
      }
   }
   else if (ufl.AssignOpNextToken(TOK::EQ, TOK::NATURAL, n, out))
   {
      out << "const int ";
      ufl.ctx.tmp.type = TOK::NATURAL;

      // could be done in primary_id_identifier_d?
      // get the name of the current variable
      const Node *id = ufl.GetToken(TOK::IDENTIFIER, n);
      assert(id);
      const std::string &name = id->Name();
      // get the value of this natural
      Node const *p = ufl.GetToken(TOK::NATURAL, n);
      assert(p);
      assert(ufl.ctx.N.find(name) == ufl.ctx.N.end());
      const int value = std::atoi(p->Name().c_str());
      auto res = ufl.ctx.N.emplace(name, value);
      if (res.second == false)
      {
         auto &m = res.first->second;
         dbg("Variable already present with value: %d", m);
         assert(false);
      }
      dbg("\033[31mid: %s => %d", name, value);
      dbg("\033[31mnew Natural '%s=%d'", name, value);
   }
   else if (!kernel)
   {
      out << "auto ";
   }
   else     /* nothing */
   {
   }
}
void Code::decl_id_list_assign_op_expr_u(Rule *) const { out << ";\n"; }

// *****************************************************************************
void Code::decl_direct_declarator_d(Rule *) const
{
   dbg();
   dbg("LHS=false");
   yy::rules.at(lhs_lhs) = false;
   out << "\t";
}
void Code::decl_direct_declarator_u(Rule *) const { out << ";\n"; }

// *****************************************************************************
void Code::postfix_id_primary_id_d(Rule *) const {}
void Code::postfix_id_primary_id_u(Rule *) const {}

// *****************************************************************************
void Code::primary_id_identifier_d(Rule *n) const
{
   // known types from decl_id_list_assign_op_expr_d
   const bool known_types = true;  // ufl.ctx.tmp.type > 0;

   if (yy::rules.at(lhs_lhs) &&                      // LHS declaration
       yy::rules.at(decl_id_list_assign_op_expr) &&  // declaration
       known_types)                                  // ? known types
   {
      dbg("\033[33mAdding variable: %s:%d", n->child->Name().c_str(),
          ufl.ctx.tmp.type);
      const std::string &name = n->child->Name();
      auto res =
         ufl.ctx.var.emplace(name, xfl::var{name, ufl.ctx.tmp.type, xfl::NONE});
      if (res.second == false)
      {
         auto &m = res.first->second;
         dbg("Variable already present with type: %d", m.type);
         // assert(false);
      }
      // if the context holds a fes, add it to this variable
      if (!ufl.ctx.tmp.fes.empty())
      {
         dbg("\033[36mAdding fes '%s' to variable '%s'", ufl.ctx.tmp.fes.c_str(),
             name.c_str());
         ufl.ctx.var.at(name).fes = ufl.ctx.tmp.fes;
         ufl.ctx.tmp.fes.clear();
      }
      // for (const auto &p : ufl.ctx.var) { dbg("%s:%d", p.first.c_str(),
      // p.second.type); }
   }
}
void Code::primary_id_identifier_u(Rule *) const {}

// *****************************************************************************
void Code::assign_expr_postfix_expr_assign_op_assign_expr_d(Rule *) const {}
void Code::assign_expr_postfix_expr_assign_op_assign_expr_u(Rule *) const {}

// *****************************************************************************
void Code::primary_expr_identifier_d(Rule *n) const
{
   dbg();
   // If we are inside a argument list, skip it
   constexpr int in_arg_list = assign_expr_postfix_expr_assign_op_assign_expr;
   if (yy::rules.at(in_arg_list))
   {
      return;
   }
   // RHS & VARDOM_DX => inputs
   if (!yy::rules.at(lhs_lhs) && yy::rules.at(extra_status_rule_dom_xt) &&
       !yy::rules.at(grad_expr_grad_op_form_args))
   {
      const std::string &name = n->child->Name();
      if (ufl.ctx.var.find(name) != ufl.ctx.var.end())
      {
         auto &var = ufl.ctx.var.at(name);
         var.mode |= xfl::VALUE;
         dbg("\033[31m%s:%d:%d", name.c_str(), var.type, var.mode);
      }
      // else { assert(false); }
   }
}
void Code::primary_expr_identifier_u(Rule *) const {}

// *****************************************************************************
void Code::assign_op_eq_d(Rule *n) const
{
   // If we are inside a argument list, skip it
   constexpr int in_arg_list = assign_expr_postfix_expr_assign_op_assign_expr;
   if (yy::rules.at(in_arg_list))
   {
      n->dfs.down = false;
      return;
   }

   dbg("LHS=false");
   yy::rules.at(lhs_lhs) = false;
}
void Code::assign_op_eq_u(Rule *) const {}

// *****************************************************************************
void Code::postfix_expr_api_d(Rule *) const
{
   out << "xfl::";
}
void Code::postfix_expr_api_u(Rule *) const { }

// *****************************************************************************
void Code::postfix_expr_pow_expr_d(Rule *) const { out << "xfl::math::Pow("; }
void Code::postfix_expr_pow_expr_u(Rule *) const { out << ")"; }

// *****************************************************************************
void Code::id_list_postfix_ids_d(Rule *n) const
{
   if (!yy::rules.at(id_list_id_list_coma_postfix_ids))
   {
      return;
   }
   assert(n);
   assert(n->next);
   assert(n->next->IsToken());
   assert(n->next->Number() == TOK::COMA);
   assert(n->root);
   assert(n->root->next);
   if (n->root->next->Number() != assign_op_eq)
   {
      return;
   }
   Node *extra = n->root->next;
   ufl.ctx.extra = extra;
}
void Code::id_list_postfix_ids_u(Rule *) const {}

// *****************************************************************************
void Code::if_statement_if_lp_expr_rp_expr_d(Rule *) const { out << "\t"; }
void Code::if_statement_if_lp_expr_rp_expr_u(Rule *) const
{
   out << ";" << std::endl;
}

// *****************************************************************************
void Code::decl_iteration_statement_d(Rule *) const { out << "\t"; }
void Code::decl_iteration_statement_u(Rule *) const { out << ";" << std::endl; }

// *****************************************************************************
void Code::decl_api_statement_d(Rule *) const
{
   out << "\tstatus |= xfl::";
   dbg("LHS=false");
   yy::rules.at(lhs_lhs) = false;
}
void Code::decl_api_statement_u(Rule *) const { out << ";" << std::endl; }

// *****************************************************************************
void Code::decl_function_d(Rule *) const { out << "\tauto "; }
void Code::decl_function_u(Rule *) const { out << ";};" << std::endl; }

// *****************************************************************************
void Code::args_expr_list_assign_expr_d(Rule *n) const
{
   if (yy::rules.at(decl_function) &&
       !yy::rules.at(args_expr_list_args_expr_list_coma_assign_expr) &&
       !ufl.ctx.nodes[0])
   {
      assert(n);
      assert(n->child);
      ufl.ctx.nodes[0] = n->child;
      // dbg("\033[31mSAVE ufl.ctx.node");
      n->dfs.down = false;  // don't continue down
   }
}
void Code::args_expr_list_assign_expr_u(Rule *) const {}

// *****************************************************************************
void Code::args_expr_list_args_expr_list_coma_assign_expr_d(Rule *n) const
{
   dbg();
   if (yy::rules.at(decl_function) &&
       !yy::rules.at(args_expr_list_args_expr_list_coma_assign_expr) &&
       !ufl.ctx.nodes[0])
   {
      assert(n);
      assert(n->child);
      ufl.ctx.nodes[0] = n->child;
      // dbg("\033[31mSAVE ufl.ctx.node");
      n->dfs.down = false;  // don't continue down, we wait for the ':'
   }
}
void Code::args_expr_list_args_expr_list_coma_assign_expr_u(Rule *) const {}

// *****************************************************************************
void Code::def_statement_nl_d(Rule *n) const
{
   if (!ufl.ctx.nodes[0])
   {
      return;
   }
   out << "[&] (int ";
   ufl.DfsPreOrder(ufl.ctx.nodes[0],
                   me);  // now dfs with the saved arguments node
   ufl.ctx.nodes[0] = nullptr;
   // dbg("\033[31mFLUSH ufl.ctx.node");
   // We delayed the lhs => rhs (from COLON), now switch to RHS
   yy::rules.at(lhs_lhs) = false;
   out << ") {";
}
void Code::def_statement_nl_u(Rule *) const {}

// *****************************************************************************
void Code::statement_decl_nl_d(Rule *n) const
{
   dbg("LHS=true");
   yy::rules.at(lhs_lhs) = true;
}
void Code::statement_decl_nl_u(Rule *) const {}

// *****************************************************************************
void Code::def_empty_empty_d(Rule *n) const
{
   if (yy::rules.at(decl_function))
   {
      def_statement_nl_d(n);
   }
}
void Code::def_empty_empty_u(Rule *) const {}

// *****************************************************************************
void Code::extra_status_rule_dom_xt_d(Rule *n) const
{
   dbg();
   assert(n->next);
   assert(n->next->child);
   out << "[&]() {\n";
   // continue with parsing the commented body to capture inputs
   out << "\t\tconstexpr const char *qs" << ufl.ctx.ker << " = \"";
}
void Code::extra_status_rule_dom_xt_u(Rule *n) const
{
   int j = 0;
   out << "\";\n";         // end of qfunction string
   out << "\t\t// var:[";  // adding names as comments
   for (auto &p : ufl.ctx.var)
   {
      xfl::var &var = p.second;
      if (var.mode != xfl::NONE)
      {
         out << (j++ == 0 ? "" : ",") << var.name;
      }
   }

   out << "], ops:[Xe,";
   mfem::internal::KernelOperations ko(ufl, out);
   ufl.DfsInOrder(n->child, ko);
   out << "]\n";

   const int k = ufl.ctx.ker;
   std::string trial_fes{}, test_fes{};
   bool linear_form = false;
   bool bilinear_form = false;
   for (auto &p : ufl.ctx.var)
   {
      xfl::var &var = p.second;
      if (var.type == TOK::TRIAL_FUNCTION && var.mode != xfl::NONE)
      {
         assert(!var.fes.empty() && trial_fes.empty());
         trial_fes = var.fes;
         out << "\t\t// Trial FES: '" << var.fes << "':" << var.name << " ("
             << (var.mode == 1 ? "Eval" : "Grad") << ")\n";
      }
      if (var.type == TOK::TEST_FUNCTION && var.mode != xfl::NONE)
      {
         assert(!var.fes.empty() && test_fes.empty());
         test_fes = var.fes;
         out << "\t\t// Test FES: '" << var.fes << "':" << var.name << " ("
             << (var.mode == 1 ? "Eval" : "Grad") << ")\n";
      }
   }
   if (trial_fes.empty() && !test_fes.empty())
   {
      linear_form = true;
   }
   if (!trial_fes.empty() && !test_fes.empty())
   {
      bilinear_form = true;
   }

   if (linear_form || bilinear_form)
   {
      out << "\t\tParFiniteElementSpace *fes" << k << " = " << test_fes << ";";
      if (linear_form)
      {
         out << "\n\t\tmfem::Operator *QM" << k << " = nullptr;\n";
      }

      // Some ufl tests use FEC directly as arguments (Poisson.ufl)
      const bool is_fec = ufl.ctx.fec.find(test_fes) != ufl.ctx.fec.end();

      // make sure fes exists
      const bool is_fes = ufl.ctx.fes.find(test_fes) != ufl.ctx.fes.end();

      if (!is_fec && is_fes && bilinear_form)
      {
         xfl::fes &fes = ufl.ctx.fes.at(test_fes);

         mfem::ParFiniteElementSpace *mfem_fes =
            static_cast<mfem::ParFiniteElementSpace *>(fes.fes);

         // make sure fec exists
         assert(ufl.ctx.fec.find(fes.fec) != ufl.ctx.fec.end());
         xfl::fec &fec = ufl.ctx.fec.at(fes.fec);
         // out << "\n\t\t// type: " << fec.type;
         if (fec.type == "quadrilateral" || fec.type == "hexahedron")
         {
            const int dim = fec.type == "quadrilateral" ? 2
                            : fec.type == "hexahedron"  ? 3
                            : 0;
            assert(fec.dim == dim);

            assert(trial_fes == test_fes);
            out << "\n\t\tstruct QMult" << k << ": public xfl::Operator<" << dim
                << ">{\n";
            out << "\t\t\tQMult" << k << "(const ParFiniteElementSpace *fes): ";
            out << "xfl::Operator<" << dim << ">(fes) { Setup(); }\n";
            out << "\t\t\t~QMult" << k << "() { }\n";

            dbg(n->Name().c_str());
            assert(n->n == extra_status_rule_dom_xt);

            std::ostringstream ker;
            std::array<int, 2> shp;

            // Extract Kernel DX Shape
            {
               mfem::internal::KernelShapes ks(k, ufl, n->child, ker, fes, fec);
               ufl.DfsInOrder(n->child, ks);
               shp = ks.GetShapes();
               assert((shp[0] * shp[1]) > 0);
            }

            //assert(false);
            const bool USE_CUDA = false;
            const bool KER_LIBP = std::getenv("KER_LIBP");
            const bool KER_SIMD = std::getenv("KER_SIMD"); assert(!KER_SIMD);
            const bool KER_CEED = std::getenv("KER_CEED");
            const bool KER_REGS = std::getenv("KER_REGS");
            const bool KER_OREO = std::getenv("KER_OREO");
            const bool KER_LEAN = true;//std::getenv("KER_LEAN");
            assert(KER_LEAN);

            const bool SIMD_SETUP = KER_SIMD || KER_OREO || KER_REGS || KER_LEAN;
            const bool KER_COLLOCATED_G = KER_REGS || KER_OREO || KER_LEAN;

            // Generate Kernel Setup Code
            if (!SIMD_SETUP)
            {
               mfem::internal::KernelSetup kc(k, ufl, n->child, ker, fes, fec);
               ufl.DfsInOrder(n->child, kc);
            }

            // Generate Kernel Mult Code
            if (KER_LIBP) // LibP
            {
               mfem::internal::KerLibPMult km(k, ufl, n->child, ker, fes, fec);
               ufl.DfsInOrder(n->child, km);
            }
            else if (KER_CEED) // libCEED
            {
               mfem::internal::KerCeedMult km(k, ufl, n->child, ker, fes, fec);
               ufl.DfsInOrder(n->child, km);
            }
            else if (KER_SIMD) // SIMD + OpenMP
            {
               {
                  mfem::internal::KerSimdSetup kc(k, ufl, n->child, ker, fes, fec);
                  ufl.DfsInOrder(n->child, kc);
               }
               mfem::internal::KerSimdMult km(k, ufl, n->child, ker, fes, fec);
               ufl.DfsInOrder(n->child, km);
            }
            else if (KER_REGS) // Minimal register usage (+ SIMD)
            {
               {
                  mfem::internal::KerRegsSetup kc(k, ufl, n->child, ker, fes, fec);
                  ufl.DfsInOrder(n->child, kc);
               }
               mfem::internal::KerRegsMult km(k, ufl, n->child, ker, fes, fec);
               ufl.DfsInOrder(n->child, km);
            }
            else if (KER_OREO) // Minimal register usage (+ SIMD)
            {
               {
                  mfem::internal::KerOreoSetup kc(k, ufl, n->child, ker, fes, fec);
                  ufl.DfsInOrder(n->child, kc);
               }
               mfem::internal::KerOreoMult km(k, ufl, n->child, ker, fes, fec);
               ufl.DfsInOrder(n->child, km);
            }
            else if (KER_LEAN) // No intermediate array
            {
               {
                  mfem::internal::KerLeanSetup kc(k, ufl, n->child, ker, fes, fec);
                  ufl.DfsInOrder(n->child, kc);
               }
               mfem::internal::KerLeanMult km(k, ufl, n->child, ker, fes, fec);
               ufl.DfsInOrder(n->child, km);
            }
            else
            {
               mfem::internal::KernelMult km(k, ufl, n->child, ker, fes, fec);
               ufl.DfsInOrder(n->child, km);
            }
            n->out = ker.str();

            const int NDOFS = mfem_fes ? mfem_fes->GetNDofs() : 0;
            const int VDIM = mfem_fes ? mfem_fes->GetVDim() : 0;
            const int NE = mfem_fes ? mfem_fes->GetParMesh()->GetNE() : 0;

            // Compute D1D, Q1D
            const int p = fec.order;
            const int node_order = 1;
            const int order_w = node_order*fec.dim - 1;
            const int ir_order = 2*p + order_w;
            //const int ir_order = 2*(p + 2) - 1; // <-----
            const int GeomType = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
            const mfem::IntegrationRule &ir = mfem::IntRules.Get(GeomType, ir_order);
            const int Q1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
            const int D1D = p + 1;

            out << "\t\t\tvoid Setup() {\n";
            out << "\t\t\t\tint myid = 0; MPI_Comm_rank(MPI_COMM_WORLD, &myid);\n";
            out << "\t\t\t\tif(myid == 0){\n";
            out << "\t\t\t\t\tstd::cout << \"XFL(SIMD_\" << SIMD_SIZE <<\") version using integration rule with "
                << (Q1D*Q1D*Q1D) << " points ...\\n\";\n";
            out << "\t\t\t\t\tstd::cout << \"D1D:" << D1D << ", Q1D:" << Q1D << "\\n\";\n";
            out << "\t\t\t\t}\n";

            out << "\t\t\t\tdx.SetSize(NQ*NE*" << shp[0] << "*" << shp[1]
                << ", Device::GetDeviceMemoryType()); ";
            out << "// DX shape: " << shp[0] << "x" << shp[1] << "\n";
            out << "\t\t\t\tKSetup" << k << "<" << dim << ","
                << shp[0] << "," << shp[1] << ","<< Q1D <<">(";
            out << "NDOFS, VDIM, NE, J0.Read(), ir.GetWeights().Read(), dx.Write()";
            out << ");\n";
            if (KER_COLLOCATED_G)
            {
               out << "\t\t\t\tCoG.SetSize("<<Q1D<<"*"<<Q1D<<");\n";
               out << "\t\t\t\t// Compute the collocated gradient d2q->CoG\n";
               out << "\t\t\t\tkernels::GetCollocatedGrad<"<<D1D<<","<<Q1D<<">(\n";
               out << "\t\t\t\t\tConstDeviceMatrix(maps->B.HostRead(),"<<Q1D<<","<<D1D<<"),\n";
               out << "\t\t\t\t\tConstDeviceMatrix(maps->G.HostRead(),"<<Q1D<<","<<D1D<<"),\n";
               out << "\t\t\t\t\tDeviceMatrix(CoG.HostReadWrite(),"<<Q1D<<","<<Q1D<<"));\n";
            }
            out << "\t\t\t}\n";  // end of setup
            out << "\t\t\tvoid Mult(const mfem::Vector &x, mfem::Vector &y) const "
                "{\n";
            out << "\t\t\t\ty = 0.0;\n";
            int SMEM;
            if (KER_CEED)
            {
               const int T1D = std::max(Q1D, D1D);
               const int NBZ = (T1D < 6) ? 4 : ((T1D<8)? 2 : 1);
               const int GRID = NE/NBZ + ((((NE/NBZ)*NBZ) < NE) ? 1 : 0);
               SMEM = NBZ*T1D*T1D*sizeof(double);
               if (USE_CUDA)
               {
                  out << "\t\t\t\tconst int GRID = " << GRID << ";\n";
                  out << "\t\t\t\tconst dim3 BLCK("<<T1D<<","<<T1D<<","<<NBZ<<");\n";
               }
            }
            //out << "\t\tdbg(\"MAP size:%d\", ER.GatherMap().Size());\n";
            out << "\t\t\t\tKMult" << k << "<" << dim << ","
                << shp[0] << "," << shp[1] << "," << D1D << "," << Q1D
                << (KER_CEED ? "," + std::to_string(NDOFS) : "")
                << (KER_CEED ? "," + std::to_string(VDIM) : "")
                << (KER_CEED ? "," + std::to_string(SMEM) : "")
                << ">";
            if (KER_CEED && USE_CUDA) { out << "<<<GRID,BLCK>>>"; }
            out << "(NDOFS /*" << NDOFS << "*/,";
            out << "VDIM /*" << VDIM << "*/, NE /*" << NE << "*/,";
            //if (KER_CEED) { out << "ir.GetWeights().Read(),"; }
            out << "maps->B.Read(), ";
            out << (KER_COLLOCATED_G ? "CoG" : "maps->G") << ".Read(), ";
            out << "ER.GatherMap().Read(), ";
            out << "dx.Read(), x.Read(), y.ReadWrite());\n";
            out << "\t\t\t}\n";  // end of mult
            out << "\t\t}; // QMult struct\n";
            out << "\t\tQMult" << k << " *QM" << k << " = new QMult" << k << "("
                << test_fes << ");\n";
         }
         else
         {
            out << "\n\t\tmfem::Operator *QM" << k
                << " = nullptr; // unknown type\n";
         }
      }
      out << "\t\txfl::QForm QForm" << k << "(fes" << k << ", qs" << k << ", QM"
          << k;
      out << ");\n\t\treturn QForm" << k;
      // for (int i = 1; i <= ufl.ctx.ker; i++) { out << " + QForm" << i; }
      out << ";\n\t}()";  // force lambda evaluation
      ufl.ctx.ker += 1;
   }
   // Flush variables of this dom_dx
   for (auto &p : ufl.ctx.var)
   {
      p.second.mode = xfl::NONE;
   }
}

// *****************************************************************************
void Code::extra_status_rule_dot_xt_d(Rule *) const { out << "dot"; }
void Code::extra_status_rule_dot_xt_u(Rule *) const {}

// *****************************************************************************
void Code::extra_status_rule_transpose_xt_d(Rule *) const {}
void Code::extra_status_rule_transpose_xt_u(Rule *) const {}

// *****************************************************************************
void Code::grad_expr_grad_op_form_args_d(Rule *n) const
{
   // Even with the postfix_expr grad changes, we can still go here
   // with functions for example
   // assert(false);
   assert(n->child);
   assert(n->child->next);
   assert(n->child->next->n == form_args_lp_additive_expr_rp);
   Node *form_expr = n->child->next;
   assert(form_expr->child->n == TOK::LP);
   Node *additive_expr = form_expr->child->next;
   const Node *id = ufl.GetToken(TOK::IDENTIFIER, additive_expr->child);
   assert(id);
   const std::string &name = id->Name();
   dbg("%s", name.c_str());
   if (ufl.ctx.var.find(name) != ufl.ctx.var.end())
   {
      xfl::var &var = ufl.ctx.var.at(name);
      var.mode |= xfl::GRAD;
      dbg("\033[31m%s:%d:%d", name.c_str(), var.type, var.mode);
   }
   // else { assert(false); }
}
void Code::grad_expr_grad_op_form_args_u(Rule *) const {}

// *****************************************************************************
void Code::shift_expr_additive_expr_d(Rule *) const {}
void Code::shift_expr_additive_expr_u(Rule *) const {}

// *****************************************************************************
// Specialized Code backend tokens
void Code::token_NL(Token *) const   /* empty */
{
}

void Code::token_QUOTE(Token *tok) const
{
   const bool expr_quote = yy::rules.at(expr_quote_expr_quote);
   if (expr_quote)
   {
      dbg();
      out << "[](const Vector &x){return ";
      string &str = tok->name;
      str.erase(std::remove(str.begin(), str.end(), '\''), str.end());
      out << tok->name;
      out << ";}";
      yy::rules.at(expr_quote_expr_quote) = false;
      return;
   }
   std::replace(tok->name.begin(), tok->name.end(), '\'', '\"');
   out << tok->name;
}

void Code::token_DOM_DX(Token *) const { }

void Code::token_POW(Token *) const { out << ","; }

void Code::token_INNER_OP(Token *) const { out << "dot"; }

void Code::token_EQ(Token *) const { out << " = "; }

void Code::token_FOR(Token *) const { out << "for(auto &&"; }

void Code::token_IN(Token *) const { out << ": cpp::Range("; }

void Code::token_RANGE(Token *) const {}

void Code::token_COLON(Token *) const
{
   dbg();
   if (yy::rules.at(decl_iteration_statement))
   {
      out << ")) ";
      return;
   }
   if (yy::rules.at(decl_function))
   {
      out << " /*function*/ = ";
      return;
   }
   out << ":";
}

void Code::token_DEF(Token *) const
{
   dbg("LHS=true");
   yy::rules.at(lhs_lhs) = true;
}

void Code::token_LP(Token *) const
{
   if (yy::rules.at(decl_function) && yy::rules.at(lhs_lhs))
   {
      return;
   }
   out << "(";
}

void Code::token_RP(Token *) const
{
   if (yy::rules.at(decl_function) && yy::rules.at(lhs_lhs))
   {
      return;
   }
   out << ")";
}

void Code::token_RETURN(Token *) const { out << "return "; }

void Code::token_COMA(Token *) const
{
   dbg();
   const bool lhs = yy::rules.at(lhs_lhs);
   const bool decl_func = lhs and yy::rules.at(decl_function);
   if (decl_func)
   {
      out << ", int ";
      return;
   }
   out << ", ";
}

void Code::token_AND_AND(Token *) const { out << " && "; }

void Code::token_OR_OR(Token *) const { out << " || "; }

void Code::token_CONSTANT_API(Token *) const { out << "Constant"; }

void Code::token_DOT_OP(Token *) const { out << "*"; }

void Code::token_GRAD_OP(Token *) const { out << "grad"; }

void Code::token_ADD(Token *t) const
{
   if (t->next && ufl.HitToken(TOK::DOM_DX, t->next))
   {
      out << ");\n";
      return;
   }
   out << "+";
}

void Code::token_EXPRESSION(Token *token) const
{
   yy::rules.at(expr_quote_expr_quote) = true;
   out << token->Name();
}

void Code::token_BOOL(Token *token) const
{
   std::string &s = token->Name();
   auto to_lower_l = [](unsigned char c) { return std::tolower(c); };
   std::transform(s.begin(), s.end(), s.begin(), to_lower_l);
   out << s;
}

/** @endcond */

}  // namespace internal

}  // namespace mfem
