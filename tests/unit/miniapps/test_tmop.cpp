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

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include "catch.hpp"

#include "mfem.hpp"
#include "general/forall.hpp"
#include "linalg/kernels.hpp"

#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
extern mfem::MPI_Session *GlobalMPISession;
#define PFesGetParMeshGetComm(pfes) pfes.GetParMesh()->GetComm()
#define PFesGetParMeshGetComm0(pfes) pfes.GetParMesh()->GetComm()
#else
typedef int MPI_Session;
#define ParMesh Mesh
#define GetParMesh GetMesh
#define GlobalTrueVSize GetVSize
#define ParBilinearForm BilinearForm
#define ParGridFunction GridFunction
#define ParFiniteElementSpace FiniteElementSpace
#define PFesGetParMeshGetComm(...)
#define PFesGetParMeshGetComm0(...) 0
#define MPI_Finalize()
#define MPI_Allreduce(src,dst,...) *dst = *src
#define MPI_Reduce(src, dst, n, T,...) *dst = *src
#endif

using namespace std;
using namespace mfem;

namespace mfem
{

int tmop(int myid, int argc, char *argv[])
{
   const char *mesh_file = "icf.mesh";
   int order             = 1;
   int rs_levels         = 0;
   constexpr double jitter = 0.0;
   int metric_id         = 1;
   int target_id         = 1;
   constexpr double lim_const = 0.0;
   constexpr double adapt_lim_const = 0.0;
   int quad_type         = 1;
   int quad_order        = 2;
   int newton_iter       = 10;
   double newton_rtol    = 1e-8;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   constexpr bool move_bnd = false;
   constexpr int combomet  = 0;
   constexpr bool normalization = false;
   constexpr int verbosity_level = 0;
   constexpr bool fdscheme = false;

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "");
   args.AddOption(&order, "-o", "--order", "");
   args.AddOption(&rs_levels, "-rs", "--refine-serial", "");
   args.AddOption(&metric_id, "-mid", "--metric-id", "");
   args.AddOption(&target_id, "-tid", "--target-id", "");
   args.AddOption(&quad_type, "-qt", "--quad-type", "");
   args.AddOption(&quad_order, "-qo", "--quad_order", "");
   args.AddOption(&newton_iter, "-ni", "--newton-iters","");
   args.AddOption(&newton_rtol, "-rtol", "--newton-rel-tolerance", "");
   args.AddOption(&lin_solver, "-ls", "--lin-solver", "");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter", "");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (verbosity_level > 0) { args.PrintOptions(cout); }

   REQUIRE(mesh_file);
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();

   REQUIRE(order > 0);
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);
   mesh->SetNodalFESpace(fespace);

   Vector b[2] = {Vector(0.0), Vector(0.0)};
   b[0].UseDevice(true);
   b[1].UseDevice(true);

   GridFunction x[2] = {GridFunction(fespace), GridFunction(fespace)};
   mesh->SetNodalGridFunction(&x[0]);
   mesh->GetNodes()->UseDevice(true);
   mesh->SetNodalGridFunction(&x[1]);

   Vector h0(fespace->GetNDofs());
   h0 = infinity();
   Array<int> dofs;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      fespace->GetElementDofs(i, dofs);
      const double hi = mesh->GetElementSize(i);
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), hi);
      }
   }

   GridFunction rdm(fespace);
   rdm.Randomize();
   rdm -= 0.25;
   rdm *= 0.0;//jitter;
   rdm.HostReadWrite();
   for (int i = 0; i < fespace->GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(fespace->DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < fespace->GetNBE(); i++)
   {
      fespace->GetBdrElementVDofs(i, vdofs);
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   for (int k : {0,1})
   {
      //x[k] -= rdm;
      x[k].SetTrueVector();
      x[k].SetFromTrueVector();
   }

   GridFunction x0[2] = {GridFunction(fespace),GridFunction(fespace)};

   for (int k : {0,1})
   {
      x0[k] = x[k];
   }

   TMOP_QualityMetric *metric = nullptr;
   switch (metric_id)
   {
      case 2: metric = new TMOP_Metric_002; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 321: metric = new TMOP_Metric_321; break;
      default: cout << "Unknown metric_id: " << metric_id << std::endl; return 3;
   }

   TargetConstructor::TargetType target_t;
   REQUIRE(target_id == 1);
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
         //case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
         //case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
         //case 4: // Analytic
         //case 5: // Discrete size 2D
         //case 6: // Discrete size + aspect ratio - 2D
         //case 7: // Discrete aspect ratio 3D
         //case 8: // shape/size + orientation 2D
   }
   TargetConstructor *target_c = new TargetConstructor(target_t);
   target_c->SetNodes(x0[0]);

   // Setup the quadrature rule for the non-linear form integrator.
   const IntegrationRule *ir = nullptr;
   IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
   IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);
   const int geom_type = fespace->GetFE(0)->GetGeomType();
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order); break;
      case 2: ir = &IntRules.Get(geom_type, quad_order); break;
      case 3: ir = &IntRulesCU.Get(geom_type, quad_order); break;
      default:
      {
         std::cout << "Unknown quad_type: " << quad_type << std::endl;
         return 3;
      }
   }
   std::cout << "Quadrature points per cell: " << ir->GetNPoints() << std::endl;

   TMOP_Integrator *he_nlf_integ[2];
   for (int k : {0,1})
   {
      he_nlf_integ[k] = new TMOP_Integrator(metric, target_c);
      he_nlf_integ[k]->SetIntegrationRule(*ir);
   }

   REQUIRE_FALSE(normalization);
   REQUIRE(lim_const==0.0);
   REQUIRE(adapt_lim_const == 0.0);
   REQUIRE_FALSE(fdscheme);

   NonlinearForm nlf[2] = { NonlinearForm(fespace), NonlinearForm(fespace) };
   nlf[0].SetAssemblyLevel(AssemblyLevel::NONE);
   nlf[1].SetAssemblyLevel(AssemblyLevel::PARTIAL);

   REQUIRE(combomet == 0);
   for (int k : {0,1}) { nlf[k].AddDomainIntegrator(he_nlf_integ[k]); }
   nlf[1].Setup();

   const double init_energy[2] =
   {
      nlf[0].GetGridFunctionEnergy(x[0]),
      nlf[1].GetGridFunctionEnergy(x[1])
   };
   REQUIRE(init_energy[0] == Approx(init_energy[1]));

   REQUIRE_FALSE(move_bnd);
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   for (int k : {0,1}) { nlf[k].SetEssentialBC(ess_bdr); }

   Solver *S[2] = { nullptr, nullptr };
   constexpr double linsol_rtol = 1e-12;
   if (lin_solver == 0)
   {
      for (int k : {0,1})
      {
         S[k] = new DSmoother(1, 1.0, max_lin_iter);
      }
   }
   else if (lin_solver == 1)
   {
      for (int k : {0,1})
      {
         CGSolver *cg = new CGSolver;
         cg->SetMaxIter(max_lin_iter);
         cg->SetRelTol(linsol_rtol);
         cg->SetAbsTol(0.0);
         cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
         S[k] = cg;
      }
   }
   else
   {
      for (int k : {0,1})
      {
         MINRESSolver *minres = new MINRESSolver;
         minres->SetMaxIter(max_lin_iter);
         minres->SetRelTol(linsol_rtol);
         minres->SetAbsTol(0.0);
         minres->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
         S[k] = minres;
      }
   }

   // 18. Compute the minimum det(J) of the starting mesh.
   double tauval = infinity();
   const int NE = mesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      ElementTransformation *transf = mesh->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir->IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   cout << "Minimum det(J) of the original mesh is " << tauval << endl;

   // 19. Finally, perform the nonlinear optimization.
   NewtonSolver *newton[2] = { nullptr, nullptr };
   REQUIRE(tauval > 0.0);
   tauval = 0.0;
   TMOPNewtonSolver *tns[2] =
   {
      new TMOPNewtonSolver(*ir),
      new TMOPNewtonSolver(*ir)
   };
   for (int k : {0,1})
   {
      newton[k] = tns[k];
      newton[k]->SetPreconditioner(*S[k]);
      newton[k]->SetMaxIter(newton_iter);
      newton[k]->SetRelTol(newton_rtol);
      newton[k]->SetAbsTol(0.0);
      newton[k]->SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
      newton[k]->SetOperator(nlf[k]);
      newton[k]->Mult(b[k], x[k].GetTrueVector());
      x[k].SetFromTrueVector();
      REQUIRE(newton[k]->GetConverged());
   }
   REQUIRE(x[0]*x[0] == Approx(x[1]*x[1]));
   REQUIRE(nlf[0].GetGridFunctionEnergy(x[0]) ==
           Approx(nlf[1].GetGridFunctionEnergy(x[1])));
   delete newton[0]; delete newton[1];
   delete S[0]; delete S[1];
   delete target_c;
   delete metric;
   delete fespace;
   delete fec;
   delete mesh;
   return 0;
}

} // namespace mfem

static int argn(const char *argv[], int argc =0)
{
   while (argv[argc]) { argc+=1; }
   return argc;
}

static void tmop_tests(int myid)
{
   const char *argv2D[]= { "tmop_tests",
                           "-m", "../../miniapps/meshing/blade.mesh",
                           "-mid", "2",
                           nullptr
                         };
   REQUIRE(tmop(myid, argn(argv2D), const_cast<char**>(argv2D))==0);

   const char *argv2Drs1[]= { "tmop_tests",
                              "-m", "../../miniapps/meshing/blade.mesh",
                              "-mid", "2",
                              "-rs", "1",
                              "-ni", "20",
                              nullptr
                            };
   REQUIRE(tmop(myid, argn(argv2Drs1), const_cast<char**>(argv2Drs1))==0);

   const char *argv3D[]= { "tmop_tests",
                           "-m", "../../data/toroid-hex.mesh",
                           "-mid", "302",
                           nullptr
                         };
   REQUIRE(tmop(myid, argn(argv3D), const_cast<char**>(argv3D))==0);

   const char *argv3Drs1[]= { "tmop_tests",
                              "-m", "../../data/toroid-hex.mesh",
                              "-mid", "302",
                              "-rs", "1",
                              "-ni", "20",
                              nullptr
                            };
   REQUIRE(tmop(myid, argn(argv3Drs1), const_cast<char**>(argv3Drs1))==0);

}

#if defined(MFEM_TMOP_MPI)
#ifndef MFEM_TMOP_TESTS
TEST_CASE("TMOP", "[TMOP], [Parallel]")
{
   tmop_tests(GlobalMPISession->WorldRank());
}
#else
TEST_CASE("TMOP", "[TMOP], [Parallel]")
{
   Device device;
   device.Configure(MFEM_TMOP_DEVICE);
   device.Print();
   tmop_tests(GlobalMPISession->WorldRank());
}
#endif
#else
#ifndef MFEM_TMOP_TESTS
TEST_CASE("TMOP", "[TMOP]")
{
   tmop_tests(0);
}
#else
TEST_CASE("TMOP", "[TMOP]")
{
   Device device;
   device.Configure(MFEM_TMOP_DEVICE);
   device.Print();
   tmop_tests(0);
}
#endif
#endif
