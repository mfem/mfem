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

#include <fstream>
#include <iostream>

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
#define ParNonlinearForm NonlinearForm
#define ParGridFunction GridFunction
#define GetParGridFunctionEnergy GetGridFunctionEnergy
#define ParFiniteElementSpace FiniteElementSpace
#define PFesGetParMeshGetComm(...)
#define MPI_Allreduce(src,dst,...) *dst = *src
#endif

using namespace std;
using namespace mfem;

namespace mfem
{

struct Req
{
   const double init_energy;
   const double tauval;
   const double dot;
   const double final_energy;
   Req(double init_energy,
       double tauval,
       double dot,
       double final_energy):
      init_energy(init_energy),
      tauval(tauval),
      dot(dot),
      final_energy(final_energy) {}
};

int tmop(int myid, const Req &res, int argc, char *argv[])
{
   bool pa               = false;
   const char *mesh_file = nullptr;
   int order             = 1;
   int rs_levels         = 0;
   int metric_id         = 1;
   int target_id         = 1;
   int quad_type         = 1;
   int quad_order        = 2;
   int newton_iter       = 10;
   double newton_rtol    = 1e-8;
   int lin_solver        = 2;
   int max_lin_iter      = 100;

   constexpr double lim_const = 0.0;
   constexpr double adapt_lim_const = 0.0;
   constexpr bool move_bnd = false;
   constexpr int combomet  = 0;
   constexpr bool normalization = false;
   constexpr int verbosity_level = 0;
   constexpr bool fdscheme = false;

   REQUIRE_FALSE(normalization);
   REQUIRE(lim_const==0.0);
   REQUIRE(adapt_lim_const == 0.0);
   REQUIRE_FALSE(fdscheme);
   REQUIRE(combomet == 0);
   REQUIRE_FALSE(move_bnd);

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
   args.AddOption(&pa, "-pa", "--pa", "-no-pa", "--no-pa", "");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (verbosity_level > 0) { args.PrintOptions(cout); }

   REQUIRE(mesh_file);
   Mesh smsh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { smsh.UniformRefinement(); }
   const int dim = smsh.Dimension();
   ParMesh *pmsh = nullptr;
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
   pmsh = new ParMesh(MPI_COMM_WORLD, smsh);
#else
   pmsh = new Mesh(smsh);
#endif

   REQUIRE(order > 0);
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(pmsh, &fec, dim);
   pmsh->SetNodalFESpace(&fes);
   ParGridFunction x0(&fes), x(&fes);
   pmsh->SetNodalGridFunction(&x);
   x.SetTrueVector();
   x.SetFromTrueVector();
   x0 = x;

   TMOP_QualityMetric *metric = nullptr;
   switch (metric_id)
   {
      case   2: metric = new TMOP_Metric_002; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 321: metric = new TMOP_Metric_321; break;
      default:
      {
         if (myid == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
         return 2;
      }
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
   TargetConstructor target_c(target_t);
   target_c.SetNodes(x0);

   // Setup the quadrature rule for the non-linear form integrator.
   const IntegrationRule *ir = nullptr;
   IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
   IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);
   const int geom_type = fes.GetFE(0)->GetGeomType();
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order); break;
      case 2: ir = &IntRules.Get(geom_type, quad_order); break;
      case 3: ir = &IntRulesCU.Get(geom_type, quad_order); break;
      default:
      {
         if (myid == 0) { cout << "Unknown quad_type: " << quad_type << endl; }
         return 3;
      }
   }

   TMOP_Integrator *he_nlf_integ = new TMOP_Integrator(metric, &target_c);
   he_nlf_integ->SetIntegrationRule(*ir);

   ParNonlinearForm nlf(&fes);
   nlf.SetAssemblyLevel(pa ? AssemblyLevel::PARTIAL : AssemblyLevel::NONE);
   nlf.AddDomainIntegrator(he_nlf_integ);
   nlf.Setup();

   const double init_energy = nlf.GetParGridFunctionEnergy(x);
   REQUIRE(init_energy == Approx(res.init_energy));

   Array<int> ess_bdr(pmsh->bdr_attributes.Max());
   ess_bdr = 1;
   nlf.SetEssentialBC(ess_bdr);

   Solver *S = nullptr;
   constexpr double linsol_rtol = 1e-12;
   if (lin_solver == 0)
   {
      S = new DSmoother(1, 1.0, max_lin_iter);
   }
   else if (lin_solver == 1)
   {
      CGSolver *cg = new CGSolver(PFesGetParMeshGetComm(fes));
      cg->SetMaxIter(max_lin_iter);
      cg->SetRelTol(linsol_rtol);
      cg->SetAbsTol(0.0);
      cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = cg;
   }
   else
   {
      MINRESSolver *minres = new MINRESSolver(PFesGetParMeshGetComm(fes));
      minres->SetMaxIter(max_lin_iter);
      minres->SetRelTol(linsol_rtol);
      minres->SetAbsTol(0.0);
      minres->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = minres;
   }

   double tauval = infinity();
   for (int i = 0; i < pmsh->GetNE(); i++)
   {
      ElementTransformation *transf = pmsh->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir->IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   double minJ0;
   MPI_Allreduce(&tauval, &minJ0, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   tauval = minJ0;
   if (myid == 0) { cout << "Min det(J) of the mesh is " << tauval << endl; }
   REQUIRE(tauval > 0.0);
   REQUIRE(tauval == Approx(res.tauval));

   Vector b(0), &x_t(x.GetTrueVector());
   b.UseDevice(true);
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
   NewtonSolver *newton = new TMOPNewtonSolver(PFesGetParMeshGetComm(fes),*ir);
#else
   NewtonSolver *newton = new TMOPNewtonSolver(*ir);
#endif
   newton->SetPreconditioner(*S);
   newton->SetMaxIter(newton_iter);
   newton->SetRelTol(newton_rtol);
   newton->SetAbsTol(0.0);
   newton->SetPrintLevel(verbosity_level >= 1 ? 1 : -1);
   newton->SetOperator(nlf);
   newton->Mult(b, x_t);
   REQUIRE(newton->GetConverged());

   double x_t_dot = x_t*x_t, dot;
   MPI_Allreduce(&x_t_dot, &dot, 1, MPI_DOUBLE, MPI_SUM, pmsh->GetComm());
   REQUIRE(dot == Approx(res.dot));

   x.SetFromTrueVector();
   const double final_energy = nlf.GetParGridFunctionEnergy(x);
   REQUIRE(final_energy == Approx(res.final_energy));

   delete S;
   delete pmsh;
   delete metric;
   delete newton;

   return 0;
}

} // namespace mfem

static int argn(const char *argv[], int argc =0)
{
   while (argv[argc]) { argc+=1; }
   return argc;
}

static void req_tmop(int myid, const char *arg[], const Req &res)
{ REQUIRE(tmop(myid, res, argn(arg), const_cast<char**>(arg))==0); }

static void tmop_launch(int myid, const char *arg[], const Req &res)
{
   (arg[1] = "-pa", req_tmop(myid, arg, res));
   (arg[1] = "-no-pa", req_tmop(myid, arg, res));
}

static void tmop_tests(int myid)
{
   constexpr int ORD = 5;
   constexpr int MID = 9;
   constexpr int QO  = 15;
   constexpr int NI  = 17;

   const char *a2D[] = { "tmop_tests", "-pa",
                         "-m", "blade.mesh",
                         "-o", "1",
                         "-rs", "0",
                         "-mid", "002",
                         "-tid", "1",
                         "-qt", "1",
                         "-qo", "2",
                         "-ni", "10",
                         "-rtol", "1e-8",
                         "-ls", "2",
                         "-li", "100",
                         nullptr
                       };
   Req r122 { 170.5301636144, 0.0000708422, 92.6143439914, 72.9039715692 };
   tmop_launch(myid, a2D, r122);

   a2D[ORD] = "2";
   a2D[NI] = "15";
   Req r222 { 171.1323988131, 0.000064793, 325.1465122229, 69.7164293181 };
   tmop_launch(myid, a2D, r222);

   a2D[QO] = "8";
   Req r228 { 170.2231639887, 0.0000663019, 325.1400405167, 69.432996753 };
   tmop_launch(myid, a2D, r228);

   const char *a3D[]= { "tmop_tests", "-pa",
                        "-m", "toroid-hex.mesh",
                        "-o", "1",
                        "-rs", "0",
                        "-mid", "302",
                        "-tid", "1",
                        "-qt", "1",
                        "-qo", "2",
                        "-ni", "10",
                        "-rtol", "1e-8",
                        "-ls", "2",
                        "-li", "100",
                        nullptr
                      };
   a3D[MID] = "302";
   Req r1302 { 3.9932136939, 0.1419261902, 27.8400002696, 3.9932136939 };
   tmop_launch(myid, a3D, r1302);

   a3D[MID] = "303";
   Req r1303 { 1.9844641874, 0.1419261902, 27.8400002696, 1.9844641874 };
   tmop_launch(myid, a3D, r1303);

   a3D[MID] = "321";
   Req r1321 { 30.9569126183, 0.1419261902, 27.8400002696, 30.9569126183 };
   tmop_launch(myid, a3D, r1321);

   a3D[ORD] = "2";
   Req r2321 { 19.0579987606, 0.2054996096, 119.4242345677, 19.0576995427 };
   tmop_launch(myid, a3D, r2321);

   a3D[QO] = "8";
   Req r23218 { 19.1695955389, 0.2002639462, 119.4426521542, 19.1687878956 };
   tmop_launch(myid, a3D, r23218);
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
