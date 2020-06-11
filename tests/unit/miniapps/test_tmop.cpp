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
#include <iostream>

#include "catch.hpp"

#include "mfem.hpp"

#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
extern mfem::MPI_Session *GlobalMPISession;
#define PFesGetParMeshGetComm(pfes) pfes.GetParMesh()->GetComm()
#else
typedef int MPI_Session;
#define ParMesh Mesh
#define ParGridFunction GridFunction
#define ParNonlinearForm NonlinearForm
#define ParFiniteElementSpace FiniteElementSpace
#define GetParGridFunctionEnergy GetGridFunctionEnergy
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
   Req(double ie, double t, double d, double fe):
      init_energy(ie), tauval(t), dot(d), final_energy(fe) { }
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
   if (verbosity_level > 0) { if (myid == 0) {args.PrintOptions(cout); } }

   REQUIRE(mesh_file);
   Mesh smesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { smesh.UniformRefinement(); }
   const int dim = smesh.Dimension();
   ParMesh *pmesh = nullptr;
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
   pmesh = new ParMesh(MPI_COMM_WORLD, smesh);
#else
   pmesh = new Mesh(smesh);
#endif

   REQUIRE(order > 0);
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(pmesh, &fec, dim);
   pmesh->SetNodalFESpace(&fes);
   ParGridFunction x0(&fes), x(&fes);
   pmesh->SetNodalGridFunction(&x);
   x.SetTrueVector();
   x.SetFromTrueVector();
   x0 = x;

   TMOP_QualityMetric *metric = nullptr;
   switch (metric_id)
   {
      case   1: metric = new TMOP_Metric_001; break;
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
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      //case 4: // Analytic
      //case 5: // Discrete size 2D
      //case 6: // Discrete size + aspect ratio - 2D
      //case 7: // Discrete aspect ratio 3D
      //case 8: // shape/size + orientation 2D
      default:
      {
         if (myid == 0) { cout << "Unknown target_id: " << target_id << endl; }
         return 3;
      }
   }
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
   TargetConstructor target_c(target_t,MPI_COMM_WORLD);
#else
   TargetConstructor target_c(target_t);
#endif
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
         return 4;
      }
   }

   TMOP_Integrator *he_nlf_integ = new TMOP_Integrator(metric, &target_c);
   he_nlf_integ->SetIntegrationRule(*ir);

   ParNonlinearForm nlf(&fes);
   nlf.SetAssemblyLevel(pa ? AssemblyLevel::PARTIAL : AssemblyLevel::NONE);
   nlf.AddDomainIntegrator(he_nlf_integ);
   nlf.Setup();

   const double init_energy = nlf.GetParGridFunctionEnergy(x);

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
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
   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      ElementTransformation *transf = pmesh->GetElementTransformation(i);
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

   REQUIRE(init_energy == Approx(res.init_energy));

   REQUIRE(newton->GetConverged());
   double x_t_dot = x_t*x_t, dot;
   MPI_Allreduce(&x_t_dot, &dot, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   REQUIRE(dot == Approx(res.dot));

   x.SetFromTrueVector();
   const double final_energy = nlf.GetParGridFunctionEnergy(x);
   REQUIRE(final_energy == Approx(res.final_energy));

   delete S;
   delete pmesh;
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
   (arg[1] = "-no-pa", req_tmop(myid, arg, res));
   (arg[1] = "-pa", req_tmop(myid, arg, res));
}

static void tmop_tests(int myid)
{
   constexpr int MSH = 3;
   constexpr int ORD = 5;
   constexpr int MID = 9;
   constexpr int TID = 11;
   constexpr int QO  = 15;
   constexpr int NI  = 17;
   constexpr int LI  = 23;

   const char *a2D[] = { "tmop_tests", "-pa",
                         "-m", "star.mesh",
                         "-o", "1",
                         "-rs", "0",
                         "-mid", "001",
                         "-tid", "1",
                         "-qt", "1",
                         "-qo", "2",
                         "-ni", "10",
                         "-rtol", "1e-8",
                         "-ls", "2",
                         "-li", "100",
                         nullptr
                       };
   Req r112 { 9.999991262, 0.2377610897, 39.1647887817, 9.7969547569 };
   tmop_launch(myid, a2D, r112);

   a2D[ORD] = "2";
   Req r212 { 9.999991262, 0.2377610897, 108.6173503755, 9.7244865423 };
   tmop_launch(myid, a2D, r212);

   a2D[QO] = "8";
   Req r218 { 9.999991262, 0.2377610897, 108.6534469132, 9.7167152788 };
   tmop_launch(myid, a2D, r218);

   a2D[ORD] = "1";
   a2D[MSH] = "blade.mesh";
   a2D[MID] = "002";
   a2D[QO]  = "2";
   a2D[LI]  = "100";
   Req r122 { 170.5301636144, 0.0000708422, 92.6143439914, 72.9039715692 };
   tmop_launch(myid, a2D, r122);

   a2D[ORD] = "2";
   a2D[NI]  = "20";
   Req r222 { 171.1323988131, 0.000064793, 325.1465122229, 69.7164293181 };
   tmop_launch(myid, a2D, r222);

   a2D[QO] = "8";
   Req r228 { 170.2231639887, 0.0000663019, 325.1400405167, 69.432996753 };
   tmop_launch(myid, a2D, r228);

   a2D[ORD] = "1";
   a2D[TID] = "2";
   a2D[NI]  = "20";
   a2D[LI]  = "400";
   Req r1228 { 1.9240173328, 0.0000663019, 92.642204486, 0.8201983284 };
   tmop_launch(myid, a2D, r1228);

   a2D[TID] = "3";
   Req r1328 { 1.0141986391, 0.0000708422, 94.9910671635, 0.5392898375 };
   tmop_launch(myid, a2D, r1328);

   const char *a3D[]= { "tmop_tests", "-pa",
                        "-m", "toroid-hex.mesh",
                        "-o", "1",
                        "-rs", "1",
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
   Req r1_302 {23.7771704247, 0.0230794426, 118.5803592995, 23.0610659099};
   tmop_launch(myid, a3D, r1_302);

   a3D[MID] = "303";
   Req r1_303 {12.2180210422, 0.0230794426, 118.2536942233, 11.765513442};
   tmop_launch(myid, a3D, r1_303);

   a3D[MID] = "321";
   Req r1_321 {1234.3897639272, 0.0230794426, 119.518581363, 1234.2552718299};
   tmop_launch(myid, a3D, r1_321);

   a3D[ORD] = "2";
   Req r2_321 {1144.8574374136, 0.0250315411, 649.0553569081, 1144.4861740306};
   tmop_launch(myid, a3D, r2_321);

   a3D[QO] = "8";
   Req r2_321_8 {1145.1635275919, 0.0250315411, 649.1136786233, 1144.7572817515};
   tmop_launch(myid, a3D, r2_321_8);

   a3D[TID] = "2";
   Req r_2_321_2_8 {2.5600626823, 0.0250315411, 644.267383563, 2.5385985405};
   tmop_launch(myid, a3D, r_2_321_2_8);

   a3D[TID] = "3";
   Req r_2_321_3_8 {2.6264613333, 0.0250315411, 639.6722100713, 2.5209961176};
   tmop_launch(myid, a3D, r_2_321_3_8);
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
