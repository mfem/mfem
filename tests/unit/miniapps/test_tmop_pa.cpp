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

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include <list>
#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "unit_tests.hpp"
#include "miniapps/meshing/mesh-optimizer.hpp"

#if defined(MFEM_TMOP_MPI) && !defined(MFEM_USE_MPI)
#error "Cannot use MFEM_TMOP_MPI without MFEM_USE_MPI!"
#endif

#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
extern mfem::MPI_Session *GlobalMPISession;
#define PFesGetParMeshGetComm(pfes) pfes.GetComm()
#define SetDiscreteTargetSize SetParDiscreteTargetSize
#define SetDiscreteTargetAspectRatio SetParDiscreteTargetAspectRatio
#define GradientClass HypreParMatrix
#else
typedef int MPI_Session;
#define ParMesh Mesh
#define ParGridFunction GridFunction
#define ParNonlinearForm NonlinearForm
#define ParFiniteElementSpace FiniteElementSpace
#define GetParGridFunctionEnergy GetGridFunctionEnergy
#define PFesGetParMeshGetComm(...)
#define MPI_Allreduce(src,dst,...) *dst = *src
#define SetDiscreteTargetSize SetSerialDiscreteTargetSize
#define SetDiscreteTargetAspectRatio SetSerialDiscreteTargetAspectRatio
#define GradientClass SparseMatrix
#define ParEnableNormalization EnableNormalization
#endif

using namespace std;
using namespace mfem;

namespace mfem
{

struct Req
{
   double init_energy;
   double tauval;
   double dot;
   double final_energy;
   double diag;
};

int tmop(int id, Req &res, int argc, char *argv[])
{
   bool pa               = false;
   const char *mesh_file = nullptr;
   int order             = 1;
   int rs_levels         = 0;
   int metric_id         = 1;
   int target_id         = 1;
   int quad_type         = 1;
   int quad_order        = 2;
   int newton_iter       = 100;
   double newton_rtol    = 1e-8;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   double lim_const      = 0.0;
   int normalization     = 0;
   double jitter         = 0.0;
   bool diag             = true;
   int newton_loop       = 1;
   int combo             = 0;

   constexpr int verbosity_level = 0;
   constexpr int seed = 0x100001b3;
   constexpr bool move_bnd = false;
   constexpr bool fdscheme = false;
   constexpr bool exactaction = false;

   REQUIRE_FALSE(fdscheme);
   REQUIRE_FALSE(move_bnd);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "");
   args.AddOption(&order, "-o", "--order", "");
   args.AddOption(&rs_levels, "-rs", "--refine-serial", "");
   args.AddOption(&metric_id, "-mid", "--metric-id", "");
   args.AddOption(&target_id, "-tid", "--target-id", "");
   args.AddOption(&quad_type, "-qt", "--quad-type", "");
   args.AddOption(&quad_order, "-qo", "--quad_order", "");
   args.AddOption(&newton_iter, "-ni", "--newton-iters","");
   args.AddOption(&newton_loop, "-nl", "--newton-loops","");
   args.AddOption(&newton_rtol, "-rtol", "--newton-rel-tolerance", "");
   args.AddOption(&lin_solver, "-ls", "--lin-solver", "");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter", "");
   args.AddOption(&lim_const, "-lc", "--limit-const", "");
   args.AddOption(&normalization, "-nor", "--normalization", "");
   args.AddOption(&pa, "-pa", "--pa", "-no-pa", "--no-pa", "");
   args.AddOption(&jitter, "-ji", "--jitter", "");
   args.AddOption(&diag, "-diag", "--diag", "-no-diag", "--no-diag", "");
   args.AddOption(&combo, "-cmb", "--combo-type", "");
   args.Parse();
   if (!args.Good())
   {
      if (id == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (verbosity_level > 0) { if (id == 0) {args.PrintOptions(cout); } }

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
   smesh.Clear();

   REQUIRE(order > 0);
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(pmesh, &fec, dim);
   ParGridFunction x0(&fes), x(&fes);
   pmesh->SetNodalGridFunction(&x);

   Vector h0(fes.GetNDofs());
   h0 = infinity();
   double volume = 0.0;
   {
      Array<int> dofs;
      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         fes.GetElementDofs(i, dofs);
         const double hi = pmesh->GetElementSize(i);
         for (int j = 0; j < dofs.Size(); j++)
         {
            h0(dofs[j]) = min(h0(dofs[j]), hi);
         }
         volume += pmesh->GetElementVolume(i);
      }
   }
   const double small_phys_size = pow(volume, 1.0 / dim) / 100.0;

   ParGridFunction rdm(&fes);
   rdm.Randomize(seed);
   rdm -= 0.5;
   rdm *= jitter;
   rdm.HostReadWrite();
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < fes.GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(fes.DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < fes.GetNBE(); i++)
   {
      fes.GetBdrElementVDofs(i, vdofs);
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   x -= rdm;
   x.SetTrueVector();
   x.SetFromTrueVector();
   x0 = x;

   TMOP_QualityMetric *metric = nullptr;
   switch (metric_id)
   {
      case   1: metric = new TMOP_Metric_001; break;
      case   2: metric = new TMOP_Metric_002; break;
      case   7: metric = new TMOP_Metric_007; break;
      case  77: metric = new TMOP_Metric_077; break;
      case  80: metric = new TMOP_Metric_080(0.5); break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 315: metric = new TMOP_Metric_315; break;
      case 321: metric = new TMOP_Metric_321; break;
      default:
      {
         if (id == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
         return 2;
      }
   }

   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = nullptr;
   HessianCoefficient *adapt_coeff = nullptr;
   constexpr int mesh_poly_deg = 1;
   H1_FECollection ind_fec(mesh_poly_deg, dim);
   ParFiniteElementSpace ind_fes(pmesh, &ind_fec);
   ParGridFunction size(&ind_fes);
   ParFiniteElementSpace ind_fesv(pmesh, &ind_fec, dim);
   ParGridFunction aspr3d(&ind_fesv);
   const AssemblyLevel al =
      pa ? AssemblyLevel::PARTIAL : AssemblyLevel::LEGACY;

   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      case 4: // Analytic
      {
         target_t = TargetConstructor::GIVEN_FULL;
         AnalyticAdaptTC *tc = new AnalyticAdaptTC(target_t);
         adapt_coeff = new HessianCoefficient(dim, metric_id);
         tc->SetAnalyticTargetSpec(NULL, NULL, adapt_coeff);
         target_c = tc;
         break;
      }
      case 5: // Discrete size 2D or 3D
      {
         target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
         DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
         tc->SetAdaptivityEvaluator(new AdvectorCG(al));
         if (dim == 2)
         {
            FunctionCoefficient ind_coeff(discrete_size_2d);
            size.ProjectCoefficient(ind_coeff);
         }
         else if (dim == 3)
         {
            FunctionCoefficient ind_coeff(discrete_size_3d);
            size.ProjectCoefficient(ind_coeff);
         }
         tc->SetDiscreteTargetSize(size);
         target_c = tc;
         break;
      }
      case 7: // aspect-ratio 3D
      {
         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
         tc->SetAdaptivityEvaluator(new AdvectorCG(al));
         VectorFunctionCoefficient fd_aspr3d(dim, discrete_aspr_3d);
         aspr3d.ProjectCoefficient(fd_aspr3d);
         tc->SetDiscreteTargetAspectRatio(aspr3d);
         target_c = tc;
         break;
      }
      default:
      {
         if (id == 0) { cout << "Unknown target_id: " << target_id << endl; }
         return 3;
      }
   }
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   }
#else
   if (target_c == nullptr)
   {
      target_c = new TargetConstructor(target_t);
   }
#endif
   target_c->SetNodes(x0);

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
         if (id == 0) { cout << "Unknown quad_type: " << quad_type << endl; }
         return 4;
      }
   }

   TMOP_Integrator *he_nlf_integ = new TMOP_Integrator(metric, target_c);
   he_nlf_integ->SetIntegrationRule(*ir);

   if (normalization == 1) { he_nlf_integ->ParEnableNormalization(x0); }

   ParFiniteElementSpace dist_fes(pmesh, &fec); // scalar space
   ParGridFunction dist(&dist_fes);
   dist = 1.0;
   if (normalization == 1) { dist = small_phys_size; }
   ConstantCoefficient lim_coeff(lim_const);
   if (lim_const != 0.0) { he_nlf_integ->EnableLimiting(x0, dist, lim_coeff); }

   ParNonlinearForm nlf(&fes);
   nlf.SetAssemblyLevel(pa ? AssemblyLevel::PARTIAL : AssemblyLevel::LEGACY);

   ConstantCoefficient *coeff1 = nullptr;
   TMOP_QualityMetric *metric2 = nullptr;
   TargetConstructor *target_c2 = nullptr;
   FunctionCoefficient coeff2(weight_fun);
   if (combo > 0)
   {
      // First metric.
      coeff1 = new ConstantCoefficient(1.0);
      he_nlf_integ->SetCoefficient(*coeff1);
      // Second metric.
      if (dim == 2) { metric2 = new TMOP_Metric_077; }
      else          { metric2 = new TMOP_Metric_315; }
      TMOP_Integrator *he_nlf_integ2 = nullptr;
      if (combo == 1)
      {
         target_c2 = new TargetConstructor(
            TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE);
         target_c2->SetVolumeScale(0.01);
         target_c2->SetNodes(x0);
         he_nlf_integ2 = new TMOP_Integrator(metric2, target_c2);
         he_nlf_integ2->SetCoefficient(coeff2);
      }
      else { he_nlf_integ2 = new TMOP_Integrator(metric2, target_c); }
      he_nlf_integ2->SetIntegrationRule(*ir);
      if (fdscheme) { he_nlf_integ2->EnableFiniteDifferences(x); }
      he_nlf_integ2->SetExactActionFlag(exactaction);
      TMOPComboIntegrator *combo = new TMOPComboIntegrator;
      combo->AddTMOPIntegrator(he_nlf_integ);
      combo->AddTMOPIntegrator(he_nlf_integ2);
      if (normalization) { combo->ParEnableNormalization(x0); }
      if (lim_const != 0.0) { combo->EnableLimiting(x0, dist, lim_coeff); }
      nlf.AddDomainIntegrator(combo);
   }
   else
   {
      nlf.AddDomainIntegrator(he_nlf_integ);
   }
   nlf.Setup();

   const double init_energy = nlf.GetParGridFunctionEnergy(x);
   res.init_energy = init_energy;

   // Fix all boundary nodes (-fix-bnd)
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   nlf.SetEssentialBC(ess_bdr);

   // Diagonal test, skip if combo
   Vector &xt(x.GetTrueVector());
   Vector d(fes.GetTrueVSize());
   d.UseDevice(true);
   res.diag = 0.0;
   if (diag && combo == 0)
   {
      if (pa) { nlf.GetGradient(xt).AssembleDiagonal(d); }
      else
      {
         ParNonlinearForm nlf_fa(&fes);
         TMOP_Integrator *nlfi_fa = new TMOP_Integrator(metric, target_c);
         nlfi_fa->SetIntegrationRule(*ir);
         if (normalization == 1) { nlfi_fa->ParEnableNormalization(x0); }
         if (lim_const != 0.0) { nlfi_fa->EnableLimiting(x0, dist, lim_coeff); }
         nlf_fa.AddDomainIntegrator(nlfi_fa);
         nlf_fa.SetEssentialBC(ess_bdr);
         dynamic_cast<GradientClass&>(nlf_fa.GetGradient(xt)).GetDiag(d);
      }
      res.diag = d*d;
   }

   // Linear solver for the system's Jacobian
   Solver *S = nullptr, *S_prec = nullptr;
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
      if (lin_solver == 3 || lin_solver == 4)
      {
         if (pa)
         {
            MFEM_VERIFY(lin_solver != 4, "PA l1-Jacobi is not implemented");
            S_prec = new OperatorJacobiSmoother;
         }
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
         else
         {
            HypreSmoother *hs = new HypreSmoother;
            hs->SetType((lin_solver == 3) ? HypreSmoother::Jacobi
                        : HypreSmoother::l1Jacobi, 1);
            S_prec = hs;
         }
#else
         else { S_prec = new DSmoother((lin_solver == 3) ? 0 : 1, 1.0, 1); }
#endif
         minres->SetPreconditioner(*S_prec);
      }
      S = minres;
   }

   // Compute the minimum det(J) of the starting mesh
   double tauval = infinity();
   const int NE = pmesh->GetNE();
   for (int i = 0; i < NE; i++)
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
   //if (id == 0) { cout << "Min det(J) of the mesh is " << tauval << endl; }
   REQUIRE(tauval > 0.0);
   double h0min = h0.Min(), h0min_all;
   MPI_Allreduce(&h0min, &h0min_all, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   tauval -= 0.01 * h0min_all; // Slightly below minJ0 to avoid div by 0.
   res.tauval = tauval;

   // Perform the nonlinear optimization
   Vector b(0);
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

   Vector x_init(x);
   for (int i = 0; i < newton_loop; i++)
   {
      x = x_init;
      x.SetTrueVector();

      DiscreteAdaptTC *datc = dynamic_cast<DiscreteAdaptTC*>(target_c);
      if (datc && target_id == 5) { datc->SetDiscreteTargetSize(size); }
      if (datc && target_id == 7) { datc->SetDiscreteTargetAspectRatio(aspr3d); }

      dist *= 0.93;
      if (normalization == 1) { dist = small_phys_size; }

      ConstantCoefficient lim_coeff(lim_const);
      if (lim_const != 0.0) { he_nlf_integ->EnableLimiting(x0, dist, lim_coeff); }

      if (normalization == 1) { he_nlf_integ->ParEnableNormalization(x); }

      nlf.Setup();

      newton->Mult(b, x.GetTrueVector());
      x.SetFromTrueVector();

      REQUIRE(newton->GetConverged());

      const double final_energy = nlf.GetParGridFunctionEnergy(x);
      res.final_energy = final_energy;
   }

   Vector &x_t(x.GetTrueVector());
   double x_t_dot = x_t*x_t, dot;
   MPI_Allreduce(&x_t_dot, &dot, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   res.dot = dot;

   delete S;
   delete S_prec;
   delete pmesh;
   delete metric;
   delete coeff1;
   delete metric2;
   delete newton;
   delete target_c;
   delete target_c2;
   delete adapt_coeff;

   return 0;
}

} // namespace mfem

static int argn(const char *argv[], int argc =0)
{
   while (argv[argc]) { argc+=1; }
   return argc;
}

static void req_tmop(int id, const char *args[], Req &res)
{ REQUIRE(tmop(id, res, argn(args), const_cast<char**>(args))==0); }

#define DEFAULT_ARGS const char *args[] = { \
    "tmop_tests", "-pa", "-m", "mesh", "-o", "0", "-rs", "0", \
    "-mid", "0", "-tid", "0", "-qt", "1", "-qo", "0", \
    "-ni", "10", "-rtol", "1e-8", "-ls", "2", "-li", "100", \
    "-lc", "0", "-nor", "0", "-ji", "0", "-nl", "1", "-cmb", "0", nullptr }
constexpr int ALV = 1;
constexpr int MSH = 3;
constexpr int POR = 5;
constexpr int RS = 7;
constexpr int MID = 9;
constexpr int TID = 11;
constexpr int QTY = 13;
constexpr int QOR = 15;
constexpr int NI  = 17;
constexpr int LS  = 21;
constexpr int LI  = 23;
constexpr int LC  = 25;
constexpr int NOR = 27;
constexpr int JI  = 29;
constexpr int NL  = 31;
constexpr int CMB = 33;

static void dump_args(int id, const char *args[])
{
   if (id != 0) { return; }
   const char *format =
      "tmop -m %s -o %s -qo %s -mid %s -tid %s -ls %s"
      "%s%s%s%s"         // Optional args: RS, QTY
      "%s%s%s%s%s%s%s%s" // Optional args: LC, NOR, JI, NL & CMB
      " %s\n";           // Assembly level
   printf(format,
          args[MSH], args[POR], args[QOR], args[MID], args[TID], args[LS],
          // Optional args: RS, QTY
          args[RS][0] == '0' ? "" : " -rs ",
          args[RS][0] == '0' ? "" : args[RS],
          args[QTY][0] == '1' ? "" : " -qt ",
          args[QTY][0] == '1' ? "" : args[QTY],
          // Optional args: LC, NOR, JI, NL & CMB
          args[LC][0] == '0' ? "" : " -lc ",
          args[LC][0] == '0' ? "" : args[LC],
          args[NOR][0] == '0' ? "" : " -nor",
          atof(args[JI]) == 0.0 ? "" : " -jitter",
          args[NL][0] == '1' ? "" : " -nl ",
          args[NL][0] == '1' ? "" : args[NL],
          args[CMB][0] == '0' ? "" : " -cmb ",
          args[CMB][0] == '0' ? "" : args[CMB],
          // Assembly level
          args[ALV]);
   fflush(0);
}

static void tmop_require(int id, const char *args[])
{
   Req res[2];
   (args[ALV] = "-pa", dump_args(id, args), req_tmop(id, args, res[0]));
   (args[ALV] = "-no-pa", dump_args(id, args), req_tmop(id, args, res[1]));
   REQUIRE(res[0].dot == MFEM_Approx(res[1].dot));
   REQUIRE(res[0].tauval == MFEM_Approx(res[1].tauval));
   REQUIRE(res[0].init_energy == MFEM_Approx(res[1].init_energy));
   REQUIRE(res[0].final_energy == MFEM_Approx(res[1].final_energy, 2e-12));
   REQUIRE(res[0].diag == MFEM_Approx(res[1].diag));
}

static inline const char *itoa(const int i, char *buf)
{
   std::sprintf(buf, "%d", i);
   return buf;
}

static inline const char *dtoa(const double d, char *buf)
{
   std::sprintf(buf, "%.4f", d);
   return buf;
}

class Launch
{
   typedef std::list<int> set;
public:
   class Args
   {
      friend class Launch;
   private:
      const char *name = nullptr;
      const char *mesh = "../../data/star.mesh";
      int newton_iter = 100;
      int rs_levels = 0;
      int max_lin_iter  = 100;
      int combo = 0;
      bool normalization = false;
      double lim_const = 0.0;
      double jitter = 0.0;
      set order = {1,2,3,4};
      set target_id = {1,2,3};
      set metric_id = {1,2};
      set quad_order = {2,4,8};
      set lin_solver = {3,2,1};
      set newton_loop = {1,3};

   public:
      Args(const char *name =nullptr): name(name) {}
      Args &MESH(const char *arg) { mesh = arg; return *this; }
      Args &NEWTON_ITERATIONS(const int arg) { newton_iter = arg; return *this; }
      Args &REFINE(const int arg) { rs_levels = arg; return *this; }
      Args &LINEAR_ITERATIONS(const int arg) { max_lin_iter = arg; return *this; }
      Args &CMB(const int arg) { combo = arg; return *this; }
      Args &NORMALIZATION(const bool arg) { normalization = arg; return *this; }
      Args &LIMITING(const double arg) { lim_const = arg; return *this; }
      Args &JI(const double arg) { jitter = arg; return *this; }

      Args &POR(set arg) { order = arg; return *this; }
      Args &TID(set arg) { target_id = arg; return *this; }
      Args &MID(set arg) { metric_id = arg; return *this; }
      Args &QOR(set arg) { quad_order = arg; return *this; }
      Args &LS(set arg) { lin_solver = arg; return *this; }
      Args &NL(set arg) { newton_loop = arg; return *this; }
   };
   const char *name, *mesh;
   int NEWTON_ITERATIONS, REFINE, LINEAR_ITERATIONS, COMBO;
   bool NORMALIZATION;
   double LIMITING, JITTER;
   set P_ORDERS, TARGET_IDS, METRIC_IDS, Q_ORDERS, LINEAR_SOLVERS, NEWTON_LOOPS;
public:
   Launch(Args a = Args()):
      name(a.name), mesh(a.mesh),
      NEWTON_ITERATIONS(a.newton_iter), REFINE(a.rs_levels),
      LINEAR_ITERATIONS(a.max_lin_iter), COMBO(a.combo),
      NORMALIZATION(a.normalization), LIMITING(a.lim_const), JITTER(a.jitter),
      P_ORDERS(a.order), TARGET_IDS(a.target_id), METRIC_IDS(a.metric_id),
      Q_ORDERS(a.quad_order), LINEAR_SOLVERS(a.lin_solver),
      NEWTON_LOOPS(a.newton_loop)
   { }

   void Run(const int id = 0, bool nr = false) const
   {
      if ((id==0) && name) { mfem::out << "[" << name << "]" << std::endl; }
      DEFAULT_ARGS;
      char ni[8] {}, rs[8] {}, li[8] {}, lc[16] {}, ji[16] {}, cmb[8] {};
      args[MSH] = mesh;
      args[RS] = itoa(REFINE,rs);
      args[NI] = itoa(NEWTON_ITERATIONS,ni);
      args[LI] = itoa(LINEAR_ITERATIONS,li);
      args[CMB] = itoa(COMBO,cmb);
      args[LC] = dtoa(LIMITING,lc);
      args[JI] = dtoa(JITTER,ji);
      args[NOR] = NORMALIZATION ? "1" : "0";
      for (int p : P_ORDERS)
      {
         char por[2] {};
         args[POR] = itoa(p, por);
         for (int t : TARGET_IDS)
         {
            char tid[2] {};
            args[TID] = itoa(t, tid);
            for (int m : METRIC_IDS)
            {
               char mid[4] {};
               args[MID] = itoa(m, mid);
               for (int q : Q_ORDERS)
               {
                  if (q <= p) { continue; }
                  char qor[2] {};
                  args[QOR] = itoa(q, qor);
                  for (int ls : LINEAR_SOLVERS)
                  {
                     // skip some linear solver & metric combinations
                     // that lead to non positive definite operators
                     if (ls == 1 && m != 1) { continue; }
                     char lsb[2] {};
                     args[LS] = itoa(ls, lsb);
                     for (int n : NEWTON_LOOPS)
                     {
                        char nl[2] {};
                        args[NL] = itoa(n, nl);
                        tmop_require(id, args);
                        if (!nr) { break; }
                     }
                     if (!nr) { break; }
                  }
                  if (!nr) { break; }
               }
               if (!nr) { break; }
            }
            if (!nr) { break; }
         }
         if (!nr) { break; }
      }
   }
};

// id: MPI rank, nr: launch all non-regression tests
static void tmop_tests(int id = 0, bool all = false)
{
   const double jitter = 1./(M_PI*M_PI);

   Launch(Launch::Args("TC_IDEAL_SHAPE_UNIT_SIZE_2D_KERNEL").
          MESH("../../data/star.mesh").REFINE(1).JI(jitter).
          POR({1,2}).QOR({2,3}).
          TID({1}).MID({2})).Run(id,all);

   Launch(Launch::Args("TC_IDEAL_SHAPE_GIVEN_SIZE_2D_KERNEL").
          MESH("../../data/star.mesh").REFINE(1).JI(jitter).
          POR({1,2}).QOR({2,3}).
          TID({3}).MID({2})).Run(id,all);

   Launch(Launch::Args("TC_IDEAL_SHAPE_UNIT_SIZE_3D_KERNEL").
          MESH("../../miniapps/meshing/cube.mesh").REFINE(1).JI(jitter).
          POR({1,2}).QOR({2,3}).
          TID({1}).MID({302})).Run(id,all);

   Launch(Launch::Args("TC_IDEAL_SHAPE_GIVEN_SIZE_3D_KERNEL").
          MESH("../../miniapps/meshing/cube.mesh").REFINE(1).JI(jitter).
          POR({1,2}).QOR({2,3}).
          TID({3}).MID({302})).Run(id,all);

   Launch(Launch::Args("Star").
          MESH("../../data/star.mesh").
          POR({1,2,3,4}).QOR({2,4,8}).
          TID({1,2,3}).MID({1,2})).Run(id,all);

   Launch(Launch::Args("Square01 + Adapted analytic Hessian").
          MESH("../../miniapps/meshing/square01.mesh").REFINE(1).
          POR({1,2}).QOR({2,4}).
          TID({4}).MID({1,2})).Run(id,all);

   Launch(Launch::Args("Square01 + Adapted discrete size").
          MESH("../../miniapps/meshing/square01.mesh").REFINE(1).
          NORMALIZATION(true).
          POR({1,2}).QOR({4,6}).
          TID({5}).MID({80}).LS({3})).Run(id,all);

   Launch(Launch::Args("Blade").
          MESH("../../miniapps/meshing/blade.mesh").
          POR({1,2}).QOR({2,4}).
          TID({1,2,3}).MID({2}).LS({2})).Run(id,all);

   Launch(Launch::Args("Blade + normalization").
          MESH("../../miniapps/meshing/blade.mesh").
          NORMALIZATION(true).
          POR({1,2}).QOR({2,4}).
          TID({1,2,3}).MID({2})).Run(id,all);

   Launch(Launch::Args("Blade + limiting + normalization").
          MESH("../../miniapps/meshing/blade.mesh").
          NORMALIZATION(true).LIMITING(M_PI).
          POR({1,2}).QOR({2,4}).
          TID({1,2,3}).MID({2})).Run(id,all);

   Launch(Launch::Args("Cube").
          MESH("../../miniapps/meshing/cube.mesh").REFINE(1).JI(jitter).
          POR({1,2}).QOR({2,4}).
          TID({2,3}).MID({302,303})).Run(id,all);

   Launch(Launch::Args("Cube + Discrete size & aspect + normalization + limiting").
          MESH("../../miniapps/meshing/cube.mesh").
          NORMALIZATION(true).LIMITING(M_PI).
          POR({1,2}).QOR({4,2}).
          TID({7}).MID({302,321})).Run(id,all);

   // Note: order 1 has no interior nodes, so all residuals are zero and the
   // Newton iteration exits immediately.
   Launch(Launch::Args("Toroid-Hex").
          MESH("../../data/toroid-hex.mesh").
          POR({1,2}).QOR({2,4,8}).
          TID({1,2,3}).MID({302,303,321})).Run(id,all);

   Launch(Launch::Args("Toroid-Hex + limiting").
          MESH("../../data/toroid-hex.mesh").
          LIMITING(M_PI).
          POR({1,2}).QOR({2,4}).NL({3,1}).
          TID({1,2}).MID({321})).Run(id,all);

   Launch(Launch::Args("Toroid-Hex + limiting + norm.").
          MESH("../../data/toroid-hex.mesh").
          LIMITING(M_PI).NORMALIZATION(true).
          POR({1,2}).QOR({2,4}).
          TID({1,2}).MID({321})).Run(id,all);

   // -m cube.mesh -rs 1 -tid 5 -mid 321 -ni 5 -ls 3 -li 100 -lc 1.0 -nor
   Launch(Launch::Args("Cube + Blast options").
          MESH("../../miniapps/meshing/cube.mesh").REFINE(1).
          TID({5}).MID({321}).LS({3}).LINEAR_ITERATIONS(100).
          LIMITING(M_PI).NORMALIZATION(true).
          POR({1,2,3}).QOR({2,4}).NL({1,2})).Run(id,all);

   // Combo 2D
   Launch(Launch::Args("Square01 + Combo").
          MESH("../../miniapps/meshing/square01.mesh").REFINE(1).JI(jitter).
          NORMALIZATION(true).
          TID({5}).MID({2}).LS({2}).
          POR({2}).QOR({8}).CMB(2)).Run(id,all);

   // Combo 3D
   Launch(Launch::Args("Cube + Combo").
          MESH("../../miniapps/meshing/cube.mesh").REFINE(1).JI(jitter).
          NORMALIZATION(true).
          TID({5}).MID({302}).LS({2}).
          POR({1,2}).QOR({2,8}).CMB(2)).Run(id,all);

   // NURBS
   Launch(Launch::Args("2D Nurbs").
          MESH("../../data/square-disc-nurbs.mesh").REFINE(1).JI(jitter).
          POR({1,2}).QOR({2,4}).
          TID({1,2,3}).MID({1,2})).Run(id,all);

   Launch(Launch::Args("3D Nurbs").
          MESH("../../data/beam-hex-nurbs.mesh").REFINE(1).JI(jitter).
          POR({1,2}).QOR({2,4}).
          TID({1,2,3}).MID({302,321})).Run(id,all);

   // The folowing tests need more iterations to converge between PA & non-PA
   // They can only be launched with the `--all` command line option

   if (!all) { return; }

   Launch(Launch::Args("Blade + Discrete size + normalization").
          MESH("../../miniapps/meshing/blade.mesh").
          LINEAR_ITERATIONS(300).NORMALIZATION(true).
          POR({1}).QOR({2}).
          TID({5}).MID({7}).LS({2}).NL({2})).Run(id,true);

   Launch(Launch::Args("Blade + Discrete size + normalization").
          MESH("../../miniapps/meshing/blade.mesh").
          LINEAR_ITERATIONS(200).NORMALIZATION(true).
          POR({1}).QOR({2}).
          TID({5}).MID({2})).Run(id,true);
}

#if defined(MFEM_TMOP_MPI)
#ifndef MFEM_TMOP_DEVICE
TEST_CASE("tmop_pa", "[TMOP_PA], [Parallel]")
{
   tmop_tests(GlobalMPISession->WorldRank(), launch_all_non_regression_tests);
}
#else
TEST_CASE("tmop_pa", "[TMOP_PA], [Parallel]")
{
   Device device;
   device.Configure(MFEM_TMOP_DEVICE);
   device.Print();
   tmop_tests(GlobalMPISession->WorldRank(), launch_all_non_regression_tests);
}
#endif
#else
#ifndef MFEM_TMOP_DEVICE
TEST_CASE("tmop_pa", "[TMOP_PA]")
{
   tmop_tests(0, launch_all_non_regression_tests);
}
#else
TEST_CASE("tmop_pa", "[TMOP_PA]")
{
   Device device;
   device.Configure(MFEM_TMOP_DEVICE);
   device.Print();
   tmop_tests(0, launch_all_non_regression_tests);
}
#endif
#endif
