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

#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
extern mfem::MPI_Session *GlobalMPISession;
#define PFesGetParMeshGetComm(pfes) pfes.GetParMesh()->GetComm()
#define SetDiscreteTargetSize SetParDiscreteTargetSize
#define SetDiscreteTargetAspectRatio SetParDiscreteTargetAspectRatio
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
};

static double discrete_size_2d(const Vector &x)
{
   int opt = 2;
   const double small = 0.001, big = 0.01;
   double val = 0.;

   if (opt == 1) // sine wave.
   {
      const double X = x(0), Y = x(1);
      val = std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) + 1) -
            std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) - 1);
   }
   else if (opt == 2) // semi-circle
   {
      const double xc = x(0) - 0.0, yc = x(1) - 0.5;
      const double r = sqrt(xc*xc + yc*yc);
      double r1 = 0.45; double r2 = 0.55; double sf=30.0;
      val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));
   }

   val = std::max(0.,val);
   val = std::min(1.,val);

   return val * small + (1.0 - val) * big;
}

static void discrete_aspr_3d(const Vector &x, Vector &v)
{
   int dim = x.Size();
   v.SetSize(dim);
   double l1, l2, l3;
   l1 = 1.;
   l2 = 1. + 5*x(1);
   l3 = 1. + 10*x(2);
   v[0] = l1/pow(l2*l3,0.5);
   v[1] = l2/pow(l1*l3,0.5);
   v[2] = l3/pow(l2*l1,0.5);
}

class HessianCoefficient : public MatrixCoefficient
{
private:
   int metric;

public:
   HessianCoefficient(int dim, int metric_id)
      : MatrixCoefficient(dim), metric(metric_id) { }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector pos(3);
      T.Transform(ip, pos);
      if (metric != 14 && metric != 87)
      {
         const double xc = pos(0) - 0.5, yc = pos(1) - 0.5;
         const double r = sqrt(xc*xc + yc*yc);
         double r1 = 0.15; double r2 = 0.35; double sf=30.0;
         const double eps = 0.5;

         const double tan1 = std::tanh(sf*(r-r1)),
                      tan2 = std::tanh(sf*(r-r2));

         K(0, 0) = eps + 1.0 * (tan1 - tan2);
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = 1.0;
      }
      else if (metric == 14) // Size + Alignment
      {
         const double xc = pos(0), yc = pos(1);
         double theta = M_PI * yc * (1.0 - yc) * cos(2 * M_PI * xc);
         double alpha_bar = 0.1;

         K(0, 0) =  cos(theta);
         K(1, 0) =  sin(theta);
         K(0, 1) = -sin(theta);
         K(1, 1) =  cos(theta);

         K *= alpha_bar;
      }
      else if (metric == 87) // Shape + Alignment
      {
         Vector x = pos;
         double xc = x(0)-0.5, yc = x(1)-0.5;
         double th = 22.5*M_PI/180.;
         double xn =  cos(th)*xc + sin(th)*yc;
         double yn = -sin(th)*xc + cos(th)*yc;
         xc = xn; yc=yn;

         double tfac = 20;
         double s1 = 3;
         double s2 = 2;
         double wgt = std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) + 1)
                      - std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) - 1);
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }

         xc = pos(0), yc = pos(1);
         double theta = M_PI * (yc) * (1.0 - yc) * cos(2 * M_PI * xc);

         K(0, 0) =  cos(theta);
         K(1, 0) =  sin(theta);
         K(0, 1) = -sin(theta);
         K(1, 1) =  cos(theta);

         double asp_ratio_tar = 0.1 + 1*(1-wgt)*(1-wgt);

         K(0, 0) *=  1/pow(asp_ratio_tar,0.5);
         K(1, 0) *=  1/pow(asp_ratio_tar,0.5);
         K(0, 1) *=  pow(asp_ratio_tar,0.5);
         K(1, 1) *=  pow(asp_ratio_tar,0.5);
      }
   }
};

int tmop(int myid, Req &res, int argc, char *argv[])
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
   double lim_const      = 0.0;
   int normalization     = 0;
   double jitter         = 0.0;

   constexpr bool move_bnd = false;
   constexpr int combomet  = 0;
   constexpr int verbosity_level = 0;
   constexpr bool fdscheme = false;

   REQUIRE_FALSE(fdscheme);
   REQUIRE(combomet == 0);
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
   args.AddOption(&newton_rtol, "-rtol", "--newton-rel-tolerance", "");
   args.AddOption(&lin_solver, "-ls", "--lin-solver", "");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter", "");
   args.AddOption(&lim_const, "-lc", "--limit-const", "");
   args.AddOption(&normalization, "-nor", "--normalization", "");
   args.AddOption(&pa, "-pa", "--pa", "-no-pa", "--no-pa", "");
   args.AddOption(&jitter, "-ji", "--jitter", "");
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
   rdm.Randomize(0x100001b3);
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
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
   TargetConstructor *target_c = nullptr;
   HessianCoefficient *adapt_coeff = nullptr;
   constexpr int mesh_poly_deg = 1;
   H1_FECollection ind_fec(mesh_poly_deg, dim);
   ParFiniteElementSpace ind_fes(pmesh, &ind_fec);
   ParGridFunction size(&ind_fes);
   ParFiniteElementSpace ind_fesv(pmesh, &ind_fec, dim);
   ParGridFunction aspr3d(&ind_fesv);
   const AssemblyLevel al = pa ? AssemblyLevel::PARTIAL : AssemblyLevel::FULL;

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
      case 5: // Discrete size 2D
      {
         target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
         DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
         tc->SetAdaptivityEvaluator(new AdvectorCG(al));
         FunctionCoefficient ind_coeff(discrete_size_2d);
         size.ProjectCoefficient(ind_coeff);
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
         if (myid == 0) { cout << "Unknown target_id: " << target_id << endl; }
         return 3;
      }
   }
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_MPI)
   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   }
#else
   if (target_c == NULL)
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
         if (myid == 0) { cout << "Unknown quad_type: " << quad_type << endl; }
         return 4;
      }
   }

   TMOP_Integrator *he_nlf_integ = new TMOP_Integrator(metric, target_c);
   he_nlf_integ->SetIntegrationRule(*ir);

   if (normalization == 1) { he_nlf_integ->EnableNormalization(x0); }

   ParGridFunction dist(&fes);
   dist = 1.0;
   if (normalization == 1) { dist = small_phys_size; }
   ConstantCoefficient lim_coeff(lim_const);
   if (lim_const != 0.0) { he_nlf_integ->EnableLimiting(x0, dist, lim_coeff); }

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
   //if (myid == 0) { cout << "Min det(J) of the mesh is " << tauval << endl; }
   REQUIRE(tauval > 0.0);
   res.tauval = tauval;

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

   res.init_energy = init_energy;

   REQUIRE(newton->GetConverged());
   double x_t_dot = x_t*x_t, dot;
   MPI_Allreduce(&x_t_dot, &dot, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   res.dot = dot;

   x.SetFromTrueVector();
   const double final_energy = nlf.GetParGridFunctionEnergy(x);
   res.final_energy = final_energy;

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

static void req_tmop(int myid, const char *args[], Req &res)
{ REQUIRE(tmop(myid, res, argn(args), const_cast<char**>(args))==0); }

#define DEFAULT_ARGS const char *args[] = { \
    "tmop_tests", "-pa", "-m", "mesh", "-o", "0", "-rs", "0", \
    "-mid", "0", "-tid", "0", "-qt", "1", "-qo", "0", \
    "-ni", "10", "-rtol", "1e-8", "-ls", "2", "-li", "100", \
    "-lc", "0", "-nor", "0", "-ji", "0", nullptr }
constexpr int ALV = 1;
constexpr int MSH = 3;
constexpr int POR = 5;
constexpr int RFS = 7;
constexpr int MID = 9;
constexpr int TID = 11;
//constexpr int QTY = 13;
constexpr int QOR = 15;
constexpr int NI  = 17;
constexpr int LS  = 21;
constexpr int LI  = 23;
constexpr int LC  = 25;
constexpr int NOR  = 27;
constexpr int JI  = 29;

static void dump_args(const char *args[])
{
   printf("tmop -m %s -o %s -qo %s -mid %s -tid %s -ls %s%s%s%s %s\n",
          args[MSH], args[POR], args[QOR],
          args[MID], args[TID], args[LS],
          args[LC][0] == '0' ? "" : " -lc",
          args[NOR][0] == '0' ? "" : " -nor",
          args[JI][0] == '0' ? "" : " -jitter",
          args[ALV]);
   fflush(0);
}

static void tmop_require(int myid, const char *args[])
{
   Req res[2];
   (args[ALV] = "-pa", dump_args(args), req_tmop(myid, args, res[0]));
   (args[ALV] = "-no-pa", dump_args(args), req_tmop(myid, args, res[1]));
   REQUIRE(res[0].dot == Approx(res[1].dot));
   REQUIRE(res[0].tauval == Approx(res[1].tauval));
   REQUIRE(res[0].init_energy == Approx(res[1].init_energy));
   REQUIRE(res[0].final_energy == Approx(res[1].final_energy));
}

static inline const char *itoa(int i, char *buf)
{
   std::sprintf(buf, "%d", i);
   return buf;
}

static void tmop_tests(int myid)
{
   static bool all = getenv("MFEM_TESTS_UNIT_TMOP_ALL");

   // 3D CUBE + Discrete size & aspect-ratio 3D + normalization + limiting
   {
      DEFAULT_ARGS;
      args[MSH] = "cube.mesh";
      args[RFS] = "0";
      args[NOR] = "1";
      args[LC] = "3.14";
      for (int p : {1,2})
      {
         char por[2] {};
         args[POR] = itoa(p, por);
         for (int q : {1,2})
         {
            if (q < p) { continue; }
            char qor[2] {};
            args[QOR] = itoa(q, qor);
            for (int m : {302})
            {
               char mid[4] {};
               args[MID] = itoa(m, mid);
               for (int t : {7})
               {
                  char tid[2] {};
                  args[TID] = itoa(t, tid);
                  for (int ls : {2})
                  {
                     char lsb[2] {};
                     args[LS] = itoa(ls, lsb);
                     tmop_require(myid, args);
                     if (!all) { break; }
                  }
                  if (!all) { break; }
               }
               if (!all) { break; }
            }
            if (!all) { break; }
         }
         if (!all) { break; }
      }
   } // 3D CUBE + Discrete size & aspect-ratio 3D + normalization + limiting

   // 2D BLADE + Discrete size 2D + normalization
   {
      DEFAULT_ARGS;
      args[MSH] = "blade.mesh";
      args[MID] = "2";
      args[NI] = "100";
      args[LI] = "100";
      args[NOR] = "1";
      for (int p : {1})
      {
         char por[2] {};
         args[POR] = itoa(p, por);
         for (int q : {2})
         {
            if (q <= p) { continue; }
            char qor[2] {};
            args[QOR] = itoa(q, qor);
            for (int t : {5})
            {
               char tid[2] {};
               args[TID] = itoa(t, tid);
               for (int ls : {2})
               {
                  char lsb[2] {};
                  args[LS] = itoa(ls, lsb);
                  tmop_require(myid, args);
                  if (!all) { break; }
               }
               if (!all) { break; }
            }
            if (!all) { break; }
         }
         if (!all) { break; }
      }
   } // 2D BLADE + Discrete size 2D + normalization

   // SQUARE01 + Adapted analytic Hessian
   {
      DEFAULT_ARGS;
      args[NI] = "100";
      args[MSH] = "square01.mesh";
      args[RFS] = "1";
      for (int p : {1, 2})
      {
         char por[2] {};
         args[POR] = itoa(p, por);
         for (int t : {4})
         {
            char tid[2] {};
            args[TID] = itoa(t, tid);
            for (int m : {2})
            {
               char mid[2] {};
               args[MID] = itoa(m, mid);
               for (int q : {2, 4})
               {
                  if (q <= p) { continue; }
                  char qor[2] {};
                  args[QOR] = itoa(q, qor);
                  for (int ls : {2})
                  {
                     char lsb[2] {};
                     args[LS] = itoa(ls, lsb);
                     tmop_require(myid, args);
                     if (!all) { break; }
                  }
                  if (!all) { break; }
               }
               if (!all) { break; }
            }
            if (!all) { break; }
         }
         if (!all) { break; }
      }
   } // SQUARE01

   // STAR
   {
      DEFAULT_ARGS;
      args[MSH] = "star.mesh";
      for (int p : {1, 2, 3, 4})
      {
         char por[2] {};
         args[POR] = itoa(p, por);
         for (int t : {1, 2, 3})
         {
            char tid[2] {};
            args[TID] = itoa(t, tid);
            for (int m : {1, 2})
            {
               char mid[2] {};
               args[MID] = itoa(m, mid);
               for (int q : {2, 4, 8})
               {
                  if (q <= p) { continue; }
                  char qor[2] {};
                  args[QOR] = itoa(q, qor);
                  for (int ls : {2, 3})
                  {
                     char lsb[2] {};
                     args[LS] = itoa(ls, lsb);
                     tmop_require(myid, args);
                     if (!all) { break; }
                  }
                  if (!all) { break; }
               }
               if (!all) { break; }
            }
            if (!all) { break; }
         }
         if (!all) { break; }
      }
   } // STAR

   // BLADE
   {
      DEFAULT_ARGS;
      args[MSH] = "blade.mesh";
      args[MID] = "2";
      args[NI] = "100";
      args[LI] = "100";
      for (int p : {1, 2})
      {
         char por[2] {};
         args[POR] = itoa(p, por);
         for (int q : {2, 4})
         {
            char qor[2] {};
            args[QOR] = itoa(q, qor);
            for (int t : {1, 2, 3})
            {
               char tid[2] {};
               args[TID] = itoa(t, tid);
               for (int ls : {2, 3})
               {
                  char lsb[2] {};
                  args[LS] = itoa(ls, lsb);
                  tmop_require(myid, args);
                  if (!all) { break; }
               }
               if (!all) { break; }
            }
            if (!all) { break; }
         }
         if (!all) { break; }
      }
   } // BLADE

   // 2D BLADE + normalization
   {
      DEFAULT_ARGS;
      args[MSH] = "blade.mesh";
      args[MID] = "2";
      args[NI] = "100";
      args[LI] = "100";
      args[NOR] = "1";
      for (int p : {1, 2})
      {
         char por[2] {};
         args[POR] = itoa(p, por);
         for (int q : {2, 4})
         {
            if (q <= p) { continue; }
            char qor[2] {};
            args[QOR] = itoa(q, qor);
            for (int t : {1, 2, 3})
            {
               char tid[2] {};
               args[TID] = itoa(t, tid);
               for (int ls : {2, 3})
               {
                  char lsb[2] {};
                  args[LS] = itoa(ls, lsb);
                  tmop_require(myid, args);
                  if (!all) { break; }
               }
               if (!all) { break; }
            }
            if (!all) { break; }
         }
         if (!all) { break; }
      }
   } // 2D BLADE + normalization

   // 2D BLADE + limiting + normalization
   {
      DEFAULT_ARGS;
      args[MSH] = "blade.mesh";
      args[MID] = "2";
      args[NI] = "100";
      args[LI] = "100";
      args[LC] = "3.14";
      args[NOR] = "1";
      for (int p : {1, 2})
      {
         char por[2] {};
         args[POR] = itoa(p, por);
         for (int q : {2, 4})
         {
            if (q <= p) { continue; }
            char qor[2] {};
            args[QOR] = itoa(q, qor);
            for (int t : {1, 2, 3})
            {
               char tid[2] {};
               args[TID] = itoa(t, tid);
               for (int ls : {2, 3})
               {
                  char lsb[2] {};
                  args[LS] = itoa(ls, lsb);
                  tmop_require(myid, args);
                  if (!all) { break; }
               }
               if (!all) { break; }
            }
            if (!all) { break; }
         }
         if (!all) { break; }
      }
   } // 2D BLADE + limiting + normalization

   // TOROID-HEX
   {
      DEFAULT_ARGS;
      args[MSH] = "toroid-hex.mesh";
      args[RFS] = "0";
      for (int p : {1, 2})
      {
         char por[2] {};
         args[POR] = itoa(p, por);
         for (int q : {2, 4, 8})
         {
            char qor[2] {};
            args[QOR] = itoa(q, qor);
            for (int m : {302, 303, 321})
            {
               char mid[4] {};
               args[MID] = itoa(m, mid);
               for (int t : {1, 2, 3})
               {
                  char tid[2] {};
                  args[TID] = itoa(t, tid);
                  for (int ls : {1, 2, 3})
                  {
                     char lsb[2] {};
                     args[LS] = itoa(ls, lsb);
                     tmop_require(myid, args);
                     if (!all) { break; }
                  }
                  if (!all) { break; }
               }
               if (!all) { break; }
            }
            if (!all) { break; }
         }
         if (!all) { break; }
      }
   } // TOROID-HEX

   // TOROID-HEX + limiting
   {
      DEFAULT_ARGS;
      args[MSH] = "toroid-hex.mesh";
      args[RFS] = "0";
      args[LC] = "3.14";
      for (int p : {1, 2})
      {
         char por[2] {};
         args[POR] = itoa(p, por);
         for (int q : {2, 4})
         {
            char qor[2] {};
            args[QOR] = itoa(q, qor);
            for (int m : {321})
            {
               char mid[4] {};
               args[MID] = itoa(m, mid);
               for (int t : {1, 2})
               {
                  char tid[2] {};
                  args[TID] = itoa(t, tid);
                  for (int ls : {1, 2})
                  {
                     char lsb[2] {};
                     args[LS] = itoa(ls, lsb);
                     tmop_require(myid, args);
                     if (!all) { break; }
                  }
                  if (!all) { break; }
               }
               if (!all) { break; }
            }
            if (!all) { break; }
         }
         if (!all) { break; }
      }
   } // TOROID-HEX + limiting

   // TOROID-HEX + limiting + normalization
   {
      DEFAULT_ARGS;
      args[MSH] = "toroid-hex.mesh";
      args[RFS] = "0";
      args[LC] = "3.14";
      args[NOR] = "1";
      for (int p : {1, 2})
      {
         char por[2] {};
         args[POR] = itoa(p, por);
         for (int q : {2, 4})
         {
            char qor[2] {};
            args[QOR] = itoa(q, qor);
            for (int m : {321})
            {
               char mid[4] {};
               args[MID] = itoa(m, mid);
               for (int t : {1, 2})
               {
                  char tid[2] {};
                  args[TID] = itoa(t, tid);
                  for (int ls : {1, 2})
                  {
                     char lsb[2] {};
                     args[LS] = itoa(ls, lsb);
                     tmop_require(myid, args);
                     if (!all) { break; }
                  }
                  if (!all) { break; }
               }
               if (!all) { break; }
            }
            if (!all) { break; }
         }
         if (!all) { break; }
      }
   } // TOROID-HEX + limiting + normalization
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
