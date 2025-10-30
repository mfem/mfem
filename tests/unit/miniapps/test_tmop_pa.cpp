// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#define CATCH_CONFIG_RUNNER
#include "mfem.hpp"
#include "run_unit_tests.hpp"

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#else
// Avoiding MSVC error C2491: 'definition of dllimport function not allowed'
#include "fem/qinterp/det.hpp"                   // IWYU pragma: keep
#include "fem/qinterp/grad.hpp"                  // IWYU pragma: keep
#include "fem/qinterp/eval.hpp"                  // IWYU pragma: keep
#include "fem/integ/bilininteg_mass_kernels.hpp" // IWYU pragma: keep
#endif

#include <iostream>
#include <list>
#include <memory>

#include "miniapps/meshing/mesh-optimizer.hpp"

#if defined(MFEM_TMOP_PA_MPI) && !defined(MFEM_USE_MPI)
#error "Cannot use MFEM_TMOP_PA_MPI without MFEM_USE_MPI!"
#endif

#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_PA_MPI)
#define PFesGetParMeshGetComm(pfes) pfes.GetComm()
#define SetDiscreteTargetSize SetParDiscreteTargetSize
#define SetDiscreteTargetAspectRatio SetParDiscreteTargetAspectRatio
#define GradientClass HypreParMatrix
#else
#define ParMesh Mesh
#define ParGridFunction GridFunction
#define ParNonlinearForm NonlinearForm
#define ParFiniteElementSpace FiniteElementSpace
#define GetParGridFunctionEnergy GetGridFunctionEnergy
#define PFesGetParMeshGetComm(...)
#define MPI_Allreduce(src, dst, ...) *dst = *src
#define SetDiscreteTargetSize SetSerialDiscreteTargetSize
#define SetDiscreteTargetAspectRatio SetSerialDiscreteTargetAspectRatio
#define GradientClass SparseMatrix
#define ParEnableNormalization EnableNormalization
#endif

using namespace mfem;

namespace mfem
{

struct Req
{
   real_t dot;
   real_t diag;
   real_t min_detJ;
   real_t bal_weights;
   real_t met_normal, lim_normal;
   real_t init_energy, final_energy;
};

int tmop(int id, Req &res, int argc, char *argv[])
{
   bool pa = false;
   const char *mesh_file = nullptr;
   int mesh_poly_deg = 1;
   int rs_levels = 0;
   int metric_id = 1;
   int target_id = 1;
   int quad_type = 1;
   int quad_order = 2;
   int newton_iter = 100;
   real_t newton_rtol = 1e-10;
   real_t linsol_rtol = 1e-10;
   int lin_solver = 2;
   int max_lin_iter = 100;
   real_t lim_const = 0.0;
   int lim_type = 0;
   bool normalization = false;
   real_t jitter = 0.0;
   bool diag = true;
   int newton_loop = 1;
   int combomet = 0;
   bool bal_expl_combo = false;
   bool periodic = false;
   const int mesh_node_order = Ordering::byNODES;

   constexpr int verbosity_level = 0;
   constexpr int seed = 0x100001b3;
   constexpr bool move_bnd = false;
   constexpr bool fdscheme = false;
   constexpr bool integ_over_targ  = true;
   constexpr bool exactaction = false;

   OptionsParser args(argc, argv);
   args.AddOption(&pa, "-pa", "--pa", "-no-pa", "--no-pa", "");
   args.AddOption(&mesh_file, "-m", "--mesh", "");
   args.AddOption(&mesh_poly_deg, "-o", "--order", "");
   args.AddOption(&rs_levels, "-rs", "--refine-serial", "");
   args.AddOption(&metric_id, "-mid", "--metric-id", "");
   args.AddOption(&target_id, "-tid", "--target-id", "");
   args.AddOption(&quad_type, "-qt", "--quad-type", "");
   args.AddOption(&quad_order, "-qo", "--quad_order", "");
   args.AddOption(&newton_iter, "-ni", "--newton-iters", "");
   args.AddOption(&newton_loop, "-nl", "--newton-loops", "");
   args.AddOption(&newton_rtol, "-nrtol", "--newton-rel-tolerance", "");
   args.AddOption(&linsol_rtol, "-lrtol", "--linsol-rel-tolerance", "");
   args.AddOption(&lin_solver, "-ls", "--lin-solver", "");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter", "");
   args.AddOption(&lim_const, "-lc", "--limit-const", "");
   args.AddOption(&lim_type, "-lt", "--limit-type", "");
   args.AddOption(&normalization, "-nor", "--normalization",
                  "-no-nor", "--no-normalization", "");
   args.AddOption(&jitter, "-ji", "--jitter", "");
   args.AddOption(&diag, "-diag", "--diag", "-no-diag", "--no-diag", "");
   args.AddOption(&combomet, "-cmb", "--combo-type", "");
   args.AddOption(&bal_expl_combo, "-bec", "--balance-explicit-combo",
                  "-no-bec", "--no-balance-explicit-combo", "");
   args.AddOption(&periodic, "-per", "--periodic", "-no-per", "--no-periodic", "");
   args.Parse();
   if (!args.Good())
   {
      args.PrintOptions(mfem::out);
      if (id == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (verbosity_level > 0)
   {
      if (id == 0) { args.PrintOptions(cout); }
   }

   // Initialize and refine the starting mesh.
   Mesh smesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { smesh.UniformRefinement(); }
   const int dim = smesh.Dimension();

   if (periodic)
   {
      auto s = smesh.GetNodalFESpace();
      REQUIRE((s && s->IsDGSpace()));
   }

   ParMesh mesh([](Mesh &mesh)
   {
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_PA_MPI)
      return ParMesh(MPI_COMM_WORLD, mesh);
#else
      return Mesh(mesh);
#endif
   } (smesh));
   smesh.Clear();

   // Define a FE space on the mesh, based on the input order.
   REQUIRE(mesh_poly_deg > 0);
   std::unique_ptr<FiniteElementCollection> fec;
   if (periodic)
   {
      fec.reset(new L2_FECollection(mesh_poly_deg, dim, BasisType::GaussLobatto));
   }
   else { fec.reset(new H1_FECollection(mesh_poly_deg, dim)); }
   ParFiniteElementSpace fespace(&mesh, fec.get(), dim, mesh_node_order);

   // Make the starting mesh curved.
   mesh.SetNodalFESpace(&fespace);

   // Get the mesh nodes (vertices and other DOFs in the FE space)
   ParGridFunction x(&fespace), x0_before_jitter(&fespace);
   mesh.SetNodalGridFunction(&x);

   // When the target is GIVEN_SHAPE_AND_SIZE, we want to call tc->SetNodes()
   // with something other than x0 (otherwise all metrics would be 0).
   x0_before_jitter = x;

   // We create an H1 space for the mesh displacement.
   H1_FECollection fec_h1(mesh_poly_deg, dim);
   ParFiniteElementSpace fes_h1(&mesh, &fec_h1, dim, mesh_node_order);
   ParGridFunction dx(&fes_h1); dx = 0.0;

   // Define a vector representing the minimal local mesh size in the mesh nodes.
   // In addition, compute average mesh size and total volume.
   Vector h0(fes_h1.GetNDofs());
   h0 = infinity();
   real_t mesh_volume = 0.0;
   Array<int> dofs;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      fes_h1.GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      const real_t hi = mesh.GetElementSize(i);
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), hi);
      }
      mesh_volume += mesh.GetElementVolume(i);
   }
   const real_t small_phys_size = pow(mesh_volume, 1.0 / dim) / 100.0;

   // Add a random perturbation to the nodes in the interior of the domain.
   if (jitter > 0)
   {
      ParGridFunction rdm(&fes_h1);
      rdm.Randomize(seed);
      rdm -= 0.25;
      rdm *= jitter;
      rdm.HostReadWrite();
      // Scale the random values to be of order of the local mesh size.
      for (int i = 0; i < fes_h1.GetNDofs(); i++)
      {
         for (int d = 0; d < dim; d++)
         {
            rdm(fes_h1.DofToVDof(i, d)) *= h0(i);
         }
      }
      // Set the boundary values to zero.
      Array<int> vdofs;
      for (int i = 0; i < fes_h1.GetNBE(); i++)
      {
         fes_h1.GetBdrElementVDofs(i, vdofs);
         for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
      }

      if (periodic)
      {
         // For H1 the perturbation is controlled by the true nodes.
         rdm.SetFromTrueVector();
         ParGridFunction rdm_l2(&fespace);
         rdm_l2.ProjectGridFunction(rdm);
         x -= rdm_l2;
      }
      else
      {
         x -= rdm;
         // For H1 the perturbation is controlled by the true nodes.
         x.SetTrueVector();
         x.SetFromTrueVector();
      }
   }

   // Store the starting (prior to the optimization) positions.
   ParGridFunction x0(x);

   // Form the integrator that uses the chosen metric and target.
   std::unique_ptr<TMOP_QualityMetric> metric;
   switch (metric_id)
   {
      case 1:   metric.reset(new TMOP_Metric_001); break;
      case 2:   metric.reset(new TMOP_Metric_002); break;
      case 7:   metric.reset(new TMOP_Metric_007); break;
      case 56:  metric.reset(new TMOP_Metric_056); break;
      case 77:  metric.reset(new TMOP_Metric_077); break;
      case 80:  metric.reset(new TMOP_Metric_080(0.5)); break; // combo
      case 94:  metric.reset(new TMOP_Metric_094); break;      // combo
      case 302: metric.reset(new TMOP_Metric_302); break;
      case 303: metric.reset(new TMOP_Metric_303); break;
      case 315: metric.reset(new TMOP_Metric_315); break;
      case 318: metric.reset(new TMOP_Metric_318); break;
      case 321: metric.reset(new TMOP_Metric_321); break;
      case 332: metric.reset(new TMOP_Metric_332(0.5)); break; // combo
      case 338: metric.reset(new TMOP_Metric_338); break;      // combo
      default:
      {
         cout << "Unknown metric_id: " << metric_id << endl;
         return 2;
      }
   }

   TargetConstructor::TargetType target_t;
   std::unique_ptr<TargetConstructor> target_c = nullptr;
   std::unique_ptr<HessianCoefficient> adapt_coeff = nullptr;
   const int ind_fec_order =
      (target_id >= 5 && target_id <= 8 && !fdscheme) ?
      1 : mesh_poly_deg;
   H1_FECollection ind_fec(ind_fec_order, dim);
   ParFiniteElementSpace ind_fes(&mesh, &ind_fec);
   ParFiniteElementSpace ind_fesv(&mesh, &ind_fec, dim);
   ParGridFunction size(&ind_fes), ori(&ind_fes);
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
         auto tc = new AnalyticAdaptTC(target_t);
         adapt_coeff.reset(new HessianCoefficient(dim, metric_id));
         tc->SetAnalyticTargetSpec(nullptr, nullptr, adapt_coeff.get());
         target_c.reset(tc);
         break;
      }
      case 5: // Discrete size 2D or 3D
      {
         target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
         auto tc = new DiscreteAdaptTC(target_t);
         tc->SetAdaptivityEvaluator(new AdvectorCG(al));
         ConstructSizeGF(size);
         tc->SetDiscreteTargetSize(size);
         tc->SetMinSizeForTargets(size.Min());
         target_c.reset(tc);
         break;
      }
      case 7: // Discrete aspect-ratio 3D
      {
         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         auto tc = new DiscreteAdaptTC(target_t);
         tc->SetAdaptivityEvaluator(new AdvectorCG(al));
         VectorFunctionCoefficient fd_aspr3d(dim, discrete_aspr_3d);
         aspr3d.ProjectCoefficient(fd_aspr3d);
         tc->SetDiscreteTargetAspectRatio(aspr3d);
         target_c.reset(tc);
         break;
      }
      case 8: // fully specified through the initial mesh, 2D or 3D.
      {
         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         break;
      }
      default:
      {
         cout << "Unknown target_id: " << target_id << endl;
         return 3;
      }
   }
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_PA_MPI)
   if (target_c == nullptr)
   {
      target_c.reset(new TargetConstructor(target_t, MPI_COMM_WORLD));
   }
#else
   if (target_c == nullptr) { target_c.reset(new TargetConstructor(target_t)); }
#endif
   target_c->SetNodes(x0_before_jitter);

   auto tmop_integ = new TMOP_Integrator(metric.get(), target_c.get());
   tmop_integ->IntegrateOverTarget(integ_over_targ);
   tmop_integ->SetExactActionFlag(exactaction);

   // Setup the quadrature rules for the TMOP integrator.
   IntegrationRules *irules = nullptr;
   IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
   IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);
   switch (quad_type)
   {
      case 1: irules = &IntRulesLo; break;
      case 2: irules = &IntRules; break;
      case 3: irules = &IntRulesCU; break;
      default: cout << "Unknown quad_type: " << quad_type << endl; return 3;
   }
   tmop_integ->SetIntegrationRules(*irules, quad_order);

   // Automatically balanced gamma in composite metrics.
   res.bal_weights = 0.0;
   auto metric_combo = dynamic_cast<TMOP_Combo_QualityMetric *>(metric.get());
   if (metric_combo && bal_expl_combo)
   {
      Vector bal_weights;
      auto ir = irules->Get(mesh.GetTypicalElementGeometry(), quad_order);
      metric_combo->ComputeBalancedWeights(x, *target_c, bal_weights, pa, &ir);
      metric_combo->SetWeights(bal_weights);
      res.bal_weights = bal_weights.Norml2();
   }

   // Limit the node movement.
   // The limiting distances can be given by a general function of space.
   ParFiniteElementSpace dist_fespace(&mesh, &fec_h1); // scalar space
   ParGridFunction dist(&dist_fespace);
   dist = 1.0;
   // The small_phys_size is relevant only with proper normalization.
   if (normalization) { dist = small_phys_size; }
   auto coeff_lim_func = [&](const Vector &x) { return x(0) + lim_const; };
   FunctionCoefficient lim_coeff(coeff_lim_func);
   if (lim_const != 0.0)
   {
      if (lim_type == 0) { tmop_integ->EnableLimiting(x0, dist, lim_coeff); }
      else
      {
         tmop_integ->EnableLimiting(x0, dist, lim_coeff,
                                    new TMOP_ExponentialLimiter);
      }
   }

   // Setup the NonlinearForm which defines the integral of interest.
   ParNonlinearForm a(&fes_h1);
   a.SetAssemblyLevel(pa ? AssemblyLevel::PARTIAL : AssemblyLevel::LEGACY);

   std::unique_ptr<FunctionCoefficient> metric_coeff1 = nullptr;
   std::unique_ptr<TMOP_QualityMetric> metric2 = nullptr;
   std::unique_ptr<TargetConstructor> target_c2 = nullptr;
   FunctionCoefficient metric_coeff2(weight_fun);
   TMOPComboIntegrator *combo = nullptr;

   if (combomet > 0)
   {
      // First metric.
      auto coeff_1_func = [&](const Vector &x) { return x(0) + M_PI; };
      metric_coeff1.reset(new FunctionCoefficient(coeff_1_func));
      tmop_integ->SetCoefficient(*metric_coeff1);

      // Second metric.
      if (dim == 2) { metric2.reset(new TMOP_Metric_077); }
      else          { metric2.reset(new TMOP_Metric_315); }
      TMOP_Integrator *tmop_integ2 = nullptr;
      if (combomet == 1)
      {
         target_c2.reset(
            new TargetConstructor(TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE));
         target_c2->SetVolumeScale(0.01);
         target_c2->SetNodes(x0); assert(false && "?");
         tmop_integ2 = new TMOP_Integrator(metric2.get(), target_c2.get());
         tmop_integ2->SetCoefficient(metric_coeff2);
      }
      else { tmop_integ2 = new TMOP_Integrator(metric2.get(), target_c.get()); }
      tmop_integ2->IntegrateOverTarget(integ_over_targ);
      tmop_integ2->SetIntegrationRules(*irules, quad_order);
      if (fdscheme) { tmop_integ2->EnableFiniteDifferences(x); }
      tmop_integ2->SetExactActionFlag(exactaction);

      combo = new TMOPComboIntegrator;
      combo->AddTMOPIntegrator(tmop_integ);
      combo->AddTMOPIntegrator(tmop_integ2);

      if (normalization) { combo->ParEnableNormalization(x0); }
      if (lim_const != 0.0) { combo->EnableLimiting(x0, dist, lim_coeff); }

      a.AddDomainIntegrator(combo);
   }
   else { a.AddDomainIntegrator(tmop_integ); }

   // The PA setup must be performed after all integrators have been added.
   if (pa) { a.Setup(); }

   // Has to be after the enabling of the limiting / alignment, as it computes
   // normalization factors for these terms as well.
   if (normalization)
   {
      tmop_integ->ParEnableNormalization(x0);
      if (combomet > 0) { combo->ParEnableNormalization(x0); }
   }
   real_t unused;
   tmop_integ->GetNormalizationFactors(res.met_normal, res.lim_normal, unused);

   // Compute the minimum det(J) of the starting mesh.
   real_t min_detJ = infinity();
   const int NE = mesh.GetNE();
   for (int i = 0; i < NE; i++)
   {
      const IntegrationRule &ir =
         irules->Get(fespace.GetFE(i)->GetGeomType(), quad_order);
      auto transf = mesh.GetElementTransformation(i);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         min_detJ = min(min_detJ, transf->Jacobian().Det());
      }
   }
   real_t minJ0;
   MPI_Allreduce(&min_detJ, &minJ0, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN,
                 MPI_COMM_WORLD);
   min_detJ = minJ0;
   REQUIRE(min_detJ > 0.0);
   res.min_detJ = min_detJ;

   if (periodic) { tmop_integ->SetInitialMeshPos(&x0); }
   const real_t init_energy = a.GetParGridFunctionEnergy(periodic ? dx : x);
   res.init_energy = init_energy;

   // Fix all boundary nodes
   REQUIRE(move_bnd == false);
   Array<int> ess_bdr(periodic ? 0 : mesh.bdr_attributes.Max());
   ess_bdr = 1;
   if (!periodic) { a.SetEssentialBC(ess_bdr); }

   // Diagonal test, skip if combo
   Vector &xt(x.GetTrueVector());
   Vector d(fespace.GetTrueVSize());
   d.UseDevice(true);
   res.diag = 0.0;
   if (diag && combomet == 0)
   {
      if (pa) { a.GetGradient(xt).AssembleDiagonal(d); }
      else
      {
         ParNonlinearForm nlf_fa(&fes_h1);
         auto *nlfi_fa = new TMOP_Integrator(metric.get(), target_c.get());
         nlfi_fa->SetIntegrationRules(*irules, quad_order);
         if (normalization) { nlfi_fa->ParEnableNormalization(x0); }
         if (lim_const != 0.0)
         {
            if (lim_type == 0) { nlfi_fa->EnableLimiting(x0, dist, lim_coeff); }
            else
            {
               nlfi_fa->EnableLimiting(x0, dist, lim_coeff,
                                       new TMOP_ExponentialLimiter);
            }
         }
         nlf_fa.AddDomainIntegrator(nlfi_fa);
         nlf_fa.SetEssentialBC(ess_bdr);
         dynamic_cast<GradientClass &>(nlf_fa.GetGradient(xt)).GetDiag(d);
      }
      res.diag = d * d;
   }

   // Linear solver for the system's Jacobian
   std::unique_ptr<Solver> S = nullptr, S_prec = nullptr;
   if (lin_solver == 0) { S.reset(new DSmoother(1, 1.0, max_lin_iter)); }
   else if (lin_solver == 1)
   {
      auto cg = new CGSolver(PFesGetParMeshGetComm(fes_h1));
      cg->SetMaxIter(max_lin_iter);
      cg->SetRelTol(linsol_rtol);
      cg->SetAbsTol(0.0);
      cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S.reset(cg);
   }
   else
   {
      auto minres = new MINRESSolver(PFesGetParMeshGetComm(fes_h1));
      minres->SetMaxIter(max_lin_iter);
      minres->SetRelTol(linsol_rtol);
      minres->SetAbsTol(0.0);
      minres->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      if (lin_solver == 3 || lin_solver == 4)
      {
         if (pa)
         {
            MFEM_VERIFY(lin_solver != 4, "PA l1-Jacobi is not implemented");
            auto js = new OperatorJacobiSmoother;
            js->SetPositiveDiagonal(true);
            S_prec.reset(js);
         }
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_PA_MPI)
         else
         {
            auto hs = new HypreSmoother;
            hs->SetType((lin_solver == 3)
                        ? HypreSmoother::Jacobi
                        : HypreSmoother::l1Jacobi, 1);
            S_prec.reset(hs);
         }
#else
         else
         {
            auto ds = new DSmoother((lin_solver == 3) ? 0 : 1, 1.0, 1);
            ds->SetPositiveDiagonal(true);
            S_prec.reset(ds);
         }
#endif
         minres->SetPreconditioner(*S_prec);
      }
      S.reset(minres);
   }

   // Perform the nonlinear optimization.
   const IntegrationRule &ir =
      irules->Get(mesh.GetTypicalElementGeometry(), quad_order);
#if defined(MFEM_USE_MPI) && defined(MFEM_TMOP_PA_MPI)
   TMOPNewtonSolver solver(PFesGetParMeshGetComm(fes_h1), ir);
#else
   TMOPNewtonSolver solver(ir);
#endif
   // Provide all integration rules in case of a mixed mesh.
   solver.SetIntegrationRules(*irules, quad_order);
   // Specify linear solver when we use a Newton-based solver.
   solver.SetPreconditioner(*S);
   solver.SetMinDetPtr(&min_detJ);
   solver.SetMaxIter(newton_iter);
   solver.SetRelTol(newton_rtol);
   solver.SetAbsTol(0.0);
   solver.SetPrintLevel(verbosity_level >= 1 ? 3 : -1);
   solver.SetOperator(a);

   Vector x_init(x), b(0);
   b.UseDevice(true);

   for (int i = 0; i < newton_loop; i++)
   {
      x = x_init;
      x.SetTrueVector();

      auto datc = dynamic_cast<DiscreteAdaptTC *>(target_c.get());
      if (datc && target_id == 5) { datc->SetDiscreteTargetSize(size); }
      if (datc && target_id == 7)
      {
         datc->SetDiscreteTargetAspectRatio(aspr3d);
      }

      dist *= 0.93;
      if (normalization) { dist = small_phys_size; }

      if (lim_const != 0.0)
      {
         if (lim_type == 0) { tmop_integ->EnableLimiting(x0, dist, lim_coeff); }
         else
         {
            tmop_integ->EnableLimiting(x0, dist, lim_coeff,
                                       new TMOP_ExponentialLimiter);
         }
      }

      a.Setup();

      if (normalization) { tmop_integ->ParEnableNormalization(x); }

      solver.Mult(b, x.GetTrueVector());
      x.SetFromTrueVector();

      REQUIRE(solver.GetConverged());

      // Report the final energy of the functional.
      if (periodic)
      {
         ParGridFunction dx_L2(x); dx_L2 -= x0;
         dx.ProjectGridFunction(dx_L2);
         tmop_integ->SetInitialMeshPos(&x0);
         res.final_energy = a.GetParGridFunctionEnergy(dx);
      }
      else
      {
         res.final_energy = a.GetParGridFunctionEnergy(x);
      }
   }

   Vector &x_t(x.GetTrueVector());
   real_t x_t_dot = x_t * x_t, dot;
   MPI_Allreduce(&x_t_dot, &dot, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 MPI_COMM_WORLD);
   res.dot = dot;

   return EXIT_SUCCESS;
}

} // namespace mfem

static inline int argn(const char *argv[], int argc = 0)
{
   while (argv[argc]) { argc += 1; }
   return argc;
}

static inline void req_tmop(int id, const char *args[], Req &res)
{
   REQUIRE(tmop(id, res, argn(args), const_cast<char **>(args)) == 0);
}

#define DEFAULT_ARGS const char *args[] = { "tmop_pa_tests", "-pa", "-m", "mesh", \
   "-o", "0", "-rs", "0", "-mid", "0", "-tid", "0", "-qt", "1", "-qo", "0", \
   "-ni", "10", "-nl", "1", "-nrtol", "1e-8", "-lrtol", "1e-12", "-ls", "2", "-li", "100", "-lc", "0", \
   "-lt", "0", "-no-nor", "-ji", "0", "-diag", "-cmb",  "0", "-no-bec", "-no-per", nullptr }

constexpr int ALV = 1, MSH = 3, POR = 5, RS = 7, MID = 9, TID = 11, QTY = 13,
              QOR = 15, NI = 17, NL = 19, NRTOL = 21, LRTOL = 23, LS = 25, LI = 27, LC = 29,
              LT = 31, NOR = 32, JI = 34, DIAG = 35, CMB = 37, BEC = 38, PER = 39;

static inline void dump_args(int id, const char *args[])
{
   if (id != 0) { return; }
   const char *format =
      "tmop_pa_tests %6.6s -m %s -o %s -rs %s -mid %s -tid %s -qt %s -qo %s "
      "-ni %s -nl %s -nrtol %s -lrtol %s -ls %s -li %s -lc %s -lt %s %s -ji %s "
      "%s -cmb %s %s %s\n";
   printf(format, args[ALV], args[MSH], args[POR], args[RS], args[MID], args[TID],
          args[QTY], args[QOR], args[NI], args[NL], args[NRTOL], args[LRTOL],
          args[LS], args[LI], args[LC], args[LT], args[NOR], args[JI], args[DIAG],
          args[CMB], args[BEC], args[PER]);
   fflush(nullptr);
}

static inline void tmop_require(int id, const char *args[])
{
   Req res[2];
   constexpr real_t eps = 2e-12;
   (args[ALV] = "-pa", dump_args(id, args), req_tmop(id, args, res[0]));
   (args[ALV] = "-no-pa", dump_args(id, args), req_tmop(id, args, res[1]));
   REQUIRE(res[0].dot == MFEM_Approx(res[1].dot));
   REQUIRE(res[0].diag == MFEM_Approx(res[1].diag));
   REQUIRE(res[0].min_detJ == MFEM_Approx(res[1].min_detJ));
   REQUIRE(res[0].met_normal == MFEM_Approx(res[1].met_normal));
   REQUIRE(res[0].lim_normal == MFEM_Approx(res[1].lim_normal));
   REQUIRE(res[0].bal_weights == MFEM_Approx(res[1].bal_weights));
   REQUIRE(res[0].init_energy == MFEM_Approx(res[1].init_energy));
   REQUIRE(res[0].final_energy == MFEM_Approx(res[1].final_energy, eps));
}

static constexpr int SZ = 32;

static inline const char *itoa(const int i, char *buf)
{
   const int rtn = std::snprintf(buf, SZ, "%d", i);
   if (rtn < 0) { MFEM_ABORT("snprintf error!"); }
   MFEM_ASSERT(rtn < SZ, "snprintf overflow!");
   return buf;
}

static inline const char *dtoa(const real_t d, char *buf)
{
   const int rtn = std::snprintf(buf, SZ, "%g", d);
   if (rtn < 0) { MFEM_ABORT("snprintf error!"); }
   MFEM_ASSERT(rtn < SZ, "snprintf overflow!");
   return buf;
}

class Launch
{
   using list_t = std::list<int>;

public:
   class Args
   {
      friend class Launch;

   private:
      const char *name = nullptr;
      const char *mesh = "../../data/star.mesh";
      int newton_iter = 100;
      real_t newton_rtol = 1e-6;
      real_t linsol_rtol = 1e-8;
      int rs_levels = 0;
      int linsol_iter = 100;
      int combo = 0;
      bool diag = true;
      bool bal_expl_combo = false;
      bool normalization = false;
      bool periodic = false;
      real_t lim_const = 0.0;
      int lim_type = 0;
      real_t jitter = 0.0;
      list_t order = { 1, 2, 3, 4 };
      list_t target_id = { 1, 2, 3 };
      list_t metric_id = { 1, 2 };
      list_t quad_order = { 2, 4, 8 };
      list_t lin_solver = { 3, 2, 1 };
      list_t newton_loop = { 1, 3 };

   public:
      Args(const char *name = nullptr): name(name) {}
      Args &MESH(const char *arg) { mesh = arg; return *this; }
      // int
      Args &NEWTON_ITERATIONS(const int arg) { newton_iter = arg; return *this; }
      Args &LINSOL_ITERATIONS(const int arg) { linsol_iter = arg; return *this; }
      Args &REFINE(const int arg) { rs_levels = arg; return *this; }
      Args &CMB(const int arg) { combo = arg; return *this; }
      Args &LIMIT_TYPE(const int arg) { lim_type = arg; return *this; }
      // bool
      Args &NORMALIZATION() { normalization = true; return *this; }
      Args &DIAGONAL(const bool arg) { diag = arg; return *this; }
      Args &BALANCE_EXPLICIT_COMBO() { bal_expl_combo = true; return *this; }
      Args &PERIODIC() { periodic = true; return *this; }
      // real_t
      Args &NEWTON_RTOLERANCE(const real_t arg) { newton_rtol = arg; return *this; }
      Args &LINSOL_RTOLERANCE(const real_t arg) { linsol_rtol = arg; return *this; }
      Args &LIMITING(const real_t arg) { lim_const = arg; return *this; }
      Args &JI(const real_t arg) { jitter = arg; return *this; }
      // lists
      Args &POR(list_t arg) { order = arg; return *this; }
      Args &TID(list_t arg) { target_id = arg; return *this; }
      Args &MID(list_t arg) { metric_id = arg; return *this; }
      Args &QOR(list_t arg) { quad_order = arg; return *this; }
      Args &LS(list_t arg) { lin_solver = arg; return *this; }
      Args &NL(list_t arg) { newton_loop = arg; return *this; }
   };
   const char *name, *mesh;
   int NEWTON_ITERATIONS, LINSOL_ITERATIONS, REFINE, COMBO, LIMIT_TYPE;
   bool NORMALIZATION, DIAGONAL, BAL_EXPL_COMBO, PERIODIC;
   real_t NEWTON_RTOLERANCE, LINSOL_RTOLERANCE, LIMITING, JITTER;
   list_t P_ORDERS, TARGET_IDS, METRIC_IDS, Q_ORDERS, LINEAR_SOLVERS, NEWTON_LOOPS;

public:
   Launch(Args a = Args()):
      name(a.name), mesh(a.mesh),
      // int
      NEWTON_ITERATIONS(a.newton_iter),
      LINSOL_ITERATIONS(a.linsol_iter),
      REFINE(a.rs_levels), COMBO(a.combo), LIMIT_TYPE(a.lim_type),
      // bool
      NORMALIZATION(a.normalization),
      DIAGONAL(a.diag),
      BAL_EXPL_COMBO(a.bal_expl_combo),
      PERIODIC(a.periodic),
      // real_t
      NEWTON_RTOLERANCE(a.newton_rtol),
      LINSOL_RTOLERANCE(a.linsol_rtol),
      LIMITING(a.lim_const),
      JITTER(a.jitter),
      // lists
      P_ORDERS(a.order),
      TARGET_IDS(a.target_id),
      METRIC_IDS(a.metric_id),
      Q_ORDERS(a.quad_order),
      LINEAR_SOLVERS(a.lin_solver),
      NEWTON_LOOPS(a.newton_loop) { }

   void Run(const int id = 0, bool all = false) const
   {
      if ((id == 0) && name) { mfem::out << "[" << name << "]" << std::endl; }
      DEFAULT_ARGS;
      char ni[SZ] {}, nrtol[SZ] {}, lrtol[SZ] {}, rs[SZ] {}, li[SZ] {},
           lc[SZ] {}, lt[SZ] {}, ji[SZ] {}, cmb[SZ] {};
      args[MSH] = mesh;
      // int
      args[NI] = itoa(NEWTON_ITERATIONS, ni);
      args[LI] = itoa(LINSOL_ITERATIONS, li);
      args[RS] = itoa(REFINE, rs);
      args[CMB] = itoa(COMBO, cmb);
      args[LT] = itoa(LIMIT_TYPE, lt);
      // bool
      args[NOR] = NORMALIZATION ? "-nor" : "-no-nor";
      args[DIAG] = DIAGONAL ? "-diag" : "-no-diag";
      args[BEC] = BAL_EXPL_COMBO ? "-bec" : "-no-bec";
      args[PER] = PERIODIC ? "-per" : "-no-per";
      // real_t
      args[NRTOL] = dtoa(NEWTON_RTOLERANCE, nrtol);
      args[LRTOL] = dtoa(LINSOL_RTOLERANCE, lrtol);
      args[LC] = dtoa(LIMITING, lc);
      args[JI] = dtoa(JITTER, ji);

      for (int p : P_ORDERS)
      {
         char por[SZ] {};
         args[POR] = itoa(p, por);
         for (int t : TARGET_IDS)
         {
            char tid[SZ] {};
            args[TID] = itoa(t, tid);
            for (int m : METRIC_IDS)
            {
               char mid[SZ] {};
               args[MID] = itoa(m, mid);
               for (int q : Q_ORDERS)
               {
                  if (q <= p) { continue; }
                  char qor[SZ] {};
                  args[QOR] = itoa(q, qor);
                  for (int ls : LINEAR_SOLVERS)
                  {
                     // skip some linear solver & metric combinations
                     // that lead to non positive definite operators
                     if (ls == 1 && m != 1) { continue; }
                     char lsb[SZ] {};
                     args[LS] = itoa(ls, lsb);
                     for (int n : NEWTON_LOOPS)
                     {
                        char nl[SZ] {};
                        args[NL] = itoa(n, nl);
                        tmop_require(id, args);
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
         if (!all) { break; }
      }
   }
};

// id: MPI rank, all: launch all non-regression tests
static void tmop_tests(int id = 0, bool all = false)
{
#if defined(MFEM_TMOP_PA_MPI)
   if (HypreUsingGPU())
   {
      cout << "\nAs of mfem-4.3 and hypre-2.22.0 (July 2021) this unit test\n"
           << "is NOT supported with the GPU version of hypre.\n\n";
      return;
   }
#endif

#ifndef _WIN32
   {
      using Det = QuadratureInterpolator::DetKernels;
      Det::Specialization<2, 2, 3, 3>::Add();
      Det::Specialization<2, 2, 5, 5>::Add();
      Det::Specialization<3, 3, 2, 3>::Add();
      Det::Specialization<3, 3, 3, 4>::Add();
      Det::Specialization<3, 3, 4, 6>::Add();

      using Grad = QuadratureInterpolator::GradKernels;
      Grad::Specialization<2, QVectorLayout::byNODES, false, 2, 3, 5>::Add();
      Grad::Specialization<2, QVectorLayout::byNODES, false, 2, 5, 5>::Add();
      Grad::Specialization<2, QVectorLayout::byNODES, false, 2, 6, 6>::Add();
      Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 4, 5>::Add();

      using TensorEval = QuadratureInterpolator::TensorEvalKernels;
      TensorEval::Specialization<2, QVectorLayout::byVDIM, 2, 2, 2>::Opt<4>::Add();
      TensorEval::Specialization<2, QVectorLayout::byVDIM, 2, 3, 3>::Opt<4>::Add();
      TensorEval::Specialization<2, QVectorLayout::byVDIM, 2, 4, 4>::Opt<2>::Add();
      TensorEval::Specialization<2, QVectorLayout::byVDIM, 2, 5, 5>::Opt<2>::Add();
      TensorEval::Specialization<3, QVectorLayout::byVDIM, 3, 2, 3>::Opt<2>::Add();
      TensorEval::Specialization<3, QVectorLayout::byVDIM, 3, 3, 4>::Opt<1>::Add();
      TensorEval::Specialization<3, QVectorLayout::byVDIM, 3, 4, 6>::Opt<1>::Add();

      using MassDiagonal = MassIntegrator::DiagonalPAKernels;
      MassDiagonal::Specialization<2, 2, 3>::Add();
      MassDiagonal::Specialization<2, 2, 4>::Add();
      MassDiagonal::Specialization<2, 2, 5>::Add();
      MassDiagonal::Specialization<3, 2, 4>::Add();
      MassDiagonal::Specialization<3, 2, 6>::Add();

      using MassApply = MassIntegrator::ApplyPAKernels;
      MassApply::Specialization<2, 2, 3>::Add();
      MassApply::Specialization<2, 2, 4>::Add();
      MassApply::Specialization<2, 2, 5>::Add();
      MassApply::Specialization<3, 2, 4>::Add();
      MassApply::Specialization<3, 2, 6>::Add();
   }
#endif

   const real_t jitter = 1. / (M_PI * M_PI);

   Launch(Launch::Args("2D Periodic + adapted discrete size")
          .MESH("../../data/periodic-square.mesh")
          .PERIODIC()
          .REFINE(1)
          .NORMALIZATION()
          .MID({ 94 })
          .TID({ 5 })
          .LS({ 3 })
          .NEWTON_RTOLERANCE(1e-6)
          .LINSOL_RTOLERANCE(1e-8)
          .LINSOL_ITERATIONS(150)
          .POR({ 1, 2, 3, 4 })
          .QOR({ 4, 8 })
          .NL({ 3 })
          .DIAGONAL(false))
   .Run(id, all);

   Launch(Launch::Args("3D Periodic + adapted discrete size")
          .MESH("../../data/periodic-cube.mesh")
          .PERIODIC()
          .MID({ 338 })
          .TID({ 5 })
          .LS({ 2 })
          .NORMALIZATION()
          .NEWTON_RTOLERANCE(1e-5)
          .LINSOL_RTOLERANCE(1e-10)
          .LINSOL_ITERATIONS(200)
          .POR({ 1, 2 })
          .QOR({ 4 })
          .NL({ 1 })
          .DIAGONAL(false))
   .Run(id, all);

   Launch(Launch::Args("2D + Combo + Balance")
          .MESH("../../miniapps/meshing/square01.mesh")
          .REFINE(1)
          .JI(jitter)
          .NORMALIZATION()
          .TID({ 5 })
          .MID({ 80 })
          .LS({ 2 })
          .LINSOL_RTOLERANCE(1e-10)
          .POR({ 2 })
          .QOR({ 6 })
          .CMB(2)
          .BALANCE_EXPLICIT_COMBO()
         )
   .Run(id, all);

   Launch(Launch::Args("3D + Combo + Balance")
          .MESH("../../miniapps/meshing/cube.mesh")
          .REFINE(1)
          .JI(jitter)
          .NORMALIZATION()
          .TID({ 5 })
          .MID({ 302 })
          .LS({ 2 })
          .POR({ 1, 2 })
          .QOR({ 2, 8 })
          .CMB(2)
          .BALANCE_EXPLICIT_COMBO())
   .Run(id, all);

   Launch(Launch::Args("TC_IDEAL_SHAPE_UNIT_SIZE_2D_KERNEL")
          .MESH("../../data/star.mesh")
          .REFINE(1)
          .JI(jitter)
          .POR({ 1, 2 })
          .QOR({ 2, 3 })
          .TID({ 1 })
          .MID({ 2 }))
   .Run(id, all);

   Launch(Launch::Args("TC_IDEAL_SHAPE_GIVEN_SIZE_2D_KERNEL")
          .MESH("../../data/star.mesh")
          .REFINE(1)
          .JI(jitter)
          .POR({ 1, 2 })
          .QOR({ 2, 3 })
          .TID({ 3 })
          .MID({ 2 }))
   .Run(id, all);

   Launch(Launch::Args("TC_GIVEN_SHAPE_AND_SIZE_2D_KERNEL")
          .MESH("../../data/star.mesh")
          .REFINE(1)
          .JI(jitter)
          .NORMALIZATION()
          .POR({ 1, 2 })
          .QOR({ 2, 3 })
          .TID({ 8 })
          .MID({ 94 })
          .LS({ 3 }))
   .Run(id, all);

   Launch(Launch::Args("TC_GIVEN_SHAPE_AND_SIZE_3D_KERNEL")
          .MESH("../../data/toroid-hex.mesh")
          .LIMITING(M_PI)
          .LIMIT_TYPE(1)
          .REFINE(1)
          .JI(jitter)
          .NORMALIZATION()
          .POR({ 2 })
          .QOR({ 4 })
          .TID({ 8 })
          .MID({ 338 })
          .LS({ 3 }))
   .Run(id, all);

   Launch(Launch::Args("TC_IDEAL_SHAPE_UNIT_SIZE_3D_KERNEL")
          .MESH("../../miniapps/meshing/cube.mesh")
          .REFINE(1)
          .JI(jitter)
          .POR({ 1, 2 })
          .QOR({ 2, 3 })
          .TID({ 1 })
          .MID({ 302 }))
   .Run(id, all);

   Launch(Launch::Args("TC_IDEAL_SHAPE_GIVEN_SIZE_3D_KERNEL")
          .MESH("../../miniapps/meshing/cube.mesh")
          .REFINE(1)
          .JI(jitter)
          .POR({ 1, 2 })
          .QOR({ 2, 6 })
          .TID({ 3 })
          .MID({ 302 }))
   .Run(id, all);

   Launch(Launch::Args("Star")
          .MESH("../../data/star.mesh")
          .POR({ 1, 2, 3, 4 })
          .QOR({ 2, 4, 8 })
          .TID({ 1, 2, 3 })
          .MID({ 1, 2 }))
   .Run(id, all);

   Launch(Launch::Args("Square01 + Adapted analytic Hessian")
          .MESH("../../miniapps/meshing/square01.mesh")
          .REFINE(1)
          .POR({ 1, 2 })
          .QOR({ 2, 4 })
          .TID({ 4 })
          .MID({ 1, 2 }))
   .Run(id, all);

   Launch(Launch::Args("Square01 + Adapted discrete size")
          .MESH("../../miniapps/meshing/square01.mesh")
          .REFINE(1)
          .NORMALIZATION()
          .POR({ 1 })
          .QOR({ 4, 6 })
          .LINSOL_RTOLERANCE(1e-12)
          .LINSOL_ITERATIONS(150)
          .TID({ 5 })
          .MID({ 80, 94 })
          .LS({ 3 }))
   .Run(id, all);

   Launch(Launch::Args("Blade")
          .MESH("../../miniapps/meshing/blade.mesh")
          .POR({ 1, 2 })
          .QOR({ 2, 4 })
          .NEWTON_RTOLERANCE(1e-13)
          .TID({ 1, 2, 3 })
          .MID({ 2 })
          .LS({ 2 }))
   .Run(id, all);

   Launch(Launch::Args("Blade + normalization")
          .MESH("../../miniapps/meshing/blade.mesh")
          .NORMALIZATION()
          .POR({ 1, 2 })
          .QOR({ 2, 4 })
          .LINSOL_ITERATIONS(200)
          .NEWTON_RTOLERANCE(1e-12)
          .NL({ 2 })
          .TID({ 1, 2, 3 })
          .MID({ 2 }))
   .Run(id, all);

   Launch(Launch::Args("Blade + limiting + normalization")
          .MESH("../../miniapps/meshing/blade.mesh")
          .NORMALIZATION()
          .LIMITING(M_PI)
          .POR({ 1, 2 })
          .QOR({ 2, 4 })
          .LINSOL_ITERATIONS(200)
          .NEWTON_RTOLERANCE(1e-12)
          .TID({ 1, 2, 3 })
          .MID({ 2 }))
   .Run(id, all);

   Launch(Launch::Args("Blade + limiting_expo + normalization")
          .MESH("../../miniapps/meshing/blade.mesh")
          .NORMALIZATION()
          .LIMITING(M_PI)
          .LIMIT_TYPE(1)
          .LINSOL_ITERATIONS(200)
          .NEWTON_RTOLERANCE(1e-12)
          .POR({ 1, 2 })
          .QOR({ 2, 4 })
          .TID({ 1, 2, 3 })
          .MID({ 2 }))
   .Run(id, all);

   Launch(Launch::Args("Cube")
          .MESH("../../miniapps/meshing/cube.mesh")
          .REFINE(1)
          .JI(jitter)
          .POR({ 1, 2 })
          .QOR({ 2, 4 })
          .TID({ 2, 3 })
          .MID({ 302, 303 }))
   .Run(id, all);

   Launch(
      Launch::Args("Cube + Discrete size & aspect + normalization + limiting")
      .MESH("../../miniapps/meshing/cube.mesh")
      .NORMALIZATION()
      .LIMITING(M_PI)
      .POR({ 1, 2 })
      .QOR({ 4, 2 })
      .TID({ 7 })
      .MID({ 302, 321 }))
   .Run(id, all);

   Launch(Launch::Args("Cube + Discrete size + normalization")
          .MESH("../../miniapps/meshing/cube.mesh")
          .NORMALIZATION()
          .POR({ 1 })
          .QOR({ 4, 2 })
          .NEWTON_RTOLERANCE(1e-12)
          .TID({ 5 })
          .MID({ 315, 318, 332, 338 }))
   .Run(id, all);

   // Note: order 1 has no interior nodes, so all residuals are zero and the
   // Newton iteration exits immediately.
   // Note: In parallel, orders > 1 fail with: Initial mesh was valid,
   //       but intermediate mesh is invalid. Contact TMOP Developers.
   Launch(Launch::Args("Toroid-Hex")
          .MESH("../../data/toroid-hex.mesh")
          .POR({ 1, 2 })
          .QOR({ 2, 4, 8 })
          .TID({ 1, 2, 3 })
          .MID({ 302, 303, 321 }))
   .Run(id, all);

   Launch(Launch::Args("Toroid-Hex + limiting")
          .MESH("../../data/toroid-hex.mesh")
          .LIMITING(M_PI)
          .POR({ 1, 2 })
          .QOR({ 2, 4 })
          .NL({ 3, 1 })
          .TID({ 1, 2 })
          .MID({ 321, 338 }))
   .Run(id, all);

   Launch(Launch::Args("Toroid-Hex + limiting + norm.")
          .MESH("../../data/toroid-hex.mesh")
          .LIMITING(M_PI)
          .NORMALIZATION()
          .POR({ 1, 2 })
          .QOR({ 2, 4 })
          .TID({ 1, 2 })
          .MID({ 321, 315, 318, 332, 338 }))
   .Run(id, all);

   Launch(Launch::Args("Toroid-Hex + limiting_expo + norm.")
          .MESH("../../data/toroid-hex.mesh")
          .LIMITING(M_PI)
          .LIMIT_TYPE(1)
          .NORMALIZATION()
          .POR({ 1, 2 })
          .QOR({ 2, 4 })
          .TID({ 1, 2 })
          .MID({ 321 }))
   .Run(id, all);

   // -m cube.mesh -rs 1 -tid 5 -mid 321 -ni 5 -ls 3 -li 100 -lc 1.0 -nor
   Launch(Launch::Args("Cube + Blast options")
          .MESH("../../miniapps/meshing/cube.mesh")
          .REFINE(1)
          .TID({ 5 })
          .MID({ 321 })
          .LS({ 3 })
          .NEWTON_ITERATIONS(10)
          .NEWTON_RTOLERANCE(1e-10)
          .LINSOL_RTOLERANCE(1e-14)
          .LIMITING(M_PI)
          .NORMALIZATION()
          .POR({ 1, 2, 3 })
          .QOR({ 3, 6 })
          .NL({ 1, 2 }))
   .Run(id, all);

   // Combo 2D
   Launch(Launch::Args("Square01 + Combo")
          .MESH("../../miniapps/meshing/square01.mesh")
          .REFINE(1)
          .JI(jitter)
          .NORMALIZATION()
          .LINSOL_RTOLERANCE(1e-10)
          .TID({ 5 })
          .MID({ 2 })
          .LS({ 2 })
          .POR({ 2 })
          .QOR({ 8 })
          .CMB(2))
   .Run(id, all);

   // Combo 3D
   Launch(Launch::Args("Cube + Combo")
          .MESH("../../miniapps/meshing/cube.mesh")
          .REFINE(1)
          .JI(jitter)
          .NORMALIZATION()
          .TID({ 5 })
          .MID({ 302 })
          .LS({ 2 })
          .POR({ 1, 2 })
          .QOR({ 2, 8 })
          .CMB(2))
   .Run(id, all);

   // NURBS
   Launch(Launch::Args("2D Nurbs")
          .MESH("../../data/square-disc-nurbs.mesh")
          .REFINE(1)
          .JI(jitter)
          .POR({ 1, 2 })
          .QOR({ 2, 4 })
          .TID({ 1, 2, 3 })
          .MID({ 1, 2 }))
   .Run(id, all);

   Launch(Launch::Args("3D Nurbs")
          .MESH("../../data/beam-hex-nurbs.mesh")
          .REFINE(1)
          .JI(jitter)
          .POR({ 1, 2 })
          .QOR({ 2, 4 })
          .TID({ 1, 2, 3 })
          .MID({ 302, 321 }))
   .Run(id, all);

   // The following tests need more iterations to converge between PA & non-PA
   // They can only be launched with the `--all` command line option

   if (!all) { return; }

   Launch(Launch::Args("Blade + Discrete size + normalization")
          .MESH("../../miniapps/meshing/blade.mesh")
          .LINSOL_ITERATIONS(1000)
          .NORMALIZATION()
          .NEWTON_RTOLERANCE(1e-14)
          .LINSOL_RTOLERANCE(1e-14)
          .POR({ 1 })
          .QOR({ 2 })
          .TID({ 5 })
          .MID({ 7 })
          .LS({ 2 })
          .NL({ 4 }))
   .Run(id, true);

   Launch(Launch::Args("Blade + Discrete size + normalization")
          .MESH("../../miniapps/meshing/blade.mesh")
          .LINSOL_ITERATIONS(500)
          .NORMALIZATION()
          .NEWTON_RTOLERANCE(1e-12)
          .LINSOL_RTOLERANCE(1e-10)
          .POR({ 1 })
          .QOR({ 2 })
          .TID({ 5 })
          .MID({ 2 }))
   .Run(id, true);
}

#ifdef MFEM_TMOP_PA_MPI
TEST_CASE("tmop_pa", "[TMOP_PA], [Parallel]")
{
   tmop_tests(Mpi::WorldRank(), launch_all_non_regression_tests);
}
#else
TEST_CASE("tmop_pa", "[TMOP_PA]")
{
   tmop_tests(0, launch_all_non_regression_tests);
}
#endif

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_SINGLE
   std::cout << "\nThe TMOP unit tests are not supported in single"
             " precision.\n\n";
   return MFEM_SKIP_RETURN_VALUE;
#endif

#ifdef MFEM_TMOP_PA_MPI
   mfem::Mpi::Init();
   mfem::Hypre::Init();
#endif
#ifdef MFEM_TMOP_PA_DEVICE
   Device device(MFEM_TMOP_PA_DEVICE);
#else
   Device device("cpu"); // make sure hypre runs on CPU, if possible
#endif
   device.Print();

#ifdef MFEM_TMOP_PA_MPI
   return RunCatchSession(argc, argv, { "[Parallel]" }, Root());
#else
   // Exclude parallel tests.
   return RunCatchSession(argc, argv, { "~[Parallel]" });
#endif
}
