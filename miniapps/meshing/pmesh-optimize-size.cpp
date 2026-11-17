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

// Optimize the shape and size of a mesh using TMOP. By default the target is
// the ideal shape scaled to match the initial Jacobian determinant at each
// quadrature point. The optimized mesh is saved in MFEM format, and a VisIt
// collection records the initial and optimized meshes at cycles 0 and 1.
//
// Compile with: make pmesh-optimize-size
// Sample runs:
//   mpirun -np 4 pmesh-optimize-size -m junction-box-tet.mesh -mid 321 -mod-bndr-attr -bnd -vl 2
//   mpirun -np 4 pmesh-optimize-size -m junction-box-tet.mesh -mod-bndr-attr -bnd -vl 2 -mid 334 -gamma 0.7
//   2D demo
//   mpirun -np 4 pmesh-optimize-size -m stretched2D-tri.mesh -rr 2 -rrp 0.2 -vl 2 -vis -mid 80

// Suggested TMOP settings
// TMOP target
// target_id = 3; // IDEAL SHAPE GIVEN SIZE
// use       = 2 for IDEAL SHAPE EQUAL SIZE but that will change element sizes
// TMOP metric
// metric_id = 321 // depends on shape+size
// Or          334 // depends on shape+size, with gamma \ in [0,1] controlling
//                 // shape-size trade-off. \gamma = 0 favors shape, \gamma = 1
//                 // favors size.
// Boundary movement
// Tangential movement on Cartesian boundary faces can be enabled with
// the -mod-bndr-attr and -bnd options.

#include "mfem.hpp"
#include "mesh-fitting.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const MPI_Comm comm = MPI_COMM_WORLD;
   const bool root = Mpi::Root();

   const char *mesh_file = "google_mesh.msh";
   const char *output_file = "optimized-size.mesh";
   int order = 1;
   int rs_levels = 0;
   int rp_levels = 0;
   int random_refine_passes = 0;
   real_t random_refine_probability = 0.5;
   int metric_id = 321;
   real_t gamma = 0.5;
   int target_id = 3;
   int quad_order = 8;
   real_t lim_const = 0.0;
   int newton_iter = 100;
#ifdef MFEM_USE_SINGLE
   real_t newton_rtol = 1e-3;
   const real_t lin_rtol = 1e-5;
#else
   real_t newton_rtol = 1e-5;
   const real_t lin_rtol = 1e-08;
#endif
   int adaptive_rtol = 0;
   int lin_solver = 2;
   int lin_iter = 100;
   int verbosity = 0;
   bool move_bnd = true;
   bool mod_bdr_attr = true;
   bool normalization = false;
   bool integ_over_target = true;
   bool visualization = false;
   const char *device_config = "cpu";
   bool pa = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Input mesh file.");
   args.AddOption(&output_file, "-out", "--output", "Optimized mesh file.");
   args.AddOption(&order, "-o", "--order", "Order of the mesh nodes.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of uniform refinements before partitioning.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of uniform refinements after partitioning.");
   args.AddOption(&random_refine_passes, "-rr", "--random-refine-passes",
                  "Number of random refinement passes before uniform refinement."
                  " Each pass samples the current mesh.");
   args.AddOption(&random_refine_probability, "-rrp",
                  "--random-refine-probability",
                  "Per-element refinement probability in [0,1]. Fractional "
                  "probabilities require a triangle-only or tetrahedron-only mesh.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "For automatic metric selection use 0 (7 in 2D, 321 in 3D). "
                  "2 and 303 measure shape. "
                  "7, 80, 321 and 334 measure shape and size. "
                  "Metric 334 requires full assembly.");
   args.AddOption(&gamma, "-gamma", "--gamma",
                  "Size weight in [0,1] for metrics 80 and 334. "
                  "The shape weight is 1 - gamma.");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Ideal shape with unit (1), equal (2), or initial (3) size. "
                  "Use 4 for the analytic Jacobian from pmesh-optimizer.");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Quadrature order (Gauss-Lobatto on tensor products).");
   args.AddOption(&lim_const, "-lc", "--limit-const",
                  "Penalty for moving nodes away from their initial positions.");
   args.AddOption(&newton_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&newton_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&adaptive_rtol, "-art", "--adaptive-rel-tol",
                  "Use a fixed linear solver tolerance (0) or "
                  "Eisenstat-Walker type 1 or 2.");
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Use CG (1), MINRES (2), or MINRES with Jacobi (3).");
   args.AddOption(&lin_iter, "-li", "--lin-iter",
                  "Maximum number of linear iterations.");
   args.AddOption(&verbosity, "-vl", "--verbosity-level",
                  "Show warnings (0), Newton iterations (1), "
                  "linear solver summaries (2), or linear iterations (3).");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary", "Allow boundary motion based on attributes. "
                  "Attributes 1/2/3 fix x/y/z, 4 fixes all, and higher tags "
                  "leave all components free.");
   args.AddOption(&mod_bdr_attr, "-mod-bndr-attr", "--modify-boundary-attribute",
                  "-no-mod-bndr-attr", "--no-modify-boundary-attribute",
                  "Temporarily assign boundary attributes 1/2/3 to constant "
                  "x/y/z and 4 elsewhere. Use with -bnd to slide along "
                  "aligned boundaries. Saved meshes retain the input attributes.");
   args.AddOption(&normalization, "-nor", "--normalization", "-no-nor",
                  "--no-normalization", "Normalize the TMOP functional.");
   args.AddOption(&integ_over_target, "-it", "--integ-over-target",
                  "-irf", "--integ-over-ref",
                  "Integrate over target (-it) or reference (-irf) elements.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Visualize TMOP metric values in GLVis.");
   args.AddOption(&device_config, "-d", "--device", "Device configuration.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Use partial assembly on quad or hex meshes.");
   args.Parse();
   if (!args.Good())
   {
      if (root) { args.PrintUsage(cout); }
      return args.Help() ? 0 : 1;
   }

   MFEM_VERIFY(order > 0 && quad_order >= 0 && rs_levels >= 0 && rp_levels >= 0,
               "Invalid mesh order, quadrature order, or refinement level.");
   MFEM_VERIFY(random_refine_passes >= 0 && random_refine_probability >= 0.0 &&
               random_refine_probability <= 1.0,
               "Use a nonnegative number of random refinement passes and a "
               "probability in [0,1].");
   MFEM_VERIFY(gamma >= 0.0 && gamma <= 1.0, "Use -gamma in [0,1].");
   MFEM_VERIFY(target_id >= 1 && target_id <= 4, "Use -tid 1, 2, 3 or 4.");
   MFEM_VERIFY(lin_solver >= 1 && lin_solver <= 3, "Use -ls 1, 2 or 3.");
   MFEM_VERIFY(newton_iter >= 0 && newton_rtol > 0.0 && lin_iter > 0 &&
               adaptive_rtol >= 0 && adaptive_rtol <= 2 && lim_const >= 0.0,
               "Invalid Newton, linear solver, or limiting parameter.");

   Device device(device_config);
   unique_ptr<ParMesh> pmesh;
   {
      Mesh mesh(mesh_file, 1, 1, false);
      const int dim = mesh.Dimension();
      MFEM_VERIFY((dim == 2 || dim == 3) && mesh.SpaceDimension() == dim,
                  "The input must be a full-dimensional 2D or 3D mesh.");
      MFEM_VERIFY(mesh.Conforming() && !mesh.NURBSext &&
                  (!mesh.GetNodalFESpace() || !mesh.GetNodalFESpace()->IsDGSpace()),
                  "The input must be conforming with continuous mesh nodes.");
      MFEM_VERIFY(!pa || (mesh.GetNumGeometries(dim) == 1 &&
                          mesh.HasGeometry(dim == 2 ? Geometry::SQUARE : Geometry::CUBE)),
                  "TMOP partial assembly requires a quad-only or hex-only mesh.");
      if (random_refine_passes > 0 && random_refine_probability > 0.0)
      {
         const bool refine_all = random_refine_probability == 1.0;
         const auto simplex = dim == 2 ? Geometry::TRIANGLE : Geometry::TETRAHEDRON;
         MFEM_VERIFY(refine_all || (mesh.GetNumGeometries(dim) == 1 &&
                                    mesh.HasGeometry(simplex)),
                     "Fractional random refinement requires a triangle-only or "
                     "tetrahedron-only mesh to keep the result conforming.");
         // Each rank refines the same serial mesh before partitioning.
         std::srand(1);
         for (int pass = 0; pass < random_refine_passes; pass++)
         {
            if (refine_all) { mesh.UniformRefinement(); }
            else { mesh.RandomRefinement(random_refine_probability, false, 0); }
         }
      }
      for (int i = 0; i < rs_levels; i++) { mesh.UniformRefinement(); }
      MFEM_VERIFY(mesh.GetNE() >= Mpi::WorldSize(),
                  "Use fewer MPI ranks or refine the mesh in serial with -rs.");
      pmesh.reset(new ParMesh(comm, mesh));
   }
   for (int i = 0; i < rp_levels; i++) { pmesh->UniformRefinement(); }
   const int dim = pmesh->Dimension();

   if (metric_id == 0) { metric_id = dim == 2 ? 7 : 321; }
   unique_ptr<TMOP_QualityMetric> metric;
   if (dim == 2 && metric_id == 2) { metric.reset(new TMOP_Metric_002); }
   if (dim == 2 && metric_id == 7) { metric.reset(new TMOP_Metric_007); }
   if (dim == 2 && metric_id == 80) { metric.reset(new TMOP_Metric_080(gamma)); }
   if (dim == 3 && metric_id == 303) { metric.reset(new TMOP_Metric_303); }
   if (dim == 3 && metric_id == 321) { metric.reset(new TMOP_Metric_321); }
   if (dim == 3 && metric_id == 334) { metric.reset(new TMOP_Metric_334(gamma)); }
   MFEM_VERIFY(metric,
               "Use -mid 2, 7 or 80 in 2D and -mid 303, 321 or 334 in 3D.");
   MFEM_VERIFY(!pa || metric_id != 334,
               "Metric 334 requires full assembly. Use -no-pa.");
   if (root) { args.PrintOptions(cout); device.Print(); }

   int min_elements = pmesh->GetNE();
   MPI_Allreduce(MPI_IN_PLACE, &min_elements, 1, MPI_INT, MPI_MIN, comm);
   MFEM_VERIFY(min_elements > 0, "The mesh partition leaves an empty MPI rank.");

   pmesh->SetCurvature(order, false, dim, Ordering::byNODES);
   auto &x = *static_cast<ParGridFunction *>(pmesh->GetNodes());
   ParFiniteElementSpace &fes = *x.ParFESpace();
   const ParGridFunction x0(x);

   auto target_type = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
   if (target_id == 1) { target_type = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; }
   if (target_id == 2) { target_type = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; }
   TargetConstructor ideal_target(target_type, comm);
   HessianCoefficient analytic_jacobian(dim, metric_id);
   AnalyticAdaptTC analytic_target(TargetConstructor::GIVEN_FULL);
   analytic_target.SetAnalyticTargetSpec(nullptr, nullptr, &analytic_jacobian);
   TargetConstructor &target = target_id == 4 ? analytic_target : ideal_target;
   target.SetNodes(x0);

   real_t min_det = infinity();
   real_t volume = 0.0;
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      const auto &ir = IntRulesLo.Get(pmesh->GetElementBaseGeometry(e), quad_order);
      auto &T = *pmesh->GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         T.SetIntPoint(&ir.IntPoint(q));
         min_det = std::min(min_det, T.Jacobian().Det());
      }
      if (normalization && lim_const > 0.0) { volume += pmesh->GetElementVolume(e); }
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_det, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, comm);
   MFEM_VERIFY(min_det > 0.0,
               "The input mesh has a nonpositive Jacobian determinant.");

   unique_ptr<ParFiniteElementSpace> scalar_fes;
   unique_ptr<ParGridFunction> distance;
   ConstantCoefficient limit(lim_const);
   auto *integ = new TMOP_Integrator(metric.get(), &target);
   integ->IntegrateOverTarget(integ_over_target);
   integ->SetIntegrationRules(IntRulesLo, quad_order);
   if (lim_const > 0.0 && normalization)
   {
      MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, comm);
      scalar_fes.reset(new ParFiniteElementSpace(pmesh.get(), fes.FEColl()));
      distance.reset(new ParGridFunction(scalar_fes.get()));
      *distance = std::pow(volume, 1.0 / dim) / 100.0;
      integ->EnableLimiting(x0, *distance, limit);
   }
   else if (lim_const > 0.0) { integ->EnableLimiting(x0, limit); }

   ParNonlinearForm form(&fes);
   if (pa) { form.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   form.AddDomainIntegrator(integ);
   if (pa) { form.Setup(); }
   real_t initial_energy = form.GetParGridFunctionEnergy(x);
   MFEM_VERIFY(std::isfinite(initial_energy),
               "The initial TMOP energy is nonfinite.");
   // Normalization divides by the initial metric energy, which can be zero.
   if (normalization && initial_energy > 0.0)
   {
      integ->ParEnableNormalization(x0);
      initial_energy = form.GetParGridFunctionEnergy(x);
   }

   MFEM_VERIFY(!move_bnd || mod_bdr_attr || dim == 3 ||
               pmesh->bdr_attributes.Find(3) < 0,
               "Attribute 3 fixes z. In 2D use -mod-bndr-attr or -fix-bnd.");

   const filesystem::path mesh_path(output_file);
   const auto visit_path = mesh_path.parent_path() /
                           ("visit-" + mesh_path.stem().string());
   const int output_precision = numeric_limits<real_t>::max_digits10;
   VisItDataCollection visit_dc(visit_path.generic_string(), pmesh.get());
   visit_dc.SetPrecision(output_precision);
   auto save_visit = [&](int cycle)
   {
      visit_dc.SetCycle(cycle);
      visit_dc.SetTime(cycle);
      visit_dc.Save();
      int error = visit_dc.Error();
      MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_INT, MPI_MAX, comm);
      MFEM_VERIFY(error == DataCollection::No_Error,
                  "Unable to save VisIt collection at cycle " << cycle);
   };

   if (visualization)
   {
      char title[] = "Initial metric values";
      vis_tmop_metric_p(order, *metric, target, *pmesh, title, 0);
   }
   save_visit(0);

   Array<int> original_bdr_attrs;
   if (mod_bdr_attr)
   {
      original_bdr_attrs.SetSize(pmesh->GetNBE());
      for (int i = 0; i < original_bdr_attrs.Size(); i++)
      {
         original_bdr_attrs[i] = pmesh->GetBdrAttribute(i);
      }
      x.HostRead();
      ModifyBoundaryAttributesForNodeMovement(pmesh.get(), x);
      pmesh->SetAttributes(false, true);
   }

   if (pmesh->bdr_attributes.Size())
   {
      const int max_attr = pmesh->bdr_attributes.Max();
      Array<int> marker(max_attr);
      if (!move_bnd)
      {
         marker = 1;
         form.SetEssentialBC(marker);
      }
      else
      {
         Array<int> essential, component_dofs;
         for (int d = 0; d < dim; d++)
         {
            marker = 0;
            if (d < max_attr) { marker[d] = 1; }
            if (max_attr >= 4) { marker[3] = 1; }
            fes.GetEssentialTrueDofs(marker, component_dofs, d);
            essential.Append(component_dofs);
         }
         form.SetEssentialTrueDofs(essential);
      }
   }

   CGSolver cg(comm);
   MINRESSolver minres(comm);
   HypreSmoother hypre_jacobi;
   OperatorJacobiSmoother pa_jacobi;
   IterativeSolver &linear = lin_solver == 1 ? static_cast<IterativeSolver &>(cg)
                             : static_cast<IterativeSolver &>(minres);
   if (lin_solver == 3)
   {
      if (pa)
      {
         pa_jacobi.SetPositiveDiagonal(true);
         linear.SetPreconditioner(pa_jacobi);
      }
      else
      {
         hypre_jacobi.SetType(HypreSmoother::Jacobi, 1);
         hypre_jacobi.SetPositiveDiagonal(true);
         linear.SetPreconditioner(hypre_jacobi);
      }
   }
   linear.SetMaxIter(lin_iter);
   linear.SetRelTol(lin_rtol);
   linear.SetAbsTol(0.0);
   IterativeSolver::PrintLevel linear_print, newton_print;
   if (verbosity >= 2) { linear_print.Errors().Warnings().FirstAndLast(); }
   if (verbosity >= 3) { linear_print.Iterations(); }
   newton_print.Errors().Warnings();
   if (verbosity >= 1) { newton_print.Iterations(); }
   linear.SetPrintLevel(linear_print);

   const auto &ir = IntRulesLo.Get(pmesh->GetTypicalElementGeometry(), quad_order);
   TMOPNewtonSolver solver(comm, ir);
   solver.SetIntegrationRules(IntRulesLo, quad_order);
   solver.SetPreconditioner(linear);
   solver.SetMaxIter(newton_iter);
   solver.SetRelTol(newton_rtol);
   solver.SetAbsTol(0.0);
   solver.SetPrintLevel(newton_print);
   if (adaptive_rtol > 0) { solver.SetAdaptiveLinRtol(adaptive_rtol, 0.5, 0.9); }
   solver.SetOperator(form);

   x.SetTrueVector();
   solver.Mult(Vector(0), x.GetTrueVector());
   x.SetFromTrueVector();
   pmesh->NodesUpdated();
   if (mod_bdr_attr)
   {
      for (int i = 0; i < original_bdr_attrs.Size(); i++)
      {
         pmesh->SetBdrAttribute(i, original_bdr_attrs[i]);
      }
      pmesh->SetAttributes(false, true);
   }
   save_visit(1);

   const real_t final_energy = form.GetParGridFunctionEnergy(x);
   if (root)
   {
      cout << "Initial minimum det(J): " << min_det << '\n'
           << "Initial TMOP energy: " << initial_energy << '\n'
           << "Final TMOP energy: " << final_energy << endl;
   }
   if (visualization)
   {
      char title[] = "Final metric values";
      vis_tmop_metric_p(order, *metric, target, *pmesh, title, 600);
   }

   ofstream output;
   int saved = 1;
   if (root)
   {
      output.open(output_file);
      output.precision(output_precision);
      saved = output.good();
   }
   MPI_Bcast(&saved, 1, MPI_INT, 0, comm);
   MFEM_VERIFY(saved, "Unable to open output mesh " << output_file);
   pmesh->PrintAsSerial(output);
   if (root)
   {
      output.close();
      saved = output.good();
   }
   MPI_Bcast(&saved, 1, MPI_INT, 0, comm);
   MFEM_VERIFY(saved, "Unable to write output mesh " << output_file);
   if (root)
   {
      cout << "Wrote " << output_file << " and VisIt collection "
           << visit_path.string() << endl;
   }
   return 0;
}
