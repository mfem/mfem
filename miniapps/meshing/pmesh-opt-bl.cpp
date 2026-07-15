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
//
//       --------------------------------------------------------------
//       Boundary-Laplace Mesh Optimizer Miniapp - Parallel Version
//       --------------------------------------------------------------
//
// This miniapp starts from a mesh, prescribes a quadratic displacement on the
// top boundary, extends that displacement to the interior by solving a vector
// Laplace problem, and improves the resulting mesh with TMOP metric 2 using an
// ideal-shape target.
//
// Compile with: make pmesh-opt-bl
//
// Sample run:
//    mpirun -np 4 ./pmesh-opt-bl -m square01-topattr4.mesh -o 2 -rs 2 -bltrans -blp 1.5 -a -0.31 -ni 40 -bnd -arw 50 -arwd 0.2 -arweps 0.05 -vl 1

#include "mfem.hpp"
#include "../common/mfem-common.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace mfem;
using namespace std;

namespace
{

real_t deformation_amplitude;

void TopBoundaryQuadraticDeformation(const Vector &x, Vector &v)
{
   v.SetSize(2);
   v = 0.0;

   if (std::abs(x(1) - real_t(1.0)) <= 1e-10)
   {
      v(1) = deformation_amplitude * 4.0 * x(0) * (1.0 - x(0));
   }
}

real_t TopBoundaryLevelSetCoefficient(const Vector &x)
{
   return real_t(1.0) - x(1);
}

class TanhDistanceWeightCoefficient : public Coefficient
{
private:
   Coefficient &distance;
   const real_t amplitude, delta, transition_width;

public:
   TanhDistanceWeightCoefficient(Coefficient &distance_,
                                 real_t amplitude_,
                                 real_t delta_,
                                 real_t transition_width_)
      : distance(distance_),
        amplitude(amplitude_),
        delta(delta_),
        transition_width(transition_width_)
   { }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      const real_t d = distance.Eval(T, ip);
      return amplitude * 0.5 *
             (1.0 - std::tanh((d - delta) / transition_width));
   }
};

void ApplyBoundaryLayerTransform(ParMesh &pmesh,
                                 ParGridFunction &nodes,
                                 real_t power,
                                 int order)
{
   MFEM_VERIFY(power > 1.0, "The boundary-layer transform power must be > 1.");

   Vector bb_min, bb_max;
   pmesh.GetBoundingBox(bb_min, bb_max, order);

   const real_t ymin = bb_min(1);
   const real_t ymax = bb_max(1);
   const real_t height = ymax - ymin;
   MFEM_VERIFY(height > 0.0, "Invalid bounding box for boundary-layer transform.");

   FiniteElementSpace *fes = nodes.FESpace();
   nodes.HostReadWrite();
   for (int i = 0; i < fes->GetNDofs(); i++)
   {
      const int vdof = fes->DofToVDof(i, 1);
      real_t eta = (nodes(vdof) - ymin) / height;
      eta = std::max(real_t(0.0), std::min(real_t(1.0), eta));
      nodes(vdof) = ymin + height * (1.0 - std::pow(1.0 - eta, power));
   }

   nodes.SetTrueVector();
   nodes.SetFromTrueVector();
   pmesh.NodesUpdated();
}

real_t MinDetJ(ParMesh &pmesh,
               const ParFiniteElementSpace &pfespace,
               IntegrationRules &irules,
               int quad_order)
{
   real_t min_detJ = infinity();
   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      const IntegrationRule &ir =
         irules.Get(pfespace.GetFE(i)->GetGeomType(), quad_order);
      ElementTransformation *trans = pmesh.GetElementTransformation(i);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         trans->SetIntPoint(&ir.IntPoint(j));
         min_detJ = min(min_detJ, trans->Jacobian().Det());
      }
   }

   real_t global_min_detJ;
   MPI_Allreduce(&min_detJ, &global_min_detJ, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MIN, pmesh.GetComm());
   return global_min_detJ;
}

void SetTMOPBoundaryConditions(ParMesh &pmesh,
                               ParFiniteElementSpace &pfespace,
                               ParNonlinearForm &a,
                               bool move_bnd)
{
   // Fix all boundary nodes, or fix only a given component depending on the
   // boundary attributes of the given mesh.
   // Attributes 1/2/3 correspond to fixed x/y/z components of the node.
   // Attribute 4 corresponds to an entirely fixed node.
   // All other attributes represent unconstrained boundary nodes.
   if (move_bnd == false)
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);
   }
   else
   {
      const int dim = pmesh.Dimension();
      int n = 0;
      for (int i = 0; i < pmesh.GetNBE(); i++)
      {
         const int nd = pfespace.GetBE(i)->GetDof();
         const int attr = pmesh.GetBdrElement(i)->GetAttribute();
         MFEM_VERIFY(!(dim == 2 && attr == 3),
                     "Boundary attribute 3 must be used only for 3D meshes. "
                     "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                     "components, rest for free nodes), or use -fix-bnd.");
         if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
         if (attr == 4) { n += nd * dim; }
      }

      Array<int> vdofs, ess_vdofs(n);
      n = 0;
      for (int i = 0; i < pmesh.GetNBE(); i++)
      {
         const int nd = pfespace.GetBE(i)->GetDof();
         const int attr = pmesh.GetBdrElement(i)->GetAttribute();
         pfespace.GetBdrElementVDofs(i, vdofs);
         if (attr == 1) // Fix x components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
         else if (attr == 2) // Fix y components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j + nd]; }
         }
         else if (attr == 3) // Fix z components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j + 2 * nd]; }
         }
         else if (attr == 4) // Fix all components.
         {
            for (int j = 0; j < vdofs.Size(); j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
      }
      a.SetEssentialVDofs(ess_vdofs);
   }
}

void SolveHarmonicDisplacement(ParMesh &pmesh,
                               ParFiniteElementSpace &pfespace,
                               VectorCoefficient &bdr_deformation,
                               ParGridFunction &displacement,
                               real_t rel_tol,
                               int max_iter,
                               int verbosity)
{
   displacement.ProjectCoefficient(bdr_deformation);

   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_tdof_list;
   pfespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ParLinearForm b(&pfespace);
   b.Assemble();

   ConstantCoefficient one(1.0);
   ParBilinearForm a(&pfespace);
   a.AddDomainIntegrator(new VectorDiffusionIntegrator(one));
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, displacement, b, A, X, B);

   HypreSmoother prec;
   prec.SetType(HypreSmoother::l1Jacobi, 1);
   prec.SetPositiveDiagonal(true);

   CGSolver cg(pfespace.GetComm());
   cg.SetRelTol(rel_tol);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(max_iter);
   IterativeSolver::PrintLevel cg_print;
   if (verbosity == 2)
   {
      cg_print.Errors().Warnings().FirstAndLast();
   }
   if (verbosity > 2)
   {
      cg_print.Errors().Warnings().Iterations();
   }
   cg.SetPrintLevel(cg_print);
   cg.SetPreconditioner(prec);
   cg.SetOperator(*A);
   cg.Mult(B, X);
   MFEM_VERIFY(cg.GetConverged(),
               "The boundary-Laplace solve did not converge.");

   a.RecoverFEMSolution(X, b, displacement);
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   const int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "square01-topattr4.mesh";
   int mesh_poly_deg = 2;
   int rs_levels = 0;
   int rp_levels = 0;
   real_t amplitude = -0.30;
   int quad_order = 8;
   int solver_iter = 40;
#ifdef MFEM_USE_SINGLE
   real_t solver_rtol = 1e-4;
   real_t solver_atol = 1e-6;
   real_t laplace_rtol = 1e-6;
   real_t linsol_rtol = 1e-5;
#else
   real_t solver_rtol = 1e-10;
   real_t solver_atol = 1e-12;
   real_t laplace_rtol = 1e-12;
   real_t linsol_rtol = 1e-12;
#endif
   int laplace_iter = 500;
   int max_lin_iter = 100;
   bool move_bnd = true;
   bool visualization = true;
   int verbosity_level = 0;
   bool bl_transform = false;
   real_t bl_power = 1.2;
   real_t aspect_ratio_weight = 0.0;
   real_t aspect_ratio_weight_delta = -1.0;
   real_t aspect_ratio_weight_transition = -1.0;
   bool experimental = false;
   const char *devopt = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&amplitude, "-a", "--amplitude",
                  "Peak vertical displacement on the top boundary.");
   args.AddOption(&quad_order, "-qo", "--quad-order",
                  "Quadrature order for the TMOP integrator.");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of TMOP Newton iterations.");
   args.AddOption(&solver_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the TMOP Newton solver.");
   args.AddOption(&solver_atol, "-atol", "--newton-abs-tolerance",
                  "Absolute tolerance for the TMOP Newton solver.");
   args.AddOption(&laplace_iter, "-li", "--laplace-iters",
                  "Maximum number of iterations for the Laplace solve.");
   args.AddOption(&laplace_rtol, "-lrtol", "--laplace-rel-tolerance",
                  "Relative tolerance for the Laplace solve.");
   args.AddOption(&max_lin_iter, "-tli", "--tmop-lin-iters",
                  "Maximum number of iterations for TMOP linear solves.");
   args.AddOption(&linsol_rtol, "-tlrtol", "--tmop-lin-rel-tolerance",
                  "Relative tolerance for TMOP linear solves.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Allow boundary nodes to move tangentially, according to "
                  "boundary attributes 1/2/3/4.");
   args.AddOption(&bl_transform, "-bltrans", "--boundary-layer-transform",
                  "-no-bltrans", "--no-boundary-layer-transform",
                  "Apply a boundary-layer-like transform to the initial mesh.");
   args.AddOption(&bl_power, "-blp", "--boundary-layer-power",
                  "Power (> 1) used by the boundary-layer transform.");
   args.AddOption(&aspect_ratio_weight, "-arw", "--aspect-ratio-weight",
                  "Weight for an additional aspect-ratio preservation term. "
                  "Use 0 to disable it.");
   args.AddOption(&aspect_ratio_weight_delta, "-arwd",
                  "--aspect-ratio-weight-distance",
                  "Distance from the top boundary over which the aspect-ratio "
                  "weight remains close to its full value. Use <= 0 for a "
                  "constant weight.");
   args.AddOption(&aspect_ratio_weight_transition, "-arweps",
                  "--aspect-ratio-weight-transition",
                  "Transition width for the tanh aspect-ratio spatial "
                  "weight. Use <= 0 for 0.25 * distance.");
   args.AddOption(&experimental, "-exp", "--experimental",
                  "-no-exp", "--no-experimental",
                  "Use the experimental modular target constructor for the "
                  "aspect-ratio preservation term.");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Verbosity level for the involved iterative solvers:\n\t"
                  "0: no output\n\t"
                  "1: Newton iterations\n\t"
                  "2: Newton iterations + linear solver summaries\n\t"
                  "3: Newton iterations + linear solver iterations");
   args.AddOption(&devopt, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   MFEM_VERIFY(aspect_ratio_weight >= 0.0,
               "The aspect-ratio weight must be nonnegative.");
   const bool use_aspect_ratio_term = (aspect_ratio_weight > 0.0);
   if (aspect_ratio_weight_delta > 0.0 && aspect_ratio_weight_transition <= 0.0)
   {
      aspect_ratio_weight_transition = 0.25 * aspect_ratio_weight_delta;
   }
   MFEM_VERIFY(aspect_ratio_weight_delta <= 0.0 ||
               aspect_ratio_weight_transition > 0.0,
               "The aspect-ratio weight transition width must be positive "
               "when distance weighting is enabled.");
   MFEM_VERIFY(!bl_transform || bl_power > 1.0,
               "The boundary-layer transform power must be > 1.");
   if (myid == 0) { args.PrintOptions(cout); }

   Device device(devopt);
   if (myid == 0) { device.Print(); }

   Mesh mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   MFEM_VERIFY(mesh.Dimension() == 2, "This miniapp expects a 2D mesh.");

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }

   if (mesh_poly_deg <= 0) { mesh_poly_deg = 2; }
   const int mesh_node_order = Ordering::byNODES;
   pmesh.SetCurvature(mesh_poly_deg, false, 2, mesh_node_order);
   ParGridFunction &x = *static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace &pfespace = *x.ParFESpace();

   if (visualization)
   {
      socketstream vis1;
      common::VisualizeMesh(vis1, "localhost", 19916, pmesh,
                            "Initial mesh", 0, 0, 400, 400, "em");
   }

   if (bl_transform)
   {
      ApplyBoundaryLayerTransform(pmesh, x, bl_power, mesh_poly_deg);
      if (myid == 0)
      {
         cout << "Applied boundary-layer transform with power "
              << bl_power << "." << endl;
      }

      if (visualization)
      {
         socketstream visbl;
         common::VisualizeMesh(visbl, "localhost", 19916, pmesh,
                               "Boundary-layer mesh", 410, 0, 400, 400, "em");
      }
   }

   ParGridFunction x_reference(x);

   deformation_amplitude = amplitude;
   VectorFunctionCoefficient bdr_deformation(2, TopBoundaryQuadraticDeformation);

   const bool use_spatial_aspect_ratio_weight =
      (use_aspect_ratio_term && aspect_ratio_weight_delta > 0.0);
   H1_FECollection distance_fec(mesh_poly_deg, pmesh.Dimension());
   ParFiniteElementSpace distance_fes(&pmesh, &distance_fec);
   ParGridFunction boundary_distance(&distance_fes);
   GridFunctionCoefficient boundary_distance_coeff(&boundary_distance);
   TanhDistanceWeightCoefficient aspect_ratio_distance_coeff(
      boundary_distance_coeff, aspect_ratio_weight, aspect_ratio_weight_delta,
      aspect_ratio_weight_transition);
   if (use_spatial_aspect_ratio_weight)
   {
      FunctionCoefficient level_set_coeff(TopBoundaryLevelSetCoefficient);
      boundary_distance.ProjectCoefficient(level_set_coeff);
      boundary_distance.SaveAsOne("boundary-distance.gf", 8);

      if (visualization)
      {
         socketstream visd;
         common::VisualizeField(visd, "localhost", 19916, boundary_distance,
                                "Aspect-ratio distance field",
                                0, 410, 400, 400, "jRcmA");
      }

      if (myid == 0)
      {
         cout << "Using analytic top-boundary distance for aspect-ratio "
              << "weight with delta " << aspect_ratio_weight_delta
              << " and transition width "
              << aspect_ratio_weight_transition << "." << endl;
      }
   }

   ParGridFunction displacement(&pfespace);
   displacement = 0.0;
   SolveHarmonicDisplacement(pmesh, pfespace, bdr_deformation,
                             displacement, laplace_rtol, laplace_iter,
                             verbosity_level);

   x += displacement;
   x.SetTrueVector();
   x.SetFromTrueVector();
   pmesh.NodesUpdated();

   if (visualization)
   {
      socketstream vis2;
      common::VisualizeMesh(vis2, "localhost", 19916, pmesh,
                            "Quadratic boundary deformation",
                            bl_transform ? 820 : 410, 0, 400, 400, "em");
   }

   IntegrationRules irules(0, Quadrature1D::GaussLobatto);
   real_t min_detJ = MinDetJ(pmesh, pfespace, irules, quad_order);
   if (myid == 0)
   {
      cout << "Minimum det(J) after boundary-Laplace deformation: "
           << min_detJ << endl;
   }
   MFEM_VERIFY(min_detJ > 0.0, "The pre-TMOP mesh is invalid.");

   ParGridFunction x0(x);

   TMOP_Metric_002 metric;
   TMOP_Metric_aspratio2D aspect_ratio_metric;
   TargetConstructor target_c(TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                              pmesh.GetComm());
   ConstantCoefficient aspect_ratio_coeff(aspect_ratio_weight);
   target_c.SetNodes(x0);

   TMOP_Integrator *tmop_integ = new TMOP_Integrator(&metric, &target_c);
   tmop_integ->SetIntegrationRules(irules, quad_order);

   std::unique_ptr<TargetConstructor> aspect_ratio_target_c;
   ParNonlinearForm a(&pfespace);
   if (use_aspect_ratio_term)
   {
      if (experimental)
      {
         auto *tc = new ModularTargetConstructor(pmesh.GetComm());
         tc->SetTargetAspectRatio(
            new ModularTargetConstructor::InitialMeshSource(x_reference));
         tc->SetTargetSkew(
            new ModularTargetConstructor::ConstantSource(M_PI/2.0));
         aspect_ratio_target_c.reset(tc);
      }
      else
      {
         auto *tc = new TargetConstructor(TargetConstructor::GIVEN_SHAPE_AND_SIZE,
                                          pmesh.GetComm());
         tc->SetNodes(x_reference);
         aspect_ratio_target_c.reset(tc);
      }

      Coefficient *aspect_ratio_weight_coeff =
         use_spatial_aspect_ratio_weight ?
         static_cast<Coefficient *>(&aspect_ratio_distance_coeff) :
         static_cast<Coefficient *>(&aspect_ratio_coeff);

      auto *aspect_ratio_integ =
         new TMOP_Integrator(&aspect_ratio_metric, aspect_ratio_target_c.get());
      aspect_ratio_integ->SetIntegrationRules(irules, quad_order);
      if (!experimental) { aspect_ratio_integ->IntegrateOverTarget(false); }
      aspect_ratio_integ->SetCoefficient(*aspect_ratio_weight_coeff);

      auto *combo = new TMOPComboIntegrator;
      combo->AddTMOPIntegrator(tmop_integ);
      combo->AddTMOPIntegrator(aspect_ratio_integ);
      a.AddDomainIntegrator(combo);
   }
   else
   {
      a.AddDomainIntegrator(tmop_integ);
   }

   SetTMOPBoundaryConditions(pmesh, pfespace, a, move_bnd);

   const real_t init_energy = a.GetParGridFunctionEnergy(x);
   const real_t init_metric_energy = init_energy;

   HypreSmoother prec;
   prec.SetType(HypreSmoother::l1Jacobi, 1);
   prec.SetPositiveDiagonal(true);

   MINRESSolver minres(pmesh.GetComm());
   minres.SetMaxIter(max_lin_iter);
   minres.SetRelTol(linsol_rtol);
   minres.SetAbsTol(0.0);
   IterativeSolver::PrintLevel minres_print;
   if (verbosity_level == 2)
   {
      minres_print.Errors().Warnings().FirstAndLast();
   }
   if (verbosity_level > 2)
   {
      minres_print.Errors().Warnings().Iterations();
   }
   minres.SetPrintLevel(minres_print);
   minres.SetPreconditioner(prec);

   const IntegrationRule &ir =
      irules.Get(pmesh.GetTypicalElementGeometry(), quad_order);
   TMOPNewtonSolver solver(pmesh.GetComm(), ir);
   solver.SetIntegrationRules(irules, quad_order);
   solver.SetPreconditioner(minres);
   solver.SetMinDetPtr(&min_detJ);
   solver.SetMaxIter(solver_iter);
   solver.SetRelTol(solver_rtol);
   solver.SetAbsTol(solver_atol);
   IterativeSolver::PrintLevel tmop_print;
   if (verbosity_level > 0)
   {
      tmop_print.Errors().Warnings().Iterations();
   }
   else
   {
      tmop_print.Errors().Warnings();
   }
   solver.SetPrintLevel(tmop_print);
   solver.SetOperator(a);

   Vector zero(0);
   zero.UseDevice(true);
   x.SetTrueVector();
   solver.Mult(zero, x.GetTrueVector());
   x.SetFromTrueVector();
   pmesh.NodesUpdated();

   const real_t fin_energy = a.GetParGridFunctionEnergy(x);
   const real_t fin_metric_energy = fin_energy;

   if (myid == 0)
   {
      std::cout << std::scientific << std::setprecision(4);
      cout << "Initial strain energy: " << init_energy
           << " = metrics: " << init_metric_energy
           << " + extra terms: " << init_energy - init_metric_energy << endl;
      cout << "  Final strain energy: " << fin_energy
           << " = metrics: " << fin_metric_energy
           << " + extra terms: " << fin_energy - fin_metric_energy << endl;
      cout << "The strain energy decreased by: "
           << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;
   }

   ParGridFunction dx(x0);
   dx -= x;

   if (visualization)
   {
      socketstream vis3;
      common::VisualizeMesh(vis3, "localhost", 19916, pmesh,
                            "Optimized mesh",
                            bl_transform ? 1230 : 820, 0, 400, 400, "em");

      socketstream sock;
      if (myid == 0)
      {
         sock.open("localhost", 19916);
         sock << "solution\n";
      }
      pmesh.PrintAsOne(sock);
      dx.SaveAsOne(sock);
      if (myid == 0)
      {
         sock << "window_title 'Displacements'\n"
              << "window_geometry "
              << (bl_transform ? 410 : 0) << " " << 410 << " "
              << 400 << " " << 400 << "\n"
              << "keys jRmclA" << endl;
      }
   }

   return 0;
}
