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
//       Boundary-Layer Mesh Optimizer Miniapp - Parallel Version
//       --------------------------------------------------------------
//
// This miniapp optimizes a 2D or 3D boundary-layer mesh. In 2D, it can also
// preserve the stretched near-wall structure. The loaded mesh is stored as the
// reference boundary-layer state for the TMOP solve.
//
// Boundary-layer preservation is handled by adding a 2D aspect-ratio
// preservation term to the base TMOP metric. With -bltarget, this term uses a
// modular target whose aspect ratio is evaluated from the reference
// boundary-layer mesh and whose skew is fixed at 90 degrees. The weight of this
// term is controlled by -arw. The weight can be localized near a selected
// boundary attribute with -arwba, -arwd, and -arweps. The wall-distance field is
// computed with the p-Laplacian DistanceSolver from a finite-element mask that
// is zero on the selected boundary and one elsewhere. This protects
// boundary-layer cells most strongly near the wall while the rest of the mesh
// can relax. The base TMOP metric still uses the ideal-shape target and can be
// selected with -mid. Supported base metric ids are 2 and 80 in 2D, and 303
// and 321 in 3D. The default is metric 2.
//
// Compile with: make pmesh-bl-fluids
//
//    make pmesh-bl-fluids -j4 && mpirun -np 8 ./pmesh-bl-fluids -m bfs-bl-coarse.mesh -o 2 -ni 80 -bnd -bltarget -arw 5 -arwba 4 -arwd 0.02 -arweps 0.02 -vis -vl 2
//    make pmesh-bl-fluids -j4 && mpirun -np 8 ./pmesh-bl-fluids -m blademultiattr.mesh -o 4 -ni 80 -bnd -bltarget -arw 10 -arwba 5 -arwd 0.02 -arweps 0.01 -vis -vl 2

#include "mfem.hpp"
#include "../common/mfem-common.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace mfem;
using namespace std;

namespace
{

class TanhDistanceWeightCoefficient : public Coefficient
{
private:
   GridFunctionCoefficient distance;
   const real_t amplitude, delta, transition_width;

public:
   TanhDistanceWeightCoefficient(const GridFunction &distance_,
                                 real_t amplitude_,
                                 real_t delta_,
                                 real_t transition_width_)
      : distance(&distance_),
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

IterativeSolver::PrintLevel LinearSolverPrintLevel(int verbosity)
{
   IterativeSolver::PrintLevel print;
   if (verbosity == 2) { print.Errors().Warnings().FirstAndLast(); }
   if (verbosity > 2) { print.Errors().Warnings().Iterations(); }
   return print;
}

IterativeSolver::PrintLevel TMOPSolverPrintLevel(int verbosity)
{
   IterativeSolver::PrintLevel print;
   if (verbosity > 0) { print.Errors().Warnings().Iterations(); }
   else               { print.Errors().Warnings(); }
   return print;
}

unique_ptr<TMOP_QualityMetric> MakeBaseMetric(int metric_id, int dim)
{
   switch (metric_id)
   {
      case 2:
         MFEM_VERIFY(dim == 2, "TMOP metric 2 is only defined for 2D meshes.");
         return make_unique<TMOP_Metric_002>();
      case 80:
         MFEM_VERIFY(dim == 2, "TMOP metric 80 is only defined for 2D meshes.");
         return make_unique<TMOP_Metric_080>(0.5);
      case 303:
         MFEM_VERIFY(dim == 3, "TMOP metric 303 is only defined for 3D meshes.");
         return make_unique<TMOP_Metric_303>();
      case 321:
         MFEM_VERIFY(dim == 3, "TMOP metric 321 is only defined for 3D meshes.");
         return make_unique<TMOP_Metric_321>();
      default:
         MFEM_ABORT("Unsupported TMOP metric id. Supported metric ids are "
                    "2, 80, 303, and 321.");
   }
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
   if (!move_bnd)
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);
      return;
   }

   MFEM_VERIFY(pfespace.GetOrdering() == Ordering::byNODES,
               "Sliding boundary constraints assume byNODES ordering.");
   const int dim = pmesh.Dimension();
   MFEM_VERIFY(pfespace.GetVDim() == dim,
               "Mesh node finite element space must have vdim equal to dim.");

   Array<int> vdofs;
   Array<int> ess_vdof_marker(pfespace.GetVSize());
   ess_vdof_marker = 0;

   for (int i = 0; i < pmesh.GetNBE(); i++)
   {
      const int nd = pfespace.GetBE(i)->GetDof();
      const int attr = pmesh.GetBdrElement(i)->GetAttribute();
      pfespace.GetBdrElementVDofs(i, vdofs);

      MFEM_VERIFY(!(dim != 3 && attr == 3),
                  "Boundary attribute 3 must be used only for 3D meshes. "
                  "Use attributes 1, 2, 3 for fixed x, y, z components and "
                  "attributes >=4 for all components, or use -fix-bnd.");

      for (int j = 0; j < nd; j++)
      {
         if (attr >= 1 && attr <= dim)
         {
            ess_vdof_marker[vdofs[j + (attr - 1) * nd]] = 1;
         }
         else if (attr >= 4)
         {
            for (int d = 0; d < dim; d++)
            {
               ess_vdof_marker[vdofs[j + d * nd]] = 1;
            }
         }
      }
   }

   Array<int> ess_vdofs;
   FiniteElementSpace::MarkerToList(ess_vdof_marker, ess_vdofs);
   a.SetEssentialVDofs(ess_vdofs);
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const int myid = Mpi::WorldRank();

   const char *mesh_file = "square01.mesh";
   int mesh_poly_deg = 2;
   int rs_levels = 0;
   int rp_levels = 0;
   int metric_id = 2;
   int quad_order = 8;
   int newton_iter = 40;
#ifdef MFEM_USE_SINGLE
   real_t newton_rtol = 1e-4;
   real_t newton_atol = 1e-6;
   real_t lin_rtol = 1e-5;
#else
   real_t newton_rtol = 1e-10;
   real_t newton_atol = 1e-12;
   real_t lin_rtol = 1e-12;
#endif
   int lin_iter = 100;
   int verbosity = 0;
   bool visualization = true;
   bool move_bnd = false;
   bool bl_target = false;
   real_t aspect_ratio_weight = 0.0;
   int aspect_ratio_weight_bdr_attr = 0;
   real_t aspect_ratio_weight_delta = 0.1;
   real_t aspect_ratio_weight_transition = -1.0;
   const char *devopt = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of the mesh nodes.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of serial uniform refinements.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of parallel uniform refinements.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "TMOP metric id for the base optimization metric: "
                  "2, 80, 303, or 321.");
   args.AddOption(&quad_order, "-qo", "--quad-order",
                  "TMOP quadrature order.");
   args.AddOption(&newton_iter, "-ni", "--newton-iters",
                  "Maximum number of TMOP Newton iterations.");
   args.AddOption(&newton_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the TMOP Newton solver.");
   args.AddOption(&newton_atol, "-atol", "--newton-abs-tolerance",
                  "Absolute tolerance for the TMOP Newton solver.");
   args.AddOption(&lin_iter, "-tli", "--tmop-lin-iters",
                  "Maximum number of iterations for TMOP linear solves.");
   args.AddOption(&lin_rtol, "-tlrtol", "--tmop-lin-rel-tolerance",
                  "Relative tolerance for TMOP linear solves.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Allow boundary nodes to slide during TMOP according to "
                  "attributes 1, 2, 3 for fixed x, y, z components and "
                  "attributes >=4 for all components. Attribute 3 is 3D only.");
   args.AddOption(&bl_target, "-bltarget", "--boundary-layer-target",
                  "-no-bltarget", "--no-boundary-layer-target",
                  "Use a modular target with aspect ratio from the initial "
                  "boundary-layer mesh and skew fixed at 90 degrees. "
                  "Supported only for 2D.");
   args.AddOption(&aspect_ratio_weight, "-arw", "--aspect-ratio-weight",
                  "Weight for an additional aspect-ratio preservation term. "
                  "Use 0 to disable it. Supported only for 2D.");
   args.AddOption(&aspect_ratio_weight_bdr_attr, "-arwba",
                  "--aspect-ratio-weight-boundary-attribute",
                  "Boundary attribute used to localize the aspect-ratio "
                  "weight. Use 0 for a constant weight.");
   args.AddOption(&aspect_ratio_weight_delta, "-arwd",
                  "--aspect-ratio-weight-distance",
                  "Distance from the selected boundary over which the "
                  "aspect-ratio weight remains close to its full value.");
   args.AddOption(&aspect_ratio_weight_transition, "-arweps",
                  "--aspect-ratio-weight-transition",
                  "Transition width for the tanh aspect-ratio spatial "
                  "weight. Use <= 0 for 0.25 * distance.");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity, "-vl", "--verbosity-level",
                  "Verbosity level for the involved iterative solvers:\n\t"
                  "0: no output\n\t"
                  "1: Newton iterations\n\t"
                  "2: Newton iterations + linear solver summaries\n\t"
                  "3: newton iterations + linear solver iterations");
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
   MFEM_VERIFY(aspect_ratio_weight_bdr_attr >= 0,
               "The aspect-ratio weight boundary attribute must be >= 0.");
   const bool use_aspect_ratio_preservation = (aspect_ratio_weight > 0.0);
   const bool use_spatial_aspect_ratio_weight =
      (use_aspect_ratio_preservation && aspect_ratio_weight_bdr_attr > 0);
   if (use_spatial_aspect_ratio_weight)
   {
      MFEM_VERIFY(aspect_ratio_weight_delta > 0.0,
                  "The aspect-ratio weight distance must be positive.");
      if (aspect_ratio_weight_transition <= 0.0)
      {
         aspect_ratio_weight_transition = 0.25 * aspect_ratio_weight_delta;
      }
      MFEM_VERIFY(aspect_ratio_weight_transition > 0.0,
                  "The aspect-ratio weight transition width must be positive.");
   }
   if (mesh_poly_deg <= 0) { mesh_poly_deg = 2; }
   if (myid == 0) { args.PrintOptions(cout); }

   Device device(devopt);
   if (myid == 0) { device.Print(); }

   Mesh mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   MFEM_VERIFY(mesh.Dimension() == mesh.SpaceDimension(),
               "This miniapp expects a full-dimensional mesh.");
   MFEM_VERIFY(mesh.Dimension() == 2 || mesh.Dimension() == 3,
               "This miniapp expects a 2D or 3D mesh.");

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }

   const int mesh_node_order = Ordering::byNODES;
   pmesh.SetCurvature(mesh_poly_deg, false, pmesh.SpaceDimension(),
                      mesh_node_order);
   const int dim = pmesh.Dimension();
   ParGridFunction &x = *static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace &pfespace = *x.ParFESpace();

   MFEM_VERIFY(dim == 2 || !use_aspect_ratio_preservation,
               "The aspect-ratio preservation term is implemented only for "
               "2D meshes.");
   MFEM_VERIFY(dim == 2 || !bl_target,
               "The boundary-layer modular target is implemented only for "
               "2D meshes.");

   if (use_spatial_aspect_ratio_weight)
   {
      MFEM_VERIFY(pmesh.bdr_attributes.Find(aspect_ratio_weight_bdr_attr) >= 0,
                  "The requested aspect-ratio weight boundary attribute is "
                  "not present in the mesh.");
   }

   if (visualization)
   {
      socketstream sock;
      common::VisualizeMesh(sock, "localhost", 19916, pmesh, "Initial mesh",
                            0, 0, 400, 400, "em");
   }

   ParGridFunction x_reference(x);

   unique_ptr<H1_FECollection> distance_fec;
   unique_ptr<ParFiniteElementSpace> distance_fes;
   unique_ptr<ParGridFunction> boundary_distance;
   unique_ptr<TanhDistanceWeightCoefficient> aspect_ratio_spatial_coeff;
   if (use_spatial_aspect_ratio_weight)
   {
      distance_fec.reset(new H1_FECollection(mesh_poly_deg, dim));
      distance_fes.reset(new ParFiniteElementSpace(&pmesh, distance_fec.get()));
      boundary_distance.reset(new ParGridFunction(distance_fes.get()));

      ParGridFunction boundary_mask_zero_set(distance_fes.get());
      ConstantCoefficient one(1.0);
      ConstantCoefficient zero(0.0);
      boundary_mask_zero_set.ProjectCoefficient(one);

      Array<int> attr_marker(pmesh.bdr_attributes.Max());
      attr_marker = 0;
      attr_marker[aspect_ratio_weight_bdr_attr - 1] = 1;
      boundary_mask_zero_set.ProjectBdrCoefficient(zero, attr_marker);
      boundary_mask_zero_set.SetTrueVector();
      boundary_mask_zero_set.SetFromTrueVector();
      boundary_mask_zero_set.SaveAsOne("boundary-zero-set-mask.gf", 8);

      GridFunctionCoefficient wall_zero_set_coeff(&boundary_mask_zero_set);

      const int p_laplacian_power = 10;
      const int p_laplacian_newton_iter = 50;
      common::PLapDistanceSolver distance_solver(p_laplacian_power,
                                                 p_laplacian_newton_iter);
      distance_solver.print_level = LinearSolverPrintLevel(verbosity);
      distance_solver.ComputeScalarDistance(wall_zero_set_coeff,
                                            *boundary_distance);
      boundary_distance->SaveAsOne("boundary-distance.gf", 8);

      if (visualization)
      {
         socketstream vis1;
         common::VisualizeField(vis1, "localhost", 19916, *boundary_distance,
                                "Boundary distance field",
                                0, 410, 400, 400, "jRcmAmp");
      }

      aspect_ratio_spatial_coeff.reset(
         new TanhDistanceWeightCoefficient(*boundary_distance,
                                           aspect_ratio_weight,
                                           aspect_ratio_weight_delta,
                                           aspect_ratio_weight_transition));

      ParGridFunction aspect_ratio_weight_field(distance_fes.get());
      aspect_ratio_weight_field.ProjectCoefficient(*aspect_ratio_spatial_coeff);
      aspect_ratio_weight_field.SaveAsOne("aspect-ratio-weight.gf", 8);

      if (myid == 0)
      {
         cout << "Using p-Laplacian distance localized aspect-ratio weight "
              << "from boundary attribute "
              << aspect_ratio_weight_bdr_attr
              << " with finite-element boundary mask zero set, delta "
              << aspect_ratio_weight_delta
              << " and transition width "
              << aspect_ratio_weight_transition << "." << endl;
      }
   }

   IntegrationRules irules(0, Quadrature1D::GaussLobatto);
   real_t min_detJ = MinDetJ(pmesh, pfespace, irules, quad_order);
   if (myid == 0)
   {
      cout << "Minimum det(J) before TMOP: " << min_detJ << endl;
   }
   MFEM_VERIFY(min_detJ > 0.0,
               "The pre-TMOP mesh is invalid.");

   ParGridFunction x0(x);

   unique_ptr<TMOP_QualityMetric> metric = MakeBaseMetric(metric_id, dim);
   TMOP_Metric_aspratio2D aspect_ratio_metric;
   TargetConstructor ideal_target_c(TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                                    pmesh.GetComm());
   ModularTargetConstructor bl_target_c(pmesh.GetComm());
   TargetConstructor *aspect_ratio_target_c = &ideal_target_c;
   if (bl_target)
   {
      const real_t right_angle = 0.5 * std::acos(real_t(-1.0));
      bl_target_c.SetTargetAspectRatio(
         new ModularTargetConstructor::InitialMeshSource(x_reference));
      bl_target_c.SetTargetSkew(
         new ModularTargetConstructor::ConstantSource(right_angle));
      aspect_ratio_target_c = &bl_target_c;
      if (myid == 0)
      {
         cout << "Using modular aspect-ratio target: initial aspect ratio, "
              << "90-degree skew. The base TMOP metric uses the ideal-shape "
              << "target." << endl;
      }
   }
   if (myid == 0)
   {
      cout << "Using base TMOP metric " << metric_id << "." << endl;
   }
   ideal_target_c.SetNodes(x_reference);
   if (bl_target)
   {
      bl_target_c.SetNodes(x_reference);
   }

   auto *tmop_integ = new TMOP_Integrator(metric.get(), &ideal_target_c);
   tmop_integ->SetIntegrationRules(irules, quad_order);
   ConstantCoefficient aspect_ratio_coeff(aspect_ratio_weight);
   Coefficient *aspect_ratio_weight_coeff =
      use_spatial_aspect_ratio_weight ?
      static_cast<Coefficient *>(aspect_ratio_spatial_coeff.get()) :
      static_cast<Coefficient *>(&aspect_ratio_coeff);

   ParNonlinearForm a(&pfespace);
   if (use_aspect_ratio_preservation)
   {
      auto *aspect_ratio_integ =
         new TMOP_Integrator(&aspect_ratio_metric, aspect_ratio_target_c);
      aspect_ratio_integ->SetIntegrationRules(irules, quad_order);
      aspect_ratio_integ->SetCoefficient(*aspect_ratio_weight_coeff);

      auto *combo = new TMOPComboIntegrator;
      combo->AddTMOPIntegrator(tmop_integ);
      combo->AddTMOPIntegrator(aspect_ratio_integ);
      a.AddDomainIntegrator(combo);

      if (myid == 0)
      {
         cout << "Added aspect-ratio preservation term with "
              << (use_spatial_aspect_ratio_weight ? "localized " : "")
              << "weight " << aspect_ratio_weight << "." << endl;
      }
   }
   else
   {
      a.AddDomainIntegrator(tmop_integ);
   }
   SetTMOPBoundaryConditions(pmesh, pfespace, a, move_bnd);

   const real_t init_energy = a.GetParGridFunctionEnergy(x);

   HypreSmoother prec;
   prec.SetType(HypreSmoother::l1Jacobi, 1);
   prec.SetPositiveDiagonal(true);

   MINRESSolver minres(pmesh.GetComm());
   minres.SetMaxIter(lin_iter);
   minres.SetRelTol(lin_rtol);
   minres.SetAbsTol(0.0);
   minres.SetPrintLevel(LinearSolverPrintLevel(verbosity));
   minres.SetPreconditioner(prec);

   const IntegrationRule &ir =
      irules.Get(pmesh.GetTypicalElementGeometry(), quad_order);
   TMOPNewtonSolver solver(pmesh.GetComm(), ir);
   solver.SetIntegrationRules(irules, quad_order);
   solver.SetPreconditioner(minres);
   solver.SetMinDetPtr(&min_detJ);
   solver.SetMaxIter(newton_iter);
   solver.SetRelTol(newton_rtol);
   solver.SetAbsTol(newton_atol);
   solver.SetPrintLevel(TMOPSolverPrintLevel(verbosity));
   solver.SetOperator(a);

   Vector zero(0);
   zero.UseDevice(true);
   x.SetTrueVector();
   solver.Mult(zero, x.GetTrueVector());
   x.SetFromTrueVector();
   pmesh.NodesUpdated();

   ofstream mesh_ofs("optimized.mesh");
   mesh_ofs.precision(8);
   pmesh.PrintAsOne(mesh_ofs);

   const real_t final_energy = a.GetParGridFunctionEnergy(x);
   const real_t final_min_detJ = MinDetJ(pmesh, pfespace, irules, quad_order);

   if (myid == 0)
   {
      cout << scientific << setprecision(4);
      cout << "Initial TMOP energy: " << init_energy << endl;
      cout << "  Final TMOP energy: " << final_energy << endl;
      cout << "Initial TMOP norm: " << solver.GetInitialNorm() << endl;
      cout << "  Final TMOP norm: " << solver.GetFinalNorm() << endl;
      cout << "  Final relative TMOP norm: "
           << solver.GetFinalRelNorm() << endl;
      cout << "TMOP solver converged: "
           << (solver.GetConverged() ? "yes" : "no") << endl;
      cout << "Minimum det(J) after TMOP optimization: "
           << final_min_detJ << endl;
      if (std::abs(init_energy) > 0.0)
      {
         cout << "The TMOP energy decreased by: "
              << (init_energy - final_energy) * 100.0 / init_energy
              << " %." << endl;
      }
   }

   ParGridFunction optimized_displacement(x0);
   optimized_displacement -= x;
   optimized_displacement.SaveAsOne("optimized-displacement.gf", 8);

   if (visualization)
   {
      socketstream sock;
      common::VisualizeMesh(sock, "localhost", 19916, pmesh, "Optimized mesh",
                            410, 0, 400, 400, "em");

      socketstream disp_sock;
      if (myid == 0)
      {
         disp_sock.open("localhost", 19916);
         disp_sock << "solution\n";
      }
      pmesh.PrintAsOne(disp_sock);
      optimized_displacement.SaveAsOne(disp_sock);
      if (myid == 0)
      {
         disp_sock << "window_title 'Optimization displacement'\n"
                   << "window_geometry "
                   << 410 << " " << 410 << " " << 400 << " " << 400 << "\n"
                   << "keys jRmclA" << endl;
      }
   }

   return 0;
}
