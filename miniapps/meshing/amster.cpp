// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
//    ---------------------------------------------------------------------
//    Amsterdam 2023 code -- AMSTER (Automatic Mesh SmooThER)
//    ---------------------------------------------------------------------
//
//
// Compile with: make amster
//
// Sample runs:
//
//    2D untangling:
//      mpirun -np 4 amster -m jagged.mesh -o 2 -qo 4 -no-wc -no-fit
//    2D untangling + worst-case:
//      mpirun -np 4 amster -m amster_q4warp.mesh -o 2 -qo 6 -no-fit
//    2D fitting:
//      mpirun -np 6 amster -m amster_q4warp.mesh -rs 1 -o 3 -no-wc -amr 7
//
//    2D orders prec:
//      mpirun -np 6 amster -m ../../data/star.mesh -rs 0 -o 1 -no-wc -amr 7 -vis
//
//    3D untangling:
//      mpirun -np 6 amster -m ../../../mfem_data/cube-holes-inv.mesh -o 3 -qo 4 -no-wc -no-fit

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "mesh-optimizer.hpp"
#include "amster.hpp"

using namespace mfem;
using namespace std;

void TransferLowToHigh(const ParGridFunction &l, ParGridFunction &h);
void TransferHighToLow(const ParGridFunction &h, ParGridFunction &l);

void Untangle(ParGridFunction &x, double min_detA, int quad_order);
void WorstCaseOptimize(ParGridFunction &x, int quad_order);

int main (int argc, char *argv[])
{
#ifndef MFEM_USE_GSLIB
   cout << "AMSTER requires GSLIB!" << endl; return 1;
#endif

   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   const int myid = Mpi::WorldRank();
   Hypre::Init();

   // Set the method's default parameters.
   const char *mesh_file = "jagged.mesh";
   int rs_levels         = 0;
   int mesh_poly_deg     = 2;
   bool worst_case       = true;
   bool fit_optimize     = true;
   int solver_iter       = 50;
   int quad_order        = 8;
   int bg_amr_steps      = 6;
   double surface_fit_const = 10.0;
   double surface_fit_adapt = 10.0;
   double surface_fit_threshold = 1e-5;
   int metric_id         = 2;
   int target_id         = 1;
   bool vis              = false;

   // Parse command-line input file.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--mesh-order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&worst_case, "-wc", "--worst-case",
                  "-no-wc", "--no-worst-case",
                  "Enable worst case optimization step.");
   args.AddOption(&fit_optimize, "-fit", "--fit_optimize",
                  "-no-fit", "--no-fit-optimize",
                  "Enable optimization with tangential relaxation.");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&bg_amr_steps, "-amr", "--amr-bg-steps",
                  "Number of AMR steps on the background mesh.");
   args.AddOption(&surface_fit_const, "-sfc", "--surface-fit-const",
                  "Surface preservation constant.");
   args.AddOption(&surface_fit_adapt, "-sfa", "--adaptive-surface-fit",
                  "Enable or disable adaptive surface fitting.");
   args.AddOption(&surface_fit_threshold, "-sft", "--surf-fit-threshold",
                  "Set threshold for surface fitting. TMOP solver will"
                  "terminate when max surface fitting error is below this limit");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric 1/2/50/58 in 2D:\n\t");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Mesh optimization metric 1/2/3 in 2D:\n\t");
   args.AddOption(&vis, "-vis", "--vis", "-no-vis", "--no-vis",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   const int dim = pmesh->Dimension();
   const int NE = pmesh->GetNE();

   delete mesh;

   // Define a finite element space on the mesh.
   H1_FECollection fec(mesh_poly_deg, dim);
   ParFiniteElementSpace pfes(pmesh, &fec, dim);
   pmesh->SetNodalFESpace(&pfes);

   // Get the mesh nodes as a finite element grid function in fespace.
   ParGridFunction x(&pfes);
   pmesh->SetNodalGridFunction(&x);

   // Save the starting (prior to the optimization) mesh to a file.
   ostringstream mesh_name;
   mesh_name << "amster_in.mesh";
   ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   pmesh->PrintAsOne(mesh_ofs);

   // Store the starting (prior to the optimization) positions.
   ParGridFunction x0(&pfes);
   x0 = x;

   // Compute the minimum det(A) of the starting mesh.
   double min_detA = infinity();
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir =
         IntRulesLo.Get(pfes.GetFE(e)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh->GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         transf->SetIntPoint(&ir.IntPoint(q));
         min_detA = fmin(min_detA, transf->Jacobian().Det());
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_detA, 1, MPI_DOUBLE,
                 MPI_MIN, pfes.GetComm());
   if (myid == 0)
   { cout << "Minimum det(J) of the original mesh is " << min_detA << endl; }

   // Metric.
   TMOP_QualityMetric *metric = NULL;
   if (dim == 2)
   {
      switch (metric_id)
      {
         case 1: metric = new TMOP_Metric_001; break;
         case 2: metric = new TMOP_Metric_002; break;
         case 50: metric = new TMOP_Metric_050; break;
         case 58: metric = new TMOP_Metric_058; break;
         case 80: metric = new TMOP_Metric_080(0.1); break;
      }
   }
   else { metric = new TMOP_Metric_302; }

   TargetConstructor::TargetType target_t;
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
   }
   auto target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   target_c->SetNodes(x0);

   // Visualize the starting mesh.
   if (vis)
   {
      socketstream vis;
      common::VisualizeMesh(vis, "localhost", 19916, *pmesh,
                            "Initial mesh", 0, 0, 400, 400, "me");
   }

   // If needed, untangle with fixed boundary.
   if (min_detA < 0.0) { Untangle(x, min_detA, quad_order); }

   // If needed, perform worst-case optimization with fixed boundary.
   if (worst_case) { WorstCaseOptimize(x, quad_order); }

   // Visualize the starting mesh and metric values.
   if (vis)
   {
      char title[] = "After Untangl / WC";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 0);
   }

   if (fit_optimize == false) { return 0; }

   // Average quality and worst-quality for the mesh.
   double integral_mu = 0.0, volume = 0.0, max_mu = -1.0;
   for (int i = 0; i < NE; i++)
   {
      const FiniteElement &fe_pos = *x.FESpace()->GetFE(i);
      const IntegrationRule &ir = IntRulesLo.Get(fe_pos.GetGeomType(), 10);
      const int nsp = ir.GetNPoints(), dof = fe_pos.GetDof();

      DenseMatrix dshape(dof, dim);
      DenseMatrix pos(dof, dim);
      pos.SetSize(dof, dim);
      Vector posV(pos.Data(), dof * dim);

      Array<int> pos_dofs;
      x.FESpace()->GetElementVDofs(i, pos_dofs);
      x.GetSubVector(pos_dofs, posV);

      DenseTensor W(dim, dim, nsp);
      DenseMatrix Winv(dim), T(dim), A(dim);
      target_c->ComputeElementTargets(i, fe_pos, ir, posV, W);

      for (int j = 0; j < nsp; j++)
      {
         const DenseMatrix &Wj = W(j);
         metric->SetTargetJacobian(Wj);
         CalcInverse(Wj, Winv);

         const IntegrationPoint &ip = ir.IntPoint(j);
         fe_pos.CalcDShape(ip, dshape);
         MultAtB(pos, dshape, A);
         Mult(A, Winv, T);

         const double mu = metric->EvalW(T);
         max_mu = fmax(mu, max_mu);
         integral_mu += mu * ip.weight * A.Det();
         volume += ip.weight * A.Det();
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &max_mu, 1, MPI_DOUBLE, MPI_MAX, pfes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &integral_mu, 1, MPI_DOUBLE, MPI_SUM,
                 pfes.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPI_DOUBLE, MPI_SUM, pfes.GetComm());
   if (myid == 0)
   {
      cout << "Max mu: " << max_mu << endl
           << "Avg mu: " << integral_mu / volume << endl;
   }

   // Compute size field.
   ParFiniteElementSpace pfes_nodes_scalar(pmesh, &fec);
   ParGridFunction size_gf(&pfes_nodes_scalar);
   for (int e = 0; e < NE; e++)
   {
      const FiniteElement &fe = *pfes.GetFE(e);
      const IntegrationRule &ir_nodes = fe.GetNodes();
      const int nqp = ir_nodes.GetNPoints();
      ElementTransformation &Tr = *pmesh->GetElementTransformation(e);
      auto n_fe = dynamic_cast<const NodalFiniteElement *>(&fe);
      const Array<int> &lex_order = n_fe->GetLexicographicOrdering();
      Vector loc_size(nqp);
      Array<int> dofs;
      pfes.GetElementDofs(e, dofs);
      for (int q = 0; q < nqp; q++)
      {
         Tr.SetIntPoint(&ir_nodes.IntPoint(q));
         loc_size(lex_order[q]) = Tr.Weight();
      }
      size_gf.SetSubVector(dofs, loc_size);
   }
   if (vis)
   {
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, size_gf,
                             "Size", 0, 0, 300, 300, "Rj");
   }
   DiffuseH1(size_gf, 2.0);
   if (vis)
   {
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, size_gf,
                             "Size", 300, 0, 300, 300, "Rj");
   }

   // Detect boundary nodes.
   Array<int> vdofs;
   ParGridFunction domain(&pfes_nodes_scalar);
   domain = 1.0;
   for (int i = 0; i < pfes_nodes_scalar.GetNBE(); i++)
   {
      pfes_nodes_scalar.GetBdrElementDofs(i, vdofs);
      for (int j = 0; j < vdofs.Size(); j++) { domain(vdofs[j]) = 0.0; }
   }

   // Distance to the boundary, on the original mesh.
   GridFunctionCoefficient coeff(&domain);
   ParGridFunction dist(&pfes_nodes_scalar);
   ComputeScalarDistanceFromLevelSet(*pmesh, coeff, dist, 10);
   dist *= -1.0;
   if (vis)
   {
      socketstream vis_b_func;
      common::VisualizeField(vis_b_func, "localhost", 19916, dist,
                             "Dist to Boundary", 0, 700, 300, 300, "Rj");

      VisItDataCollection visit_dc("amster_in", pmesh);
      visit_dc.RegisterField("distance", &dist);
      visit_dc.SetFormat(DataCollection::SERIAL_FORMAT);
      visit_dc.Save();
   }

   // Setup distance field on the background mesh.
   BackgroundData backgrnd(*pmesh, dist, bg_amr_steps);
   if (vis)
   {
      socketstream vis_b_func;
      common::VisualizeField(vis_b_func, "localhost", 19916, *backgrnd.dist_bg,
                             "Dist on Background", 300, 700, 300, 300, "Rj");
   }
   backgrnd.ComputeBackgroundDistance();
   if (vis)
   {
      socketstream vis_b_func;
      common::VisualizeField(vis_b_func, "localhost", 19916, *backgrnd.dist_bg,
                             "Final Background LS", 600, 700, 300, 300, "Rjmm");

      VisItDataCollection visit_dc("amster_bg", backgrnd.pmesh_bg);
      visit_dc.RegisterField("distance", backgrnd.dist_bg);
      visit_dc.SetFormat(DataCollection::SERIAL_FORMAT);
      visit_dc.Save();
   }
   backgrnd.ComputeGradientAndHessian();

   // Setup the quadrature rules for the TMOP integrator.
   IntegrationRules *irules = &IntRulesLo;
   if (myid == 0 && dim == 2)
   {
      cout << "Triangle quadrature points: "
           << irules->Get(Geometry::TRIANGLE, quad_order).GetNPoints()
           << "\nQuadrilateral quadrature points: "
           << irules->Get(Geometry::SQUARE, quad_order).GetNPoints() << endl;
   }
   if (myid == 0 && dim == 3)
   {
      cout << "Tetrahedron quadrature points: "
           << irules->Get(Geometry::TETRAHEDRON, quad_order).GetNPoints()
           << "\nHexahedron quadrature points: "
           << irules->Get(Geometry::CUBE, quad_order).GetNPoints()
           << "\nPrism quadrature points: "
           << irules->Get(Geometry::PRISM, quad_order).GetNPoints() << endl;
   }

   MeshOptimizer mesh_opt;
   mesh_opt.Setup(pfes, metric_id, quad_order);

   ConstantCoefficient surf_fit_coeff(surface_fit_const);
   if (surface_fit_const > 0.0)
   {
      mesh_opt.SetupSurfaceFit(pfes_nodes_scalar, surf_fit_coeff, backgrnd);
      mesh_opt.GetSolver()->
      SetAdaptiveSurfaceFittingScalingFactor(surface_fit_adapt);
      mesh_opt.GetSolver()->
      SetTerminationWithMaxSurfaceFittingError(surface_fit_threshold);

      if (vis)
      {
         ParGridFunction surf_fit_mat_gf(&pfes_nodes_scalar);
         surf_fit_mat_gf = 0.0;
         for (int i = 0; i < pmesh->GetNBE(); i++)
         {
            pfes_nodes_scalar.GetBdrElementVDofs(i, vdofs);
            for (int j = 0; j < vdofs.Size(); j++)
            {
               surf_fit_mat_gf(vdofs[j]) = 1.0;
            }
         }
         socketstream vis1;
         common::VisualizeField(vis1, "localhost", 19916, surf_fit_mat_gf,
                                "Boundary DOFs to Fit",
                                900, 600, 300, 300);
      }

      double err_avg, err_max;
      mesh_opt.GetIntegrator()->GetSurfaceFittingErrors(x, err_avg, err_max);
      if (myid == 0)
      {
         cout << "Initial Avg fitting error: " << err_avg << endl
              << "Initial Max fitting error: " << err_max << endl;
      }
   }

   mesh_opt.OptimizeNodes(x, vis);
   // MFEM_ABORT(" ");

   MeshOptimizer mesh_opt_2;
   H1_FECollection fec_2(2, dim);
   ParFiniteElementSpace pfes_2(pmesh, &fec_2, dim);
   ParFiniteElementSpace pfes_2_scalar(pmesh, &fec_2, dim);
   ParGridFunction x_2(&pfes_2);
   TransferLowToHigh(x, x_2);
   mesh_opt_2.Setup(pfes_2, metric_id, quad_order);
   if (surface_fit_const > 0.0)
   {
      surf_fit_coeff.constant = surface_fit_const;
      mesh_opt_2.SetupSurfaceFit(pfes_2_scalar, surf_fit_coeff, backgrnd);
      mesh_opt_2.GetSolver()->
      SetAdaptiveSurfaceFittingScalingFactor(surface_fit_adapt);
      mesh_opt_2.GetSolver()->
      SetTerminationWithMaxSurfaceFittingError(1e-7);
   }
   mesh_opt_2.OptimizeNodes(x_2, vis);

   // Save the optimized mesh to files.
   {
      ostringstream mesh_name;
      mesh_name << "amster-out.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);

      VisItDataCollection visit_dc("amster_opt", pmesh);
      visit_dc.SetFormat(DataCollection::SERIAL_FORMAT);
      visit_dc.Save();
   }

   if (surface_fit_const > 0.0)
   {
      double err_avg, err_max;
      mesh_opt.GetIntegrator()->GetSurfaceFittingErrors(x, err_avg, err_max);
      if (myid == 0)
      {
         std::cout << "Avg fitting error: " << err_avg << std::endl
                   << "Max fitting error: " << err_max << std::endl;
      }
   }

   // Visualize the mesh displacement.
   if (vis)
   {
      socketstream vis;
      x0 -= x;
      common::VisualizeField(vis, "localhost", 19916, x0,
                             "Displacements", 1200, 0, 400, 400, "jRmclA");
   }

   delete target_c;
   delete metric;
   delete pmesh;

   return 0;
}

void Untangle(ParGridFunction &x, double min_detA, int quad_order)
{
   ParFiniteElementSpace &pfes = *x.ParFESpace();
   const int dim = pfes.GetParMesh()->Dimension();

   if (pfes.GetMyRank() == 0) { cout << "*** \nUntangle Phase\n***\n"; }

   // The metrics work in terms of det(T).
   const DenseMatrix &Wideal =
      Geometries.GetGeomToPerfGeomJac(pfes.GetFE(0)->GetGeomType());
   // Slightly below the minimum to avoid division by 0.
   double min_detT = min_detA / Wideal.Det();

   // Metric / target / integrator.
   auto btype = TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType::Shifted;
   auto wctype = TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType::None;
   TMOP_QualityMetric *metric = NULL;
   if (dim == 2) { metric = new TMOP_Metric_004; }
   else          { metric = new TMOP_Metric_360; }
   TMOP_WorstCaseUntangleOptimizer_Metric u_metric(*metric, 1.0, 1.0, 2, 1.5,
                                                   0.001, 0.001,
                                                   btype, wctype);
   TargetConstructor::TargetType target =
      TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
   TargetConstructor target_c(target, pfes.GetComm());
   auto tmop_integ = new TMOP_Integrator(&u_metric, &target_c, nullptr);
   tmop_integ->EnableFiniteDifferences(x);
   tmop_integ->SetIntegrationRules(IntRulesLo, quad_order);


   // Nonlinear form.
   ParNonlinearForm nlf(&pfes);
   nlf.AddDomainIntegrator(tmop_integ);

   Array<int> ess_bdr(pfes.GetParMesh()->bdr_attributes.Max());
   ess_bdr = 1;
   nlf.SetEssentialBC(ess_bdr);

   // Linear solver.
   MINRESSolver minres(pfes.GetComm());
   minres.SetMaxIter(100);
   minres.SetRelTol(1e-12);
   minres.SetAbsTol(0.0);
   IterativeSolver::PrintLevel minres_pl;
   minres.SetPrintLevel(minres_pl.FirstAndLast().Summary());

   // Nonlinear solver.
   const IntegrationRule &ir =
      IntRulesLo.Get(pfes.GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfes.GetComm(), ir);
   solver.SetIntegrationRules(IntRulesLo, quad_order);
   solver.SetOperator(nlf);
   solver.SetPreconditioner(minres);
   solver.SetMinDetPtr(&min_detT);
   solver.SetMaxIter(200);
   solver.SetRelTol(1e-8);
   solver.SetAbsTol(0.0);
   IterativeSolver::PrintLevel newton_pl;
   solver.SetPrintLevel(newton_pl.Iterations().Summary());

   // Optimize.
   x.SetTrueVector();
   Vector b;
   solver.Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();

   delete metric;

   return;
}

void WorstCaseOptimize(ParGridFunction &x, int quad_order)
{
   ParFiniteElementSpace &pfes = *x.ParFESpace();
   const int dim = pfes.GetParMesh()->Dimension();

   if (pfes.GetMyRank() == 0) { cout << "*** \nWorst Quality Phase\n***\n"; }

   // Metric / target / integrator.
   auto btype = TMOP_WorstCaseUntangleOptimizer_Metric::BarrierType::None;
   auto wctype = TMOP_WorstCaseUntangleOptimizer_Metric::WorstCaseType::Beta;
   TMOP_QualityMetric *metric = NULL;
   if (dim == 2) { metric = new TMOP_Metric_002; }
   else          { metric = new TMOP_Metric_304; }
   TMOP_WorstCaseUntangleOptimizer_Metric u_metric(*metric, 1.0, 1.0, 2, 1.5,
                                                   0.001, 0.001,
                                                   btype, wctype);
   TargetConstructor::TargetType target =
      TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
   TargetConstructor target_c(target, pfes.GetComm());
   auto tmop_integ = new TMOP_Integrator(&u_metric, &target_c, nullptr);
   tmop_integ->EnableFiniteDifferences(x);
   tmop_integ->SetIntegrationRules(IntRulesLo, quad_order);

   // Nonlinear form.
   ParNonlinearForm nlf(&pfes);
   nlf.AddDomainIntegrator(tmop_integ);

   Array<int> ess_bdr(pfes.GetParMesh()->bdr_attributes.Max());
   ess_bdr = 1;
   nlf.SetEssentialBC(ess_bdr);

   // Linear solver.
   MINRESSolver minres(pfes.GetComm());
   minres.SetMaxIter(100);
   minres.SetRelTol(1e-12);
   minres.SetAbsTol(0.0);
   IterativeSolver::PrintLevel minres_pl;
   minres.SetPrintLevel(minres_pl.FirstAndLast().Summary());

   // Nonlinear solver.
   const IntegrationRule &ir =
      IntRulesLo.Get(pfes.GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfes.GetComm(), ir);
   solver.SetIntegrationRules(IntRulesLo, quad_order);
   solver.SetOperator(nlf);
   solver.EnableWorstCaseOptimization();
   solver.SetPreconditioner(minres);
   solver.SetMaxIter(1000);
   solver.SetRelTol(1e-8);
   solver.SetAbsTol(0.0);
   IterativeSolver::PrintLevel newton_pl;
   solver.SetPrintLevel(newton_pl.Iterations().Summary());

   // Optimize.
   x.SetTrueVector();
   Vector b;
   solver.Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();

   delete metric;

   return;
}

void Interpolate(const ParGridFunction &src, const Array<int> &y_fixed_marker,
                 ParGridFunction &y)
{
   const int dim = y.ParFESpace()->GetVDim();
   Array<int> dofs;
   for (int e = 0; e < y.ParFESpace()->GetNE(); e++)
   {
      const IntegrationRule &ir = y.ParFESpace()->GetFE(e)->GetNodes();
      const int ndof = ir.GetNPoints();
      y.ParFESpace()->GetElementVDofs(e, dofs);

      for (int i = 0; i < ndof; i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         for (int d = 0; d < dim; d++)
         {
            if (y_fixed_marker[dofs[d*ndof + i]] == 0)
            {
               y(dofs[d*ndof + i]) = src.GetValue(e, ip, d+1);
            }
         }
      }
   }
}

void TransferLowToHigh(const ParGridFunction &l, ParGridFunction &h)
{
   Array<int> h_ess_marker(h.Size());
   h_ess_marker = 0;

   Interpolate(l, h_ess_marker, h);
}

void TransferHighToLow(const ParGridFunction &h, ParGridFunction &l)
{
   Array<int> l_ess_vdof_marker(l.Size());
   l_ess_vdof_marker = 0;
   // wrong.
   // PRefinementTransferOperator transfer(*l.ParFESpace(), *h.ParFESpace());
   // transfer.MultTranspose(h, l);

   // Projects, doesn't interpolate.
   //l.ProjectGridFunction(h);

   Interpolate(h, l_ess_vdof_marker, l);
}
