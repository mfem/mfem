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
// Compile with: make orders
//
// Sample runs:
//

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "mesh-optimizer.hpp"

using namespace mfem;
using namespace std;

double test_func(const Vector &coord)
{
   return sin(M_PI*coord(0)) * sin(2.0*M_PI*coord(1));
}


void OptimizeMesh(ParGridFunction &x,
                  TMOP_QualityMetric &metric, int quad_order);

int main (int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   const int myid = Mpi::WorldRank();
   Hypre::Init();

   // Set the method's default parameters.
   const char *mesh_file = "blade.mesh";
   int rs_levels         = 0;
   int mesh_poly_deg     = 2;
   int solver_iter       = 50;
   int quad_order        = 8;
   int metric_id         = 2;

   // Parse command-line input file.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--mesh-order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric 1/2/50/58 in 2D:\n\t");
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

   TargetConstructor::TargetType target_t =
         TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
   auto target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   target_c->SetNodes(x0);

   // Visualize the starting mesh and metric values.
   {
      char title[] = "Initial metric values";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 0);
   }

   // If needed, perform worst-case optimization with fixed boundary.
   OptimizeMesh(x, *metric, quad_order);

   ParFiniteElementSpace pfes_s(pmesh, &fec);
   ParGridFunction g(&pfes_s);
   FunctionCoefficient fc(test_func);
   g.ProjectCoefficient(fc);
   {
      socketstream vis_b_func;
      common::VisualizeField(vis_b_func, "localhost", 19916, g,
                             "Dist on Background", 0, 700, 300, 300, "Rj");
   }

#ifndef MFEM_USE_GSLIB
   MFEM_ABORT("GSLIB needed for this functionality.");
#endif

   // Visualize the final mesh and metric values.
   {
      char title[] = "Final metric values";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 600);
   }

   // Visualize the mesh displacement.
   {
      x0 -= x;
      socketstream sock;
      if (myid == 0)
      {
         sock.open("localhost", 19916);
         sock << "solution\n";
      }
      pmesh->PrintAsOne(sock);
      x0.SaveAsOne(sock);
      if (myid == 0)
      {
         sock << "window_title 'Displacements'\n"
              << "window_geometry "
              << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
              << "keys jRmclA" << endl;
      }
   }

   delete target_c;
   delete metric;
   delete pmesh;

   return 0;
}

void OptimizeMesh(ParGridFunction &x,
                  TMOP_QualityMetric &metric, int quad_order)
{
   ParFiniteElementSpace &pfes = *x.ParFESpace();

   if (pfes.GetMyRank() == 0) { cout << "*** \nWorst Quality Phase\n***\n"; }

   // Metric / target / integrator.
   TargetConstructor::TargetType target =
         TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
   TargetConstructor target_c(target, pfes.GetComm());
   auto tmop_integ = new TMOP_Integrator(&metric, &target_c, nullptr);
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

   return;
}
