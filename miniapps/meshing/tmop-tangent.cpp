// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
//    ------------------------------------------------------------------
//      Fitting of Selected Mesh Nodes to Specified Physical Positions
//    ------------------------------------------------------------------
//
// This example fits a selected set of the mesh nodes to given physical
// positions while maintaining a valid mesh with good quality.
//
// Sample runs:
// mpirun -np 4 tmop-tangent -rs 2 -m square01.mesh -o 2 -qo 6

#include "../common/mfem-common.hpp"
#include "tmop-tangent.hpp"

using namespace mfem;
using namespace std;

char vishost[] = "localhost";
int  visport   = 19916;
int  wsize     = 350;

int main (int argc, char *argv[])
{
   // Initialize MPI.
   Mpi::Init();
   int myid = Mpi::WorldRank();
   const char *mesh_file = "square01.mesh";
   int rs_levels     = 1;
   int mesh_poly_deg = 2;
   int quad_order    = 5;
   bool glvis        = true;
   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
		 "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
		 "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
       "Polynomial degree of mesh finite element space.");
   args.AddOption(&quad_order, "-qo", "--quad_order",
		 "Order of the quadrature rule.");
   args.AddOption(&glvis, "-vis", "--visualization", "-no-vis",
		 "--no-visualization",
		 "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Read and refine the mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   const int dim = pmesh.Dimension();

   // Setup mesh curvature and GridFunction that stores the coordinates.
   H1_FECollection fec_mesh(mesh_poly_deg, dim);
   ParFiniteElementSpace pfes_mesh(&pmesh, &fec_mesh, dim);
   pmesh.SetNodalFESpace(&pfes_mesh);
   ParGridFunction coord_x(&pfes_mesh), coord_t(&pfes_mesh);
   pmesh.SetNodalGridFunction(&coord_x);
   ParGridFunction x0(coord_x);

   // Move the mesh nodes to have non-trivial problem.
   const int N = coord_x.Size() / 2;
   for (int i = 0; i < N; i++)
   {
      double x = coord_x(i);
      double y = coord_x(i+N);

      // Displace all x and y, so that the spacing is non-uniform.
      x = x + x*(1-x)*0.4;
      y = y + y*(1-y)*0.4;

      // a adds deformation inside.
      // b pulls the top-right corner out.
      // c adds boundary deformation.
      double a = 0.2, b = 0.5, c = 1.3;
      // coord_x(i)     = x + a * sin(M_PI * x) * sin(M_PI * y) + d * x * y;
      // coord_x(i + N) = y + a * sin(M_PI * x) * sin(M_PI * y) + d * x * y;
      coord_x(i)     = x + a * sin(0.5 * M_PI * x) * sin(c * M_PI * y)   + b * x * y;
      coord_x(i + N) = y + a * sin(c * M_PI * x)   * sin(0.5 * M_PI * y) + b * x * y;
   }

   // Compute the minimum det(J) of the starting mesh.
   double min_detJ = infinity();
   const int NE = pmesh.GetNE();
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir =
          IntRules.Get(pfes_mesh.GetFE(0)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh.GetElementTransformation(e);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         min_detJ = min(min_detJ, transf->Jacobian().Det());
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_detJ, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   if (myid == 0)
   { cout << "Minimum det(J) of the original mesh is " << min_detJ << endl; }
   MFEM_VERIFY(min_detJ > 0.0, "Inverted initial meshes are not supported.");

   // Mark which nodes to move tangentially.
   Array<int> fit_marker_0(pfes_mesh.GetNDofs()),
              fit_marker_1(pfes_mesh.GetNDofs()),
              fit_marker_2(pfes_mesh.GetNDofs());
   ParFiniteElementSpace pfes_scalar(&pmesh, &fec_mesh, 1);
   ParGridFunction fit_marker_vis_gf(&pfes_scalar);
   Array<int> vdofs, ess_vdofs;
   fit_marker_0 = -1;
   fit_marker_1 = -1;
   fit_marker_2 = -1;
   fit_marker_vis_gf = 0.0;
   for (int e = 0; e < pmesh.GetNBE(); e++)
   {
      const int attr = pmesh.GetBdrElement(e)->GetAttribute();
      const int nd = pfes_mesh.GetBE(e)->GetDof();
      pfes_mesh.GetBdrElementVDofs(e, vdofs);

      // Top boundary.
      if (attr == 1)
      {
         for (int j = 0; j < nd; j++)
         {
            // Eliminate y component.
            ess_vdofs.Append(vdofs[j+nd]);
            fit_marker_0[vdofs[j]] = 0;
         }
      }
      // Right boundary.
      else if (attr == 2)
      {
         for (int j = 0; j < nd; j++)
         {
            // Eliminate y component.
            ess_vdofs.Append(vdofs[j+nd]);
            fit_marker_1[vdofs[j]] = 1;
         }
      }
      else if (attr == 3)
      {
         // Fix y components.
         for (int j = 0; j < nd; j++)
         {
            fit_marker_2[vdofs[j]] = 2;
            ess_vdofs.Append(vdofs[j+nd]);
         }
      }
      else if (attr == 4)
      {
         // Fix x components.
         for (int j = 0; j < nd; j++)
         {
            fit_marker_2[vdofs[j]] = 2;
            ess_vdofs.Append(vdofs[j]);
         }
      }
   }

   Array<int> dof_to_surface(pfes_mesh.GetNDofs());
   dof_to_surface = -1;
   for (int e = 0; e < pmesh.GetNBE(); e++)
   {
      pfes_mesh.GetBdrElementVDofs(e, vdofs);
      const int nd = pfes_mesh.GetBE(e)->GetDof();

      for (int j = 0; j < nd; j++)
      {
         int cnt = 0;
         if (fit_marker_0[vdofs[j]] >= 0) { cnt++; }
         if (fit_marker_1[vdofs[j]] >= 0) { cnt++; }
         if (fit_marker_2[vdofs[j]] >= 0) { cnt++; }

         fit_marker_vis_gf(vdofs[j]) = cnt;

         if (cnt > 1) { ess_vdofs.Append(vdofs[j]); }
         else if (fit_marker_0[vdofs[j]] >= 0) { dof_to_surface[vdofs[j]] = 0; }
         else if (fit_marker_1[vdofs[j]] >= 0) { dof_to_surface[vdofs[j]] = 1; }
      }
   }

   // Visualize the selected nodes and their target positions.
   if (glvis)
   {
      socketstream vis1, vis2, vis3;
      common::VisualizeField(vis1, "localhost", 19916, fit_marker_vis_gf,
			     "Target positions (DOFS with value 1)",
			     0, 0, 400, 400, (dim == 2) ? "Rjm" : "");
      common::VisualizeMesh(vis2, "localhost", 19916, pmesh, "Initial mesh",
                            400, 0, 400, 400, "me");
   }

   Array<const AnalyticSurface *> surf_array;
   Line_Top line_top(dof_to_surface, 0);
   Curve_Sine_Top curve_top(dof_to_surface, 0);
   Line_Right line_right(dof_to_surface, 1);
   Curve_Sine_Right curve_right(dof_to_surface, 1);
   //surf_array.Append(&line_top);
   surf_array.Append(&curve_top);
   //surf_array.Append(&line_right);
   surf_array.Append(&curve_right);

   AnalyticCompositeSurface surfaces(dof_to_surface, surf_array);
   surfaces.ConvertPhysCoordToParam(coord_x, coord_t);

   if (glvis)
   {
      surfaces.ConvertParamCoordToPhys(coord_t, coord_x);
      socketstream vis1;
      common::VisualizeMesh(vis1, "localhost", 19916, pmesh, "Mesh x->t->x",
                            400, 0, 400, 400, "me");
   }

   // return 0;

   // TMOP setup.
   TMOP_QualityMetric *metric;
   if (dim == 2) { metric = new TMOP_Metric_002; }
   else          { metric = new TMOP_Metric_302; }
   metric->use_old_invariants_code = true;
   TargetConstructor target(TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                            pfes_mesh.GetComm());
   auto integ = new TMOP_Integrator(metric, &target, nullptr);
   integ->EnableTangentialMovement(dof_to_surface, surfaces, pfes_mesh);

   // Linear solver.
   MINRESSolver minres(pfes_mesh.GetComm());
   minres.SetMaxIter(100);
   minres.SetRelTol(1e-8);
   minres.SetAbsTol(0.0);

   // Nonlinear solver.
   ParNonlinearForm a(&pfes_mesh);
   a.SetEssentialVDofs(ess_vdofs);
   a.AddDomainIntegrator(integ);
   const IntegrationRule &ir =
    IntRules.Get(pfes_mesh.GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfes_mesh.GetComm(), ir, 0);
   solver.SetOperator(a);
   solver.SetPreconditioner(minres);
   solver.SetPrintLevel(1);
   solver.SetMaxIter(50);
   solver.SetRelTol(1e-6);
   solver.SetAbsTol(0.0);

   // Solve.
   Vector b(0);
   coord_t.SetTrueVector();
   solver.Mult(b, coord_t.GetTrueVector());
   coord_t.SetFromTrueVector();
   surfaces.ConvertParamCoordToPhys(coord_t, coord_x);
   if(glvis)
   {
      socketstream vis2;
      common::VisualizeMesh(vis2, "localhost", 19916, pmesh, "Final mesh",
                            800, 0, 400, 400, "me");
   }

   delete metric;
   return 0;
}
