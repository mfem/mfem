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
//    ------------------------------------------------------------------
//      Fitting of Selected Mesh Nodes to Specified Physical Positions
//    ------------------------------------------------------------------
//
// This example fits a selected set of the mesh nodes to given physical
// positions while maintaining a valid mesh with good quality.
//
// Sample runs:
//   mpirun -np 4 fit-node-position
//   mpirun -np 4 fit-node-position -m square01-tri.mesh
//   mpirun -np 4 fit-node-position -m ./cube.mesh
//   mpirun -np 4 fit-node-position -m ./cube-tet.mesh -rs 0

#include "mfem.hpp"
#include "../common/mfem-common.hpp"

using namespace mfem;
using namespace std;

char vishost[] = "localhost";
int  wsize     = 350;

int main (int argc, char *argv[])
{
   // Initialize MPI.
   Mpi::Init();
   int myid = Mpi::WorldRank();

   const char *mesh_file = "square01.mesh";
   int rs_levels     = 2;
   int mesh_poly_deg = 2;
   int quad_order    = 5;
   bool glvis        = true;
   int visport       = 19916;

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
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
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
   FiniteElementCollection *fec_mesh;
   if (mesh_poly_deg <= 0)
   {
      fec_mesh = new QuadraticPosFECollection;
      mesh_poly_deg = 2;
   }
   else { fec_mesh = new H1_FECollection(mesh_poly_deg, dim); }
   ParFiniteElementSpace pfes_mesh(&pmesh, fec_mesh, dim);
   pmesh.SetNodalFESpace(&pfes_mesh);
   ParGridFunction coord(&pfes_mesh);
   pmesh.SetNodalGridFunction(&coord);
   ParGridFunction x0(coord);

   // Pick which nodes to fit and select the target positions.
   // (attribute 2 would have a prescribed deformation in y-direction, same x).
   Array<bool> fit_marker(pfes_mesh.GetNDofs());
   ParGridFunction fit_marker_vis_gf(&pfes_mesh);
   ParGridFunction coord_target(&pfes_mesh);
   Array<int> vdofs;
   fit_marker = false;
   coord_target = coord;
   fit_marker_vis_gf = 0.0;
   for (int e = 0; e < pmesh.GetNBE(); e++)
   {
      const int nd = pfes_mesh.GetBE(e)->GetDof();
      const int attr = pmesh.GetBdrElement(e)->GetAttribute();
      if (attr != 2) { continue; }

      pfes_mesh.GetBdrElementVDofs(e, vdofs);
      for (int j = 0; j < nd; j++)
      {
         int j_x = vdofs[j], j_y = vdofs[nd+j];
         const real_t x = coord(j_x),
                      z = (dim == 2) ? 0.0 : coord(vdofs[2*nd + j]);
         fit_marker[pfes_mesh.VDofToDof(j_x)] = true;
         fit_marker_vis_gf(j_x) = 1.0;
         if (coord(j_y) < 0.5)
         {
            coord_target(j_y) = 0.1 * sin(4 * M_PI * x) * cos(M_PI * z);
         }
         else
         {
            if (coord(j_x) < 0.5)
            {
               coord_target(j_y) = 1.0 + 0.1 * sin(2 * M_PI * x);
            }
            else
            {
               coord_target(j_y) = 1.0 + 0.1 * sin(2 * M_PI * (x + 0.5));
            }

         }
      }
   }

   // Visualize the selected nodes and their target positions.
   if (glvis)
   {
      socketstream vis1;
      coord = coord_target;
      common::VisualizeField(vis1, "localhost", 19916, fit_marker_vis_gf,
                             "Target positions (DOFS with value 1)",
                             0, 0, 400, 400, (dim == 2) ? "Rjm" : "");
      coord = x0;
   }

   // Allow slipping along the remaining boundaries.
   // (attributes 1 and 3 would slip, while 4 is completely fixed).
   int n = 0;
   for (int i = 0; i < pmesh.GetNBE(); i++)
   {
      const int nd = pfes_mesh.GetBE(i)->GetDof();
      const int attr = pmesh.GetBdrElement(i)->GetAttribute();
      MFEM_VERIFY(!(dim == 2 && attr == 3),
                  "Boundary attribute 3 must be used only for 3D meshes. "
                  "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                  "components, rest for free nodes), or use -fix-bnd.");
      if (attr == 1 || attr == 3) { n += nd; }
      if (attr == 4) { n += nd * dim; }
   }
   Array<int> ess_vdofs(n);
   n = 0;
   for (int i = 0; i < pmesh.GetNBE(); i++)
   {
      const int nd = pfes_mesh.GetBE(i)->GetDof();
      const int attr = pmesh.GetBdrElement(i)->GetAttribute();
      pfes_mesh.GetBdrElementVDofs(i, vdofs);
      if (attr == 1) // Fix x components.
      {
         for (int j = 0; j < nd; j++)
         { ess_vdofs[n++] = vdofs[j]; }
      }
      else if (attr == 3) // Fix z components.
      {
         for (int j = 0; j < nd; j++)
         { ess_vdofs[n++] = vdofs[j+2*nd]; }
      }
      else if (attr == 4) // Fix all components.
      {
         for (int j = 0; j < vdofs.Size(); j++)
         { ess_vdofs[n++] = vdofs[j]; }
      }
   }

   // TMOP setup.
   TMOP_QualityMetric *metric;
   if (dim == 2) { metric = new TMOP_Metric_002; }
   else          { metric = new TMOP_Metric_302; }
   TargetConstructor target(TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                            pfes_mesh.GetComm());
   ConstantCoefficient fit_weight(100.0);
   auto integ = new TMOP_Integrator(metric, &target, nullptr);
   integ->EnableSurfaceFitting(coord_target, fit_marker, fit_weight);

   // Linear solver.
   MINRESSolver minres(pfes_mesh.GetComm());
   minres.SetMaxIter(100);
   minres.SetRelTol(1e-12);
   minres.SetAbsTol(0.0);

   // Nonlinear solver.
   ParNonlinearForm a(&pfes_mesh);
   a.SetEssentialVDofs(ess_vdofs);
   a.AddDomainIntegrator(integ);
   const IntegrationRule &ir =
      IntRules.Get(pmesh.GetTypicalElementGeometry(), quad_order);
   TMOPNewtonSolver solver(pfes_mesh.GetComm(), ir, 0);
   solver.SetOperator(a);
   solver.SetPreconditioner(minres);
   solver.SetPrintLevel(1);
   solver.SetMaxIter(200);
   solver.SetRelTol(1e-10);
   solver.SetAbsTol(0.0);
   solver.SetAdaptiveSurfaceFittingScalingFactor(10);
   solver.SetTerminationWithMaxSurfaceFittingError(1e-3);

   // Solve.
   Vector b(0);
   coord.SetTrueVector();
   solver.Mult(b, coord.GetTrueVector());
   coord.SetFromTrueVector();

   if (glvis)
   {
      socketstream vis2;
      common::VisualizeMesh(vis2, "localhost", 19916, pmesh, "Final mesh",
                            400, 0, 400, 400);
   }

   delete metric;
   delete fec_mesh;
   return 0;
}
