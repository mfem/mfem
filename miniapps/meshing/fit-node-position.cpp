// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

//   square into circle 1
//   mpirun -np 1 fit-node-position -rs 2 -o 2 -s 2

//   square into circle 2
//   mpirun -np 1 fit-node-position -m icf.mesh -rs 1 -o 2 -s 3

#include "mfem.hpp"
#include "../common/mfem-common.hpp"

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
   int rs_levels     = 2;
   int mesh_poly_deg = 2;
   int quad_order    = 7;
   int fit_setup     = 1;
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
   args.AddOption(&fit_setup, "-s", "--setup",
                  "Setup of the fitting problem (1 or 2).");
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

   //delete mesh;
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
   if (fit_setup == 1)
   {
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
   }
   else if (fit_setup == 2)
   {
      const int angles_cnt = (pmesh.GetNBE() / 2) * mesh_poly_deg;
      const real_t angle = 0.5 * M_PI / angles_cnt;

      const int pts_in_1D = angles_cnt/ 2 + 1;
      const double spacing = 1.0 / (pts_in_1D - 1);

      std::cout << angles_cnt << " " << pts_in_1D << std::endl;

      for (int e = 0; e < pmesh.GetNBE(); e++)
      {
         const int nd = pfes_mesh.GetBE(e)->GetDof();
         pfes_mesh.GetBdrElementVDofs(e, vdofs);
         for (int j = 0; j < nd; j++)
         {
            int j_x = vdofs[j], j_y = vdofs[nd+j];
            const real_t x = coord(j_x), y = coord(j_y),
                         z = (dim == 2) ? 0.0 : coord(vdofs[2*nd + j]);

            if (fabs(x - 1.0) < 1e-12)
            {
               fit_marker[pfes_mesh.VDofToDof(j_x)] = true;
               fit_marker_vis_gf(j_x) = 1.0;

               int id = floor((y + 1e-12) / spacing);
               coord_target(j_x) = cos(angle * id) * sqrt(2.0);
               coord_target(j_y) = sin(angle * id) * sqrt(2.0);
            }

            if (fabs(y - 1.0) < 1e-12)
            {
               fit_marker[pfes_mesh.VDofToDof(j_x)] = true;
               fit_marker_vis_gf(j_x) = 1.0;

               int id = floor((1-x + 1e-12) / spacing);
               coord_target(j_x) = cos(angle * (id+angles_cnt/2)) * sqrt(2.0);
               coord_target(j_y) = sin(angle * (id+angles_cnt/2)) * sqrt(2.0);
            }
         }
      }
   }
   else if (fit_setup == 3)
   {
      MFEM_VERIFY(rs_levels == 1, "The setup assumes exactly 1 refinement.");

      //
      // Internal interface.
      //
      const int zones_1D = 8; // change this for more refinements.

      int angles_cnt = 2 * zones_1D * mesh_poly_deg;
      real_t angle = 0.5 * M_PI / angles_cnt;

      const int pts_in_1D = zones_1D * mesh_poly_deg + 1;
      double spacing = 0.03 / (pts_in_1D - 1);

      real_t rad = sqrt(0.03*0.03 + 0.03*0.03);

      std::cout << angles_cnt << " " << pts_in_1D << std::endl;

      for (int e = 0; e < pmesh.GetNE(); e++)
      {
         const int nd = pfes_mesh.GetFE(e)->GetDof();
         pfes_mesh.GetElementVDofs(e, vdofs);
         for (int j = 0; j < nd; j++)
         {
            int j_x = vdofs[j], j_y = vdofs[nd+j];
            const real_t x = coord(j_x), y = coord(j_y),
                         z = (dim == 2) ? 0.0 : coord(vdofs[2*nd + j]);

            if (fabs(x - 0.03) < 1e-10 && y < 0.03 + 1e-10)
            {
               fit_marker[pfes_mesh.VDofToDof(j_x)] = true;
               fit_marker_vis_gf(j_x) = 1.0;

               int id = floor((y + 1e-12) / spacing);
               coord_target(j_x) = cos(angle * id) * rad;
               coord_target(j_y) = sin(angle * id) * rad;
            }

            if (fabs(y - 0.03) < 1e-10 && x < 0.03 + 1e-10)
            {
               fit_marker[pfes_mesh.VDofToDof(j_x)] = true;
               fit_marker_vis_gf(j_x) = 1.0;

               int id = floor((0.03 - x + 1e-12) / spacing);
               coord_target(j_x) = cos(angle * (id+angles_cnt/2)) * rad;
               coord_target(j_y) = sin(angle * (id+angles_cnt/2)) * rad;
            }
         }
      }

      //
      // Boundary (same angles, different positions).
      //
      rad = 0.3;
      angles_cnt *= 2; // we were counting only angles in 1D above.
      angle = 0.5 * M_PI / angles_cnt;
      for (int e = 0; e < pmesh.GetNBE(); e++)
      {
         const int nd = pfes_mesh.GetBE(e)->GetDof();
         pfes_mesh.GetBdrElementVDofs(e, vdofs);
         for (int j = 0; j < nd; j++)
         {
            int j_x = vdofs[j], j_y = vdofs[nd+j];
            const real_t x = coord(j_x), y = coord(j_y),
                         z = (dim == 2) ? 0.0 : coord(vdofs[2*nd + j]);

            if (sqrt(x*x + y*y) < 0.3 - 1e-3) { continue; }

            fit_marker[pfes_mesh.VDofToDof(j_x)] = true;
            fit_marker_vis_gf(j_x) = 1.0;

            double angle_old = atan2(y, x);
            cout << " --- " << angle_old << endl;
            int id = -1;
            for (int a = 0; a < angles_cnt+1; a++)
            {
               std::cout << a*angle << endl;
               if (fabs(angle_old - a*angle) < 1e-2)
               {
                  cout << a << endl;
                  id = a; break;
               }
            }
            if (id == -1)
            {
               std::cout << "***" << M_PI / 2 << std::endl;
               std::cout << angle_old << " " << x << " " << y << endl;
            }
            MFEM_VERIFY(id >= 0, "couldn't match the angle");

            coord_target(j_x) = cos(angle * id) * rad;
            coord_target(j_y) = sin(angle * id) * rad;
         }
      }
   }
   else if (fit_setup == 4)
   {
      for (int e = 0; e < pmesh.GetNBE(); e++)
      {
         const int nd = pfes_mesh.GetBE(e)->GetDof();
         const int attr = pmesh.GetBdrElement(e)->GetAttribute();

         pfes_mesh.GetBdrElementVDofs(e, vdofs);
         for (int j = 0; j < nd; j++)
         {
            int j_x = vdofs[j], j_y = vdofs[nd+j];
            const real_t x = coord(j_x), y = coord(j_y);

            fit_marker[pfes_mesh.VDofToDof(j_x)] = true;
            fit_marker_vis_gf(j_x) = attr;

            double r_new;
            if (attr == 5) { r_new = 1.0; }
            if (attr == 7) { r_new = 0.2; }

            const double r_now = x*x + y*y;
            coord_target(j_x) = x * sqrt(r_new / r_now);
            coord_target(j_y) = y * sqrt(r_new / r_now);
         }

         if (attr == 4)
         {
            continue;
         }
      }
   }
   else { MFEM_ABORT("Wrong problem setup id."); }

   // Visualize the selected nodes and their target positions.
   if (glvis)
   {
      socketstream vis1;
      coord = coord_target;
      common::VisualizeField(vis1, "localhost", 19916, fit_marker_vis_gf,
                             "Target positions (DOFS with value 1)",
                             0, 0, 400, 400, (dim == 2) ? "Rjm" : "");
      coord = x0;
      socketstream vis2;
      common::VisualizeMesh(vis2, "localhost", 19916, pmesh, "Initial mesh",
                            400, 0, 400, 400, "me");
   }

   // Allow slipping along the remaining boundaries.
   // (attributes 1 and 3 would slip, while 4 is completely fixed).
   Array<int> ess_vdofs;
   for (int i = 0; i < pmesh.GetNBE(); i++)
   {
      const int nd = pfes_mesh.GetBE(i)->GetDof();
      const int attr = pmesh.GetBdrElement(i)->GetAttribute();
      pfes_mesh.GetBdrElementVDofs(i, vdofs);

      if (attr == 1) // Fix x components.
      {
         for (int j = 0; j < nd; j++)
         {
            int j_x = vdofs[j], j_y = vdofs[nd+j];
            const real_t x = coord(j_x), y = coord(j_y),
                         z = (dim == 2) ? 0.0 : coord(vdofs[2*nd + j]);

            if (fit_setup == 1 || fit_setup == 3)
            {
               ess_vdofs.Append(vdofs[j]);
            }
            else if (fit_setup == 2 && x < 0.5)
            {
               ess_vdofs.Append(vdofs[j]);
            }
         }
      }
      else if (attr == 2) // Fix y components.
      {
         for (int j = 0; j < nd; j++)
         {
            int j_x = vdofs[j], j_y = vdofs[nd+j];
            const real_t x = coord(j_x), y = coord(j_y),
                z = (dim == 2) ? 0.0 : coord(vdofs[2*nd + j]);

            if (fit_setup == 2 && y < 0.5)
            {
               ess_vdofs.Append(vdofs[j+nd]);
            }
            else if (fit_setup == 3)
            {
               ess_vdofs.Append(vdofs[j+nd]);
            }
         }
      }
      else if (attr == 3) // Fix z components.
      {
         for (int j = 0; j < nd; j++)
         { ess_vdofs.Append(vdofs[j+2*nd]); }
      }
      else if (attr == 4) // Fix all components.
      {
         if (fit_setup == 3) { continue; }

         for (int j = 0; j < vdofs.Size(); j++)
         { ess_vdofs.Append(vdofs[j]); }
      }
   }

   // TMOP setup.
   TMOP_QualityMetric *metric;
   if (dim == 2) { metric = new TMOP_Metric_058; }
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
      IntRules.Get(pfes_mesh.GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfes_mesh.GetComm(), ir, 0);
   solver.SetOperator(a);
   solver.SetPreconditioner(minres);
   solver.SetPrintLevel(1);
   solver.SetMaxIter(400);
   solver.SetRelTol(1e-12);
   solver.SetAbsTol(0.0);
   solver.SetAdaptiveSurfaceFittingScalingFactor(10);
   //solver.SetSurfaceFittingMaxErrorLimit(1e-3);
   solver.SetTerminationWithMaxSurfaceFittingError(1e-7);

   // Solve.
   Vector b(0);
   coord.SetTrueVector();
   solver.Mult(b, coord.GetTrueVector());
   coord.SetFromTrueVector();

   if (glvis)
   {
      socketstream vis2;
      common::VisualizeMesh(vis2, "localhost", 19916, pmesh, "Final mesh",
                            800, 0, 400, 400, "me");
   }

   {
      mesh->SetNodalGridFunction(&coord);
      ofstream mesh_ofs("optimized.mesh");
      mesh_ofs.precision(14);
      mesh->Print(mesh_ofs);
   }

   delete metric;
   return 0;
}
