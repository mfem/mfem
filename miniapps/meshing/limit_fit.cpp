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
// Sample runs:
//   mpirun -np 4 limit_fit
//   mpirun -np 4 limit_fit -m square01-tri.mesh
//   mpirun -np 4 limit_fit -m ./cube.mesh
//   mpirun -np 4 limit_fit -m ./cube01_tet.mesh -rs 1

#include "mfem.hpp"
#include "../common/mfem-common.hpp"

using namespace mfem;
using namespace std;

char vishost[] = "localhost";
int  visport   = 19916;
int  wsize     = 350;

int problem = 0;

int main (int argc, char *argv[])
{
   // 0. Initialize MPI.
   Mpi::Init();
   int myid = Mpi::WorldRank();

   const char *mesh_file = "square01.mesh";
   int rs_levels     = 2;
   int mesh_poly_deg = 2;
   int quad_order    = 5;
   bool fin_diff     = false;

   // 2. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&fin_diff, "-fd", "--fd_approximation",
                  "-no-fd", "--no-fd-approx",
                  "Enable finite difference based derivative computations.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   const int dim = pmesh.Dimension();

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
      pfes_mesh.GetBdrElementVDofs(e, vdofs);
      for (int j = 0; j < nd; j++)
      {
         int j_x = vdofs[j], j_y = vdofs[nd+j];
         const double x = coord(j_x), y = coord(j_y),
                      z = (dim == 2) ? 0.0 : coord(vdofs[2*nd + j]);
         fit_marker[j_x] = true;
         fit_marker_vis_gf(j_x) = 1.0;
         if (attr == 2)
         {
            if (coord(j_y) < 0.5)
            {
               coord_target(j_y) = 0.1 * std::sin(4 * M_PI * x) *
                                         std::cos(M_PI * z);
            }
            else
            {
               if (coord(j_x) < 0.5)
               {
                  coord_target(j_y) = 1.0 + 0.1 * std::sin(2 * M_PI * x);
               }
               else
               {
                  coord_target(j_y) = 1.0 + 0.1 * std::sin(2 * M_PI * (x + 0.5));
               }

            }
         }
         else { coord_target(j_y) = y; }
      }
   }
   // Show the target positions.
   socketstream vis1;
   coord = coord_target;
   common::VisualizeField(vis1, "localhost", 19916, fit_marker_vis_gf,
                          "Marked dofs", 0, 0, 300, 300);
   coord = x0;

   TMOP_QualityMetric *metric;
   if (dim == 2) { metric = new TMOP_Metric_002; }
   else          { metric = new TMOP_Metric_302; }
   TargetConstructor target(TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                            pfes_mesh.GetComm());
   ConstantCoefficient one(1.0);
   auto integ = new TMOP_Integrator(metric, &target, nullptr);
   integ->EnableSurfaceFitting(coord_target, fit_marker, one);
   if (fin_diff) { integ->EnableFiniteDifferences(coord); }

   ParNonlinearForm a(&pfes_mesh);
   a.AddDomainIntegrator(integ);

   MINRESSolver minres(pfes_mesh.GetComm());
   minres.SetMaxIter(100);
   minres.SetRelTol(1e-12);
   minres.SetAbsTol(0.0);

   const IntegrationRule &ir =
      IntRules.Get(pfes_mesh.GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfes_mesh.GetComm(), ir, 0);
   solver.SetOperator(a);
   solver.SetPreconditioner(minres);
   solver.SetPrintLevel(1);
   solver.SetMaxIter(200);
   solver.SetRelTol(1e-10);
   solver.SetAbsTol(0.0);
   solver.EnableAdaptiveSurfaceFitting();
   solver.SetTerminationWithMaxSurfaceFittingError(1e-5);

   Vector b(0);
   coord.SetTrueVector();
   solver.Mult(b, coord.GetTrueVector());
   coord.SetFromTrueVector();

   char title[] = "Final metric values";
   vis_tmop_metric_p(mesh_poly_deg, *metric, target, pmesh, title, 600);

   Vector zvec(dim); zvec = 0.0;
   VectorConstantCoefficient zero(zvec);
   double norm = coord.ComputeL1Error(zero);
   if (myid == 0)
   {
      std::cout << setprecision(12) << "L1 norm = " << norm << std::endl;
   }

   delete metric;
   return 0;
}
