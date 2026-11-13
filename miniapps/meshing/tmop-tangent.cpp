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
//    -------------------------------------------------------------------
//    Tangential relaxation of boundary nodes to a given analytic surface
//    -------------------------------------------------------------------
//
//
// Sample runs:
// mpirun -np 4 tmop-tangent -dim 2 -rs 2 -o 2 -qo 6
// mpirun -np 4 tmop-tangent -dim 3 -rs 2 -o 2 -qo 6

#include "../common/mfem-common.hpp"
#include "tmop-tangent.hpp"
#include <memory>
#include <utility>
#include <vector>

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
   // Initialize MPI.
   Mpi::Init();
   int myid = Mpi::WorldRank();
   int dim           = 2;
   int rs_levels     = 1;
   int mesh_poly_deg = 2;
   int quad_order    = 5;
   real_t d          = 100;
   bool glvis        = true;
   bool pa           = false;
   const char *device_config = "cpu";
   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension",
                  "Dimension of the Cartesian mesh (2 or 3).");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
       "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
       "Polynomial degree of mesh finite element space.");
   args.AddOption(&quad_order, "-qo", "--quad_order",
		 "Order of the quadrature rule.");
   args.AddOption(&d, "-dist", "--distance",
                  "Physical distance for limiting.");
   args.AddOption(&glvis, "-vis", "--visualization", "-no-vis",
		 "--no-visualization",
		 "Enable or disable GLVis visualization.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable partial assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   Device device(device_config);
   if (myid == 0) { device.Print(); }

   MFEM_VERIFY(dim == 2 || dim == 3, "Mesh dimension must be 2 or 3.");

   // Create and refine the Cartesian mesh. These initial resolutions match
   // square01.mesh and cube.mesh, respectively.
   Mesh mesh = (dim == 2) ?
               Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL) :
               Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   // Setup mesh curvature and GridFunction that stores the coordinates.
   H1_FECollection fec_mesh(mesh_poly_deg, dim);
   ParFiniteElementSpace pfes_mesh(&pmesh, &fec_mesh, dim);
   pmesh.SetNodalFESpace(&pfes_mesh);
   ParGridFunction coord_x(&pfes_mesh), coord_t(&pfes_mesh);
   pmesh.SetNodalGridFunction(&coord_x);

   // Move the mesh nodes to have non-trivial problem.
   const int N = coord_x.Size() / dim;
   double a, b, c;
   for (int i = 0; i < N; i++)
   {
      double x = coord_x(i);
      double y = coord_x(i + N);

      // Displace all x and y, so that the spacing is non-uniform.
      x = x + x * (1 - x) * 0.4;
      y = y + y * (1 - y) * 0.4;

      if (dim == 2)
      {
         // a adds deformation inside.
         // b pulls the top-right corner out.
         // c adds boundary deformation.
         // a = 0.0, b = 0.5, c = 0.0; // linear.
         a = 0.2, b = 0.5, c = 1.5; // curved.
         coord_x(i)     = x + a * sin(0.5 * M_PI * x) * sin(c * M_PI * y)   + b * x * y;
         coord_x(i + N) = y + a * sin(c * M_PI * x)   * sin(0.5 * M_PI * y) + b * x * y;
      }

      else if (dim == 3)
      {
         double z = coord_x(i + 2*N);
         z = z + z * (1 - z) * 0.4;

         // a, b, and c have similar actions to 2D case.
         a = 0.2, b = 0.1, c = 1.1;

         coord_x(i)       = x + a * sin(0.5 * M_PI * x) * sin(c * M_PI * y) * sin(c * M_PI * z) + b * x * y * z;
         coord_x(i + N)   = y + a * sin(c * M_PI * x) * sin(0.5 * M_PI * y) * sin(c * M_PI * z) + b * x * y * z;
         coord_x(i + 2*N) = z + a * sin(c * M_PI * x) * sin(c * M_PI * y) * sin(0.5 * M_PI * z) + b * x * y * z;
      }

   }

   ParGridFunction x0(coord_x);

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
   Array<int> dof_to_surface(pfes_mesh.GetNDofs());
   dof_to_surface = -1;
   Array<int> ess_vdofs, ess_vdofs_marker;
   ess_vdofs_marker.SetSize(pfes_mesh.GetVSize());
   ess_vdofs_marker = 0;
   ParFiniteElementSpace pfes_scalar(&pmesh, &fec_mesh, 1);
   ParGridFunction fit_marker_vis_gf(&pfes_scalar);
   fit_marker_vis_gf = 0.0;

   if (dim == 2)
   {
      enum FitMarkerId
      {
         TOP_MARKER = 0,
         RIGHT_MARKER,
         BOTTOM_MARKER,
         LEFT_MARKER
      };
      Array<int> top_bdr(pmesh.bdr_attributes.Max());
      Array<int> right_bdr(pmesh.bdr_attributes.Max());
      Array<int> bottom_bdr(pmesh.bdr_attributes.Max());
      Array<int> left_bdr(pmesh.bdr_attributes.Max());
      top_bdr = 0;
      right_bdr = 0;
      bottom_bdr = 0;
      left_bdr = 0;

      // Cartesian mesh boundary attributes are:
      // 1 -> bottom, 2 -> right, 3 -> top, 4 -> left.
      bottom_bdr[0] = 1;
      right_bdr[1] = 1;
      top_bdr[2] = 1;
      left_bdr[3] = 1;

      Array<int> top_x_marker, right_x_marker, bottom_x_marker, left_x_marker;
      pfes_mesh.GetEssentialVDofs(top_bdr, top_x_marker, 0);
      pfes_mesh.GetEssentialVDofs(right_bdr, right_x_marker, 0);
      pfes_mesh.GetEssentialVDofs(bottom_bdr, bottom_x_marker, 0);
      pfes_mesh.GetEssentialVDofs(left_bdr, left_x_marker, 0);

      for (int dof = 0; dof < pfes_mesh.GetNDofs(); dof++)
      {
         const int vdof = pfes_mesh.DofToVDof(dof, 0);
         const bool on_top = top_x_marker[vdof];
         const bool on_right = right_x_marker[vdof];
         const bool on_bottom = bottom_x_marker[vdof];
         const bool on_left = left_x_marker[vdof];
         const int boundary_count =
            int(on_top) + int(on_right) + int(on_bottom) + int(on_left);

         if (boundary_count == 2)
         {
            ess_vdofs_marker[pfes_mesh.DofToVDof(dof, 0)] = 1;
            ess_vdofs_marker[pfes_mesh.DofToVDof(dof, 1)] = 1;
         }
         else if (boundary_count == 1)
         {
            if (on_top) { dof_to_surface[dof] = TOP_MARKER; }
            if (on_right) { dof_to_surface[dof] = RIGHT_MARKER; }
            if (on_bottom) { dof_to_surface[dof] = BOTTOM_MARKER; }
            if (on_left) { dof_to_surface[dof] = LEFT_MARKER; }

            // The parameter value is stored in the x-component of coord_t for
            // all 2D analytic curves, so the y-component is constrained.
            ess_vdofs_marker[pfes_mesh.DofToVDof(dof, 1)] = 1;
         }
      }
   }

   else if (dim == 3)
   {
      enum FitMarkerId
      {
         FACE_X0_MARKER = 0,
         FACE_X1_MARKER,
         FACE_Y0_MARKER,
         FACE_Y1_MARKER,
         FACE_Z0_MARKER,
         FACE_Z1_MARKER,
         EDGE_X0Y0_MARKER,
         EDGE_X0Y1_MARKER,
         EDGE_X1Y0_MARKER,
         EDGE_X1Y1_MARKER,
         EDGE_X0Z0_MARKER,
         EDGE_X0Z1_MARKER,
         EDGE_X1Z0_MARKER,
         EDGE_X1Z1_MARKER,
         EDGE_Y0Z0_MARKER,
         EDGE_Y0Z1_MARKER,
         EDGE_Y1Z0_MARKER,
         EDGE_Y1Z1_MARKER
      };

      // x0 <--> face aligned with plane x = 0.
      // x0y0 <--> edge between x0 and y0.
      Array<int> x0_bdr(pmesh.bdr_attributes.Max());
      Array<int> x1_bdr(pmesh.bdr_attributes.Max());
      Array<int> y0_bdr(pmesh.bdr_attributes.Max());
      Array<int> y1_bdr(pmesh.bdr_attributes.Max());
      Array<int> z0_bdr(pmesh.bdr_attributes.Max());
      Array<int> z1_bdr(pmesh.bdr_attributes.Max());
      x0_bdr = 0;
      x1_bdr = 0;
      y0_bdr = 0;
      y1_bdr = 0;
      z0_bdr = 0;
      z1_bdr = 0;

      // Cartesian mesh boundary attributes are:
      // 1 -> z = 0, 2 -> y = 0, 3 -> x = 1,
      // 4 -> y = 1, 5 -> x = 0, 6 -> z = 1.
      z0_bdr[0] = 1;
      y0_bdr[1] = 1;
      x1_bdr[2] = 1;
      y1_bdr[3] = 1;
      x0_bdr[4] = 1;
      z1_bdr[5] = 1;

      Array<int> x0_marker, x1_marker, y0_marker, y1_marker, z0_marker, z1_marker;
      pfes_mesh.GetEssentialVDofs(x0_bdr, x0_marker, 0);
      pfes_mesh.GetEssentialVDofs(x1_bdr, x1_marker, 0);
      pfes_mesh.GetEssentialVDofs(y0_bdr, y0_marker, 0);
      pfes_mesh.GetEssentialVDofs(y1_bdr, y1_marker, 0);
      pfes_mesh.GetEssentialVDofs(z0_bdr, z0_marker, 0);
      pfes_mesh.GetEssentialVDofs(z1_bdr, z1_marker, 0);

      for (int dof = 0; dof < pfes_mesh.GetNDofs(); dof++)
      {
         const int vdof = pfes_mesh.DofToVDof(dof, 0);
         const bool on_x0 = x0_marker[vdof];
         const bool on_x1 = x1_marker[vdof];
         const bool on_y0 = y0_marker[vdof];
         const bool on_y1 = y1_marker[vdof];
         const bool on_z0 = z0_marker[vdof];
         const bool on_z1 = z1_marker[vdof];

         const int boundary_face_count =
            int(on_x0) + int(on_x1) + int(on_y0) +
            int(on_y1) + int(on_z0) + int(on_z1);
         if (boundary_face_count == 3)
         {
            for (int c = 0; c < dim; c++)
            {
               ess_vdofs_marker[pfes_mesh.DofToVDof(dof, c)] = 1;
            }
         }
         else if (boundary_face_count == 2)
         {
            if (on_x0 && on_y0) { dof_to_surface[dof] = EDGE_X0Y0_MARKER; }
            if (on_x0 && on_y1) { dof_to_surface[dof] = EDGE_X0Y1_MARKER; }
            if (on_x1 && on_y0) { dof_to_surface[dof] = EDGE_X1Y0_MARKER; }
            if (on_x1 && on_y1) { dof_to_surface[dof] = EDGE_X1Y1_MARKER; }
            if (on_x0 && on_z0) { dof_to_surface[dof] = EDGE_X0Z0_MARKER; }
            if (on_x0 && on_z1) { dof_to_surface[dof] = EDGE_X0Z1_MARKER; }
            if (on_x1 && on_z0) { dof_to_surface[dof] = EDGE_X1Z0_MARKER; }
            if (on_x1 && on_z1) { dof_to_surface[dof] = EDGE_X1Z1_MARKER; }
            if (on_y0 && on_z0) { dof_to_surface[dof] = EDGE_Y0Z0_MARKER; }
            if (on_y0 && on_z1) { dof_to_surface[dof] = EDGE_Y0Z1_MARKER; }
            if (on_y1 && on_z0) { dof_to_surface[dof] = EDGE_Y1Z0_MARKER; }
            if (on_y1 && on_z1) { dof_to_surface[dof] = EDGE_Y1Z1_MARKER; }
            ess_vdofs_marker[pfes_mesh.DofToVDof(dof, 1)] = 1;
            ess_vdofs_marker[pfes_mesh.DofToVDof(dof, 2)] = 1;
         }
         else if (boundary_face_count == 1)
         {
            if (on_x0) { dof_to_surface[dof] = FACE_X0_MARKER; }
            if (on_x1) { dof_to_surface[dof] = FACE_X1_MARKER; }
            if (on_y0) { dof_to_surface[dof] = FACE_Y0_MARKER; }
            if (on_y1) { dof_to_surface[dof] = FACE_Y1_MARKER; }
            if (on_z0) { dof_to_surface[dof] = FACE_Z0_MARKER; }
            if (on_z1) { dof_to_surface[dof] = FACE_Z1_MARKER; }
            ess_vdofs_marker[pfes_mesh.DofToVDof(dof, 2)] = 1;
         }
      }
   }

   for (int dof = 0; dof < pfes_mesh.GetNDofs(); dof++)
   {
      int nconstrained = 0;
      for (int c = 0; c < dim; c++)
      {
         nconstrained += ess_vdofs_marker[pfes_mesh.DofToVDof(dof, c)];
      }
      fit_marker_vis_gf(dof) = nconstrained;
   }
   FiniteElementSpace::MarkerToList(ess_vdofs_marker, ess_vdofs);

   // Visualize the selected nodes and their target positions.
   if (glvis)
   {
      socketstream vis1, vis2, vis3;
      common::VisualizeField(vis1, "localhost", 19916, fit_marker_vis_gf,
			     "Target positions (DOFS with value 1)",
			     0, 0, 400, 400, (dim == 2) ? "Rjm" : "");
      common::VisualizeMesh(vis2, "localhost", 19916, pmesh, "Initial mesh",
                            400, 0, 400, 400, (dim == 2) ? "me" : "M");
   }

   Array<AnalyticSurface *> surf_array;
   std::vector<std::unique_ptr<AnalyticSurface>> owned_surfaces;

   if (dim == 2)
   {
      owned_surfaces.emplace_back(new Curve_Sine_Top(a, b, c));
      owned_surfaces.emplace_back(new Curve_Sine_Right(a, b, c));
      owned_surfaces.emplace_back(new Line_Bottom);
      owned_surfaces.emplace_back(new Line_Left);
   }
   else if (dim == 3)
   {
      // Each 2D surface and 1D edge of cube has its own AnalyticSurface.
      owned_surfaces.emplace_back(new AxisAlignedPlane(0, 0.0));
      owned_surfaces.emplace_back(new CubeFace_X(a, b, c));
      owned_surfaces.emplace_back(new AxisAlignedPlane(1, 0.0));
      owned_surfaces.emplace_back(new CubeFace_Y(a, b, c));
      owned_surfaces.emplace_back(new AxisAlignedPlane(2, 0.0));
      owned_surfaces.emplace_back(new CubeFace_Z(a, b, c));
      owned_surfaces.emplace_back(new AxisAlignedEdge(2, 0.0, 0.0));
      owned_surfaces.emplace_back(new AxisAlignedEdge(2, 0.0, 1.0));
      owned_surfaces.emplace_back(new AxisAlignedEdge(2, 1.0, 0.0));
      owned_surfaces.emplace_back(new CubeEdge_XY(a, b, c));
      owned_surfaces.emplace_back(new AxisAlignedEdge(1, 0.0, 0.0));
      owned_surfaces.emplace_back(new AxisAlignedEdge(1, 0.0, 1.0));
      owned_surfaces.emplace_back(new AxisAlignedEdge(1, 1.0, 0.0));
      owned_surfaces.emplace_back(new CubeEdge_XZ(a, b, c));
      owned_surfaces.emplace_back(new AxisAlignedEdge(0, 0.0, 0.0));
      owned_surfaces.emplace_back(new AxisAlignedEdge(0, 0.0, 1.0));
      owned_surfaces.emplace_back(new AxisAlignedEdge(0, 1.0, 0.0));
      owned_surfaces.emplace_back(new CubeEdge_YZ(a, b, c));
   }
   for (auto &surface : owned_surfaces)
   {
      surf_array.Append(surface.get());
   }

   AnalyticCompositeSurface surfaces(surf_array, std::move(dof_to_surface));
   surfaces.ConvertPhysCoordToParam(coord_x, coord_t);

   if (glvis)
   {
      surfaces.ConvertParamCoordToPhys(coord_t, coord_x);
      socketstream vis1;
      common::VisualizeMesh(vis1, "localhost", 19916, pmesh, "Mesh x->t->x",
                            400, 0, 400, 400, (dim == 2) ? "me" : "M");
   }

   // TMOP setup.
   TMOP_QualityMetric *metric;
   if (dim == 2) { metric = new TMOP_Metric_002; }
   else          { metric = new TMOP_Metric_302; }
   TargetConstructor target(TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                            pfes_mesh.GetComm());
   target.SetNodes(coord_x);
   auto integ = new TMOP_Integrator(metric, &target, nullptr);
   integ->EnableTangentialMovement(surfaces, pfes_mesh);

   ParFiniteElementSpace pfes_dist(&pmesh, pfes_mesh.FEColl(), 1);
   ParGridFunction dist(&pfes_dist);
   dist = d;
   ConstantCoefficient limit_coeff(1.0);
   integ->EnableLimiting(x0, dist, limit_coeff);

   // Linear solver.
   MINRESSolver minres(pfes_mesh.GetComm());
   minres.SetMaxIter(100);
   minres.SetRelTol(1e-8);
   minres.SetAbsTol(0.0);

   // Nonlinear solver.
   ParNonlinearForm nlf(&pfes_mesh);
   if (pa) { nlf.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   nlf.SetEssentialVDofs(ess_vdofs);
   nlf.AddDomainIntegrator(integ);
   if (pa) { nlf.Setup(); }
   const IntegrationRule &ir =
    IntRules.Get(pfes_mesh.GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfes_mesh.GetComm(), ir, 0);
   solver.SetOperator(nlf);
   solver.SetPreconditioner(minres);
   solver.SetPrintLevel(1);
   solver.SetMaxIter(50);
   solver.SetRelTol(1e-6);
   solver.SetAbsTol(0.0);

   // Solve.
   Vector zero(0);
   coord_t.SetTrueVector();
   solver.Mult(zero, coord_t.GetTrueVector());
   coord_t.SetFromTrueVector();
   surfaces.ConvertParamCoordToPhys(coord_t, coord_x);
   if(glvis)
   {
      socketstream vis2;
      common::VisualizeMesh(vis2, "localhost", 19916, pmesh, "Final mesh",
                            800, 0, 400, 400, (dim == 2) ? "me" : "M");
   }

   delete metric;
   return 0;
}
