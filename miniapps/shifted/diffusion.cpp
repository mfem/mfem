// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
//  ---------------------------------------------------------------------------
//  Shifted Boundary Diffusion Miniapp: Finite element immersed boundary solver
//  ---------------------------------------------------------------------------
//
// This miniapp solves the Poisson problem with prescribed boundary conditions
// on a surrogate domain using a high-order extension of the shifted boundary
// method [1]. Using a level-set prescribed to represent the true boundary, a
// distance function is computed for the immersed background mesh. A surrogate
// domain is also computed to solve the Poisson problem, based on intersection
// of the zero level-set with the background mesh. The distance function is used
// with a Taylor expansion to enforce Dirichlet boundary conditions on the
// (non-aligned) mesh faces of the surrogate domain, therefore "shifting" the
// location where boundary conditions are imposed.
//
// [1] Atallah, Nabil M., Claudio Canuto and Guglielmo Scovazzi. "The
// second-generation Shifted Boundary Method and its numerical analysis."
// Computer Methods in Applied Mechanics and Engineering 372 (2020): 113341.
//
// Compile with: make diffusion
//
// Sample runs:
//
//   Problem 1: Circular hole of radius 0.2 at the center of the domain.
//              Solves -nabla^2 u = 1 with homogeneous boundary conditions.
//     mpirun -np 4 diffusion -m ../../data/inline-quad.mesh -rs 3 -o 1 -vis -lst 1
//     mpirun -np 4 diffusion -m ../../data/inline-hex.mesh -rs 2 -o 2 -vis -lst 1 -ho 1 -alpha 10
//
//   Problem 2: Circular hole of radius 0.2 at the center of the domain.
//              Solves -nabla^2 u = f with inhomogeneous boundary conditions, and
//              f is setup such that u = x^p + y^p, where p = 2 by default.
//              This is a 2D convergence test.
//     mpirun -np 4 diffusion -rs 2 -o 2 -vis -lst 2
//
//   Problem 3: Domain is y = [0, 1] but mesh is shifted to [-1.e-4, 1].
//              Solves -nabla^2 u = f with inhomogeneous boundary conditions,
//              and f is setup such that u = sin(pi*x*y).  This is a 2D
//              convergence test. Second-order can be demonstrated by changing
//              refinement level (-rs) for the sample run below.  Higher-order
//              convergence can be realized by also increasing the order (-o) of
//              the finite element space and the number of high-order terms
//              (-ho) to be included from the Taylor expansion used to enforce
//              the boundary conditions.
//     mpirun -np 4 diffusion -rs 2 -o 1 -vis -lst 3
//
//   Problem 4: Complex 2D shape:
//              Solves -nabla^2 u = 1 with homogeneous boundary conditions.
//     mpirun -np 4 diffusion -rs 5 -lst 4 -alpha 2

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include "sbm_aux.hpp"
#include "sbm_solver.hpp"
#include "marking.hpp"
#include "dist_solver.hpp"

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 2;
   bool visualization = true;
   int ser_ref_levels = 0;
   int level_set_type = 1;
   int ho_terms = 0;
   double alpha = 1;
   bool include_cut_cell = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&level_set_type, "-lst", "--level-set-type",
                  "level-set-type:");
   args.AddOption(&ho_terms, "-ho", "--high-order",
                  "Additional high-order terms to include");
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "Nitsche penalty parameter (~1 for 2D, ~10 for 3D).");
   args.AddOption(&include_cut_cell, "-cut", "--cut", "-no-cut-cell",
                  "--no-cut-cell",
                  "Include or not include elements cut by true boundary.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Enable hardware devices such as GPUs, and programming models such as CUDA,
   // OCCA, RAJA and OpenMP based on command line options.
   Device device("cpu");
   device.Print();

   // Refine the mesh.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   for (int lev = 0; lev < ser_ref_levels; lev++) { mesh.UniformRefinement(); }

   // MPI distribution.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Define a finite element space on the mesh. Here we use continuous Lagrange
   // finite elements of the specified order. If order < 1, we fix it to 1.
   if (order < 1) { order = 1; }
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace pfespace(&pmesh, &fec);

   Vector vxyz;

   // Set the nodal grid function for the mesh, and modify the nodal positions
   // for level_set_type = 3 such that some of the mesh elements are intersected
   // by the true boundary (y = 0).
   ParFiniteElementSpace pfespace_mesh(&pmesh, &fec, dim);
   pmesh.SetNodalFESpace(&pfespace_mesh);
   ParGridFunction x_mesh(&pfespace_mesh);
   pmesh.SetNodalGridFunction(&x_mesh);
   vxyz = *pmesh.GetNodes();
   int nodes_cnt = vxyz.Size()/dim;
   if (level_set_type == 3)
   {
      for (int i = 0; i < nodes_cnt; i++)
      {
         // Shift the mesh from y = [0, 1] to [-1.e-4, 1].
         vxyz(i+nodes_cnt) = (1.+1.e-4)*vxyz(i+nodes_cnt)-1.e-4;
      }
   }
   pmesh.SetNodes(vxyz);
   pfespace.ExchangeFaceNbrData();
   const int gtsize = pfespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << gtsize << endl;
   }

   // Define the solution vector x as a finite element grid function
   // corresponding to pfespace.
   ParGridFunction x(&pfespace);
   // ParGridFunction for level_set_value.
   ParGridFunction level_set_val(&pfespace);

   // Determine if each element in the ParMesh is inside the actual domain,
   // partially cut by its boundary, or completely outside the domain.
   Dist_Level_Set_Coefficient dist_fun_level_coef(level_set_type);
   level_set_val.ProjectCoefficient(dist_fun_level_coef);
   // Exchange information for ghost elements i.e. elements that share a face
   // with element on the current processor, but belong to another processor.
   level_set_val.ExchangeFaceNbrData();
   // Setup the class to mark all elements based on whether they are located
   // inside or outside the true domain, or intersected by the true boundary.
   ShiftedFaceMarker marker(pmesh, level_set_val, pfespace, include_cut_cell);
   Array<int> elem_marker;
   marker.MarkElements(elem_marker);

   // Visualize the element markers.
   if (visualization)
   {
      L2_FECollection fecl2 = L2_FECollection(0, dim);
      ParFiniteElementSpace pfesl2(&pmesh, &fecl2);
      ParGridFunction elem_marker_gf(&pfesl2);
      for (int i = 0; i < elem_marker_gf.Size(); i++)
      {
         elem_marker_gf(i) = (double)elem_marker[i];
      }
      char vishost[] = "localhost";
      int  visport   = 19916, s = 350;
      socketstream sol_sock;
      common::VisualizeField(sol_sock, vishost, visport, elem_marker_gf,
                             "Element Flags", 0, 0, s, s, "Rjmpc");
   }

   // Get a list of dofs associated with shifted boundary (SB) faces.
   Array<int> sb_dofs; // Array of dofs on SB faces
   marker.ListShiftedFaceDofs(elem_marker, sb_dofs);

   // Visualize the shifted boundary face dofs.
   if (visualization)
   {
      ParGridFunction face_dofs(&pfespace);
      face_dofs = 0.0;
      for (int i = 0; i < sb_dofs.Size(); i++)
      {
         face_dofs(sb_dofs[i]) = 1.0;
      }
      char vishost[] = "localhost";
      int  visport   = 19916, s = 350;
      socketstream sol_sock;
      common::VisualizeField(sol_sock, vishost, visport, face_dofs,
                             "Shifted Face Dofs", 0, s, s, s, "Rjmp");
   }

   // Make a list of inactive tdofs that will be eliminated from the system.
   // The inactive tdofs are the dofs for the elements located outside the true
   // domain (and optionally, for the elements cut by the true boundary, if
   // include_cut_cell = false) minus the dofs that are located on the surrogate
   // boundary.
   Array<int> ess_tdof_list;
   Array<int> ess_shift_bdr;
   marker.ListEssentialTDofs(elem_marker, sb_dofs, ess_tdof_list,
                             ess_shift_bdr);

   // Compute distance vector to the actual boundary.
   ParFiniteElementSpace distance_vec_space(&pmesh, &fec, dim);
   ParGridFunction distance(&distance_vec_space);
   VectorCoefficient *dist_vec = NULL;
   // Compute the distance field using the HeatDistanceSolver for
   // level_set_type == 4 or analytically for all other level set types.
   if (level_set_type == 4)
   {
      // Discrete distance vector.
      double dx = AvgElementSize(pmesh);
      ParGridFunction filt_gf(&pfespace);
      PDEFilter *filter = new PDEFilter(pmesh, 2.0 * dx);
      filter->Filter(dist_fun_level_coef, filt_gf);
      delete filter;
      GridFunctionCoefficient ls_filt_coeff(&filt_gf);

      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916, s = 350;
         socketstream sol_sock;
         common::VisualizeField(sol_sock, vishost, visport, filt_gf,
                                "Input Level Set", 0, 2*s, s, s, "Rjmm");
      }

      HeatDistanceSolver dist_func(2.0 * dx* dx);
      dist_func.print_level = 1;
      dist_func.smooth_steps = 1;
      dist_func.ComputeVectorDistance(ls_filt_coeff, distance);
      dist_vec = new VectorGridFunctionCoefficient(&distance);
   }
   else
   {
      // Analytic distance vector.
      dist_vec = new Dist_Vector_Coefficient(dim, level_set_type);
      distance.ProjectDiscCoefficient(*dist_vec);
   }

   // Visualize the distance vector.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916, s = 350;
      socketstream sol_sock;
      common::VisualizeField(sol_sock, vishost, visport, distance,
                             "Distance Vector", s, s, s, s, "Rjmmpcvv", 1);
   }

   // Set up a list to indicate element attributes to be included in assembly,
   // so that inactive elements are excluded.
   const int max_elem_attr = pmesh.attributes.Max();
   Array<int> ess_elem(max_elem_attr);
   ess_elem = 1;
   bool inactive_elements = false;
   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      if (!include_cut_cell &&
          (elem_marker[i] == ShiftedFaceMarker::SBElementType::OUTSIDE ||
           elem_marker[i] == ShiftedFaceMarker::SBElementType::CUT))
      {
         pmesh.SetAttribute(i, max_elem_attr+1);
         inactive_elements = true;
      }
      if (include_cut_cell &&
          elem_marker[i] == ShiftedFaceMarker::SBElementType::OUTSIDE)
      {
         pmesh.SetAttribute(i, max_elem_attr+1);
         inactive_elements = true;
      }
   }
   bool inactive_elements_global;
   MPI_Allreduce(&inactive_elements, &inactive_elements_global, 1, MPI_C_BOOL,
                 MPI_LOR, MPI_COMM_WORLD);
   if (inactive_elements_global) { ess_elem.Append(0); }
   pmesh.SetAttributes();

   // Set up the linear form b(.) which corresponds to the right-hand side of
   // the FEM linear system.
   ParLinearForm b(&pfespace);
   FunctionCoefficient *rhs_f = NULL;
   if (level_set_type == 1 || level_set_type == 4)
   {
      rhs_f = new FunctionCoefficient(rhs_fun_circle);
   }
   else if (level_set_type == 2)
   {
      rhs_f = new FunctionCoefficient(rhs_fun_xy_exponent);
   }
   else if (level_set_type == 3)
   {
      rhs_f = new FunctionCoefficient(rhs_fun_xy_sinusoidal);
   }
   else { MFEM_ABORT("RHS function not set for level set type.\n"); }
   b.AddDomainIntegrator(new DomainLFIntegrator(*rhs_f), ess_elem);

   // Dirichlet BC that must be imposed on the true boundary.
   ShiftedFunctionCoefficient *dbcCoef = NULL;
   if (level_set_type == 1 || level_set_type == 4)
   {
      dbcCoef = new ShiftedFunctionCoefficient(dirichlet_velocity_circle);
   }
   else if (level_set_type == 2)
   {
      dbcCoef = new ShiftedFunctionCoefficient(dirichlet_velocity_xy_exponent);
   }
   else if (level_set_type == 3)
   {
      dbcCoef = new ShiftedFunctionCoefficient(dirichlet_velocity_xy_sinusoidal);
   }
   else
   {
      MFEM_ABORT("Dirichlet velocity function not set for level set type.\n");
   }
   // Add integrators corresponding to the shifted boundary method (SBM).
   b.AddInteriorFaceIntegrator(new SBM2DirichletLFIntegrator(&pmesh, *dbcCoef,
                                                             alpha, *dist_vec,
                                                             elem_marker,
                                                             include_cut_cell,
                                                             ho_terms));
   b.AddBdrFaceIntegrator(new SBM2DirichletLFIntegrator(&pmesh, *dbcCoef,
                                                        alpha, *dist_vec,
                                                        elem_marker,
                                                        include_cut_cell,
                                                        ho_terms), ess_shift_bdr);
   b.Assemble();

   // Set up the bilinear form a(.,.) on the finite element space corresponding
   // to the Laplacian operator -Delta, by adding the Diffusion domain
   // integrator and SBM integrator.
   ParBilinearForm a(&pfespace);
   ConstantCoefficient one(1.);
   a.AddDomainIntegrator(new DiffusionIntegrator(one), ess_elem);
   a.AddInteriorFaceIntegrator(new SBM2DirichletIntegrator(&pmesh, alpha,
                                                           *dist_vec,
                                                           elem_marker,
                                                           include_cut_cell,
                                                           ho_terms));
   a.AddBdrFaceIntegrator(new SBM2DirichletIntegrator(&pmesh, alpha, *dist_vec,
                                                      elem_marker,
                                                      include_cut_cell,
                                                      ho_terms), ess_shift_bdr);

   // Assemble the bilinear form and the corresponding linear system,
   // applying any necessary transformations.
   a.Assemble();

   // Project the exact solution as an initial condition for Dirichlet boundary.
   x.ProjectCoefficient(*dbcCoef);

   // Form the linear system and solve it.
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   Solver *prec = new HypreBoomerAMG;
   BiCGSTABSolver *bicg = new BiCGSTABSolver(MPI_COMM_WORLD);
   bicg->SetRelTol(1e-12);
   bicg->SetMaxIter(2000);
   bicg->SetPrintLevel(1);
   bicg->SetPreconditioner(*prec);
   bicg->SetOperator(*A);
   bicg->Mult(B, X);

   // Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // Save the mesh and the solution.
   ofstream mesh_ofs("diffusion.mesh");
   mesh_ofs.precision(8);
   pmesh.PrintAsOne(mesh_ofs);
   ofstream sol_ofs("diffusion.gf");
   sol_ofs.precision(8);
   x.SaveAsOne(sol_ofs);

   // Save the solution in ParaView format
   if (visualization)
   {
      ParaViewDataCollection dacol("ParaViewDiffusion", &pmesh);
      dacol.SetLevelsOfDetail(order);
      dacol.RegisterField("distance", &distance);
      dacol.RegisterField("solution", &x);
      dacol.SetTime(1.0);
      dacol.SetCycle(1);
      dacol.Save();
   }


   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916, s = 350;
      socketstream sol_sock;
      common::VisualizeField(sol_sock, vishost, visport, x,
                             "Solution", s, 0, s, s, "Rj");
   }

   // Construct an error grid function if the exact solution is known.
   if (level_set_type == 2 || level_set_type == 3)
   {
      ParGridFunction err(x);
      Vector pxyz(dim);
      pxyz(0) = 0.;
      for (int i = 0; i < nodes_cnt; i++)
      {
         pxyz(0) = vxyz(i);
         pxyz(1) = vxyz(i+nodes_cnt);
         double exact_val = 0.;
         if (level_set_type == 2)
         {
            exact_val = dirichlet_velocity_xy_exponent(pxyz);
         }
         else if (level_set_type == 3)
         {
            exact_val = dirichlet_velocity_xy_sinusoidal(pxyz);
         }
         err(i) = std::fabs(x(i) - exact_val);
      }

      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916, s = 350;
         socketstream sol_sock;
         common::VisualizeField(sol_sock, vishost, visport, err,
                                "Error", 2*s, 0, s, s, "Rj");
      }

      const double global_error = x.ComputeL2Error(*dbcCoef);
      if (myid == 0)
      {
         std::cout << "Global L2 error: " << global_error << endl;
      }
   }

   // Free the used memory.
   delete prec;
   delete bicg;
   delete dbcCoef;
   delete rhs_f;
   delete dist_vec;

   MPI_Finalize();

   return 0;
}
