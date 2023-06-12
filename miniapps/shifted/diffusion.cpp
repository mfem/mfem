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
//     ------------------------------------------------------------------
//     Shifted Diffusion Miniapp: Finite element immersed boundary solver
//     ------------------------------------------------------------------
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
//     Dirichlet boundary condition
//     mpirun -np 4 diffusion -rs 3 -o 1 -vis -lst 1
//     mpirun -np 4 diffusion -m ../../data/inline-hex.mesh -rs 2 -o 2 -vis -lst 1 -ho 1 -alpha 10
//     Neumann boundary condition
//     mpirun -np 4 diffusion -rs 3 -o 1 -vis -nlst 1 -ho 1
//
//   Problem 2: Circular hole of radius 0.2 at the center of the domain.
//              Solves -nabla^2 u = f with inhomogeneous boundary conditions,
//              and f is setup such that u = x^p + y^p, where p = 2 by default.
//              This is a 2D convergence test.
//     Dirichlet BC
//     mpirun -np 4 diffusion -rs 2 -o 2 -vis -lst 2
//     Neumann BC (inhomogeneous condition derived using exact solution)
//     mpirun -np 4 diffusion -rs 2 -o 2 -vis -nlst 2 -ho 1
//
//   Problem 3: Domain is y = [0, 1] but mesh is shifted to [-1.e-4, 1].
//              Solves -nabla^2 u = f with inhomogeneous boundary conditions,
//              and f is setup such that u = sin(pi*x*y). This is a 2D
//              convergence test. Second-order can be demonstrated by changing
//              refinement level (-rs) for the sample run below. Higher-order
//              convergence can be realized by also increasing the order (-o) of
//              the finite element space and the number of high-order terms
//              (-ho) to be included from the Taylor expansion used to enforce
//              the boundary conditions.
//     mpirun -np 4 diffusion -rs 2 -o 1 -vis -lst 3
//
//   Problem 4: Complex 2D / 3D shapes:
//              Solves -nabla^2 u = 1 with homogeneous boundary conditions.
//     mpirun -np 4 diffusion -m ../../data/inline-quad.mesh -rs 4 -lst 4 -alpha 10
//     mpirun -np 4 diffusion -m ../../data/inline-tri.mesh  -rs 4 -lst 4 -alpha 10
//     mpirun -np 4 diffusion -m ../../data/inline-hex.mesh  -rs 3 -lst 8 -alpha 10
//     mpirun -np 4 diffusion -m ../../data/inline-tet.mesh  -rs 3 -lst 8 -alpha 10
//
//   Problem 5: Circular hole with homogeneous Neumann, triangular hole with
//            inhomogeneous Dirichlet, and a square hole with homogeneous
//            Dirichlet boundary condition.
//     mpirun -np 4 diffusion -rs 3 -o 1 -vis -lst 5 -ho 1 -nlst 7 -alpha 10.0 -dc

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include "sbm_aux.hpp"
#include "sbm_solver.hpp"
#include "marking.hpp"

using namespace mfem;
using namespace std;
using namespace common;

int main(int argc, char *argv[])
{
#ifdef HYPRE_USING_GPU
   cout << "\nAs of mfem-4.3 and hypre-2.22.0 (July 2021) this miniapp\n"
        << "is NOT supported with the GPU version of hypre.\n\n";
   return 242;
#endif

   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 2;
   bool visualization = true;
   int ser_ref_levels = 0;
   int dirichlet_level_set_type = -1;
   int neumann_level_set_type = -1;
   bool dirichlet_combo = false;
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
   args.AddOption(&dirichlet_level_set_type, "-lst", "--level-set-type",
                  "level-set-type.");
   args.AddOption(&neumann_level_set_type, "-nlst", "--neumann-level-set-type",
                  "neumann-level-set-type.");
   args.AddOption(&dirichlet_combo, "-dc", "--dcombo",
                  "no-dc", "--no-dcombo",
                  "Combination of two Dirichlet level sets.");
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
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Use Dirichlet level set if no level sets are specified.
   if (dirichlet_level_set_type <= 0 && neumann_level_set_type <= 0)
   {
      dirichlet_level_set_type = 1;
   }
   MFEM_VERIFY((neumann_level_set_type > 0 && ho_terms < 1) == false,
               "Shifted Neumann BC requires extra terms, i.e., -ho >= 1.");

   // Enable hardware devices such as GPUs, and programming models such as CUDA,
   // OCCA, RAJA and OpenMP based on command line options.
   Device device("cpu");
   if (myid == 0) { device.Print(); }

   // Refine the mesh.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   for (int lev = 0; lev < ser_ref_levels; lev++) { mesh.UniformRefinement(); }
   if (myid == 0)
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }
   MFEM_VERIFY(mesh.Conforming(), "AMR capability is not implemented yet!");

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
   // for dirichlet_level_set_type = 3 such that some of the mesh elements are
   // intersected by the true boundary (y = 0).
   ParFiniteElementSpace pfespace_mesh(&pmesh, &fec, dim);
   pmesh.SetNodalFESpace(&pfespace_mesh);
   ParGridFunction x_mesh(&pfespace_mesh);
   pmesh.SetNodalGridFunction(&x_mesh);
   vxyz = *pmesh.GetNodes();
   int nodes_cnt = vxyz.Size()/dim;
   if (dirichlet_level_set_type == 3)
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

   // Determine if each element in the ParMesh is inside the actual domain,
   // partially cut by its boundary, or completely outside the domain.
   // Setup the level-set coefficients, and mark the elements.
   Dist_Level_Set_Coefficient *dirichlet_dist_coef = NULL;
   Dist_Level_Set_Coefficient *dirichlet_dist_coef_2 = NULL;
   Dist_Level_Set_Coefficient *neumann_dist_coef = NULL;
   Combo_Level_Set_Coefficient combo_dist_coef;

   ParGridFunction level_set_gf(&pfespace);
   ShiftedFaceMarker marker(pmesh, pfespace, include_cut_cell);
   Array<int> elem_marker;

   // Dirichlet level-set.
   if (dirichlet_level_set_type > 0)
   {
      dirichlet_dist_coef = new Dist_Level_Set_Coefficient(dirichlet_level_set_type);
      const double dx = AvgElementSize(pmesh);
      PDEFilter filter(pmesh, dx);
      filter.Filter(*dirichlet_dist_coef, level_set_gf);
      //level_set_gf.ProjectCoefficient(*dirichlet_dist_coef);
      // Exchange information for ghost elements i.e. elements that share a face
      // with element on the current processor, but belong to another processor.
      level_set_gf.ExchangeFaceNbrData();
      // Setup the class to mark all elements based on whether they are located
      // inside or outside the true domain, or intersected by the true boundary.
      marker.MarkElements(level_set_gf, elem_marker);
      combo_dist_coef.Add_Level_Set_Coefficient(*dirichlet_dist_coef);
   }

   // Second Dirichlet level-set.
   if (dirichlet_combo)
   {
      MFEM_VERIFY(dirichlet_level_set_type == 5,
                  "The combo level set example has been only set for"
                  " dirichlet_level_set_type == 5.");
      dirichlet_dist_coef_2 = new Dist_Level_Set_Coefficient(6);
      level_set_gf.ProjectCoefficient(*dirichlet_dist_coef_2);
      level_set_gf.ExchangeFaceNbrData();
      marker.MarkElements(level_set_gf, elem_marker);
      combo_dist_coef.Add_Level_Set_Coefficient(*dirichlet_dist_coef_2);
   }

   // Neumann level-set.
   if (neumann_level_set_type > 0)
   {
      neumann_dist_coef = new Dist_Level_Set_Coefficient(neumann_level_set_type);
      level_set_gf.ProjectCoefficient(*neumann_dist_coef);
      level_set_gf.ExchangeFaceNbrData();
      marker.MarkElements(level_set_gf, elem_marker);
      combo_dist_coef.Add_Level_Set_Coefficient(*neumann_dist_coef);
   }

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
                             "Shifted Face Dofs", 0, s, s, s, "Rjmplo");
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
   // Compute the distance field analytically or using the HeatDistanceSolver.
   if (dirichlet_level_set_type == 1 || dirichlet_level_set_type == 2 ||
       dirichlet_level_set_type == 3)
   {
      // Analytic distance vector.
      dist_vec = new Dist_Vector_Coefficient(dim, dirichlet_level_set_type);
      distance.ProjectDiscCoefficient(*dist_vec);
   }
   else
   {
      // Discrete distance vector.
      double dx = AvgElementSize(pmesh);
      ParGridFunction filt_gf(&pfespace);
      PDEFilter filter(pmesh, 2.0 * dx);
      filter.Filter(combo_dist_coef, filt_gf);
      GridFunctionCoefficient ls_filt_coeff(&filt_gf);

      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916, s = 350;
         socketstream sol_sock;
         common::VisualizeField(sol_sock, vishost, visport, filt_gf,
                                "Input Level Set", 0, 2*s, s, s, "Rjmm");
      }

      HeatDistanceSolver dist_func(2.0 * dx * dx);
      dist_func.print_level.FirstAndLast().Summary();
      dist_func.smooth_steps = 1;
      dist_func.ComputeVectorDistance(ls_filt_coeff, distance);
      dist_vec = new VectorGridFunctionCoefficient(&distance);
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
           elem_marker[i] >= ShiftedFaceMarker::SBElementType::CUT))
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
   if (dirichlet_level_set_type == 1 || dirichlet_level_set_type == 4 ||
       dirichlet_level_set_type == 5 || dirichlet_level_set_type == 6 ||
       dirichlet_level_set_type == 8 ||
       neumann_level_set_type == 1 || neumann_level_set_type == 7)
   {
      rhs_f = new FunctionCoefficient(rhs_fun_circle);
   }
   else if (dirichlet_level_set_type == 2 || neumann_level_set_type == 2)
   {
      rhs_f = new FunctionCoefficient(rhs_fun_xy_exponent);
   }
   else if (dirichlet_level_set_type == 3)
   {
      rhs_f = new FunctionCoefficient(rhs_fun_xy_sinusoidal);
   }
   else { MFEM_ABORT("RHS function not set for level set type.\n"); }
   b.AddDomainIntegrator(new DomainLFIntegrator(*rhs_f), ess_elem);

   // Exact solution to project for Dirichlet boundaries
   FunctionCoefficient *exactCoef = NULL;
   // Dirichlet BC that must be imposed on the true boundary.
   ShiftedFunctionCoefficient *dbcCoef = NULL;
   if (dirichlet_level_set_type == 1 || dirichlet_level_set_type >= 4)
   {
      dbcCoef = new ShiftedFunctionCoefficient(homogeneous);
      exactCoef = new FunctionCoefficient(homogeneous);
   }
   else if (dirichlet_level_set_type == 2)
   {
      dbcCoef = new ShiftedFunctionCoefficient(dirichlet_velocity_xy_exponent);
      exactCoef = new FunctionCoefficient(dirichlet_velocity_xy_exponent);
   }
   else if (dirichlet_level_set_type == 3)
   {
      dbcCoef = new ShiftedFunctionCoefficient(dirichlet_velocity_xy_sinusoidal);
      exactCoef = new FunctionCoefficient(dirichlet_velocity_xy_sinusoidal);
   }

   ShiftedFunctionCoefficient *dbcCoefCombo = NULL;
   if (dirichlet_combo)
   {
      dbcCoefCombo = new ShiftedFunctionCoefficient(0.015);
   }

   // Homogeneous Neumann boundary condition coefficient
   ShiftedFunctionCoefficient *nbcCoef = NULL;
   ShiftedVectorFunctionCoefficient *normalbcCoef = NULL;
   if (neumann_level_set_type == 1)
   {
      nbcCoef = new ShiftedFunctionCoefficient(homogeneous);
      normalbcCoef = new ShiftedVectorFunctionCoefficient(dim, normal_vector_1);
   }
   else if (neumann_level_set_type == 2)
   {
      nbcCoef = new ShiftedFunctionCoefficient(traction_xy_exponent);
      normalbcCoef = new ShiftedVectorFunctionCoefficient(dim, normal_vector_1);
      exactCoef = new FunctionCoefficient(dirichlet_velocity_xy_exponent);
   }
   else if (neumann_level_set_type == 7)
   {
      nbcCoef = new ShiftedFunctionCoefficient(homogeneous);
      normalbcCoef = new ShiftedVectorFunctionCoefficient(dim, normal_vector_2);
   }
   else if (neumann_level_set_type > 0)
   {
      MFEM_ABORT(" Normal vector coefficient not implemented for level set.");
   }

   // Add integrators corresponding to the shifted boundary method (SBM)
   // for Dirichlet boundaries.
   // For each LinearFormIntegrator, we indicate the marker that we have used
   // for the cut-cell corresponding to the level-set.
   int ls_cut_marker = ShiftedFaceMarker::SBElementType::CUT;
   // For each BilinearFormIntegrators, we make a list of the markers
   // corresponding to the cut-cell whose faces they will be applied to.
   Array<int> bf_dirichlet_marker(0), bf_neumann_marker(0);

   if (dirichlet_level_set_type > 0)
   {
      b.AddInteriorFaceIntegrator(new SBM2DirichletLFIntegrator(&pmesh, *dbcCoef,
                                                                alpha, *dist_vec,
                                                                elem_marker,
                                                                include_cut_cell,
                                                                ho_terms,
                                                                ls_cut_marker));
      b.AddBdrFaceIntegrator(new SBM2DirichletLFIntegrator(&pmesh, *dbcCoef,
                                                           alpha, *dist_vec,
                                                           elem_marker,
                                                           include_cut_cell,
                                                           ho_terms,
                                                           ls_cut_marker),
                             ess_shift_bdr);
      bf_dirichlet_marker.Append(ls_cut_marker);
      ls_cut_marker += 1;
   }

   if (dirichlet_combo)
   {
      b.AddInteriorFaceIntegrator(new SBM2DirichletLFIntegrator(&pmesh, *dbcCoefCombo,
                                                                alpha, *dist_vec,
                                                                elem_marker,
                                                                include_cut_cell,
                                                                ho_terms,
                                                                ls_cut_marker));
      bf_dirichlet_marker.Append(ls_cut_marker);
      ls_cut_marker += 1;
   }

   // Add integrators corresponding to the shifted boundary method (SBM)
   // for Neumann boundaries.
   if (neumann_level_set_type > 0)
   {
      MFEM_VERIFY(!include_cut_cell, "include_cut_cell option must be set to"
                  " false for Neumann boundary conditions.");
      b.AddInteriorFaceIntegrator(new SBM2NeumannLFIntegrator(
                                     &pmesh, *nbcCoef, *dist_vec,
                                     *normalbcCoef, elem_marker,
                                     ho_terms, include_cut_cell,
                                     ls_cut_marker));
      bf_neumann_marker.Append(ls_cut_marker);
      ls_cut_marker += 1;
   }

   b.Assemble();

   // Set up the bilinear form a(.,.) on the finite element space corresponding
   // to the Laplacian operator -Delta, by adding the Diffusion domain
   // integrator and SBM integrator.
   ParBilinearForm a(&pfespace);
   ConstantCoefficient one(1.);
   a.AddDomainIntegrator(new DiffusionIntegrator(one), ess_elem);
   if (dirichlet_level_set_type > 0)
   {
      a.AddInteriorFaceIntegrator(new SBM2DirichletIntegrator(&pmesh, alpha,
                                                              *dist_vec,
                                                              elem_marker,
                                                              bf_dirichlet_marker,
                                                              include_cut_cell,
                                                              ho_terms));
      a.AddBdrFaceIntegrator(new SBM2DirichletIntegrator(&pmesh, alpha, *dist_vec,
                                                         elem_marker,
                                                         bf_dirichlet_marker,
                                                         include_cut_cell,
                                                         ho_terms), ess_shift_bdr);
   }

   // Add neumann bilinearform integrator.
   if (neumann_level_set_type > 0)
   {
      a.AddInteriorFaceIntegrator(new SBM2NeumannIntegrator(&pmesh,
                                                            *dist_vec,
                                                            *normalbcCoef,
                                                            elem_marker,
                                                            bf_neumann_marker,
                                                            include_cut_cell,
                                                            ho_terms));
   }

   // Assemble the bilinear form and the corresponding linear system,
   // applying any necessary transformations.
   a.Assemble();

   // Project the exact solution as an initial condition for Dirichlet boundary.
   if (!exactCoef)
   {
      exactCoef = new FunctionCoefficient(homogeneous);
   }
   x.ProjectCoefficient(*exactCoef);

   // Form the linear system and solve it.
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   Solver *prec = new HypreBoomerAMG;
   BiCGSTABSolver bicg(MPI_COMM_WORLD);
   bicg.SetRelTol(1e-12);
   bicg.SetMaxIter(500);
   bicg.SetPrintLevel(1);
   bicg.SetPreconditioner(*prec);
   bicg.SetOperator(*A);
   bicg.Mult(B, X);

   // Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // Save the mesh and the solution.
   ofstream mesh_ofs("diffusion.mesh");
   mesh_ofs.precision(8);
   pmesh.PrintAsOne(mesh_ofs);
   ofstream sol_ofs("diffusion.gf");
   sol_ofs.precision(8);
   x.SaveAsOne(sol_ofs);

   if (visualization)
   {
      // Save the solution in ParaView format.
      ParaViewDataCollection dacol("ParaViewDiffusion", &pmesh);
      dacol.SetLevelsOfDetail(order);
      dacol.RegisterField("distance", &distance);
      dacol.RegisterField("level_set", &level_set_gf);
      dacol.RegisterField("solution", &x);
      dacol.SetTime(1.0);
      dacol.SetCycle(1);
      dacol.Save();

      // Send the solution by socket to a GLVis server.
      char vishost[] = "localhost";
      int  visport   = 19916, s = 350;
      socketstream sol_sock;
      common::VisualizeField(sol_sock, vishost, visport, x,
                             "Solution", s, 0, s, s, "Rj");
   }

   // Construct an error grid function if the exact solution is known.
   if (dirichlet_level_set_type == 2 || dirichlet_level_set_type == 3 ||
       (dirichlet_level_set_type == -1 && neumann_level_set_type == 2))
   {
      ParGridFunction error(x);
      Vector pxyz(dim);
      pxyz(0) = 0.;
      for (int i = 0; i < nodes_cnt; i++)
      {
         pxyz(0) = vxyz(i);
         pxyz(1) = vxyz(i+nodes_cnt);
         double exact_val = 0.;
         if (dirichlet_level_set_type == 2 || neumann_level_set_type == 2)
         {
            exact_val = dirichlet_velocity_xy_exponent(pxyz);
         }
         else if (dirichlet_level_set_type == 3)
         {
            exact_val = dirichlet_velocity_xy_sinusoidal(pxyz);
         }
         error(i) = std::fabs(x(i) - exact_val);
      }

      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916, s = 350;
         socketstream sol_sock;
         common::VisualizeField(sol_sock, vishost, visport, error,
                                "Error", 2*s, 0, s, s, "Rj");
      }

      const double global_error = x.ComputeL2Error(*exactCoef);
      if (myid == 0)
      {
         std::cout << "Global L2 error: " << global_error << endl;
      }
   }

   const double norm = x.ComputeL1Error(one);
   if (myid == 0) { std::cout << setprecision(10) << norm << std::endl; }

   // Free the used memory.
   delete prec;
   delete normalbcCoef;
   delete nbcCoef;
   delete dbcCoefCombo;
   delete dbcCoef;
   delete exactCoef;
   delete rhs_f;
   delete dist_vec;
   delete neumann_dist_coef;
   delete dirichlet_dist_coef;
   delete dirichlet_dist_coef_2;

   return 0;
}
