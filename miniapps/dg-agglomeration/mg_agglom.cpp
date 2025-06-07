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

#include "mg_agglom.hpp"
#include "partition.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>


using namespace std;
using namespace mfem;

std::vector<std::vector<int>> Agglomerate(Mesh &mesh)
{
   std::vector<std::vector<int>> E;

   // Recursive METIS partitioning to create 'E' data 
   int ncoarse = 4;
   DG_FECollection fec(0, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction p(&fes);
   const int ne = mesh.GetNE();
   const int num_partitions = std::ceil(std::log(ne)/std::log(ncoarse));
   p = 0;
   Array<int> partitioning = PartitionMesh(mesh, ncoarse);
   for (int i = 0; i < p.Size(); ++i)
   {
      p[i] = partitioning[i];
   }
   for (int i = 0; i < ncoarse; ++i)
   {
      E[0].push_back(0);
   }
   std::vector<std::vector<int>> macro_elements(ncoarse);
   for (int k = 0; k < ne; ++k)
   {
      const int i = partitioning[k];
      macro_elements[i].push_back(k);
   }
   for (int j = 1; j < num_partitions; ++j)
   {
      std::vector<std::vector<int>> macro_elements(E[j-1].size());
      for (int i = 0; i < p.Size(); ++i)
      {
         const int k = p[i];
         macro_elements[k].push_back(i);
      }
      int num_total_parts = 0;
      for (int e = 0; e < E[j-1].size(); ++e)
      {
         const int num_el_part = macro_elements[e].size();
         Array<int> subset(num_el_part);
         for (int i=0; i<num_el_part; i++) {subset[i] = macro_elements[e][i];}
         Array<int> partitioning = PartitionMesh(mesh, ncoarse, subset);
         int num_actual_parts = 0;
         for (int ip = 0; ip < partitioning.Size(); ++ip)
         {
            const int i = partitioning[ip];
            num_actual_parts = (i > num_actual_parts) ? i : num_actual_parts;
            p[subset[ip]] = i + num_total_parts;
         }
         for (int k = 0; k <= num_actual_parts; ++k) {E[j].push_back(e);}
         num_total_parts = num_total_parts + num_actual_parts + 1;
      }
   }

   return E;
}

AgglomerationMultigrid::AgglomerationMultigrid(
   FiniteElementSpace &fes, SparseMatrix &Af)
{
   MFEM_VERIFY(fes.GetMaxElementOrder() == 1, "Only linear elements supported.");
   Mesh* mesh = fes.GetMesh();
   // Create the mesh hierarchy
   auto E = Agglomerate(*fes.GetMesh());
   int num_levels = E.size()+1;

   // Populate the arrays: operators, smoothers, ownedOperators, ownedSmoothers
   // from the MultigridBase class. (All smoothers are owned, all operators
   // except the finest are owned).
   operators.SetSize(num_levels);
   smoothers.SetSize(num_levels);
   ownedOperators.SetSize(num_levels);
   ownedSmoothers.SetSize(num_levels); 
   prolongations.SetSize(num_levels-1);
   ownedProlongations.SetSize(num_levels-1); 
   //Set the ownership
   for(int l = 0; l < num_levels-1; ++l){
      ownedOperators[l] = true;
      ownedSmoothers[l] = true;
      ownedProlongations[l] = true;
   }
   ownedOperators[num_levels-1] = false;
   ownedSmoothers[num_levels-1] = true; 

   // Populate the arrays: prolongations, ownedProlongations from the Multigrid
   // class. All prolongations are owned.
   // Create the prolongations using 'E' using the SparseMatrix class

   // Make the final prolongation
   operators[num_levels - 1] = &Af;
   SparseMatrix p_level = SparseMatrix(4*E[num_levels-2].size(), 4*E[num_levels-3].size());
   GridFunction verts(&fes);
   mesh->GetNodes(verts);
   int nnodes = verts.Size()/2;
   for(int r = 0; r < E[num_levels-2].size(); ++r)
   {
      int c = E[num_levels-2][r]; 
      double x0 = verts(r*4); double x1 = verts(r*4+1); double x2 = verts(r*4+2); double x3 = verts(r*4+3);
      double y0 = verts(nnodes+ r*4); double y1 = verts(nnodes + (r*4+1)); double y2 = verts(nnodes + r*4+2); double y3 = verts(nnodes+r*4+3);
      //column 1 of block
      p_level.Set(4*r, 4*c, 1); p_level.Set(4*r+1, 4*c, 1); p_level.Set(4*r+1, 4*c, 1); p_level.Set(4*r+1, 4*c, 1);
      //column 2 of block
      p_level.Set(4*r, 4*c+1, x0); p_level.Set(4*r+1, 4*c+1, x1); p_level.Set(4*r+1, 4*c+1, x2); p_level.Set(4*r+1, 4*c+1, x3);
      //column 3 of block
      p_level.Set(4*r, 4*c+2, y0); p_level.Set(4*r+1, 4*c+2, y1); p_level.Set(4*r+1, 4*c+2, y2); p_level.Set(4*r+1, 4*c+2, y3);
      //column 3 of block
      p_level.Set(4*r, 4*c+3, x0*y0); p_level.Set(4*r+1, 4*c+3, x1*y1); p_level.Set(4*r+1, 4*c+3, x2*y2); p_level.Set(4*r+1, 4*c+3, x3*y3);
   }
   SparseMatrix* PT = Transpose(p_level);
   RAPOperator Ac(*PT, *operators[num_levels-1], p_level);
   operators[num_levels-2] = &Ac;
   //SparseMatrix* p_level_p = &p_level;
   prolongations[num_levels-2] = &p_level;

   //Make the rest of the prolongations
   for(int l = num_levels-3; l > 0; --l){
      SparseMatrix p_level = SparseMatrix(4*E[l].size(), 4*E[l-1].size());
      for(int r = 1; r < E[l].size(); ++r)
      {
         int c = E[l][r];
         p_level.Set(4*r, 4*c, 1);
         p_level.Set(4*r+1, 4*c+1, 1);
         p_level.Set(4*r+2, 4*c+2, 1);
         p_level.Set(4*r+3, 4*c+3, 1);
      }
      SparseMatrix* PT = Transpose(p_level);
      RAPOperator Ac(*PT, *operators[l+1], p_level);
      operators[l] = &Ac;
      //SparseMatrix* p_level_p = &p_level;
      prolongations[l] = &p_level;
   }

   //Make First Prolongation
   SparseMatrix p_levelc = SparseMatrix(4*E[0].size(), 4);
   for(int r = 0; r < 4; ++r)
   {
      p_levelc.Set(r, r, 1); 
      p_levelc.Set(r+4, r, 1);
      p_levelc.Set(r+8, r, 1);
      p_levelc.Set(r+12, r, 1);
   }
   SparseMatrix* PTc = Transpose(p_levelc);
   RAPOperator Acc(*PTc, *operators[1], p_levelc);
   operators[0] = &Acc;
   //SparseMatrix* p_level_p = &p_level;
   prolongations[0] = &p_levelc;

   // Create the smoothers (using BlockILU for now) remember block size is num degrees of freedom per element
   for(int l=0; l < num_levels; ++l)
   {
      BlockILU smooth_l(*operators[l], 4);
      smoothers[l] = &smooth_l;
   }
}

int main(int argc, char *argv[]){
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = -1;
   int order = 1;
   real_t sigma = -1.0;
   real_t kappa = -1.0;
   real_t eta = 0.0;
   bool pa = false;
   bool visualization = 1;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the three DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the three DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&eta, "-e", "--eta", "BR2 penalty parameter.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   args.PrintOptions(cout);

     // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   //    NURBS meshes are projected to second order meshes.
   Mesh mesh(mesh_file);
   const int dim = mesh.Dimension();

      // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. By default, or if ref_levels < 0,
   //    we choose it to be the largest number that gives a final mesh with no
   //    more than 50,000 elements.
   {
      if (ref_levels < 0)
      {
         ref_levels = (int)floor(log(50000./mesh.GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }
   if (mesh.NURBSext)
   {
      mesh.SetCurvature(max(order, 1));
   }

   // 5. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   const auto bt = pa ? BasisType::GaussLobatto : BasisType::GaussLegendre;
   DG_FECollection fec(order, dim, bt);
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetVSize() << endl;

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.AddBdrFaceIntegrator(
      new DGDirichletLFIntegrator(zero, one, sigma, kappa));
   b.Assemble();

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero.
   GridFunction x(&fespace);
   x = 0.0;

   // 8. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and the interior and boundary DG face integrators.
   //    Note that boundary conditions are imposed weakly in the form, so there
   //    is no need for dof elimination. After assembly and finalizing we
   //    extract the corresponding sparse matrix A.
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   if (eta > 0)
   {
      MFEM_VERIFY(!pa, "BR2 not yet compatible with partial assembly.");
      a.AddInteriorFaceIntegrator(new DGDiffusionBR2Integrator(fespace, eta));
      a.AddBdrFaceIntegrator(new DGDiffusionBR2Integrator(fespace, eta));
   }
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.Assemble();
   a.Finalize();

   // 9. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG in the symmetric case, and GMRES in the
   //    non-symmetric one. (Note that tolerances are squared: 1e-12 corresponds
   //    to a relative tolerance of 1e-6).
   //
   //    If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   if (pa)
   {
      MFEM_VERIFY(sigma == -1.0,
                  "The case of PA with sigma != -1 is not yet supported.");
      CG(a, b, x, 1, 500, 1e-12, 0.0);
   }
   else
   {
      SparseMatrix &A = a.SpMat();
#ifndef MFEM_USE_SUITESPARSE
      //GSSmoother M(A);

      AgglomerationMultigrid M(fespace, A);
      if (sigma == -1.0)
      {
         PCG(A, M, b, x, 1, 500, 1e-12, 0.0);
      }
      else
      {
         GMRES(A, M, b, x, 1, 500, 10, 1e-12, 0.0);
      }
#else
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(A);
      umf_solver.Mult(b, x);
#endif
   }

   // 10. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 11. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
   }

   return 0;
}

 // namespace mfem