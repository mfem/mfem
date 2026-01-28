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
//                     -----------------------
//                     DG Agglomeration Solver
//                     -----------------------

#include "mfem.hpp"
#include <iostream>
#include <memory>

#include "mg_agglom.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/inline-tri.mesh";
   int ref_levels = 2;
   int order = 1;
   real_t kappa_0 = 1.0;
   int ncoarse = 4; 
   int num_levels = 2;
   int smoother = 0; // 0 - Block GS, 1 - Block L1 Jacobi, 2 - Block ILU
   bool paraview_vis = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file.");
   args.AddOption(&ref_levels, "-r", "--refine", "Refinement levels.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&kappa_0, "-k", "--kappa", "DG penalty parameter.");
   args.AddOption(&ncoarse, "-nc", "--ncoarse", "Number of Fine Elements per Coarse.");
   args.AddOption(&num_levels, "-nl", "--levels", "Number of Multigrid Levels.");
   args.AddOption(&smoother, "-s", "--smoother", "Choice of Multigrid Smoother.");
   args.AddOption(&paraview_vis, "-pv", "--paraview", "-npv", "--no-paraview", "Enable ParaView visualization.");
   args.ParseCheck();

   Mesh mesh(mesh_file);
   const int dim = mesh.Dimension();

   for (int i = 0; i < ref_levels; ++i) { mesh.UniformRefinement(); }

   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetVSize() << endl;

   const real_t sigma = -1.0;
   const real_t kappa = kappa_0 * (order + 1) * (order + 1);

   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.AddBdrFaceIntegrator(
      new DGDirichletLFIntegrator(zero, one, sigma, kappa));
   b.Assemble();

   GridFunction x(&fespace);
   x = 0.0;

   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a.Assemble();
   a.Finalize();

   SparseMatrix &A = a.SpMat();

   AgglomerationMultigrid mg(fespace, A, ncoarse, num_levels, smoother, paraview_vis);

   CGSolver cg;
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(1);
   cg.SetOperator(A);
   cg.SetPreconditioner(mg);
   cg.Mult(b, x);

   return 0;
}
