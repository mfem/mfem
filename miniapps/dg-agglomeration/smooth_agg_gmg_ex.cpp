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
//                     ------------------------
//                     DG Smooth-Agg GMG Solver
//                     ------------------------

#include "mfem.hpp"
#include <iostream>
#include <memory>

#include "mg_agglom.hpp"

using namespace std;
using namespace mfem;

// RHS
real_t rhs_function(const Vector &x);

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/ref-cube.mesh";
   int order = 1;
   real_t kappa_0 = 1.0;
   int num_levels = 2;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file.");
   // args.AddOption(&ref_levels, "-r", "--refine", "Refinement levels.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&kappa_0, "-k", "--kappa", "DG penalty parameter.");
   // args.AddOption(&ncoarse, "-nc", "--ncoarse", "Number of Fine Elements per Coarse.");
   args.AddOption(&num_levels, "-nl", "--levels", "Number of Multigrid Levels.");
   args.ParseCheck();

   Mesh mesh(mesh_file);
   const int dim = mesh.Dimension();
   int ncoarse = pow(2, dim);
   int ref_levels = num_levels;


   for (int i = 0; i < ref_levels; ++i) { mesh.UniformRefinement(); }

   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetVSize() << endl;

   const real_t sigma = -1.0;
   const real_t kappa = kappa_0 * (order + 1) * (order + 1) / 2;

   // Array<int> ess_bdr(mesh.bdr_attributes.Max());
   // ess_bdr = 0;
   // ess_bdr[0] = 1;

   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   FunctionCoefficient rhs(rhs_function);
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

   //std::string file_name = "A_nl" + std::to_string(num_levels)+".mtx";
   // std::ofstream ofs1(file_name);
   // A.PrintMM(ofs1);
   // ofs1.close();

   {
      std::ofstream f("A.txt");
      A.PrintMatlab(f);
   }
   {
      const auto &e2e = mesh.ElementToElementTable();
      Array<int> fn;

      std::ofstream f("t2t.txt");
      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         e2e.GetRow(e, fn);
         int i = 0;
         for (; i < fn.Size(); ++i)
         {
            if (fn[i] != e)
            {
               f << (fn[i] + 1) << " ";
            }
         }
         const int nf = dim == 2 ? Geometry::NumEdges[mesh.GetElementGeometry(e)]
                        : Geometry::NumFaces[mesh.GetElementGeometry(e)];
         for (; i < nf; ++i)
         {
            f << -1 << " ";
         }
         f << '\n';
      }
   }

   SmoothedAggregationGMG mg(fespace, A, ncoarse, num_levels, false);
   mg.SetCycleType(mfem::MultigridBase::CycleType::VCYCLE, 3, 3);

   CGSolver cg;
   cg.SetRelTol(1e-7);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(1);
   cg.SetOperator(A);
   cg.SetPreconditioner(mg);
   cg.Mult(b, x);

   return 0;
}

// Initial condition
real_t rhs_function(const Vector &x)
{
   int dim = x.Size();

   real_t px = M_PI*x(0);
   real_t py = M_PI*x(1);
   real_t pz = M_PI*x(2);

   real_t pi_s = M_PI*M_PI;

   real_t sss = sin(px)*sin(py)*sin(pz);

   real_t css = cos(px)*sin(py)*sin(pz)*cos(px)*sin(py)*sin(pz);
   real_t scs = sin(px)*cos(py)*sin(pz)*sin(px)*cos(py)*sin(pz);
   real_t ssc = sin(px)*sin(py)*cos(pz)*sin(px)*sin(py)*cos(pz);
   return pi_s*exp(sss)*(-3*sss + css + scs + ssc);
}