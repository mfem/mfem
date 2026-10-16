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
//                     ---------------------------------
//                     Low-Order Refined Solvers Miniapp
//                     ---------------------------------
//
// This miniapp illustrates the use of low-order refined preconditioners for
// finite element problems defined using H1, H(curl), H(div), or L2 finite
// element spaces. The following problems are solved, depending on the chosen
// finite element space:
//
// H1 and L2: definite Helmholtz problem, u - Delta u = f
//            (in L2 discretized using the symmetric interior penalty DG method)
//
// H(curl):   definite Maxwell problem, u + curl curl u = f
//
// H(div):    grad-div problem, u - grad(div u) = f
//
// In each case, the high-order finite element problem is preconditioned using a
// low-order finite element discretization defined on a Gauss-Lobatto refined
// mesh. The low-order problem is solved using a direct solver if MFEM is
// compiled with SuiteSparse enabled, or with one iteration of symmetric
// Gauss-Seidel smoothing otherwise.
//
// For vector finite element spaces, the special "Integrated" basis type is used
// to obtain spectral equivalence between the high-order and low-order refined
// discretizations. This basis is defined in reference [1] and spectral
// equivalence is shown in [2]:
//
// [1]. M. Gerritsma. Edge functions for spectral element methods. Spectral and
//      High Order Methods for Partial Differential Equations. (2010)
// [2]. C. Dohrmann. Spectral equivalence properties of higher-order tensor
//      product finite elements and applications to preconditioning. (2021)
//
// The action of the high-order operator is computed using MFEM's partial
// assembly/matrix-free algorithms (except in the case of L2, which remains
// future work).
//
// Compile with: make lor_solvers
//
// Sample runs:
//
//    lor_solvers -fe h
//    lor_solvers -fe n
//    lor_solvers -fe r
//    lor_solvers -fe l
//    lor_solvers -m ../../data/amr-quad.mesh -fe h
//    lor_solvers -m ../../data/amr-quad.mesh -fe n
//    lor_solvers -m ../../data/amr-quad.mesh -fe r
//    lor_solvers -m ../../data/star-surf.mesh -fe h
//    lor_solvers -m ../../data/star-surf.mesh -fe n
//    lor_solvers -m ../../data/star-surf.mesh -fe r
//
// Device sample runs:
//  * lor_solvers -fe h -d cuda
//  * lor_solvers -fe n -d cuda
//  * lor_solvers -fe r -d cuda
//  * lor_solvers -fe l -d cuda

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>

#include "lor_mms.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/star.mesh";
   int ref_levels = 1;
   int order = 3;
   const char *fe = "h";
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&fe, "-fe", "--fe-type",
                  "FE type. h for H1, n for Hcurl, r for Hdiv, l for L2");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.ParseCheck();

   Device device(device_config);
   device.Print();

   bool H1 = false, ND = false, RT = false, L2 = false;
   if (string(fe) == "h") { H1 = true; }
   else if (string(fe) == "n") { ND = true; }
   else if (string(fe) == "r") { RT = true; }
   else if (string(fe) == "l") { L2 = true; }
   else { MFEM_ABORT("Bad FE type. Must be 'h', 'n', 'r', or 'l'."); }

   real_t kappa = 10*(order+1)*(order+1); // Penalty used for DG discretizations

   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Mesh dimension must be 2 or 3.");
   MFEM_VERIFY(!L2 || dim == sdim, "DG surface meshes not supported.");
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }

   FunctionCoefficient f_coeff(f(1.0)), u_coeff(u);
   VectorFunctionCoefficient f_vec_coeff(sdim, f_vec(RT)),
                             u_vec_coeff(sdim, u_vec);

   int b1 = BasisType::GaussLobatto, b2 = BasisType::IntegratedGLL;
   unique_ptr<FiniteElementCollection> fec;
   if (H1) { fec.reset(new H1_FECollection(order, dim, b1)); }
   else if (ND) { fec.reset(new ND_FECollection(order, dim, b1, b2)); }
   else if (RT) { fec.reset(new RT_FECollection(order-1, dim, b1, b2)); }
   else { fec.reset(new L2_FECollection(order, dim, b1)); }

   FiniteElementSpace fes(&mesh, fec.get());
   cout << "Number of DOFs: " << fes.GetTrueVSize() << endl;

   Array<int> ess_dofs;
   // In DG, boundary conditions are enforced weakly, so no essential DOFs.
   if (!L2) { fes.GetBoundaryTrueDofs(ess_dofs); }

   BilinearForm a(&fes);
   if (H1 || L2)
   {
      a.AddDomainIntegrator(new MassIntegrator);
      a.AddDomainIntegrator(new DiffusionIntegrator);
   }
   else
   {
      a.AddDomainIntegrator(new VectorFEMassIntegrator);
   }

   if (ND) { a.AddDomainIntegrator(new CurlCurlIntegrator); }
   else if (RT) { a.AddDomainIntegrator(new DivDivIntegrator); }
   else if (L2)
   {
      a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(-1.0, kappa));
      a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(-1.0, kappa));
   }
   // Partial assembly not currently supported for DG or for surface meshes with
   // vector finite elements (ND or RT).
   if (H1 || sdim == dim) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.Assemble();

   LinearForm b(&fes);
   if (H1 || L2) { b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff)); }
   else { b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_vec_coeff)); }
   if (L2)
   {
      // DG boundary conditions are enforced weakly with this integrator.
      b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(u_coeff, -1.0, kappa));
   }
   if (H1) { b.UseFastAssembly(true); }
   b.Assemble();

   GridFunction x(&fes);
   if (H1 || L2) { x.ProjectCoefficient(u_coeff);}
   else { x.ProjectCoefficient(u_vec_coeff); }

   Vector X, B;
   OperatorHandle A;
   a.FormLinearSystem(ess_dofs, x, b, A, X, B);

#ifdef MFEM_USE_SUITESPARSE
   LORSolver<UMFPackSolver> solv_lor(a, ess_dofs);
#else
   LORSolver<GSSmoother> solv_lor(a, ess_dofs);
#endif

   CGSolver cg;
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   cg.SetPreconditioner(solv_lor);
   cg.Mult(B, X);

   a.RecoverFEMSolution(X, b, x);

   if (sdim == dim)
   {
      real_t er =
         (H1 || L2) ? x.ComputeL2Error(u_coeff) : x.ComputeL2Error(u_vec_coeff);
      cout << "L2 error: " << er << endl;
   }

   if (visualization)
   {
      // Save the solution and mesh to disk. The output can be viewed using
      // GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
      x.Save("sol.gf");
      mesh.Save("mesh.mesh");

      // Also save the solution for visualization using ParaView
      ParaViewDataCollection dc("LOR", &mesh);
      dc.SetPrefixPath("ParaView");
      dc.SetHighOrderOutput(true);
      dc.SetLevelsOfDetail(order);
      dc.RegisterField("u", &x);
      dc.SetCycle(0);
      dc.SetTime(0.0);
      dc.Save();
   }

   return 0;
}
