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
//            -------------------------------------------------
//            Overlapping Grids Miniapp: Poisson problem (ex1p)
//            -------------------------------------------------
//
// This example code demonstrates use of MFEM to solve the Poisson problem:
//
//                -Delta u = 1 \in [0, 1]^2, u_b = 0 \in \dO
//
// on two overlapping grids. Using simultaneous Schwarz iterations, the Poisson
// equation is solved iteratively, with boundary data interpolated between the
// overlapping boundaries for each grid. The overlapping Schwarz method was
// introduced by H. A. Schwarz in 1870, see also Section 2.2 of "Stability
// analysis of a singlerate and multirate predictor-corrector scheme for
// overlapping grids" by Mittal, Dutta and Fischer, arXiv:2010.00118.
//
// Compile with: make schwarz_ex1
//
// Sample runs:  schwarz_ex1
//               schwarz_ex1 -m1 ../../data/star.mesh -m2 ../../data/beam-quad.mesh

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Method to use FindPointsGSLIB to determine the boundary points of a mesh that
// are interior to another mesh.
void GetInterdomainBoundaryPoints(FindPointsGSLIB &finder1,
                                  FindPointsGSLIB &finder2,
                                  Vector &mesh_nodes_1, Vector &mesh_nodes_2,
                                  Array<int> ess_tdof_list1,
                                  Array<int> ess_tdof_list2,
                                  Array<int> &ess_tdof_list1_int,
                                  Array<int> &ess_tdof_list2_int,
                                  const int dim);

int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file_1   = "../../data/square-disc.mesh";
   const char *mesh_file_2   = "../../data/inline-quad.mesh";
   int order                 = 2;
   bool visualization        = true;
   int r1_levels             = 0;
   int r2_levels             = 0;
   int visport               = 19916;
   double rel_tol            = 1.e-8;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file_1, "-m1", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_file_2, "-m2", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&r1_levels, "-r1", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&r2_levels, "-r2", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rel_tol, "-rt", "--relative tolerance",
                  "Tolerance for Schwarz iteration convergence criterion.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   const int nmeshes = 2;

   // Read the mesh from the given mesh file. We can handle triangular,
   // quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   // the same code.
   Array <Mesh*> mesharr(2);
   mesharr[0] = new Mesh(mesh_file_1, 1, 1);
   mesharr[1] = new Mesh(mesh_file_2, 1, 1);
   int dim = mesharr[0]->Dimension();


   // Refine the mesh to increase the resolution. In this example we do
   // 'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   // largest number that gives a final mesh with no more than 50,000 elements.
   for (int lev = 0; lev < r1_levels; lev++) { mesharr[0]->UniformRefinement(); }
   for (int lev = 0; lev < r2_levels; lev++) { mesharr[1]->UniformRefinement(); }

   // Define a finite element space on the mesh. Here we use continuous
   // Lagrange finite elements of the specified order. If order < 1, we
   // instead use an isoparametric/isogeometric space.
   Array <FiniteElementCollection*> fecarr(nmeshes);
   Array <FiniteElementSpace*> fespacearr(nmeshes);
   for (int i = 0; i< nmeshes; i ++)
   {
      if (order > 0)
      {
         fecarr[i] =  new H1_FECollection(order, dim);
      }
      else if (mesharr[i]->GetNodes())
      {
         fecarr[i] =  mesharr[i]->GetNodes()->OwnFEC();
         cout << "Using isoparametric FEs: " << fecarr[0]->Name() << endl;
      }
      else
      {
         fecarr[i] = new H1_FECollection(order = 1, dim);
      }
      fespacearr[i] = new FiniteElementSpace(mesharr[i], fecarr[i]);
   }

   // Determine the list of true (i.e. conforming) essential boundary dofs.
   // In this example, the boundary conditions are defined by marking all
   // the boundary attributes from the mesh as essential (Dirichlet) and
   // converting them to a list of true dofs.
   Array<int> ess_tdof_list1, ess_tdof_list2;
   if (mesharr[0]->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesharr[0]->bdr_attributes.Max());
      ess_bdr = 1;
      fespacearr[0]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list1);
   }

   if (mesharr[1]->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesharr[1]->bdr_attributes.Max());
      ess_bdr = 1;
      fespacearr[1]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list2);
   }

   // Set up the linear form b(.) which corresponds to the right-hand side of
   // the FEM linear system, which in this case is (1,phi_i) where phi_i are
   // the basis functions in the finite element fespace1.
   ConstantCoefficient one(1.0);
   Array<LinearForm*> b_ar(nmeshes);
   for (int i = 0; i< nmeshes; i ++)
   {
      b_ar[i] = new LinearForm(fespacearr[i]);
      b_ar[i]->AddDomainIntegrator(new DomainLFIntegrator(one));
      b_ar[i]->Assemble();
   }

   // Define the solution vector x as a finite element grid function
   // corresponding to fespace1. Initialize x with initial guess of zero,
   // which satisfies the boundary conditions.
   GridFunction x1(fespacearr[0]), x2(fespacearr[1]);
   x1 = 0;
   x2 = 0;

   // Setup FindPointsGSLIB and determine points on each mesh's boundary that
   // are interior to another mesh.
   mesharr[0]->SetCurvature(order, false, dim, Ordering::byNODES);
   Vector mesh_nodes_1 = *mesharr[0]->GetNodes();
   mesharr[1]->SetCurvature(order, false, dim, Ordering::byNODES);
   Vector mesh_nodes_2 = *mesharr[1]->GetNodes();

   // For the default mesh inputs, we need to rescale inline-quad.mesh
   // such that it does not cover the entire domain [0, 1]^2 and still has a
   // non-trivial overlap with the other mesh.
   if (strcmp(mesh_file_1, "../../data/square-disc.mesh") == 0 &&
       strcmp(mesh_file_2, "../../data/inline-quad.mesh") == 0 )
   {
      for (int i = 0; i < mesh_nodes_2.Size(); i++)
      {
         mesh_nodes_2(i) = 0.5 + 0.5*(mesh_nodes_2(i)-0.5);
      }
      mesharr[1]->SetNodes(mesh_nodes_2);
   }

   FindPointsGSLIB finder1, finder2;
   finder1.Setup(*mesharr[0]);
   finder2.Setup(*mesharr[1]);

   Array<int> ess_tdof_list1_int, ess_tdof_list2_int;
   GetInterdomainBoundaryPoints(finder1, finder2, mesh_nodes_1, mesh_nodes_2,
                                ess_tdof_list1, ess_tdof_list2,
                                ess_tdof_list1_int, ess_tdof_list2_int, dim);

   // Use FindPointsGSLIB to interpolate the solution at interdomain boundary
   // points.
   const int number_boundary_1 = ess_tdof_list1_int.Size(),
             number_boundary_2 = ess_tdof_list2_int.Size(),
             number_true_1 = mesh_nodes_1.Size()/dim,
             number_true_2 = mesh_nodes_2.Size()/dim;

   MFEM_VERIFY(number_boundary_1 != 0 ||
               number_boundary_2 != 0, " Please use overlapping grids.");

   Vector bnd1(number_boundary_1*dim);
   for (int i = 0; i < number_boundary_1; i++)
   {
      int idx = ess_tdof_list1_int[i];
      for (int d = 0; d < dim; d++)
      {
         bnd1(i+d*number_boundary_1) = mesh_nodes_1(idx + d*number_true_1);
      }
   }

   Vector bnd2(number_boundary_2*dim);
   for (int i = 0; i < number_boundary_2; i++)
   {
      int idx = ess_tdof_list2_int[i];
      for (int d = 0; d < dim; d++)
      {
         bnd2(i+d*number_boundary_2) = mesh_nodes_2(idx + d*number_true_2);
      }
   }

   Vector interp_vals1(number_boundary_1), interp_vals2(number_boundary_2);
   finder1.Interpolate(bnd2, x1, interp_vals2);
   finder2.Interpolate(bnd1, x2, interp_vals1);

   // Set up the bilinear form a(.,.) on the finite element space corresponding
   // to the Laplacian operator -Delta, by adding a Diffusion integrator.
   Array <BilinearForm*> a_ar(2);
   a_ar[0] = new BilinearForm(fespacearr[0]);
   a_ar[1] = new BilinearForm(fespacearr[1]);
   a_ar[0]->AddDomainIntegrator(new DiffusionIntegrator(one));
   a_ar[1]->AddDomainIntegrator(new DiffusionIntegrator(one));

   // Assemble the bilinear form and the corresponding linear system,
   // applying any necessary transformations such as: eliminating boundary
   // conditions, applying conforming constraints for non-conforming AMR,
   // static condensation, etc.
   a_ar[0]->Assemble();
   a_ar[1]->Assemble();

   delete b_ar[0];
   delete b_ar[1];

   // Use simultaneous Schwarz iterations to iteratively solve the PDE and
   // interpolate interdomain boundary data to impose Dirichlet boundary
   // conditions.

   int NiterSchwarz = 100;
   for (int schwarz = 0; schwarz < NiterSchwarz; schwarz++)
   {
      for (int i = 0; i < nmeshes; i ++)
      {
         b_ar[i] = new LinearForm(fespacearr[i]);
         b_ar[i]->AddDomainIntegrator(new DomainLFIntegrator(one));
         b_ar[i]->Assemble();
      }

      OperatorPtr A1, A2;
      Vector B1, X1, B2, X2;

      a_ar[0]->FormLinearSystem(ess_tdof_list1, x1, *b_ar[0], A1, X1, B1);
      a_ar[1]->FormLinearSystem(ess_tdof_list2, x2, *b_ar[1], A2, X2, B2);

      // Solve the linear system A X = B.
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M1((SparseMatrix&)(*A1));
      PCG(*A1, M1, B1, X1, 0, 200, 1e-12, 0.0);
      GSSmoother M2((SparseMatrix&)(*A2));
      PCG(*A2, M2, B2, X2, 0, 200, 1e-12, 0.0);

      // Recover the solution as a finite element grid function.
      a_ar[0]->RecoverFEMSolution(X1, *b_ar[0], x1);
      a_ar[1]->RecoverFEMSolution(X2, *b_ar[1], x2);

      // Interpolate boundary condition
      finder1.Interpolate(x1, interp_vals2);
      finder2.Interpolate(x2, interp_vals1);

      double dxmax = std::numeric_limits<float>::min();
      double x1inf = x1.Normlinf();
      double x2inf = x2.Normlinf();
      for (int i = 0; i < number_boundary_1; i++)
      {
         int idx = ess_tdof_list1_int[i];
         double dx = std::abs(x1(idx)-interp_vals1(i))/x1inf;
         if (dx > dxmax) { dxmax = dx; }
         x1(idx) = interp_vals1(i);
      }

      for (int i = 0; i < number_boundary_2; i++)
      {
         int idx = ess_tdof_list2_int[i];
         double dx = std::abs(x2(idx)-interp_vals2(i))/x2inf;
         if (dx > dxmax) { dxmax = dx; }
         x2(idx) = interp_vals2(i);
      }

      delete b_ar[0];
      delete b_ar[1];

      std::cout << std::setprecision(8)    <<
                "Iteration: "           << schwarz <<
                ", Relative residual: " << dxmax   << endl;
      if (dxmax < rel_tol) { break; }
   }

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      for (int ip = 0; ip<mesharr.Size(); ++ip)
      {
         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "parallel " << mesharr.Size() << " " << ip << "\n";
         if (ip==0) { sol_sock << "solution\n" << *mesharr[ip] << x1 << flush; }
         if (ip==1) { sol_sock << "solution\n" << *mesharr[ip] << x2 << flush; }
         sol_sock << "window_title 'Overlapping grid solution'\n"
                  << "window_geometry "
                  << 0 << " " << 0 << " " << 450 << " " << 350 << "\n"
                  << "keys jmcA]]]" << endl;
      }
   }

   for (int i = 0; i < nmeshes; i++)
   {
      delete a_ar[i];
      delete fespacearr[i];
      if (order > 0) { delete fecarr[i]; }
      delete mesharr[i];
   }

   return 0;
}

void GetInterdomainBoundaryPoints(FindPointsGSLIB &finder1,
                                  FindPointsGSLIB &finder2,
                                  Vector &mesh_nodes_1, Vector &mesh_nodes_2,
                                  Array<int> ess_tdof_list1,
                                  Array<int> ess_tdof_list2,
                                  Array<int> &ess_tdof_list1_int,
                                  Array<int> &ess_tdof_list2_int,
                                  const int dim)
{
   int number_boundary_1 = ess_tdof_list1.Size(),
       number_boundary_2 = ess_tdof_list2.Size(),
       number_true_1 = mesh_nodes_1.Size()/dim,
       number_true_2 = mesh_nodes_2.Size()/dim;

   Vector bnd1(number_boundary_1*dim);
   for (int i = 0; i < number_boundary_1; i++)
   {
      int idx = ess_tdof_list1[i];
      for (int d = 0; d < dim; d++)
      {
         bnd1(i+d*number_boundary_1) = mesh_nodes_1(idx + d*number_true_1);
      }
   }

   Vector bnd2(number_boundary_2*dim);
   for (int i = 0; i < number_boundary_2; i++)
   {
      int idx = ess_tdof_list2[i];
      for (int d = 0; d < dim; d++)
      {
         bnd2(i+d*number_boundary_2) = mesh_nodes_2(idx + d*number_true_2);
      }
   }

   finder1.FindPoints(bnd2);
   finder2.FindPoints(bnd1);

   const Array<unsigned int> &code_out1 = finder1.GetCode();
   const Array<unsigned int> &code_out2 = finder2.GetCode();

   // Setup ess_tdof_list_int
   for (int i = 0; i < number_boundary_1; i++)
   {
      int idx = ess_tdof_list1[i];
      if (code_out2[i] != 2) { ess_tdof_list1_int.Append(idx); }
   }

   for (int i = 0; i < number_boundary_2; i++)
   {
      int idx = ess_tdof_list2[i];
      if (code_out1[i] != 2) { ess_tdof_list2_int.Append(idx); }
   }
}
