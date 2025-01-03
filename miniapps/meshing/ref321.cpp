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
//      -----------------------------------------------------------------
//      3:1 Refinement Miniapp:  Perform 3:1 anisotropic mesh refinements
//      -----------------------------------------------------------------
//
// This miniapp performs random 3:1 refinements of a quadrilateral or hexahedral
// mesh. A diffusion equation is solved in an H1 finite element space defined on
// the refined mesh, and its continuity is verified.
//
// Compile with: make ref321
//
// Sample runs:  ref321 -mm -dim 2 -o 2 -r 100
//               ref321 -mm -dim 3 -o 2 -r 100
//               ref321 -m ../../data/star.mesh -o 2 -r 100

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

real_t CheckH1Continuity(GridFunction & x);

// Find the two children of parent element `elem` after its refinement in one
// direction.
void FindChildren(const Mesh & mesh, int elem, Array<int> & children)
{
   const CoarseFineTransformations& cf = mesh.ncmesh->GetRefinementTransforms();
   MFEM_ASSERT(mesh.GetNE() == cf.embeddings.Size(), "");

   // Note that row `elem` of the table constructed by cf.MakeCoarseToFineTable
   // is an alternative to this global loop, but constructing the table is also
   // a global operation with global storage.
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      const int p = cf.embeddings[i].parent;
      if (p == elem)
      {
         children.Append(i);
      }
   }
}

// Refine 3:1 via 2 refinements with scalings 2/3 and 1/2.
void Refine31(Mesh & mesh, int elem, char type)
{
   Array<Refinement> refs; // Refinement is defined in ncmesh.hpp
   refs.Append(Refinement(elem, type, 2.0 / 3.0));
   mesh.GeneralRefinement(refs);

   // Find the elements with parent `elem`
   Array<int> children;
   FindChildren(mesh, elem, children);
   MFEM_ASSERT(children.Size() == 2, "");

   const int elem1 = children[0];

   refs.SetSize(0);
   refs.Append(Refinement(elem1, type)); // Default scaling of 0.5
   mesh.GeneralRefinement(refs);
}

// Deterministic, somewhat random integer generator
int MyRand(int & s)
{
   s++;
   const double a = 1000 * sin(s * 1.1234 * M_PI);
   return int(std::abs(a));
}

// Randomly select elements for 3:1 refinements in random directions.
void TestAnisoRefRandom(int iter, int dim, Mesh & mesh)
{
   int seed = 0;
   for (int i = 0; i < iter; i++)
   {
      const int elem = MyRand(seed) % mesh.GetNE();
      const int t = MyRand(seed) % dim;
      auto type = t == 0 ? Refinement::X :
                  (t == 1 ? Refinement::Y : Refinement::Z);
      Refine31(mesh, elem, type);
   }

   mesh.EnsureNodes();
   mesh.SetScaledNCMesh();
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool visualization = true;
   bool makeMesh = false;
   int num_refs = 1;
   int tdim = 2;  // Mesh dimension

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&makeMesh, "-mm", "--make-mesh", "-no-mm",
                  "--no-make-mesh", "Create Cartesian mesh");
   args.AddOption(&tdim, "-dim", "--dimension", "Dimension for Cartesian mesh");
   args.AddOption(&num_refs, "-r", "--refs", "Number of 3:1 refinements");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Create or read the mesh from the given mesh file.
   Mesh mesh;
   if (makeMesh)
   {
      mesh = tdim == 3 ? Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON) :
             Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   }
   else
   {
      mesh = Mesh::LoadFromFile(mesh_file, 1, 1);
   }

   const int dim = mesh.Dimension();

   // 3. Randomly perform 3:1 refinements in the mesh.
   TestAnisoRefRandom(num_refs, tdim, mesh);

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 5. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Solve the Laplace problem, as in ex1.
   GridFunction x(&fespace);

   {
      x = 0.0;
      LinearForm b(&fespace);
      ConstantCoefficient one(1.0);
      b.AddDomainIntegrator(new DomainLFIntegrator(one));
      b.Assemble();

      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator());
      a.Assemble();

      OperatorPtr A;
      Vector B, X;
      Array<int> ess_tdof_list;
      if (mesh.bdr_attributes.Size())
      {
         Array<int> ess_bdr(mesh.bdr_attributes.Max());
         ess_bdr = 1;
         fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 2000, 1e-12, 0.0);
      a.RecoverFEMSolution(X, b, x);
   }

   // 6. Verify the continuity of the projected function in H1.
   const real_t h1err = CheckH1Continuity(x);
   cout << "Error of H1 continuity: " << h1err << endl;
   MFEM_VERIFY(h1err < 1.0e-8, "");

   // 7. Save the refined mesh and the solution. This output can be viewed later
   //    using GLVis: "glvis -m ref321.mesh -g sol.gf".
   ofstream mesh_ofs("ref321.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 8. Send the solution by socket to a GLVis server.
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

real_t CheckH1Continuity(GridFunction & x)
{
   const FiniteElementSpace *fes = x.FESpace();
   Mesh *mesh = fes->GetMesh();

   const int dim = mesh->Dimension();

   // Following the example of KellyErrorEstimator::ComputeEstimates(), we loop
   // over interior faces and then shared faces.

   // Compute error contribution from local interior faces
   real_t errorMax = 0.0;
   for (int f = 0; f < mesh->GetNumFaces(); f++)
   {
      if (mesh->FaceIsInterior(f))
      {
         int Inf1, Inf2, NCFace;
         mesh->GetFaceInfos(f, &Inf1, &Inf2, &NCFace);

         auto FT = mesh->GetFaceElementTransformations(f);

         const int faceOrder = dim == 3 ? fes->GetFaceOrder(f) :
                               fes->GetEdgeOrder(f);
         auto &int_rule = IntRules.Get(FT->FaceGeom, 2 * faceOrder);
         const auto nip = int_rule.GetNPoints();

         // Convention:
         // * Conforming face: Face side with smaller element id handles the
         //   integration
         // * Non-conforming face: The slave handles the integration.
         // See FaceInfo documentation for details.
         bool isNCSlave    = FT->Elem2No >= 0 && NCFace >= 0;
         bool isConforming = FT->Elem2No >= 0 && NCFace == -1;
         if ((FT->Elem1No < FT->Elem2No && isConforming) || isNCSlave)
         {
            for (int i = 0; i < nip; i++)
            {
               const auto &fip = int_rule.IntPoint(i);
               IntegrationPoint ip;

               FT->Loc1.Transform(fip, ip);
               const real_t v1 = x.GetValue(FT->Elem1No, ip);

               FT->Loc2.Transform(fip, ip);
               const real_t v2 = x.GetValue(FT->Elem2No, ip);

               const real_t err_i = std::abs(v1 - v2);
               errorMax = std::max(errorMax, err_i);
            }
         }
      }
   }

   return errorMax;
}
