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
//      -----------------------------------------------------------------
//      3:1 Refinement Miniapp: Parallel 3:1 anisotropic mesh refinements
//      -----------------------------------------------------------------
//
// This miniapp performs random 3:1 refinements of a quadrilateral or hexahedral
// mesh. A diffusion equation is solved in an H1 finite element space defined on
// the refined mesh, and its continuity is verified across local and shared
// faces.
//
// Compile with: make pref321
//
// Sample runs:  mpirun -np 4 pref321 -mm -dim 2 -o 2 -r 100
//               mpirun -np 4 pref321 -mm -dim 3 -o 2 -r 100
//               mpirun -np 4 pref321 -m ../../data/star.mesh -o 2 -r 100

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

real_t CheckH1Continuity(ParGridFunction &x);

// Find the two children of parent element `elem` after its refinement in one
// direction.
void FindChildren(const Mesh &mesh, int elem, Array<int> &children)
{
   const CoarseFineTransformations &cf = mesh.ncmesh->GetRefinementTransforms();
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
void Refine31(Mesh &mesh, int elem, char type)
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

// Deterministic, somewhat random integer generator.
int MyRand(int &s)
{
   s++;
   const double a = 1000 * sin(s * 1.1234 * M_PI);
   return int(std::abs(a));
}

// Randomly select elements for 3:1 refinements in random directions.
void TestAnisoRefRandom(int iter, int dim, ParMesh &mesh, int myid,
                        int seed = 0)
{
   for (int i = 0; i < iter; i++)
   {
      const int elem = MyRand(seed) % mesh.GetNE();
      const int t = MyRand(seed) % dim;
      auto type = t == 0 ? Refinement::X :
                  (t == 1 ? Refinement::Y : Refinement::Z);

      // In 3D, check for conflicts in the parallel refinements.
      if (dim == 3)
      {
         std::set<int> conflicts; // Indices in refs of conflicting elements
         Array<Refinement> refs;
         refs.Append(Refinement(elem, type));
         const bool conflict = mesh.AnisotropicConflict(refs, conflicts);
         if (conflict)
         {
            if (myid == 0)
               cout << "Anisotropic conflict on iteration " << i
                    << ", retrying\n";
            i--;
            continue;
         }
      }

      Refine31(mesh, elem, type);
   }

   mesh.EnsureNodes();
   mesh.SetScaledNCMesh();
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const int num_procs = Mpi::WorldSize();
   const int myid = Mpi::WorldRank();

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool visualization = true;
   bool makeMesh = false;
   int num_refs = 1;
   int tdim = 2;  // Mesh dimension for Cartesian meshes.

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
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
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 2. Create or read the serial mesh on all ranks, then apply the same
   //    deterministic 3:1 refinement sequence before partitioning it.
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

   mesh.EnsureNCMesh();
   mesh.SetScaledNCMesh();

   // 3. Partition the refined serial mesh.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   TestAnisoRefRandom(num_refs, tdim, pmesh, myid, myid);

   // 4. Define a parallel H1 finite element space and report its global size.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: "
           << fespace.GlobalTrueVSize() << endl;
   }

   // 5. Assemble and solve the Poisson problem, following ex1p.
   ParGridFunction x(&fespace);
   x = 0.0;

   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator());
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 0;
      pmesh.MarkExternalBoundaries(ess_bdr);
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   HypreBoomerAMG M;
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetPreconditioner(M);
   cg.SetOperator(*A);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.Mult(B, X);

   a.RecoverFEMSolution(X, b, x);

   // 6. Verify the continuity of the solution in H1 over local and shared
   //    faces and compute the global maximum jump.
   const real_t h1err = CheckH1Continuity(x);
   if (myid == 0)
   {
      cout << "Error of H1 continuity: " << h1err << endl;
   }
   MFEM_VERIFY(h1err < 1.0e-7, "");

   // 7. Save the refined mesh and the solution in parallel. This output can
   //    be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 8. Send the parallel solution to GLVis.
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   return 0;
}

real_t CheckH1Continuity(ParGridFunction &x)
{
   const ParFiniteElementSpace *pfes = x.ParFESpace();
   ParMesh *pmesh = pfes->GetParMesh();
   const int dim = pmesh->Dimension();

   real_t errorMax = 0.0;

   // Shared-face values require face-neighbor data.
   x.ExchangeFaceNbrData();

   // First handle faces for which both elements are local to this rank.
   for (int f = 0; f < pmesh->GetNumFaces(); f++)
   {
      const auto info = pmesh->GetFaceInformation(f);
      if (!info.IsLocal())
      {
         continue;
      }

      FaceElementTransformations *FT = pmesh->GetFaceElementTransformations(f);
      const int faceOrder = dim == 3 ? pfes->GetFaceOrder(f) :
                            pfes->GetEdgeOrder(f);
      const IntegrationRule &ir = IntRules.Get(FT->FaceGeom, 2 * faceOrder);

      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &fip = ir.IntPoint(i);
         IntegrationPoint ip1, ip2;

         FT->Loc1.Transform(fip, ip1);
         FT->Loc2.Transform(fip, ip2);

         const real_t v1 = x.GetValue(*FT->Elem1, ip1);
         const real_t v2 = x.GetValue(*FT->Elem2, ip2);
         errorMax = std::max(errorMax, std::abs(v1 - v2));
      }
   }

   // Then check partition interfaces. Conforming shared faces are handled on
   // the lower-rank side, while shared slave nonconforming faces are handled
   // only on the slave side and therefore do not need additional filtering.
   for (int sf = 0; sf < pmesh->GetNSharedFaces(); sf++)
   {
      const int f = pmesh->GetSharedFace(sf);
      const auto info = pmesh->GetFaceInformation(f);
      if (!info.IsShared())
      {
         continue;
      }

      FaceElementTransformations *FT = pmesh->GetSharedFaceTransformations(sf);
      const int faceOrder = dim == 3 ? pfes->GetFaceOrder(f) :
                            pfes->GetEdgeOrder(f);
      const IntegrationRule &ir = IntRules.Get(FT->FaceGeom, 2 * faceOrder);

      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &fip = ir.IntPoint(i);
         IntegrationPoint ip1, ip2;

         FT->Loc1.Transform(fip, ip1);
         FT->Loc2.Transform(fip, ip2);

         const real_t v1 = x.GetValue(*FT->Elem1, ip1);
         const real_t v2 = x.GetValue(*FT->Elem2, ip2);
         errorMax = std::max(errorMax, std::abs(v1 - v2));
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &errorMax, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, pmesh->GetComm());

   return errorMax;
}
