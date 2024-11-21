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

#include "mfem.hpp"
#include "unit_tests.hpp"

namespace mfem
{
#ifdef MFEM_USE_MPI

TEST_CASE("ParMeshGlobalIndices",  "[Parallel], [ParMesh]")
{
   const int ne = 5;

   for (int dimension = 1; dimension < 4; ++dimension)
   {
      for (int amr=0; amr < 1 + (dimension > 1); ++amr)
      {
         Mesh mesh;
         if (dimension == 1)
         {
            mesh = Mesh::MakeCartesian1D(ne, 1.0);
         }
         else if (dimension == 2)
         {
            if (amr)
            {
               const char *mesh_file = "../../data/amr-quad.mesh";
               mesh = Mesh::LoadFromFile(mesh_file, 1, 1);
            }
            else
            {
               mesh = Mesh::MakeCartesian2D(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
            }
         }
         else
         {
            if (amr)
            {
               const char *mesh_file = "../../data/amr-hex.mesh";
               mesh = Mesh::LoadFromFile(mesh_file, 1, 1);
            }
            else
            {
               mesh = Mesh::MakeCartesian3D(ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
            }
         }


         ParMesh pmesh(MPI_COMM_WORLD, mesh);

         int globalN = 0;

         enum EntityType { VERTEX, EDGE, FACE, ELEMENT };
         // Loop over all types of mesh entities
         for (int e=EntityType::VERTEX; e<=EntityType::ELEMENT; ++e)
         {
            if (amr && dimension > 1 && e != EntityType::ELEMENT)
            {
               continue;
            }

            Array<HYPRE_BigInt> gi;

            switch (e)
            {
               case EntityType::VERTEX:
                  globalN = mesh.GetNV();
                  pmesh.GetGlobalVertexIndices(gi);
                  break;
               case EntityType::EDGE:
                  globalN = dimension == 1 ? mesh.GetNV() : mesh.GetNEdges();
                  pmesh.GetGlobalEdgeIndices(gi);
                  break;
               case EntityType::FACE:
                  globalN = mesh.GetNumFaces();
                  pmesh.GetGlobalFaceIndices(gi);
                  break;
               case EntityType::ELEMENT:
                  globalN = mesh.GetNE();
                  pmesh.GetGlobalElementIndices(gi);
                  break;
            }

            // Verify that the local entities do not share a global index.
            {
               std::set<HYPRE_BigInt> localGI;
               for (int i=0; i<gi.Size(); ++i)
               {
                  localGI.insert(gi[i]);
               }

               REQUIRE(localGI.size() == (std::size_t) gi.Size());
            }

            // Verify that the global indices range from 0 to globalN-1.
            {
               const HYPRE_BigInt localMin = gi.Size() > 0 ? gi.Min() :
                                             std::numeric_limits<HYPRE_BigInt>::max();
               const HYPRE_BigInt localMax = gi.Size() > 0 ? gi.Max() :
                                             std::numeric_limits<HYPRE_BigInt>::min();

               HYPRE_BigInt globalMin, globalMax;
               MPI_Allreduce(&localMin, &globalMin, 1, HYPRE_MPI_BIG_INT, MPI_MIN,
                             MPI_COMM_WORLD);
               MPI_Allreduce(&localMax, &globalMax, 1, HYPRE_MPI_BIG_INT, MPI_MAX,
                             MPI_COMM_WORLD);

               REQUIRE((globalMin == 0 && globalMax == globalN-1));
            }
         }
      }
   }
}

namespace simplicial
{

double exact(const Vector &xvec)
{
   // The exact solution is linear and is harmonic
   return xvec[0] + xvec[1] + xvec[2];
}

void SolveDiffusionProblem(ParMesh &mesh, Vector &x_out)
{
   H1_FECollection fec(1, mesh.Dimension());
   ParFiniteElementSpace fes(&mesh, &fec);

   // Right-hand side is zero since exact solution is harmonic
   ParLinearForm b(&fes);
   b.Assemble();

   ParBilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   Array<int> ess_tdof_list, ess_bdr;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // Use the exact solution as boundary conditions
   ParGridFunction x(&fes);
   FunctionCoefficient exact_coeff(exact);
   x.ProjectBdrCoefficient(exact_coeff, ess_bdr);

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   // X = 0.0;
   cg.Mult(B, X);
   x_out = X;
}

}

TEST_CASE("ParMeshMakeSimplicial", "[Parallel], [ParMesh]")
{
   // Test that the parallel mesh obtained by ParMesh::MakeSimplicial is valid.
   // This test solves a Poisson problem on a 3x3x3 hex mesh, and on the tet
   // mesh obtained by splitting the hexes into tets. The finite element space
   // is linear in both cases, and the exact solution is also linear, so it will
   // be recovered exactly in both cases. The vertices of both meshes are the
   // same, so we check that the resulting discrete solutions are identical up
   // to solver tolerance.

   Mesh mesh = Mesh::MakeCartesian3D(3, 3, 3, Element::HEXAHEDRON);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   ParMesh pmesh_tet = ParMesh::MakeSimplicial(pmesh);

   Vector x, x_tet;
   simplicial::SolveDiffusionProblem(pmesh, x);
   simplicial::SolveDiffusionProblem(pmesh_tet, x_tet);

   x -= x_tet;
   REQUIRE(x.Normlinf() == MFEM_Approx(0.0));
}

#endif // MFEM_USE_MPI

} // namespace mfem
