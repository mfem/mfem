// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
         Mesh* mesh;
         if (dimension == 1)
         {
            mesh = new Mesh(ne, 1.0);
         }
         else if (dimension == 2)
         {
            if (amr)
            {
               const char *mesh_file = "../../data/amr-quad.mesh";
               mesh = new Mesh(mesh_file, 1, 1);
            }
            else
            {
               mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
            }
         }
         else
         {
            if (amr)
            {
               const char *mesh_file = "../../data/amr-hex.mesh";
               mesh = new Mesh(mesh_file, 1, 1);
            }
            else
            {
               mesh = new Mesh(ne, ne, ne, Element::HEXAHEDRON, 1, 1.0, 1.0, 1.0);
            }
         }

         ParMesh pmesh(MPI_COMM_WORLD, *mesh);

         int globalN = 0;

         enum EntityType { VERTEX, EDGE, FACE, ELEMENT };
         // Loop over all types of mesh entities
         for (int e=EntityType::VERTEX; e<=EntityType::ELEMENT; ++e)
         {
            if (amr && dimension > 1 && e != EntityType::ELEMENT)
            {
               continue;
            }

            Array<HYPRE_Int> gi;

            switch (e)
            {
               case EntityType::VERTEX:
                  globalN = mesh->GetNV();
                  pmesh.GetGlobalVertexIndices(gi);
                  break;
               case EntityType::EDGE:
                  globalN = dimension == 1 ? mesh->GetNV() : mesh->GetNEdges();
                  pmesh.GetGlobalEdgeIndices(gi);
                  break;
               case EntityType::FACE:
                  globalN = mesh->GetNumFaces();
                  pmesh.GetGlobalFaceIndices(gi);
                  break;
               case EntityType::ELEMENT:
                  globalN = mesh->GetNE();
                  pmesh.GetGlobalElementIndices(gi);
                  break;
            }

            // Verify that the local entities do not share a global index.
            {
               std::set<HYPRE_Int> localGI;
               for (int i=0; i<gi.Size(); ++i)
               {
                  localGI.insert(gi[i]);
               }

               REQUIRE(localGI.size() == (std::size_t) gi.Size());
            }

            // Verify that the global indices range from 0 to globalN-1.
            {
               const HYPRE_Int localMin = gi.Min();
               const HYPRE_Int localMax = gi.Max();

               HYPRE_Int globalMin, globalMax;
               MPI_Allreduce(&localMin, &globalMin, 1, HYPRE_MPI_INT, MPI_MIN, MPI_COMM_WORLD);
               MPI_Allreduce(&localMax, &globalMax, 1, HYPRE_MPI_INT, MPI_MAX, MPI_COMM_WORLD);

               REQUIRE((globalMin == 0 && globalMax == globalN-1));
            }
         }

         delete mesh;
      }
   }
}

#endif // MFEM_USE_MPI

} // namespace mfem
