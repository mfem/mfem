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

   for (int dimension = 2; dimension < 4; ++dimension)
   {
      Mesh* mesh;
      if (dimension == 2)
      {
         mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
      }
      else
      {
         mesh = new Mesh(ne, ne, ne, Element::HEXAHEDRON, 1, 1.0, 1.0,
                         1.0);
      }

      ParMesh pmesh(MPI_COMM_WORLD, *mesh);

      int globalN = 0;

      enum EntityType { VERTEX, EDGE, FACE, ELEMENT };
      // Loop over all types of mesh entities
      for (int e=EntityType::VERTEX; e<=EntityType::ELEMENT; ++e)
      {
         if (e == EntityType::FACE && dimension == 2)
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
               globalN = mesh->GetNEdges();
               pmesh.GetGlobalEdgeIndices(gi);
               break;
            case EntityType::FACE:
               globalN = mesh->GetNFaces();
               pmesh.GetGlobalFaceIndices(gi);
               break;
            case EntityType::ELEMENT:
               globalN = mesh->GetNE();
               pmesh.GetGlobalElementIndices(gi);
               break;
         }

         // Verify that global indices are unique on each process.
         {
            std::set<HYPRE_Int> localGI;
            for (int i=0; i<gi.Size(); ++i)
            {
               localGI.insert(gi[i]);
            }

            REQUIRE(localGI.size() == gi.Size());
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

#endif // MFEM_USE_MPI

} // namespace mfem
