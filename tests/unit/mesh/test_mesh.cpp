// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "mfem.hpp"
using namespace mfem;

#include "catch.hpp"

#ifdef MFEM_USE_GECKO

TEST_CASE("Gecko integration in MFEM", "[Mesh]")
{
   Array<int> perm;

   SECTION("Permutation from Gecko is valid")
   {
      Array<bool> elem_covered;

      SECTION("Hex meshes")
      {
         Mesh mesh(3, 4, 5, Element::HEXAHEDRON);
         mesh.GetGeckoElementReordering(perm);
         REQUIRE(perm.Size() == mesh.GetNE());
         REQUIRE(perm.Min() == 0);
         REQUIRE(perm.Max() == mesh.GetNE() - 1);

         elem_covered.SetSize(perm.Size(), false);
         for (int i = 0; i < perm.Size(); ++i)
         {
            elem_covered[perm[i]] = true;
         }

         bool all_elems_covered = true;
         for (int i = 0; i < perm.Size(); ++i)
         {
            all_elems_covered &= elem_covered[i];
         }
         REQUIRE(all_elems_covered);
      }

      SECTION("Tet meshes")
      {
         Mesh mesh(5, 4, 3, Element::TETRAHEDRON);
         mesh.GetGeckoElementReordering(perm);
         REQUIRE(perm.Size() == mesh.GetNE());
         REQUIRE(perm.Min() == 0);
         REQUIRE(perm.Max() == mesh.GetNE() - 1);

         elem_covered.SetSize(perm.Size(), false);
         for (int i = 0; i < perm.Size(); ++i)
         {
            elem_covered[perm[i]] = true;
         }

         bool all_elems_covered = true;
         for (int i = 0; i < perm.Size(); ++i)
         {
            all_elems_covered &= elem_covered[i];
         }
         REQUIRE(all_elems_covered);
      }
   }

   SECTION("Reorder preserves physical vertex locations")
   {
      Mesh mesh(3, 4, 5, Element::HEXAHEDRON);
      Mesh mesh_reordered(3, 4, 5, Element::HEXAHEDRON);
      mesh_reordered.GetGeckoElementReordering(perm);
      mesh_reordered.ReorderElements(perm);

      for (int old_elid = 0; old_elid < perm.Size(); ++old_elid)
      {
         int new_elid = perm[old_elid];
         Array<int> old_dofs, new_dofs;
         mesh.GetElementVertices(old_elid, old_dofs);
         mesh_reordered.GetElementVertices(new_elid, new_dofs);
         for (int dofi = 0; dofi < old_dofs.Size(); ++dofi)
         {
            for (int d = 0; d < 3; ++d)
            {
               REQUIRE(mesh.GetVertex(old_dofs[dofi])[d] == mesh_reordered.GetVertex(
                          new_dofs[dofi])[d]);
            }
         }
      }
   }
}

#endif
