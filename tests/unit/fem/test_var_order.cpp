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

// Check basic functioning of variable order spaces plus some corner cases.
//
TEST_CASE("Variable Order FiniteElementSpace",
          "[FiniteElementCollection]"
          "[FiniteElementSpace]"
          "[NCMesh]")
{
   SECTION("Quad mesh basics")
   {
      // 2-element quad mesh
      Mesh mesh(2, 1, Element::QUADRILATERAL);
      mesh.EnsureNCMesh();

      // standard H1 conforming space (order 1 elements)
      H1_FECollection fec(1, mesh.Dimension());
      FiniteElementSpace fespace(&mesh, &fec);

      REQUIRE(fespace.GetNDofs() == 6);
      REQUIRE(fespace.GetNConformingDofs() == 6);

      // convert to variable order space: p-refine second element
      fespace.SetElementOrder(1, 2);
      fespace.Update(false);

      REQUIRE(fespace.GetNDofs() == 11);
      REQUIRE(fespace.GetNConformingDofs() == 10);

      // h-refine first element in the y axis
      Array<Refinement> refs;
      refs.Append(Refinement(0, 2));
      mesh.GeneralRefinement(refs);

      fespace.Update(true);

      REQUIRE(fespace.GetNDofs() == 13);
      REQUIRE(fespace.GetNConformingDofs() == 11);

      // relax the master edge to be quadratic
      fespace.SetRelaxedHpConformity(true);

      REQUIRE(fespace.GetNDofs() == 13);
      REQUIRE(fespace.GetNConformingDofs() == 12);

      // increase order

      // refine

      // relaxed off

   }

   // TODO: hex mesh

   // TODO: hex mesh with min order propagation

   // TODO: prism mesh with edge-face constraint
}



}
