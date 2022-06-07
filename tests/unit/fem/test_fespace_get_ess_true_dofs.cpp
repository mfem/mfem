// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

using namespace mfem;

TEST_CASE("FESpace Get Essential True DOFs",
          "[FESpace Get Essential True DOFs]")
{
   std::cout << "Testing get essential true dofs" << std::endl;
   int order_h1 = 3, n = 2, dim = 3;

   Mesh mesh = Mesh::MakeCartesian3D(
                  n, n, n, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   mesh.SetCurvature(order_h1);

   H1_FECollection fec(order_h1, dim);
   FiniteElementSpace fe_space(&mesh, &fec, dim);


   const int num_bdr_attr = fe_space.GetMesh()->bdr_attributes.Max();


   Array<int> ess_tdofs_2d, ess_tdofs_1d, ess_tdofs_tmp, ess_bdrs;
   Array2D<bool> comps(num_bdr_attr, dim);
   ess_bdrs.SetSize(num_bdr_attr);
   comps = false;
   ess_bdrs = 0;

   // simple xy boundary condition on all surfaces
   // could do something more complex but don't really want to...
   for (int i = 0; i < num_bdr_attr; i++)
   {
      ess_bdrs[i] = 1;
      comps(i, 0) = true;
      comps(i, 2) = true;
   }

   fe_space.GetEssentialTrueDofs(ess_bdrs, ess_tdofs_2d, comps);

   // Now for the old way
   fe_space.GetEssentialTrueDofs(ess_bdrs, ess_tdofs_tmp, 0);
   ess_tdofs_1d.Append(ess_tdofs_tmp);
   ess_tdofs_tmp.DeleteAll();
   fe_space.GetEssentialTrueDofs(ess_bdrs, ess_tdofs_tmp, 2);
   ess_tdofs_1d.Append(ess_tdofs_tmp);
   // Sort the 2 arrays in order to compare them
   ess_tdofs_2d.Sort();
   ess_tdofs_1d.Sort();

   int diff = 0;

   for (int i = 0; i < ess_tdofs_2d.Size(); i++)
   {
      diff += std::abs(ess_tdofs_2d[i] - ess_tdofs_1d[i]);
   }

   std::cout << "Difference in essential tdofs approaches is: " << diff <<
             std::endl;

   REQUIRE(diff == 0);

}

