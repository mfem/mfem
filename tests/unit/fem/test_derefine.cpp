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

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;

namespace derefine
{

int dimension;
double coeff(const Vector& x)
{
   if (dimension == 2)
   {
      return sin(10.0*(x[0]+x[1]));
   }
   else
   {
      return sin(10.0*(x[0]+x[1]+x[2]));
   }
}

TEST_CASE("Derefine")
{
   for (dimension = 2; dimension <= 3; ++dimension)
   {
      for (int order = 0; order <= 2; ++order)
      {
         Mesh *mesh = nullptr;
         const int ne = 8;
         if (dimension == 2)
         {
            mesh = new Mesh(ne, ne, Element::QUADRILATERAL, true, 1.0, 1.0);
         }
         else
         {
            mesh = new Mesh(ne, ne, ne, Element::HEXAHEDRON, true, 1.0, 1.0, 1.0);
         }

         mesh->EnsureNCMesh();
         mesh->SetCurvature(2, false, 2, Ordering::byNODES);

         L2_FECollection fec(order, dimension, BasisType::Positive);
         FiniteElementSpace fespace(mesh, &fec);

         GridFunction x(&fespace);

         FunctionCoefficient c(coeff);
         x.ProjectCoefficient(c);

         fespace.Update();
         x.Update();

         Array<Refinement> refinements;
         refinements.Append(mfem::Refinement(1));
         refinements.Append(mfem::Refinement(2));

         int nonconformity_limit = 0; // 0 meaning allow unlimited ratio

         // First refine two elements.
         mesh->GeneralRefinement(refinements, 1, nonconformity_limit);

         fespace.Update();
         x.Update();

         // Now refine one more element and then derefine it, comparing x before and after.
         Vector diff(x);

         refinements.DeleteAll();
         refinements.Append(mfem::Refinement(2));
         //refinements.Append(mfem::Refinement(1));
         mesh->GeneralRefinement(refinements, 1, nonconformity_limit);

         fespace.Update();
         x.Update();

         // Derefine by setting 0 error on the fine elements in coarse element 2.
         Table coarse_to_fine_;
         Table ref_type_to_matrix;
         Array<int> coarse_to_ref_type;
         Array<Geometry::Type> ref_type_to_geom;
         const CoarseFineTransformations &rtrans = mesh->GetRefinementTransforms();
         rtrans.GetCoarseToFineMap(*mesh, coarse_to_fine_, coarse_to_ref_type,
                                   ref_type_to_matrix, ref_type_to_geom);
         Array<int> tabrow;

         Vector local_err(mesh->GetNE());
         double threshold = 1.;
         local_err = 2*threshold;
         coarse_to_fine_.GetRow(2, tabrow);
         for (int j = 0; j < tabrow.Size(); j++) { local_err(tabrow[j]) = 0.; }
         bool derefs = mesh->DerefineByError(local_err, threshold, 0, 1);

         fespace.Update();
         x.Update();

         diff -= x;
         REQUIRE(diff.Norml2() / x.Norml2() < 1e-11);

         delete mesh;
      }
   }
}

#ifdef MFEM_USE_MPI

TEST_CASE("ParDerefine", "[Parallel]")
{

}

#endif
}
