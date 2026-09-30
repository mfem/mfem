// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

#include <type_traits>

using namespace mfem;

namespace
{

void CheckPackedMassEA(const char *mesh_file, const int order)
{
   Mesh mesh(mesh_file);
   const int dim = mesh.Dimension();
   const int ne = mesh.GetNE();

   L2_FECollection fec(order, dim, BasisType::Positive);
   FiniteElementSpace fes(&mesh, &fec);

   MFEM_VERIFY(UsesTensorBasis(fes),
               "This test requires a tensor-product finite element space.");

   const FiniteElement *fe0 = fes.GetFE(0);
   MFEM_VERIFY(fe0 != nullptr, "");
   const int ndof = fe0->GetDof();

   const TensorBasisElement *tbe =
      dynamic_cast<const TensorBasisElement*>(fe0);
   MFEM_VERIFY(tbe, "");
   const Array<int> &dof_map = tbe->GetDofMap();
   const bool has_dof_map = (dof_map.Size() == ndof);

   MassIntegrator mass;

   TriPackLowerMatrix packed;
   mass.AssembleEATriangular(fes, packed, false);

   REQUIRE(packed.GetNumRows() == ndof);
   REQUIRE(packed.GetNumMatrices() == ne);

   TriPackLowerMatrix packed_add = packed;
   mass.AssembleEATriangular(fes, packed_add, true);

   const double tol = std::is_same<real_t, float>::value ? 1e-5 : 1e-12;

   {
      const real_t *a = packed.Data().HostRead();
      const real_t *b = packed_add.Data().HostRead();
      for (int i = 0; i < packed.Size(); ++i)
      {
         REQUIRE((double)b[i] == MFEM_Approx(2.0*(double)a[i], tol, tol));
      }
   }

   DenseMatrix elmat;
   const real_t *packed_data = packed.Data().HostRead();
   const int packed_size = packed.GetPackedSize();

   for (int e = 0; e < ne; ++e)
   {
      const FiniteElement &el = *fes.GetFE(e);
      ElementTransformation &T = *mesh.GetElementTransformation(e);
      mass.AssembleElementMatrix(el, T, elmat);

      const real_t *pe = packed_data + e*packed_size;
      for (int i = 0; i < ndof; ++i)
      {
         const int ii_s = has_dof_map ? dof_map[i] : i;
         const int ii = ii_s >= 0 ? ii_s : -1 - ii_s;
         const int s_i = ii_s >= 0 ? 1 : -1;
         for (int j = 0; j <= i; ++j)
         {
            const int jj_s = has_dof_map ? dof_map[j] : j;
            const int jj = jj_s >= 0 ? jj_s : -1 - jj_s;
            const int s_j = jj_s >= 0 ? 1 : -1;
            const real_t val =
               s_i*s_j*pe[TriPackLowerMatrix::LowerIndex(i, j, ndof)];

            elmat(ii, jj) -= val;
            if (i != j) { elmat(jj, ii) -= val; }
         }
      }

      REQUIRE(elmat.MaxMaxNorm() == MFEM_Approx(0.0, 100*tol));
   }
}

} // namespace

TEST_CASE("MassIntegrator packed triangular EA matches element assembly",
          "[AssembleEA][Mass][TriPackLowerMatrix]")
{
   const auto mesh_file = "../../data/inline-quad.mesh";
   const int order = GENERATE(2, 3);

   CAPTURE(mesh_file, order);

   CheckPackedMassEA(mesh_file, order);
}
