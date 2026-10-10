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

void CheckMassEA(const char *mesh_file, const int order)
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

   Vector ea(ne*ndof*ndof);
   ea.UseDevice(true);
   mass.AssembleEA(fes, ea, false);
   MFEM_DEVICE_SYNC;

   Vector ea_add = ea;
   ea_add.UseDevice(true);
   mass.AssembleEA(fes, ea_add, true);
   MFEM_DEVICE_SYNC;

   const double tol = std::is_same<real_t, float>::value ? 1e-5 : 1e-12;

   {
      const real_t *a = ea.HostRead();
      const real_t *b = ea_add.HostRead();
      for (int i = 0; i < ea.Size(); ++i)
      {
         REQUIRE((double)b[i] == MFEM_Approx(2.0*(double)a[i], tol, tol));
      }
   }

   const auto ea_mats = Reshape(ea.HostRead(), ndof, ndof, ne);

   DenseMatrix elmat;
   for (int e = 0; e < ne; ++e)
   {
      const FiniteElement &el = *fes.GetFE(e);
      ElementTransformation &T = *mesh.GetElementTransformation(e);
      mass.AssembleElementMatrix(el, T, elmat);

      for (int i = 0; i < ndof; ++i)
      {
         const int ii_s = has_dof_map ? dof_map[i] : i;
         const int ii = ii_s >= 0 ? ii_s : -1 - ii_s;
         const int s_i = ii_s >= 0 ? 1 : -1;
         for (int j = 0; j < ndof; ++j)
         {
            const int jj_s = has_dof_map ? dof_map[j] : j;
            const int jj = jj_s >= 0 ? jj_s : -1 - jj_s;
            const int s_j = jj_s >= 0 ? 1 : -1;
            elmat(ii, jj) -= s_i*s_j*ea_mats(i, j, e);
         }
      }

      REQUIRE(elmat.MaxMaxNorm() == MFEM_Approx(0.0, 100*tol));
   }
}

} // namespace

TEST_CASE("MassIntegrator full EA matches element assembly (tensor quads/hexes)",
          "[AssembleEA][Mass][GPU]")
{
   const auto mesh_file = GENERATE("../../data/inline-quad.mesh",
                                   "../../data/inline-hex.mesh");
   const int order = GENERATE(2, 3);

   CAPTURE(mesh_file, order);

   CheckMassEA(mesh_file, order);
}
