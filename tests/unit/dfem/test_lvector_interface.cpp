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

#include "../unit_tests.hpp"
#include "mfem.hpp"
#include <utility>

#ifdef MFEM_USE_MPI

using namespace mfem;
using namespace mfem::future;
using mfem::future::tensor;

constexpr int DIM = 3;

namespace kernels
{
struct MFApply
{
   MFEM_HOST_DEVICE inline auto operator()(const tensor<real_t, DIM> &dudxi,
                                           const tensor<real_t, DIM, DIM> &J,
                                           const real_t &w) const
   {
      const auto invJ = inv(J);
      return tuple{ (dudxi * invJ) * transpose(invJ) * det(J) * w };
   }
};
}

TEST_CASE("DFEM L-Vector interface", "[Parallel][DFEM]")
{
   constexpr int p = 2; // Polynomial order
   constexpr int r = 1;
   constexpr int q = 2 * p + r;

   const auto filename = GENERATE("../../data/fichera.mesh");
   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   smesh.Clear();

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), q);

   ParGridFunction x(&pfes), y(&pfes), z(&pfes);
   Vector X(pfes.GetTrueVSize()), Y(pfes.GetTrueVSize()), Z(pfes.GetTrueVSize());

   X.Randomize(1);
   x.SetFromTrueDofs(X);

   ParBilinearForm blf_fa(&pfes);
   blf_fa.AddDomainIntegrator(new DiffusionIntegrator(ir));
   blf_fa.Assemble();
   blf_fa.Finalize();

   static constexpr int U = 0, Coords = 1;

   const auto solution = std::vector{FieldDescriptor{U, &pfes}};
   DifferentiableOperator dop(solution, {{Coords, mfes}}, pmesh);

   kernels::MFApply mf_apply_qf;
   dop.AddDomainIntegrator(mf_apply_qf,
                           tuple{Gradient<U>{}, Gradient<Coords>{}, Weight{}},
                           tuple{Gradient<U>{}}, *ir, all_domain_attr);

   // Use the L-vector interface to multiply
   dop.SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
   dop.SetParameters({nodes});
   dop.Mult(x, z);

   blf_fa.Mult(x, y);

   z -= y;
   REQUIRE(z.Normlinf() == MFEM_Approx(0.0));
}

#endif
