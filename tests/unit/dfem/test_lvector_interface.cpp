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

#ifdef MFEM_USE_MPI

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"

using namespace mfem;
using namespace mfem::future;

template <int DIM>
struct MFApply
{
   MFEM_HOST_DEVICE inline auto operator()(
      const tensor<real_t, DIM> &dudxi,
      const tensor<real_t, DIM, DIM> &J,
      const real_t &w,
      tensor<real_t, DIM> &dvdxi) const
   {
      const auto invJ = inv(J);
      const auto invJt = transpose(invJ);
      dvdxi = (dudxi * invJ) * invJt * det(J) * w;
   }
};

template <int DIM, typename QFBackend = LocalQFBackend>
void l_vector_interface(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   smesh.Clear();

   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   const int q = 2 * p + 1;

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

   const auto in_fds = std::vector
   {
      FieldDescriptor{ U, &pfes },
      FieldDescriptor{ Coords, mfes }
   };
   const auto out_fds = std::vector{ FieldDescriptor{ U, &pfes } };

   DifferentiableOperator dop(in_fds, out_fds, pmesh);

   MFApply<DIM> mf_apply;
   dop.AddDomainIntegrator<LocalQFBackend>(
      mf_apply,
      tuple{Gradient<U>{}, Gradient<Coords>{}, Weight{}},
      tuple{Gradient<U>{}},
      *ir, all_domain_attr);

   // Use the L-vector interface to multiply
   dop.SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);

   MultiVector mx{x, *nodes};
   MultiVector mz{z};
   dop.Mult(mx, mz);

   blf_fa.Mult(x, y);

   z -= y;
   REQUIRE(z.Normlinf() == MFEM_Approx(0.0));
}

TEST_CASE("dFEM L-Vector 2D", "[Parallel][dFEM][GPU][2D]")
{
   const auto p = GenAll({1}, {2, 3});
   const auto meshs = { "../../data/inline-quad.mesh" };
   const auto extra = { "../../data/star.mesh",
                        "../../data/star-q3.mesh",
                        "../../data/rt-2d-q3.mesh",
                        "../../data/periodic-square.mesh"
                      };
   l_vector_interface<2>(GenAll(meshs, extra), p);
}

TEST_CASE("dFEM L-Vector 3D", "[Parallel][dFEM][GPU][3D]")
{
   const auto p = GenAll({1}, {2, 3});
   const auto meshs = { "../../data/inline-hex.mesh" };
   const auto extra = { "../../data/fichera.mesh",
                        "../../data/fichera-q3.mesh",
                        "../../data/toroid-hex.mesh",
                        "../../data/periodic-cube.mesh"
                      };
   l_vector_interface<3>(GenAll(meshs, extra), p);
}

#endif // MFEM_USE_MPI
