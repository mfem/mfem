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

using namespace mfem;
using namespace mfem::future;
using mfem::future::tensor;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using mfem::future::dual;
using dscalar_t = dual<real_t, real_t>;
#endif

namespace dfem_pa_kernels
{

template <int DIM>
void dFemVectorDivergence(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   smesh.Clear();

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace psfes(&pmesh, &fec);
   ParFiniteElementSpace pvfes(&pmesh, &fec, DIM);

   const int d1d(p + 1), q = 3 * p + 1;
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), q);
   const int q1d(IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints());
   MFEM_VERIFY(d1d <= q1d, "q1d should be >= d1d");

   ParGridFunction vx(&pvfes);
   ParGridFunction sy(&psfes), sz(&psfes);
   Vector vX(pvfes.GetTrueVSize());
   Vector sY(psfes.GetTrueVSize()), sZ(psfes.GetTrueVSize());

   vX.Randomize(1), vx.SetFromTrueDofs(vX);

   MixedBilinearForm mblf_fa(&pvfes, &psfes);
   mblf_fa.AddDomainIntegrator(new VectorDivergenceIntegrator);
   mblf_fa.Assemble(), mblf_fa.Finalize();
   mblf_fa.Mult(vx, sy);

   MixedBilinearForm mblf_pa(&pvfes, &psfes);
   mblf_pa.AddDomainIntegrator(new VectorDivergenceIntegrator);
   mblf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mblf_pa.Assemble();
   mblf_pa.Mult(vx, sz);
   sy -= sz;
   REQUIRE(sy.Normlinf() == MFEM_Approx(0.0));
   MPI_Barrier(MPI_COMM_WORLD);

   {
      static constexpr int P = 0, V = 1, Coords = 2;
      ParFiniteElementSpace *mfes = nodes->ParFESpace();

      const auto solutions = std::vector{ FieldDescriptor{ P, &psfes } };
      const auto parameters = std::vector
      {
         FieldDescriptor{ V, &pvfes },
         FieldDescriptor{ Coords, mfes }
      };

      DifferentiableOperator dop_mf(solutions, parameters, pmesh);

      const auto mf_vector_divergence_qf =
         [] MFEM_HOST_DEVICE(const tensor<dscalar_t, DIM, DIM> &dudxi,
                             const tensor<mfem::real_t, DIM, DIM> &J,
                             const real_t &w)
      {
         const auto invJ = inv(J);
         const auto dudx = dudxi * invJ;
         return tuple{ tr(dudx) * det(J) * w };
      };

      dop_mf.AddDomainIntegrator(mf_vector_divergence_qf,
                                 tuple{ Gradient<V>{}, Gradient<Coords>{}, Weight{} },
                                 tuple{ Value<P>{} },
                                 *ir, all_domain_attr);

      dop_mf.SetParameters({ &vx, nodes });
      Vector unused(pvfes.GetTrueVSize());
      dop_mf.Mult(unused, sZ);

      mblf_fa.Mult(vx, sy);
      psfes.GetProlongationMatrix()->MultTranspose(sy, sY);

      sY -= sZ;
      real_t norm_global = M_PI, norm_local = sY.Normlinf();
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                    pmesh.GetComm());
      REQUIRE(norm_global == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }
}

TEST_CASE("dFEM VectorDivergence", "[Parallel][DFEM][VectorDivergence]")
{
   const bool all_tests = launch_all_non_regression_tests;

   const auto p = !all_tests ? 2 : GENERATE(1, 2, 3);

   SECTION("2D p=" + std::to_string(p))
   {
      const auto filename =
         GENERATE("../../data/star.mesh",
                  "../../data/star-q3.mesh",
                  "../../data/rt-2d-q3.mesh",
                  "../../data/inline-quad.mesh",
                  "../../data/periodic-square.mesh");
      dFemVectorDivergence<2>(filename, p);
   }

   SECTION("3D p=" + std::to_string(p))
   {
      const auto filename =
         GENERATE("../../data/fichera.mesh",
                  "../../data/fichera-q3.mesh",
                  "../../data/inline-hex.mesh",
                  "../../data/toroid-hex.mesh",
                  "../../data/periodic-cube.mesh");
      dFemVectorDivergence<3>(filename, p);
   }
}

} // namespace dfem_pa_kernels

#endif // MFEM_USE_MPI
