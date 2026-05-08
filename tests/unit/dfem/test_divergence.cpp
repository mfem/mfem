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
#include "../fem/dfem/doperator.hpp"
#include "../fem/dfem/backends/local_qf/prelude.hpp"

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

template <int DIM>
void vectordivergence(const char *filename, int p)
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

   const int q = 3 * p + 1;
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), q);

   ParGridFunction xv(&pvfes);
   ParGridFunction ys(&psfes), sz(&psfes);
   Vector Xv(pvfes.GetTrueVSize());
   Vector Ys(psfes.GetTrueVSize()), Zs(psfes.GetTrueVSize());

   Xv.Randomize(1), xv.SetFromTrueDofs(Xv);

   MixedBilinearForm mblf_fa(&pvfes, &psfes);
   mblf_fa.AddDomainIntegrator(new VectorDivergenceIntegrator);
   mblf_fa.Assemble(), mblf_fa.Finalize();
   mblf_fa.Mult(xv, ys);

   {
      static constexpr int P = 0, V = 1, Coords = 2;
      ParFiniteElementSpace *mfes = nodes->ParFESpace();

      const auto inputs = std::vector
      {
         FieldDescriptor{V, &pvfes},
         FieldDescriptor{Coords, mfes}
      };
      const auto outputs = std::vector
      {
         FieldDescriptor{P, &psfes}
      };

      DifferentiableOperator dop_mf(inputs, outputs, pmesh);

      const auto mf_vector_divergence_qf =
         [] MFEM_HOST_DEVICE(const tensor<dscalar_t, DIM, DIM> &dudxi,
                             const tensor<mfem::real_t, DIM, DIM> &J,
                             const real_t &w,
                             real_t &v)
      {
         const auto invJ = inv(J);
         const auto dudx = dudxi * invJ;
         v = tr(dudx) * det(J) * w;
      };

      const auto derivatives = std::integer_sequence<size_t, V> {};
      dop_mf.AddDomainIntegrator<LocalQFBackend>(
         mf_vector_divergence_qf,
         tuple{Gradient<V>{}, Gradient<Coords>{}, Weight{}},
         tuple{Value<P>{}},
         *ir, all_domain_attr, derivatives);

      SECTION("Action")
      {
         Vector nodestv;
         nodes->GetTrueDofs(nodestv);
         MultiVector X{Xv, nodestv};
         MultiVector Z{Zs};
         dop_mf.Mult(X, Z);

         mblf_fa.Mult(xv, ys);
         psfes.GetProlongationMatrix()->MultTranspose(ys, Ys);

         Ys -= Zs;
         real_t norm_global = 0.0;
         real_t norm_local = Ys.Normlinf();
         MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                       pmesh.GetComm());
         REQUIRE(norm_global == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
      }

      SECTION("Derivative Action")
      {
         Vector nodestv;
         nodes->GetTrueDofs(nodestv);
         MultiVector X{Xv, nodestv};
         MultiVector Z{Zs};
         auto dRdV = dop_mf.GetDerivative(V, X);
         dRdV->Mult(X[0], Z);

         mblf_fa.Mult(xv, ys);
         psfes.GetProlongationMatrix()->MultTranspose(ys, Ys);

         Ys -= Zs;
         real_t norm_global = 0.0;
         real_t norm_local = Ys.Normlinf();
         MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                       pmesh.GetComm());
         REQUIRE(norm_global == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
      }

      SECTION("Derivative Transpose Action")
      {
         Vector nodestv;
         nodes->GetTrueDofs(nodestv);

         // Build cache with full primal state
         MultiVector state{Xv, nodestv};
         auto dRdV = dop_mf.GetDerivative(V, state);

         // Direction in output (test) T-space: use Ys computed from mblf_fa.
         psfes.GetProlongationMatrix()->MultTranspose(ys, Ys);
         MultiVector direction{Ys};

         // Result in derivative (trial) T-space.
         Vector result_v(pvfes.GetTrueVSize());
         result_v = 0.0;
         MultiVector result{result_v};
         dRdV->MultTranspose(direction, result);

         // Reference: mblf_fa.MultTranspose(ys, xv) -> restrict to T-dofs.
         mblf_fa.MultTranspose(ys, xv);
         Vector ref_v(pvfes.GetTrueVSize());
         pvfes.GetProlongationMatrix()->MultTranspose(xv, ref_v);

         result_v -= ref_v;
         real_t norm_global = 0.0;
         real_t norm_local = result_v.Normlinf();
         MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                       pmesh.GetComm());
         REQUIRE(norm_global == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }
}

TEST_CASE("dFEM VectorDivergence", "[Parallel][dFEM]")
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
      vectordivergence<2>(filename, p);
   }

   SECTION("3D p=" + std::to_string(p))
   {
      const auto filename =
         GENERATE("../../data/fichera.mesh",
                  "../../data/fichera-q3.mesh",
                  "../../data/inline-hex.mesh",
                  "../../data/toroid-hex.mesh",
                  "../../data/periodic-cube.mesh");
      vectordivergence<3>(filename, p);
   }
}

#endif // MFEM_USE_MPI
