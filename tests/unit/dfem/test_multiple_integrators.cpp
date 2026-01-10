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
#include "../linalg/test_same_matrices.hpp"
#include "mfem.hpp"
#include "fem/dfem/doperator.hpp"

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
void mult_integ(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.EnsureNodes();
   auto* nodes = static_cast<ParGridFunction*>(pmesh.GetNodes());
   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   smesh.Clear();

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);

   ParGridFunction x(&fes), y(&fes), z(&fes);
   Vector X(fes.GetTrueVSize()), Y(fes.GetTrueVSize()), Z(fes.GetTrueVSize());

   X.Randomize(1);
   x.SetFromTrueDofs(X);

   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   ParBilinearForm blf(&fes);
   blf.AddDomainIntegrator(new DiffusionIntegrator(ir));
   blf.AddDomainIntegrator(new MassIntegrator(ir));
   blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf.Assemble();
   blf.Mult(x, y);
   fes.GetProlongationMatrix()->MultTranspose(y, Y);

   const auto mass_qf =
      [] MFEM_HOST_DEVICE(
         const dscalar_t &u,
         const tensor<real_t, DIM, DIM> &J,
         const real_t &w)
   {
      return tuple{u * w * det(J)};
   };

   const auto diffusion_qf =
      [] MFEM_HOST_DEVICE(
         const tensor<dscalar_t, DIM> &dudxi,
         const tensor<real_t, DIM, DIM> &J,
         const real_t &w)
   {
      return tuple{(dudxi * inv(J)) * transpose(inv(J)) * w * det(J)};
   };

   static constexpr int U = 0, Coords = 1;
   const auto sol = std::vector{ FieldDescriptor{ U, &fes } };
   DifferentiableOperator dop(sol, {{Coords, nodes->ParFESpace()}}, pmesh);

   auto derivatives = std::integer_sequence<size_t, U> {};

   dop.AddDomainIntegrator(diffusion_qf,
                           tuple{ Gradient<U>{}, Gradient<Coords>{}, Weight{} },
                           tuple{ Gradient<U>{} },
                           *ir, all_domain_attr, derivatives);

   dop.AddDomainIntegrator(mass_qf,
                           tuple{ Value<U>{}, Gradient<Coords>{}, Weight{} },
                           tuple{ Value<U>{} },
                           *ir, all_domain_attr, derivatives);

   SECTION("action")
   {
      dop.SetParameters({ nodes });

      fes.GetRestrictionMatrix()->Mult(x, X);
      dop.Mult(X, Z);

      Y -= Z;
      real_t norm_g, norm_l = Y.Normlinf();
      MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
      REQUIRE(norm_g == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

   SECTION("linearized action")
   {
      auto ddopdu = dop.GetDerivative(U, {&x}, {nodes});

      fes.GetRestrictionMatrix()->Mult(x, X);
      ddopdu->Mult(X, Z);

      Y -= Z;
      real_t norm_g, norm_l = Y.Normlinf();
      MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
      REQUIRE(norm_g == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

   SECTION("linearized assembled SparseMatrix")
   {
      auto ddopdu = dop.GetDerivative(U, {&x}, {nodes});

      SparseMatrix *A = nullptr;
      ddopdu->Assemble(A);

      fes.GetRestrictionMatrix()->Mult(x, X);
      A->Mult(X, Z);

      Y -= Z;
      real_t norm_g, norm_l = Y.Normlinf();
      MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
      REQUIRE(norm_g == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

   SECTION("linearized assembled HypreParMatrix")
   {
      auto ddopdu = dop.GetDerivative(U, {&x}, {nodes});

      HypreParMatrix *A = nullptr;
      ddopdu->Assemble(A);

      fes.GetRestrictionMatrix()->Mult(x, X);
      A->Mult(X, Z);

      Y -= Z;
      real_t norm_g, norm_l = Y.Normlinf();
      MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
      REQUIRE(norm_g == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

}

// no GPU tag to avoid failing 'hypre parallel mat' section
TEST_CASE("dFEM Multiple Integrators", "[Parallel][dFEM][XXX]")
{
   const bool all_tests = launch_all_non_regression_tests;

   const auto p = !all_tests ? 2 : GENERATE(1, 2, 3);

   SECTION("2d")
   {
      const auto filename2d =
         GENERATE(
            "../../data/star.mesh",
            "../../data/star-q3.mesh",
            "../../data/rt-2d-q3.mesh",
            "../../data/inline-quad.mesh",
            "../../data/periodic-square.mesh"
         );
      mult_integ<2>(filename2d, p);
   }

   SECTION("3d")
   {
      const auto filename3d =
         GENERATE(
            "../../data/fichera.mesh",
            "../../data/fichera-q3.mesh",
            "../../data/inline-hex.mesh",
            "../../data/toroid-hex.mesh",
            "../../data/periodic-cube.mesh"
         );
      mult_integ<3>(filename3d, p);
   }
}

#endif // MFEM_USE_MPI
