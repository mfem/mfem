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

#include "fem/dfem/crtp_default.hpp"
#include "./crtp_other.hpp"

#ifdef NVTX_DEBUG_HPP
#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kNvidia
#include NVTX_DEBUG_HPP
#else
#define dbg(...)
#endif


using namespace mfem::future;

template <int DIM>
void crtp_action(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.EnsureNodes();
   auto* nodes = static_cast<ParGridFunction*>(pmesh.GetNodes());
   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   smesh.Clear();

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);

   ParGridFunction x(&fes), y(&fes), z(&fes);
   Vector X(fes.GetTrueVSize()), Y(fes.GetTrueVSize()), Z(fes.GetTrueVSize());

   X.Randomize(1);
   x.SetFromTrueDofs(X);

   ConstantCoefficient one(1.0);

   static constexpr int U = 0, Coords = 1;
   const auto sol = std::vector{ FieldDescriptor{ U, &fes } };

   ParBilinearForm blf(&fes);
   blf.AddDomainIntegrator(new MassIntegrator(one, ir));
   blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf.Assemble();

   const auto mf_mass_qf =
      [] MFEM_HOST_DEVICE(const real_t &u,
                          const tensor<real_t, DIM, DIM> &J,
                          const real_t &w)
   {
      return tuple{u * w * det(J)};
   };

   // -------------------------------------------
   // Test current Backend Implementation
   // -------------------------------------------
   SECTION("Current Domain Integrator Implementation")
   {
      blf.Mult(x, y);
      fes.GetProlongationMatrix()->MultTranspose(y, Y);

      DifferentiableOperator dop(sol, {{Coords, nodes->ParFESpace()}}, pmesh);
      dop.AddDomainIntegrator(mf_mass_qf,
                              tuple{ Value<U>{}, Gradient<Coords>{}, Weight{} },
                              tuple{ Value<U>{} },
                              *ir, all_domain_attr);
      dop.SetParameters({ nodes });

      fes.GetRestrictionMatrix()->Mult(x, X);
      dop.Mult(X, Z);
      Y -= Z;

      real_t norm_g, norm_l = Y.Normlinf();
      MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
      REQUIRE(norm_g == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

   const auto verity_against_blf = [&](auto &op)
   {
      op.SetMultLevel(DifferentiableOperator::LVECTOR);
      op.SetName("Default")
      .SetBlocks(42)
      .Print();

      op.AddDomainIntegrator(mf_mass_qf,
                             tuple{ Value<U>{}, Gradient<Coords>{}, Weight{} },
                             tuple{ Value<U>{} },
                             *ir, all_domain_attr);
      op.SetParameters({ nodes });

      X.Randomize(1);
      x.SetFromTrueDofs(X);

      blf.Mult(x, y);
      fes.GetProlongationMatrix()->MultTranspose(y, Y);

      fes.GetRestrictionMatrix()->Mult(x, X);
      op.Mult(X, Z);
      Y -= Z;

      real_t norm_g, norm_l = Y.Normlinf();
      MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
      REQUIRE(norm_g == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   };

   // -------------------------------------------
   // Test CRTP DEFAULT Backend Implementation
   // -------------------------------------------
   SECTION("CRTP Default Backend Implementation")
   {
      DefaultDifferentiableOperator crtp(sol, {{Coords, nodes->ParFESpace()}}, pmesh);
      verity_against_blf(crtp);
   }

   // -------------------------------------------
   // Test CRTP OTHER Backend Implementation
   // -------------------------------------------
   SECTION("CRTP Other Backend Implementation")
   {
      OtherDifferentiableOperator crtp(sol, {{Coords, nodes->ParFESpace()}}, pmesh);
      verity_against_blf(crtp);
   }

   // -------------------------------------------
   // Test CRTP STORED Backend Implementation
   // -------------------------------------------
   SECTION("Stored Operator Backend Implementation")
   {
      StoredOp mop(sol, {{Coords, nodes->ParFESpace()}}, pmesh);
      verity_against_blf(*mop.op);
   }
}

TEST_CASE("dFEM CRTP", "[Parallel][dFEM][CRTP]")
{
   dbg();
   const auto p = GENERATE(2);

   SECTION("2d")
   {
      const auto filename2d = GENERATE("../../data/star.mesh");
      crtp_action<2>(filename2d, p);
   }
}

#endif // MFEM_USE_MPI
