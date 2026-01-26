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

TEST_CASE("dFEM Multiple Outputs", "[Parallel][dFEM]")
{
   const bool all_tests = launch_all_non_regression_tests;

   const auto p = !all_tests ? 2 : GENERATE(1, 2, 3);

   constexpr int DIM = 2;
   const char *filename = "../../data/inline-quad.mesh";
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.EnsureNodes();
   auto* nodes = static_cast<ParGridFunction*>(pmesh.GetNodes());
   smesh.Clear();

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);

   ParGridFunction x(&fes), y(&fes), z(&fes);
   Vector X(fes.GetTrueVSize()), Y(fes.GetTrueVSize()), Z(fes.GetTrueVSize());

   X.Randomize(1);
   x.SetFromTrueDofs(X);

   ConstantCoefficient one(1.0);

   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

   UniformParameterSpace ups(pmesh, *ir, 1);
   Vector scalar_out(ups.GetTrueVSize());

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   ParBilinearForm blf(&fes);
   blf.AddDomainIntegrator(new MassIntegrator(one, ir));
   blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf.Assemble();
   blf.Mult(x, y);
   fes.GetProlongationMatrix()->MultTranspose(y, Y);

   static constexpr int U = 0, SCALAR = 1, COORDINATES = 2;
   const auto sol = std::vector{ FieldDescriptor{ U, &fes } };
   DifferentiableOperator dop(sol, {{COORDINATES, nodes->ParFESpace()}}, pmesh);

   const auto mf_mass_qf =
      [] MFEM_HOST_DEVICE (
         const real_t &u,
         const tensor<real_t, DIM, DIM> &J, const real_t &w)
   {
      return tuple{u * w * det(J), w};
   };

   dop.AddDomainIntegrator(mf_mass_qf,
                           tuple{ Value<U>{}, Gradient<COORDINATES>{}, Weight{} },
                           tuple{ Value<U>{}, Identity<SCALAR>{} },
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

#endif // MFEM_USE_MPI
