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
#include "linalg/tensor_arrays.hpp"

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

constexpr int DIM = 2;

struct massqf
{
   inline MFEM_HOST_DEVICE
   void operator()(
      const tensor_array<const real_t> &u,
      const tensor_array<const real_t, DIM, DIM> &J,
      const tensor_array<const real_t> &w,
      const tensor_array<real_t> &out1,
      const tensor_array<real_t> &out2) const
   {
      for (size_t q = 0; q < u.size(); q++)
      {
         const auto v = u(q) * det(J(q)) * w(q);
         out1(q) = v;
         out2(q) = v;
      }
   }
};

TEST_CASE("dFEM Multiple Outputs", "[Parallel][dFEM]")
{
   const bool all_tests = launch_all_non_regression_tests;

   const auto p = !all_tests ? 2 : GENERATE(1, 2, 3);
   const char *filename = "../../data/skewed-square.mesh";
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.EnsureNodes();
   auto* nodes = static_cast<ParGridFunction*>(pmesh.GetNodes());
   smesh.Clear();

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);

   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

   UniformParameterSpace ups(pmesh, *ir, 1);
   Vector scalar_out(ups.GetTrueVSize());

   ParGridFunction x(&fes), y(&fes), z(&fes);

   Array<int> inoffsets(3);
   inoffsets[0] = 0;
   inoffsets[1] = fes.GetTrueVSize();
   inoffsets[2] = nodes->ParFESpace()->GetTrueVSize();
   inoffsets.PartialSum();

   BlockVector X(inoffsets);
   X = 1.0;
   X.GetBlock(1) = *nodes;
   x.SetFromTrueDofs(X.GetBlock(0));

   Array<int> outoffsets(2);
   outoffsets[0] = 0;
   outoffsets[1] = fes.GetTrueVSize();
   outoffsets.PartialSum();
   BlockVector Z(outoffsets);

   ConstantCoefficient one(1.0);

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
   Vector Y(fes.GetTrueVSize());
   fes.GetProlongationMatrix()->MultTranspose(y, Y);

   static constexpr int U = 0, COORDINATES = 1, V = 2;
   const std::vector<FieldDescriptor> in
   {
      {U, &fes},
      {COORDINATES, nodes->ParFESpace()}
   };

   const std::vector<FieldDescriptor> out // test spaces?
   {
      {V, &fes},
   };
   DifferentiableOperator dop(in, out, pmesh);

   auto derivatives = std::integer_sequence<size_t, U> {};
   auto mass_qfunc = massqf {};
   dop.AddDomainIntegrator(mass_qfunc,
                           tuple{ Value<U>{}, Gradient<COORDINATES>{}, Weight{} },
                           tuple{ Value<V>{}, Value<V>{} },
                           *ir, all_domain_attr, derivatives);

   fes.GetRestrictionMatrix()->Mult(x, X.GetBlock(0));
   dop.Mult(X, Z);

   Vector Y0(Y);
   Y0 *= 2.0;
   Y0 -= Z.GetBlock(0);

   real_t norm_g, norm_l = Y0.Normlinf();
   MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
   REQUIRE(norm_g == MFEM_Approx(0.0));
   MPI_Barrier(MPI_COMM_WORLD);

   auto ddop = dop.GetDerivative(U, X);

   ddop->Mult(X.GetBlock(0), Z);
   Y0 = Y;
   Y0 *= 2.0;
   Y0 -= Z.GetBlock(0);

   norm_l = Y0.Normlinf();
   MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
   REQUIRE(norm_g == MFEM_Approx(0.0));
   MPI_Barrier(MPI_COMM_WORLD);
}

#endif // MFEM_USE_MPI
