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

#include "../unit_tests.hpp"

#include "mfem.hpp"

#include <limits>

#ifdef MFEM_USE_MPI

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"

using namespace mfem;
using namespace mfem::future;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using dscalar_t = dual<real_t, real_t>;
#endif

namespace numerics_test
{

using numerics_detail::encode;
using numerics_detail::decode;

using limits = std::numeric_limits<double>;

static_assert(encode(1.0) == 0x3FF0000000000000 &&
              encode(1e-9) == 0x3E112E0BE826D695 &&
              encode(limits::min()) == 0x0010000000000000 &&
              encode(limits::max()) == 0x7FEFFFFFFFFFFFFF);
static_assert(decode(encode(1e-6)) == 1e-6 && decode(encode(5e-13)) == 5e-13 &&
              decode(encode(3.0)) == 3.0 && decode(encode(1e300)) == 1e300 &&
              decode(encode(limits::min())) == limits::min() &&
              decode(encode(limits::max())) == limits::max());
static_assert(encode(0.0) == 0 && encode(-1e-9) == 0 &&
              encode(limits::denorm_min()) == 0 &&
              encode(limits::infinity()) == 0 &&
              encode(limits::quiet_NaN()) == 0);

constexpr double tol = 1e-9;
using PrimalNumerics = decltype(MFEM_NUMERICS(Tolerance, 1e-9));
using AllNumerics = decltype(MFEM_NUMERICS(DerivativeTolerance, 1e-6) |
                             MFEM_NUMERICS(Tolerance, 1e-9));

static_assert(std::is_same_v<decltype(MFEM_NUMERICS(Tolerance, tol)),
              PrimalNumerics> &&
              !std::is_same_v<decltype(MFEM_NUMERICS(Tolerance, 2e-9)),
              PrimalNumerics> &&
              !std::is_same_v<
              decltype(MFEM_NUMERICS(DerivativeTolerance, 1e-9)),
              PrimalNumerics>);
static_assert(std::is_same_v<AllNumerics,
              Numerics<DerivativeTolerance<encode(1e-6)>,
              Tolerance<encode(1e-9)>>>);
static_assert(numerics_has_v<Tolerance, AllNumerics> &&
              numerics_has_v<DerivativeTolerance, AllNumerics> &&
              !numerics_has_v<DerivativeTolerance, PrimalNumerics> &&
              !numerics_has_v<Tolerance, Numerics<>>);
static_assert(numerics_item_t<Tolerance, AllNumerics>::value == 1e-9 &&
              numerics_item_t<DerivativeTolerance, AllNumerics>::value == 1e-6);
static_assert(is_numerics_v<AllNumerics> && !is_numerics_v<Derivatives<>>);

// The pointwise residual of the minimal surface equation, see
// miniapps/dfem/dfem-minimal-surface.cpp.
struct MinimalSurface
{
   MFEM_HOST_DEVICE inline void operator()(
      const tensor<dscalar_t, 2> &dudxi,
      const tensor<real_t, 2, 2> &J,
      const real_t &w,
      tensor<dscalar_t, 2> &dvdx) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      dvdx = 1_r / sqrt(1_r + sqnorm(dudx)) * dudx * transpose(invJ) * det(J) *
             w;
   }
};

// A q-function whose output is written at the quadrature points.
struct GeometricFactors
{
   MFEM_HOST_DEVICE inline void operator()(
      const tensor<real_t, 2, 2> &J,
      const real_t &w,
      tensor<real_t, 2, 2> &qdata) const
   {
      qdata = inv(J) * transpose(inv(J)) * det(J) * w;
   }
};

real_t MaxNorm(const Vector &a, MPI_Comm comm)
{
   real_t local = a.Normlinf(), global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 comm);
   return global;
}

real_t MaxDifference(const Vector &a, const Vector &b, MPI_Comm comm)
{
   Vector d(a);
   d -= b;
   return MaxNorm(d, comm);
}

} // namespace numerics_test

// An integrator with numerical requirements computes what the integrator
// without them computes: exactly when its sites are compiled as written, and
// well within the bounds below when Poseidon instruments or rewrites them.
TEST_CASE("dFEM Numerics", "[Parallel][dFEM][GPU]")
{
   static constexpr int U = 0, Coords = 1, QData = 2;
   constexpr int p = 2;

   Mesh smesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
   smesh.SetCurvature(p);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   smesh.Clear();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *mfes = nodes->ParFESpace();
   Vector nodestv;
   nodes->GetTrueDofs(nodestv);

   Array<int> all_domain_attr(pmesh.attributes.Max());
   all_domain_attr = 1;
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

   SECTION("Tolerance and DerivativeTolerance")
   {
      H1_FECollection fec(p, 2);
      ParFiniteElementSpace pfes(&pmesh, &fec);
      const auto in_fds = std::vector
      {
         FieldDescriptor{ U, &pfes },
         FieldDescriptor{ Coords, mfes }
      };
      const auto out_fds = std::vector{ FieldDescriptor{ U, &pfes } };
      const auto inputs = Inputs<Gradient<U>, Gradient<Coords>, Weight> {};
      const auto outputs = Outputs<Gradient<U>> {};
      numerics_test::MinimalSurface qf;
      constexpr auto kernels = DerivativeKernels::Action;

      DifferentiableOperator dop(in_fds, out_fds, pmesh), ndop(in_fds, out_fds,
                                                               pmesh);
      dop.AddDomainIntegrator<LocalQFBackend, kernels>(
         qf, inputs, outputs, *ir, all_domain_attr, Derivatives<U> {});
      ndop.AddDomainIntegrator<LocalQFBackend, kernels>(
         qf, inputs, outputs, *ir, all_domain_attr, Derivatives<U> {},
         numerics_test::AllNumerics {});

      const int n = pfes.GetTrueVSize();
      Vector x(n), dx(n), y(n), dy(n), z(n), dz(n);
      x.Randomize(1);
      dx.Randomize(2);
      MultiVector X{x, nodestv}, Y{y}, DY{dy}, Z{z}, DZ{dz};
      dop.Mult(X, Y);
      dop.GetDerivative(U, X)->Mult(dx, DY);
      ndop.Mult(X, Z);
      ndop.GetDerivative(U, X)->Mult(dx, DZ);
      REQUIRE(numerics_test::MaxDifference(y, z, MPI_COMM_WORLD) <=
              1e-7 * numerics_test::MaxNorm(y, MPI_COMM_WORLD));
      REQUIRE(numerics_test::MaxDifference(dy, dz, MPI_COMM_WORLD) <=
              1e-4 * numerics_test::MaxNorm(dy, MPI_COMM_WORLD));
   }

   SECTION("Tolerance with an output at the quadrature points")
   {
      QuadratureSpace qspace(pmesh, *ir);
      VectorQuadratureSpace qspace_vec(qspace, 4);
      QuadratureFunction qd(qspace_vec), nqd(qspace_vec);
      const auto in_fds = std::vector{ FieldDescriptor{ Coords, mfes } };
      const auto out_fds = std::vector{ FieldDescriptor{ QData, &qspace_vec } };
      const auto inputs = Inputs<Gradient<Coords>, Weight> {};
      const auto outputs = Outputs<Identity<QData>> {};
      numerics_test::GeometricFactors gf;

      DifferentiableOperator dop(in_fds, out_fds, pmesh), ndop(in_fds, out_fds,
                                                               pmesh);
      dop.AddDomainIntegrator<LocalQFBackend>(
         gf, inputs, outputs, *ir, all_domain_attr);
      ndop.AddDomainIntegrator<LocalQFBackend>(
         gf, inputs, outputs, *ir, all_domain_attr,
         numerics_test::PrimalNumerics {});

      MultiVector N{nodestv}, Q{qd}, NQ{nqd};
      dop.Mult(N, Q);
      ndop.Mult(N, NQ);
      REQUIRE(numerics_test::MaxDifference(qd, nqd, MPI_COMM_WORLD) <=
              1e-7 * numerics_test::MaxNorm(qd, MPI_COMM_WORLD));
   }
}

#endif // MFEM_USE_MPI
