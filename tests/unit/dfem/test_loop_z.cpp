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

#ifdef MFEM_USE_MPI

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"

using namespace mfem;
using namespace mfem::future;

namespace loop_z_test
{

static constexpr int DIM = 3, U = 0, V = 1, W = 2, Coords = 3, QData = 4;
static constexpr int QDIM = 1 + DIM + DIM + 1;

// Direct-value q-function covering the LOOP_Z input contractions: scalar
// value (EvalZ), vector value, scalar gradient (GradZ) and the coordinates
// Jacobian (VectorGradZ). TAG gives each registration its own kernel table.
template <int TAG>
struct QF
{
   MFEM_HOST_DEVICE inline void operator()(const real_t &u,
                                           const tensor<real_t, DIM> &w,
                                           const tensor<real_t, DIM> &dv,
                                           const tensor<real_t, DIM, DIM> &J,
                                           const real_t &weight,
                                           tensor<real_t, QDIM> &qd) const
   {
      const auto invJ = inv(J);
      const auto g = dv * invJ;
      const real_t detJ = det(J);
      qd(0) = u * detJ * weight;
      for (int i = 0; i < DIM; i++) { qd(1 + i) = w(i); }
      for (int i = 0; i < DIM; i++) { qd(1 + DIM + i) = g(i); }
      qd(1 + 2 * DIM) = tr(J) * weight;
   }
};

using IT = Inputs<Value<U>, Value<W>, Gradient<V>, Gradient<Coords>, Weight>;
using OT = Outputs<Identity<QData>>;

template <int TAG, int ZTHREADS, int Q1D>
void Register()
{
   AddAction<DIM, Q1D, QF<TAG>, IT, OT, ZTHREADS>();
   // The key is (dim, tile) only: check that it holds the LOOP_Z kernel
   using ActionLO = typename LocalQFImpl::Action<QF<TAG>, IT, OT>::ActionLO;
   REQUIRE(ActionLO::GetDispatchTable().at(std::make_tuple(DIM, Q1D)) ==
           ActionLO::template Kernel<DIM, Q1D, ZTHREADS>());
}

template <int TAG, int ZTHREADS>
void Register()
{
   Register<TAG, ZTHREADS, 5>();
   Register<TAG, ZTHREADS, 6>();
   Register<TAG, ZTHREADS, 7>();
   Register<TAG, ZTHREADS, 8>();
}

} // namespace loop_z_test

TEST_CASE("dFEM LOOP_Z Action 3D", "[Parallel][dFEM][GPU]")
{
   using namespace loop_z_test;

   // Tile p + 1: 5 to 8, i.e. 4 or 3 z-threads with 2 or 3 qz layers each
   const auto p = GENERATE(4, 5, 6, 7);
   CAPTURE(p);

   Mesh smesh = Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   smesh.EnsureNodes();
   // Make the Jacobian non-constant
   smesh.Transform([](const Vector &x, Vector &y)
   {
      y = x;
      y(0) += 0.1 * std::sin(M_PI * x(1)) * x(2);
      y(1) += 0.1 * std::cos(M_PI * x(2)) * x(0);
   });
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   smesh.Clear();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace sfes(&pmesh, &fec);
   ParFiniteElementSpace vfes(&pmesh, &fec, DIM);
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);
   QuadratureSpace qspace(pmesh, *ir);
   VectorQuadratureSpace qspace_vec(qspace, QDIM);

   Array<int> all_domain_attr(pmesh.attributes.Max());
   all_domain_attr = 1;

   Vector ut(sfes.GetTrueVSize()), vt(sfes.GetTrueVSize()),
          wt(vfes.GetTrueVSize()), nodestv;
   ut.Randomize(1);
   vt.Randomize(2);
   wt.Randomize(3);
   nodes->GetTrueDofs(nodestv);

   const auto in_fds = std::vector
   {
      FieldDescriptor{ U, &sfes },
      FieldDescriptor{ V, &sfes },
      FieldDescriptor{ W, &vfes },
      FieldDescriptor{ Coords, mfes }
   };
   const auto out_fds = std::vector{ FieldDescriptor{ QData, &qspace_vec } };

   const auto run = [&](auto qf)
   {
      DifferentiableOperator dop(in_fds, out_fds, pmesh);
      dop.AddDomainIntegrator<LocalQFBackend>(qf, IT{}, OT{}, *ir,
                                              all_domain_attr);
      QuadratureFunction qd(qspace_vec);
      MultiVector X{ut, vt, wt, nodestv};
      MultiVector Y{qd};
      dop.Mult(X, Y);
      return Vector(qd);
   };

   // Reference: the unregistered fallback, without LOOP_Z
   const Vector ref = run(QF<0> {});

   Register<1, 4>();
   Register<2, 3>();

   for (const Vector &qd : { run(QF<1> {}), run(QF<2> {}) })
   {
      Vector diff(qd);
      diff -= ref;
      real_t norm_local = diff.Normlinf(), norm_global = 0.0;
      MPI_Allreduce(&norm_local, &norm_global, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, pmesh.GetComm());
      REQUIRE(norm_global == MFEM_Approx(0.0));
   }
}

#endif // MFEM_USE_MPI
