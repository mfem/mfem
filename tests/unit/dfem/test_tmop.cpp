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

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using dscalar_t = dual<real_t, real_t>;
#endif

// ────────────────────────────────────────────────────────────────────────────
namespace
{

static constexpr int DIM = 2;

struct VectorValueCopy
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<real_t, DIM> &x,
                   tensor<real_t, DIM> &y) const
   {
      y(0) = x(0);
      y(1) = x(1);
   }
};

struct VectorMassCopy
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<real_t, DIM> &x,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   tensor<real_t, DIM> &y) const
   {
      y = x * w * det(J);
   }
};

}

// ────────────────────────────────────────────────────────────────────────────
void test_vqspace_identity_copy(int order)
{
   Mesh serial_mesh =
      Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, true, 1.0, 1.0);
   ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);

   const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * order + 1);
   QuadratureSpace qspace(pmesh, ir);
   VectorQuadratureSpace vqspace(qspace, DIM);
   const auto qvsize = vqspace.GetVSize();

   Array<int> domain_attr(pmesh.attributes.Max());
   domain_attr = 1;

   static constexpr int U = 0, V = 1;

   const std::vector fdi { FieldDescriptor{U, &vqspace} };
   const std::vector fdo { FieldDescriptor{V, &vqspace} };

   DifferentiableOperator dop(fdi, fdo, pmesh);

   VectorValueCopy qfunc;
   dop.AddDomainIntegrator<LocalQFBackend>(
      qfunc,
      Inputs<Identity<U>> {},
      Outputs<Identity<V>> {},
      ir, domain_attr);

   Vector vX(qvsize), vY(qvsize);
   vX.Randomize(0x9e3779b9);
   vY = 0.0;

   MultiVector Xmv{vX}, Ymv{vY};
   dop.Mult(Xmv, Ymv);

   Vector diff(vY);
   diff -= vX;
   REQUIRE(diff.Normlinf() == MFEM_Approx(0.0));
}

// ────────────────────────────────────────────────────────────────────────────
void test_value_vector_mass(int order)
{
   Mesh serial_mesh =
      Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, true, 1.0, 1.0);
   ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   const auto *nfes = nodes->ParFESpace();

   H1_FECollection fec(order, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec, DIM, Ordering::byVDIM);
   const auto tvsize = fes.GetTrueVSize();

   Array<int> domain_attr(pmesh.attributes.Max());
   domain_attr = 1;

   const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * order + 1);
   ConstantCoefficient one(1.0), zero(0.0);

   ParGridFunction input_gf(&fes);
   ParGridFunction reference_gf(&fes);
   ParGridFunction error_gf(&fes);
   Vector vX(tvsize), vY_ref(tvsize), vY(tvsize), N(nfes->GetTrueVSize());
   vX.Randomize(0x9e3779b9);
   input_gf.SetFromTrueDofs(vX);
   nodes->GetTrueDofs(N);
   vY = 0.0;

   ParBilinearForm blf(&fes);
   blf.AddDomainIntegrator(new VectorMassIntegrator(one, &ir));
   blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf.Assemble();
   blf.Mult(input_gf, reference_gf);
   fes.GetProlongationMatrix()->MultTranspose(reference_gf, vY_ref);

   static constexpr int U = 0, V = 1, Coords = 2;
   const std::vector fdi { FieldDescriptor{U, &fes}, FieldDescriptor{Coords, nfes} };
   const std::vector fdo { FieldDescriptor{V, &fes} };

   DifferentiableOperator dop(fdi, fdo, pmesh);
   VectorMassCopy qfunc;
   dop.AddDomainIntegrator<LocalQFBackend>(
      qfunc,
      Inputs<Value<U>, Gradient<Coords>, Weight> {},
      Outputs<Value<V>> {},
      ir, domain_attr);

   MultiVector Xmv{vX, N}, Ymv{vY};
   dop.Mult(Xmv, Ymv);

   vY_ref -= vY;
   error_gf.SetFromTrueDofs(vY_ref);
   REQUIRE(error_gf.ComputeMaxError(zero) == MFEM_Approx(0.0));
}

// ────────────────────────────────────────────────────────────────────────────
TEST_CASE("dFEM TMOP", "[Parallel][dFEM][GPU]")
{
   const auto p = GenAll({1}, {2, 3});
   SECTION("VectorQuadratureSpace identity copy")
   {
      test_vqspace_identity_copy(p);
   }
   SECTION("Value vector mass matches PA MFEM")
   {
      test_value_vector_mass(p);
   }
}

#endif // MFEM_USE_MPI
