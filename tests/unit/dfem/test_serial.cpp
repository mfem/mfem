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

// DifferentiableOperator on a serial Mesh and FiniteElementSpace, compared with
// the legacy serial forms. These tests run with and without MFEM_USE_MPI.
// This is a temporary test of the serial path of dFEM, until the other dFEM
// tests are made to run on both serial and parallel spaces, which would
// require extensive changes and longer compile and run times.

#include "../unit_tests.hpp"

#include "mfem.hpp"

#include "../linalg/test_same_matrices.hpp"

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"

using namespace mfem;
using namespace mfem::future;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using dscalar_t = dual<real_t, real_t>;
#endif

namespace dfem_serial_test
{

template <int DIM> struct local_mass_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const dscalar_t &u,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   dscalar_t &v) const
   {
      v = u * w * det(J);
   }
};

template <int DIM> struct global_mass_qf
{
   void operator()(tensor_array<const dscalar_t> &u,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<dscalar_t> &v) const
   {
      mfem::forall<UseEnzyme>(u.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         v(q) = (dscalar_t)(u(q)) * w(q) * det(J(q));
      });
   }
};

template <int DIM> struct diffusion_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<dscalar_t, DIM> &dudxi,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   tensor<dscalar_t, DIM> &dvdxi) const
   {
      const auto invJ = inv(J);
      dvdxi = (dudxi * invJ) * transpose(invJ) * det(J) * w;
   }
};

template <int DIM> struct bdr_mass_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const dscalar_t &u,
                   const tensor<real_t, DIM, DIM - 1> &J,
                   const real_t &w,
                   dscalar_t &v) const
   {
      v = u * weight(J) * w;
   }
};

/// E(u) = int 1/2 |grad u|^2 + 1/4 u^2 dx, so that E(u) = 1/2 u^T K u +
/// 1/4 u^T M u and dE/du = K u + 1/2 M u with the diffusion and mass matrices
/// K and M.
template <int DIM> struct energy_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const dscalar_t &u,
                   const tensor<dscalar_t, DIM> &dudxi,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   dscalar_t &f) const
   {
      const auto dudx = dudxi * inv(J);
      f = (0.5_r * dot(dudx, dudx) + 0.25_r * u * u) * det(J) * w;
   }
};

template <int DIM>
void serial_operators(const char *filename, int p, bool nc)
{
   CAPTURE(filename, DIM, p, nc);

   Mesh mesh(filename);
   REQUIRE(mesh.Dimension() == DIM);
   if (nc)
   {
      // Refine every other element to get hanging nodes, so that the
      // conforming prolongation of the spaces is not the identity.
      mesh.EnsureNCMesh();
      Array<int> refs;
      for (int e = 0; e < mesh.GetNE(); e += 2) { refs.Append(e); }
      mesh.GeneralRefinement(refs, 1);
   }
   mesh.EnsureNodes();
   GridFunction *nodes = mesh.GetNodes();
   FiniteElementSpace *mfes = nodes->FESpace();
   p = std::max(p, mfes->GetMaxElementOrder());

   Array<int> all_domain_attr(mesh.attributes.Max());
   all_domain_attr = 1;
   const auto *ir = &IntRules.Get(mesh.GetTypicalElementGeometry(), 2 * p);

   H1_FECollection fec(p, DIM);
   FiniteElementSpace fes(&mesh, &fec);
   const int tvsize = fes.GetTrueVSize();
   const Operator *P = fes.GetProlongationMatrix();
   REQUIRE((P != nullptr) == nc);

   // L-vector to T-vector, the transpose of the prolongation.
   const auto to_tdofs = [&](const Vector &l, Vector &t)
   {
      t.SetSize(tvsize);
      if (P) { P->MultTranspose(l, t); }
      else { t = l; }
   };

   GridFunction x(&fes), y(&fes);
   Vector X(tvsize), Y(tvsize), Z(tvsize);
   X.Randomize(1);
   x.SetFromTrueDofs(X);

   Vector N;
   nodes->GetTrueDofs(N);

   ConstantCoefficient one(1.0);

   static constexpr int U = 0, Coords = 1, Q = 2;
   const auto in_fds = std::vector
   {
      FieldDescriptor{ U, &fes },
      FieldDescriptor{ Coords, mfes }
   };
   const auto out_fds = std::vector{ FieldDescriptor{ U, &fes } };

   using IT = Inputs<Value<U>, Gradient<Coords>, Weight>;
   using OT = Outputs<Value<U>>;
   using DT = Derivatives<U>;

   SECTION("Mass Action")
   {
      // One mass integrator for each of the local and global QF backends
      BilinearForm blf(&fes);
      blf.AddDomainIntegrator(new MassIntegrator(one, ir));
      blf.AddDomainIntegrator(new MassIntegrator(one, ir));
      blf.Assemble();
      blf.Mult(x, y);
      to_tdofs(y, Y);

      DifferentiableOperator dop(in_fds, out_fds, mesh);
      constexpr auto kernels = DerivativeKernels::Action | DerivativeKernels::PA;
      local_mass_qf<DIM> local_qf;
      dop.AddDomainIntegrator<LocalQFBackend, kernels>(
         local_qf, IT{}, OT{}, *ir, all_domain_attr, DT{});
      global_mass_qf<DIM> global_qf;
      dop.AddDomainIntegrator<GlobalQFBackend, kernels>(
         global_qf, IT{}, OT{}, *ir, all_domain_attr, DT{});

      MultiVector MX{X, N}, MZ{Z};
      dop.Mult(MX, MZ);
      Z -= Y;
      REQUIRE(Z.Normlinf() == MFEM_Approx(0.0));

      for (const bool use_cache : { false, true })
      {
         dop.GetDerivative(U, MX, use_cache)->Mult(MX[0], MZ);
         Z -= Y;
         REQUIRE(Z.Normlinf() == MFEM_Approx(0.0));
      }

      MultiVector direction{X};
      dop.GetDerivative(U, MX)->MultTranspose(direction, Z);
      Z -= Y;
      REQUIRE(Z.Normlinf() == MFEM_Approx(0.0));
   }

   SECTION("Mass Assemble")
   {
      BilinearForm blf(&fes);
      blf.AddDomainIntegrator(new MassIntegrator(one, ir));
      blf.Assemble();
      blf.Finalize();

      DifferentiableOperator dop(in_fds, out_fds, mesh);
      constexpr auto kernels = DerivativeKernels::AllAssembly;
      local_mass_qf<DIM> local_qf;
      dop.AddDomainIntegrator<LocalQFBackend, kernels>(
         local_qf, IT{}, OT{}, *ir, all_domain_attr, DT{});

      MultiVector MX{X, N};
      auto dRdU = dop.GetDerivative(U, MX);

      SparseMatrix *A = nullptr;
      dRdU->Assemble(A);
      TestSameMatrices(*A, blf.SpMat());
      TestSameMatrices(blf.SpMat(), *A);
      delete A;

      Vector dfem_D(tvsize), mfem_D_l, mfem_D;
      dRdU->AssembleDiagonal(dfem_D);
      blf.SpMat().GetDiag(mfem_D_l);
      to_tdofs(mfem_D_l, mfem_D);
      mfem_D -= dfem_D;
      REQUIRE(mfem_D.Normlinf() == MFEM_Approx(0.0));
   }

   SECTION("Diffusion")
   {
      BilinearForm blf(&fes);
      blf.AddDomainIntegrator(new DiffusionIntegrator(ir));
      blf.Assemble();
      blf.Finalize();
      blf.Mult(x, y);
      to_tdofs(y, Y);

      DifferentiableOperator dop(in_fds, out_fds, mesh);
      constexpr auto kernels =
         DerivativeKernels::Action | DerivativeKernels::AssembleMatrix;
      diffusion_qf<DIM> qf;
      dop.AddDomainIntegrator<LocalQFBackend, kernels>(
         qf, Inputs<Gradient<U>, Gradient<Coords>, Weight> {},
         Outputs<Gradient<U>> {}, *ir, all_domain_attr, DT{});

      MultiVector MX{X, N}, MZ{Z};
      dop.Mult(MX, MZ);
      Z -= Y;
      REQUIRE(Z.Normlinf() == MFEM_Approx(0.0));

      SparseMatrix *A = nullptr;
      dop.GetDerivative(U, MX)->Assemble(A);
      TestSameMatrices(*A, blf.SpMat());
      TestSameMatrices(blf.SpMat(), *A);
      delete A;
   }

   SECTION("Boundary Mass")
   {
      // Boundary integrators need every boundary element to be a face
      if (nc || mesh.bdr_attributes.Size() == 0 ||
          mesh.GetNFbyType(FaceType::Boundary) != mesh.GetNBE()) { return; }

      const auto *bdr_ir =
         &IntRules.Get(mesh.GetTypicalFaceGeometry(), 2 * p);
      Array<int> all_bdr_attr(mesh.bdr_attributes.Max());
      all_bdr_attr = 1;

      BilinearForm blf(&fes);
      blf.AddBoundaryIntegrator(new MassIntegrator(one, bdr_ir));
      blf.Assemble();
      blf.Finalize();
      blf.Mult(x, y);
      to_tdofs(y, Y);

      DifferentiableOperator dop(in_fds, out_fds, mesh);
      constexpr auto kernels =
         DerivativeKernels::Action | DerivativeKernels::AssembleMatrix;
      bdr_mass_qf<DIM> qf;
      dop.AddBoundaryIntegrator<LocalQFBackend, kernels>(
         qf, IT{}, OT{}, *bdr_ir, all_bdr_attr, DT{});

      MultiVector MX{X, N}, MZ{Z};
      dop.Mult(MX, MZ);
      Z -= Y;
      REQUIRE(Z.Normlinf() == MFEM_Approx(0.0));

      SparseMatrix *A = nullptr;
      dop.GetDerivative(U, MX)->Assemble(A);
      TestSameMatrices(*A, blf.SpMat());
      TestSameMatrices(blf.SpMat(), *A);
      delete A;
   }

   SECTION("Functional")
   {
      QuadratureSpace qspace(mesh, *ir);
      VectorQuadratureSpace qspace_vec(qspace, 1);

      DifferentiableOperator dop(
         in_fds, { FieldDescriptor{ Q, &qspace_vec } }, mesh);
      energy_qf<DIM> qf;
      dop.AddDomainIntegrator<LocalQFBackend>(
         qf, Inputs<Value<U>, Gradient<U>, Gradient<Coords>, Weight> {},
         Outputs<FunctionalValue<Q>> {}, *ir, all_domain_attr, DT{});

      BilinearForm K(&fes), M(&fes);
      K.AddDomainIntegrator(new DiffusionIntegrator(ir));
      M.AddDomainIntegrator(new MassIntegrator(one, ir));
      K.Assemble();
      M.Assemble();
      Vector Kx(x.Size()), Mx(x.Size());
      K.Mult(x, Kx);
      M.Mult(x, Mx);

      MultiVector MX{X, N};
      Vector E(1);
      MultiVector ME{E};
      dop.Mult(MX, ME);
      const real_t E_ref = 0.5_r * (x * Kx) + 0.25_r * (x * Mx);
      REQUIRE(E(0) == MFEM_Approx(E_ref));

      Vector g, g_ref;
      dop.GetDerivative(U, MX)->Assemble(g);
      Kx.Add(0.5_r, Mx);
      to_tdofs(Kx, g_ref);
      REQUIRE(g.Size() == tvsize);
      g -= g_ref;
      REQUIRE(g.Normlinf() == MFEM_Approx(0.0));
   }
}

} // namespace dfem_serial_test

TEST_CASE("dFEM Serial 2D", "[dFEM][GPU]")
{
   const auto nc = GENERATE(false, true);
   const auto p = GenAll({1, 2}, {3});
   const auto meshs = { "../../data/inline-quad.mesh" };
   const auto extra = { "../../data/star-q3.mesh" };
   dfem_serial_test::serial_operators<2>(GenAll(meshs, extra), p, nc);
}

TEST_CASE("dFEM Serial 3D", "[dFEM][GPU]")
{
   const auto nc = GENERATE(false, true);
   const auto p = GenAll({1}, {2});
   const auto meshs = { "../../data/inline-hex.mesh" };
   const auto extra = { "../../data/fichera.mesh" };
   dfem_serial_test::serial_operators<3>(GenAll(meshs, extra), p, nc);
}
