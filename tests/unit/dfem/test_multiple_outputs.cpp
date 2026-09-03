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

#include <memory>

#include "../unit_tests.hpp"

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"
#include "../../../linalg/tensor_arrays.hpp"
#include "../linalg/test_same_matrices.hpp"

using namespace mfem;
using namespace mfem::future;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using dscalar_t = dual<real_t, real_t>;
#endif

constexpr int DIM = 2;

class DummyParameterSpace : public ParameterSpace
{
public:
   class Bimpl : public Operator
   {
      void Mult(const Vector &x, Vector &y) const override
      {
         const bool use_dev = x.UseDevice() || y.UseDevice();
         const auto xr = x.Read(use_dev);
         auto yw = y.Write(use_dev);
         mfem::forall_switch(use_dev, y.Size(), [=] MFEM_HOST_DEVICE (int i)
         {
            yw[i] = xr[0];
         });
      }
   };

   class Btimpl : public Operator
   {
      void Mult(const Vector &x, Vector &y) const override
      {
         const bool use_dev = x.UseDevice() || y.UseDevice();
         const auto xr = x.Read(use_dev);
         auto yw = y.Write(use_dev);
         mfem::forall_switch(use_dev, 1, [=] MFEM_HOST_DEVICE (int)
         {
            yw[0] = xr[0];
         });
      }
   };

   DummyParameterSpace() : ParameterSpace(1) {}

   int GetTrueVSize() const override
   {
      return 1;
   }

   int GetVSize() const override
   {
      return 1;
   }

   const Operator* GetB() const override
   {
      if (!B)
      {
         B = std::make_unique<Bimpl>();
      }
      return B.get();
   }

   const Operator* GetBt() const override
   {
      if (!Bt)
      {
         Bt = std::make_unique<Btimpl>();
      }
      return Bt.get();
   }
};

/*struct mass_global_qf
{
   void operator()(
      tensor_array<const dscalar_t> &u,
      tensor_array<const real_t, DIM, DIM> &J,
      tensor_array<const real_t> &w,
      tensor_array<dscalar_t> &out1,
      tensor_array<dscalar_t> &out2) const
   {
      mfem::forall(u.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         const auto v = u(q) * det(J(q)) * w(q);
         out1(q) = v;
         out2(q) = v;
      });
   }
};*/

// __enzyme_fwddiff(.....
struct mass_diffusion_global_qf
{
   void operator()(
      tensor_array<const dscalar_t> &u,
      tensor_array<const dscalar_t, DIM> &dudxi,
      tensor_array<const real_t, DIM, DIM> &J,
      [[maybe_unused]] tensor_array<const real_t, DIM, DIM> &qdata,
      tensor_array<const real_t> &w,
      [[maybe_unused]] tensor_array<const real_t> &dummy_parameter,
      tensor_array<dscalar_t> &out1,
      tensor_array<dscalar_t, DIM> &out2,
      tensor_array<real_t, DIM, DIM> &out3) const
   {
      mfem::forall<UseEnzyme>(u.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         const auto invJq = inv(J(q));
         const auto detJq = det(J(q));
         const real_t weight = detJq * w(q);
         out1(q) = u(q) * weight;
         out2(q) = (dudxi(q) * invJq) * transpose(invJq) * (detJq * w(q));
         out3(q) = J(q);
      });
   }
};
// ..... );

struct mass_local_qf
{
   inline MFEM_HOST_DEVICE
   void operator()(
      const dscalar_t &u,
      const tensor<real_t, DIM, DIM> &J,
      const real_t &w,
      dscalar_t &out1,
      dscalar_t &out2) const
   {
      const auto v = u * det(J) * w;
      out1 = v;
      out2 = v;
   }
};

struct mass_diffusion_local_qf
{
   inline MFEM_HOST_DEVICE
   void operator()(
      const dscalar_t &u,
      const tensor<dscalar_t, DIM> &dudxi,
      const tensor<real_t, DIM, DIM> &J,
      const real_t &w,
      dscalar_t &out1,
      tensor<dscalar_t, DIM> &out2) const
   {
      const auto invJ = inv(J);
      const auto detJ = det(J);
      out1 = u * detJ * w;
      out2 = (dudxi * invJ) * transpose(invJ) * (detJ * w);
   }
};

// Three outputs across two test fields: V carries mass + diffusion (two
// outputs on one field), P carries a diffusion scaled by kappa. The
// two blocks are different, so we should spot if the row blocks
// get mismatched/corrupted inadvertedly
constexpr real_t kappa = 2.0;

struct two_field_local_qf
{
   inline MFEM_HOST_DEVICE
   void operator()(
      const dscalar_t &u,
      const tensor<dscalar_t, DIM> &dudxi,
      const tensor<real_t, DIM, DIM> &J,
      const real_t &w,
      dscalar_t &out_v,
      tensor<dscalar_t, DIM> &out_dv,
      tensor<dscalar_t, DIM> &out_dp) const
   {
      const auto invJ = inv(J);
      const auto detJ = det(J);
      const auto grad = (dudxi * invJ) * transpose(invJ);
      out_v = u * detJ * w;
      out_dv = grad * (detJ * w);
      out_dp = grad * (kappa * detJ * w);
   }
};

// Mass + diffusion on an FE test field, plus quadrature point data on a
// VectorQuadratureSpace. The second row block has no basis to contract against,
// so it is not assemblable while the first one still is.
struct mass_diffusion_qdata_local_qf
{
   inline MFEM_HOST_DEVICE
   void operator()(
      const dscalar_t &u,
      const tensor<dscalar_t, DIM> &dudxi,
      const tensor<real_t, DIM, DIM> &J,
      const real_t &w,
      dscalar_t &out1,
      tensor<dscalar_t, DIM> &out2,
      tensor<real_t, DIM, DIM> &out3) const
   {
      const auto invJ = inv(J);
      const auto detJ = det(J);
      out1 = u * detJ * w;
      out2 = (dudxi * invJ) * transpose(invJ) * (detJ * w);
      out3 = J;
   }
};

TEST_CASE("dFEM Multiple Outputs", "[Parallel][dFEM][GPU]")
{
   const bool all_tests = launch_all_non_regression_tests;

   const auto p = !all_tests ? 2 : GENERATE(1, 2, 3);
   const char *filename = "../../data/inline-quad.mesh";
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   MFEM_ASSERT(smesh.Dimension() == DIM, "DIM and mesh dimension have to match");

   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.EnsureNodes();
   auto* nodes = static_cast<ParGridFunction*>(pmesh.GetNodes());
   smesh.Clear();

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);

   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

   ParGridFunction x(&fes), y(&fes), z(&fes);

   ConstantCoefficient one(1.0);

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   // {
   //    Array<int> inoffsets(3);
   // inoffsets[0] = 0;
   // inoffsets[1] = fes.GetTrueVSize();
   // inoffsets[2] = nodes->ParFESpace()->GetTrueVSize();
   // inoffsets.PartialSum();

   //    BlockVector X(inoffsets);
   //    X.GetBlock(0).Randomize(1);
   //    X.GetBlock(1) = *nodes;
   //    x.SetFromTrueDofs(X.GetBlock(0));

   //    Array<int> outoffsets(2);
   //    outoffsets[0] = 0;
   //    outoffsets[1] = fes.GetTrueVSize();
   //    outoffsets.PartialSum();
   //    BlockVector Z(outoffsets);

   //    ParBilinearForm blf(&fes);
   //    blf.AddDomainIntegrator(new MassIntegrator(one, ir));
   //    blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   //    blf.Assemble();
   //    blf.Mult(x, y);
   //    Vector Y(fes.GetTrueVSize());
   //    fes.GetProlongationMatrix()->MultTranspose(y, Y);

   //    static constexpr int U = 0, COORDINATES = 1, V = 2;
   //    const std::vector<FieldDescriptor> in
   //    {
   //       {U, &fes},
   //       {COORDINATES, nodes->ParFESpace()}
   //    };

   //    const std::vector<FieldDescriptor> out // test spaces?
   //    {
   //       {V, &fes},
   //    };
   //    DifferentiableOperator dop(in, out, pmesh);

   //    auto derivatives = std::integer_sequence<size_t, U> {};
   //    auto mass_qfunc = massqf{};
   //    dop.AddDomainIntegrator(mass_qfunc,
   //                            tuple{ Value<U>{}, Gradient<COORDINATES>{}, Weight{} },
   //                            tuple{ Value<V>{}, Value<V>{} },
   //                            *ir, all_domain_attr, derivatives);

   //    fes.GetRestrictionMatrix()->Mult(x, X.GetBlock(0));
   //    dop.Mult(X, Z);

   //    Vector Y0(Y);
   //    Y0 *= 2.0;
   //    Y0 -= Z.GetBlock(0);

   //    real_t norm_g, norm_l = Y0.Normlinf();
   //    MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
   //    REQUIRE(norm_g == MFEM_Approx(0.0));
   //    MPI_Barrier(MPI_COMM_WORLD);

   //    auto ddop = dop.GetDerivative(U, X);

   //    ddop->Mult(X.GetBlock(0), Z);
   //    Y0 = Y;
   //    Y0 *= 2.0;
   //    Y0 -= Z.GetBlock(0);

   //    norm_l = Y0.Normlinf();
   //    MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
   //    REQUIRE(norm_g == MFEM_Approx(0.0));
   //    MPI_Barrier(MPI_COMM_WORLD);
   // }

   {
      QuadratureSpace qs(pmesh, *ir);
      VectorQuadratureSpace vqs(qs, DIM * DIM);
      QuadratureFunction qdata(vqs);

      DummyParameterSpace dps;
      ParameterFunction dpf(dps);
      dpf = 9.12345;

      auto coef_func = [](const Vector &coords)
      {
         return coords[0] * coords[1] * (DIM == 3 ? coords[2] : 1.0);
      };
      FunctionCoefficient coef(coef_func);
      x.ProjectCoefficient(coef);

      Vector xtvec, ytvec, ytvecmfem;
      x.GetTrueDofs(xtvec);
      ytvec.SetSize(xtvec.Size());
      ytvecmfem.SetSize(xtvec.Size());

      Vector nodestvec;
      nodes->GetTrueDofs(nodestvec);

      qdata = 123.0;
      Vector yqdata(qdata.Size());

      static constexpr int U = 0, COORDINATES = 1, V = 2, S = 3, L = 4;

#if !defined(MFEM_USE_HIP)
      {
         xtvec.UseDevice(true);
         nodestvec.UseDevice(true);
         qdata.UseDevice(true);
         dpf.UseDevice(true);
         ytvec.UseDevice(true);
         yqdata.UseDevice(true);
         MultiVector X{xtvec, nodestvec, qdata, dpf};
         MultiVector Z{ytvec, yqdata};

         ParBilinearForm blf(&fes);
         blf.AddDomainIntegrator(new MassIntegrator(ir));
         blf.AddDomainIntegrator(new DiffusionIntegrator(ir));
         blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         blf.Assemble();
         blf.Mult(x, y);
         fes.GetProlongationMatrix()->MultTranspose(y, ytvecmfem);

         const std::vector<FieldDescriptor> in_fds
         {
            {U, &fes},
            {COORDINATES, nodes->ParFESpace()},
            {S, &vqs},
            {L, &dps}
         };

         const std::vector<FieldDescriptor> out_fds
         {
            {V, &fes},
            {S, &vqs}
         };

         DifferentiableOperator dop(in_fds, out_fds, pmesh);

         dop.SetQLayouts({{Value<U>{}, {1, 0}}}, {});

         auto derivatives = Derivatives<U> {};
         auto mass_diffusion_qfunc = mass_diffusion_global_qf{};
         constexpr auto kernels = DerivativeKernels::Action;
         dop.AddDomainIntegrator<GlobalQFBackend, kernels>(
            mass_diffusion_qfunc,
            Inputs<Value<U>, Gradient<U>, Gradient<COORDINATES>, Identity<S>, Weight, Value<L>> {},
            Outputs<Value<V>, Gradient<V>, Identity<S>> {},
            *ir, all_domain_attr, derivatives);

         fes.GetRestrictionMatrix()->Mult(x, xtvec);
         dop.Mult(X, Z);

         Vector Y0(ytvecmfem);
         Y0.UseDevice(true);

         Y0 -= Z[0];

         real_t norm_l = Y0.Normlinf();
         real_t norm_g = norm_l;
         MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         REQUIRE(norm_g == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);

         auto ddop = dop.GetDerivative(U, X);

         ddop->Mult(X[0], Z);
         Z[0].HostRead();
         Y0 = ytvecmfem;
         Y0.HostRead();
         Y0 -= Z[0];

         norm_l = Y0.Normlinf();
         norm_g = norm_l;
         MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         REQUIRE(norm_g == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
      }
#endif

      {
         static constexpr int W = 0;

         ParBilinearForm blf(&fes);
         blf.AddDomainIntegrator(new MassIntegrator(ir));
         blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         blf.Assemble();
         blf.Mult(x, y);
         fes.GetProlongationMatrix()->MultTranspose(y, ytvecmfem);

         const std::vector<FieldDescriptor> in_fds
         {
            {U, &fes},
            {COORDINATES, nodes->ParFESpace()},
         };

         const std::vector<FieldDescriptor> out_fds
         {
            {V, &fes},
            {W, &fes},
         };

         DifferentiableOperator dop(in_fds, out_fds, pmesh);

         auto mass_qfunclocal = mass_local_qf{};
         dop.AddDomainIntegrator<LocalQFBackend>(
            mass_qfunclocal,
            tuple{Value<U>{}, Gradient<COORDINATES>{}, Weight{}},
            tuple{Value<V>{}, Value<W>{}},
            *ir, all_domain_attr);

         Vector nodestv;
         nodes->GetTrueDofs(nodestv);
         fes.GetRestrictionMatrix()->Mult(x, xtvec);
         Vector ztvec(xtvec.Size());
         Vector zztvec(xtvec.Size());

         MultiVector X{xtvec, nodestv};
         MultiVector Z{ztvec, zztvec};

         dop.Mult(X, Z);

         Vector Y0(ytvecmfem);
         Y0 -= Z[0];

         real_t norm_l = Y0.Normlinf();
         real_t norm_g = norm_l;
         MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         REQUIRE(norm_g == MFEM_Approx(0.0));

         Vector Y1(ytvecmfem);
         Y1 -= Z[1];

         norm_l = Y1.Normlinf();
         norm_g = norm_l;
         MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         REQUIRE(norm_g == MFEM_Approx(0.0));

         MPI_Barrier(MPI_COMM_WORLD);
      }

      {
         ParBilinearForm blf(&fes);
         blf.AddDomainIntegrator(new MassIntegrator(ir));
         blf.AddDomainIntegrator(new DiffusionIntegrator(ir));
         blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         blf.Assemble();
         blf.Mult(x, y);
         fes.GetProlongationMatrix()->MultTranspose(y, ytvecmfem);

         const std::vector<FieldDescriptor> in_fds
         {
            {U, &fes},
            {COORDINATES, nodes->ParFESpace()},
         };

         const std::vector<FieldDescriptor> out_fds
         {
            {V, &fes},
         };

         DifferentiableOperator dop(in_fds, out_fds, pmesh);

         auto mass_diffusion_qfunclocal = mass_diffusion_local_qf{};
         dop.AddDomainIntegrator<LocalQFBackend>(
            mass_diffusion_qfunclocal,
            tuple{Value<U>{}, Gradient<U>{}, Gradient<COORDINATES>{}, Weight{}},
            tuple{Value<V>{}, Gradient<V>{}},
            *ir, all_domain_attr);

         Vector nodestv;
         nodes->GetTrueDofs(nodestv);
         fes.GetRestrictionMatrix()->Mult(x, xtvec);
         Vector ztvec(xtvec.Size());

         MultiVector X{xtvec, nodestv};
         MultiVector Z{ztvec};

         dop.Mult(X, Z);

         Vector Y0(ytvecmfem);
         Y0 -= Z[0];

         real_t norm_l = Y0.Normlinf();
         real_t norm_g = norm_l;
         MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         REQUIRE(norm_g == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
      }

      // Test for Assembly of dFEM with multiple outputs
      {
         ParBilinearForm blf_fa(&fes);
         blf_fa.AddDomainIntegrator(new MassIntegrator(ir));
         blf_fa.AddDomainIntegrator(new DiffusionIntegrator(ir));
         blf_fa.SetAssemblyLevel(AssemblyLevel::LEGACY);
         blf_fa.Assemble();
         blf_fa.Finalize();

         const std::vector<FieldDescriptor> in_fds
         {
            {U, &fes},
            {COORDINATES, nodes->ParFESpace()},
         };

         // The test field is named V even though it is the same space as the
         // trial field U, which is the usual "V is the test function" idiom.
         // The diagonal of dR_V/dU is then named explicitly.
         const std::vector<FieldDescriptor> out_fds
         {
            {V, &fes},
         };

         DifferentiableOperator dop(in_fds, out_fds, pmesh);

         auto qf = mass_diffusion_local_qf{};
         constexpr auto kernels =
            DerivativeKernels::AssembleMatrix |
            DerivativeKernels::AssembleDiagonal;
         dop.AddDomainIntegrator<LocalQFBackend, kernels>(
            qf,
            tuple{Value<U>{}, Gradient<U>{}, Gradient<COORDINATES>{}, Weight{}},
            tuple{Value<V>{}, Gradient<V>{}},
            *ir, all_domain_attr, Derivatives<U> {});

         Vector nodestv;
         nodes->GetTrueDofs(nodestv);
         fes.GetRestrictionMatrix()->Mult(x, xtvec);
         MultiVector X{xtvec, nodestv};
         auto dRdU = dop.GetDerivative(U, X);

         SECTION("Multiple Outputs SparseMatrix")
         {
            SparseMatrix *A = nullptr;
            dRdU->Assemble(A);

            // TestSameMatrices only walks the first matrix' sparsity pattern,
            // so compare both ways to catch entries missing from either side.
            TestSameMatrices(*A, blf_fa.SpMat());
            TestSameMatrices(blf_fa.SpMat(), *A);
            delete A;
            MPI_Barrier(MPI_COMM_WORLD);
         }

         SECTION("Multiple Outputs Assemble Diagonal")
         {
            Vector diag(fes.GetTrueVSize()), diag_ref_l(fes.GetVSize());
            dRdU->AssembleDiagonal(V, diag);
            blf_fa.SpMat().GetDiag(diag_ref_l);

            Vector diag_ref(fes.GetTrueVSize());
            fes.GetProlongationMatrix()->MultTranspose(diag_ref_l, diag_ref);

            diag -= diag_ref;
            real_t dnorm_l = diag.Normlinf(), dnorm_g = dnorm_l;
            MPI_Allreduce(&dnorm_l, &dnorm_g, 1, MPI_DOUBLE, MPI_MAX,
                          pmesh.GetComm());
            REQUIRE(dnorm_g == MFEM_Approx(0.0));
            MPI_Barrier(MPI_COMM_WORLD);
         }
      }

      // Outputs spanning two test fields. dR/dU is a block column, one row
      // block per output field, all sharing the trial space of U, and each row
      // block assembles into its own matrix.
      // This would look smth like:
      //
      // dR/dU = [ dR_P/dU; dR_U/dU ]
      //
      {
         static constexpr int P = 5;

         ParBilinearForm blf_u(&fes);
         blf_u.AddDomainIntegrator(new MassIntegrator(ir));
         blf_u.AddDomainIntegrator(new DiffusionIntegrator(ir));
         blf_u.SetAssemblyLevel(AssemblyLevel::LEGACY);
         blf_u.Assemble();
         blf_u.Finalize();

         ConstantCoefficient kappa_coeff(kappa);
         ParBilinearForm blf_p(&fes);
         blf_p.AddDomainIntegrator(new DiffusionIntegrator(kappa_coeff, ir));
         blf_p.SetAssemblyLevel(AssemblyLevel::LEGACY);
         blf_p.Assemble();
         blf_p.Finalize();

         const std::vector<FieldDescriptor> in_fds
         {
            {U, &fes},
            {COORDINATES, nodes->ParFESpace()},
         };

         // U is deliberately *not* the first output field: A[f] and the
         // blocks of Mult are indexed by position in out_fds, and
         // AssembleDiagonal has to pick the block whose field is the
         // differentiated one rather than simply the first.
         const std::vector<FieldDescriptor> out_fds
         {
            {P, &fes},
            {U, &fes},
         };

         DifferentiableOperator dop(in_fds, out_fds, pmesh);

         auto qf = two_field_local_qf{};
         constexpr auto kernels =
            DerivativeKernels::Action |
            DerivativeKernels::AssembleMatrix |
            DerivativeKernels::AssembleDiagonal;
         dop.AddDomainIntegrator<LocalQFBackend, kernels>(
            qf,
            tuple{Value<U>{}, Gradient<U>{}, Gradient<COORDINATES>{}, Weight{}},
            tuple{Value<U>{}, Gradient<U>{}, Gradient<P>{}},
            *ir, all_domain_attr, Derivatives<U> {});

         Vector nodestv;
         nodes->GetTrueDofs(nodestv);
         fes.GetRestrictionMatrix()->Mult(x, xtvec);
         MultiVector X{xtvec, nodestv};
         auto dRdU = dop.GetDerivative(U, X);

         REQUIRE(dRdU->Height() == 2 * fes.GetTrueVSize());
         REQUIRE(dRdU->Width() == fes.GetTrueVSize());

         SECTION("Multiple Fields SparseMatrix")
         {
            std::vector<SparseMatrix *> A;
            dRdU->Assemble(A);

            REQUIRE(A.size() == 2);
            REQUIRE(A[0] != nullptr);
            REQUIRE(A[1] != nullptr);

            // TestSameMatrices only goes thru the first matrix' sparsity pattern,
            // so if we compare both ways we can catch entries missing from either side.
            TestSameMatrices(*A[0], blf_p.SpMat());
            TestSameMatrices(blf_p.SpMat(), *A[0]);
            TestSameMatrices(*A[1], blf_u.SpMat());
            TestSameMatrices(blf_u.SpMat(), *A[1]);

            delete A[0];
            delete A[1];

            // The element matrix banks are reused, so assembling a second time
            // (what a Newton loop does every iteration) has to give the same
            // matrices instead of accumulating on top of the first ones.
            dRdU->Assemble(A);
            TestSameMatrices(*A[0], blf_p.SpMat());
            TestSameMatrices(*A[1], blf_u.SpMat());

            delete A[0];
            delete A[1];
            MPI_Barrier(MPI_COMM_WORLD);
         }

         SECTION("Multiple Fields HypreParMatrix")
         {
            std::vector<HypreParMatrix *> A;
            dRdU->Assemble(A);

            REQUIRE(A.size() == 2);
            REQUIRE(A[0] != nullptr);
            REQUIRE(A[1] != nullptr);

            std::unique_ptr<HypreParMatrix> ref_u(blf_u.ParallelAssemble());
            std::unique_ptr<HypreParMatrix> ref_p(blf_p.ParallelAssemble());

            // The assembled blocks have to agree with the matrix free action
            // block by block, which is also what pins A[f] to output field f.
            Vector dir(fes.GetTrueVSize());
            dir.Randomize(7);

            Vector yp(fes.GetTrueVSize()), yu(fes.GetTrueVSize());
            MultiVector Y{yp, yu};
            dRdU->Mult(dir, Y);

            Vector a(fes.GetTrueVSize());
            A[0]->Mult(dir, a);
            a -= Y[0];
            real_t norm_l = a.Normlinf(), norm_g = norm_l;
            MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX,
                          pmesh.GetComm());
            REQUIRE(norm_g == MFEM_Approx(0.0));
            Vector r(fes.GetTrueVSize());
            ref_p->Mult(dir, r);
            a = Y[0];
            a -= r;
            norm_l = a.Normlinf();
            MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX,
                          pmesh.GetComm());
            REQUIRE(norm_g == MFEM_Approx(0.0));

            A[1]->Mult(dir, a);
            a -= Y[1];
            norm_l = a.Normlinf();
            MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX,
                          pmesh.GetComm());
            REQUIRE(norm_g == MFEM_Approx(0.0));
            ref_u->Mult(dir, r);
            a = Y[1];
            a -= r;
            norm_l = a.Normlinf();
            MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX,
                          pmesh.GetComm());
            REQUIRE(norm_g == MFEM_Approx(0.0));

            delete A[0];
            delete A[1];
            MPI_Barrier(MPI_COMM_WORLD);
         }

         SECTION("Multiple Fields Assemble Diagonal")
         {
            // Both row blocks are square here, since P and U share a space, so
            // squareness alone cannot pick one. They carry different operators
            // though, so reading the wrong row block cannot pass unnoticed.
            const auto check_diag = [&](const Vector &diag,
                                        ParBilinearForm &ref)
            {
               REQUIRE(diag.Size() == fes.GetTrueVSize());

               Vector diag_ref_l(fes.GetVSize());
               ref.SpMat().GetDiag(diag_ref_l);
               Vector diag_ref(fes.GetTrueVSize());
               fes.GetProlongationMatrix()->MultTranspose(diag_ref_l, diag_ref);

               Vector d(diag);
               d -= diag_ref;
               real_t norm_l = d.Normlinf(), norm_g = norm_l;
               MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX,
                             pmesh.GetComm());
               REQUIRE(norm_g == MFEM_Approx(0.0));
            };

            // Default: the row block of the differentiated field, dR_U/dU.
            Vector diag_u;
            dRdU->AssembleDiagonal(diag_u);
            check_diag(diag_u, blf_u);

            // Named: the other row block, dR_P/dU.
            Vector diag_p;
            dRdU->AssembleDiagonal(P, diag_p);
            check_diag(diag_p, blf_p);

            MPI_Barrier(MPI_COMM_WORLD);
         }
      }

      // The same two field structure, but now the two output fields live on
      // different spaces. That is what makes AssembleDiagonal's choice of row
      // block observable at all: with both fields on one space, reading the
      // wrong block still lands on an identically sized vector holding the
      // same numbers, so the check above cannot tell the two apart.
      {
         static constexpr int P = 5;

         H1_FECollection fec_p(p + 1, DIM);
         ParFiniteElementSpace fes_p(&pmesh, &fec_p);

         ParBilinearForm blf_u(&fes);
         blf_u.AddDomainIntegrator(new MassIntegrator(ir));
         blf_u.AddDomainIntegrator(new DiffusionIntegrator(ir));
         blf_u.SetAssemblyLevel(AssemblyLevel::LEGACY);
         blf_u.Assemble();
         blf_u.Finalize();

         const std::vector<FieldDescriptor> in_fds
         {
            {U, &fes},
            {COORDINATES, nodes->ParFESpace()},
         };

         // U is the second output field on purpose, and fes_p is a different
         // space, so picking out_fds[0] would give a differently sized vector.
         const std::vector<FieldDescriptor> out_fds
         {
            {P, &fes_p},
            {U, &fes},
         };

         DifferentiableOperator dop(in_fds, out_fds, pmesh);

         auto qf = two_field_local_qf{};
         constexpr auto kernels = DerivativeKernels::AssembleDiagonal;
         dop.AddDomainIntegrator<LocalQFBackend, kernels>(
            qf,
            tuple{Value<U>{}, Gradient<U>{}, Gradient<COORDINATES>{}, Weight{}},
            tuple{Value<U>{}, Gradient<U>{}, Gradient<P>{}},
            *ir, all_domain_attr, Derivatives<U> {});

         Vector nodestv;
         nodes->GetTrueDofs(nodestv);
         fes.GetRestrictionMatrix()->Mult(x, xtvec);
         MultiVector X{xtvec, nodestv};
         auto dRdU = dop.GetDerivative(U, X);

         SECTION("Assemble Diagonal Picks The Square Block")
         {
            Vector diag;
            dRdU->AssembleDiagonal(diag);

            // Sized by U's space, not by the first output field's.
            REQUIRE(diag.Size() == fes.GetTrueVSize());

            Vector diag_ref_l(fes.GetVSize());
            blf_u.SpMat().GetDiag(diag_ref_l);
            Vector diag_ref(fes.GetTrueVSize());
            fes.GetProlongationMatrix()->MultTranspose(diag_ref_l, diag_ref);

            diag -= diag_ref;
            real_t norm_l = diag.Normlinf(), norm_g = norm_l;
            MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX,
                          pmesh.GetComm());
            REQUIRE(norm_g == MFEM_Approx(0.0));
            MPI_Barrier(MPI_COMM_WORLD);
         }
      }

      // One assemblable row block and one that is not: S lives on quadrature
      // points, so it has no basis to contract against and stays null, while
      // the mass + diffusion block on V assembles as usual.
      {
         ParBilinearForm blf_fa(&fes);
         blf_fa.AddDomainIntegrator(new MassIntegrator(ir));
         blf_fa.AddDomainIntegrator(new DiffusionIntegrator(ir));
         blf_fa.SetAssemblyLevel(AssemblyLevel::LEGACY);
         blf_fa.Assemble();
         blf_fa.Finalize();

         const std::vector<FieldDescriptor> in_fds
         {
            {U, &fes},
            {COORDINATES, nodes->ParFESpace()},
         };

         const std::vector<FieldDescriptor> out_fds
         {
            {V, &fes},
            {S, &vqs},
         };

         DifferentiableOperator dop(in_fds, out_fds, pmesh);

         auto qf = mass_diffusion_qdata_local_qf{};
         constexpr auto kernels = DerivativeKernels::AssembleMatrix;
         dop.AddDomainIntegrator<LocalQFBackend, kernels>(
            qf,
            tuple{Value<U>{}, Gradient<U>{}, Gradient<COORDINATES>{}, Weight{}},
            tuple{Value<V>{}, Gradient<V>{}, Identity<S>{}},
            *ir, all_domain_attr, Derivatives<U> {});

         Vector nodestv;
         nodes->GetTrueDofs(nodestv);
         fes.GetRestrictionMatrix()->Mult(x, xtvec);
         MultiVector X{xtvec, nodestv};
         auto dRdU = dop.GetDerivative(U, X);

         SECTION("Partly Assemblable Outputs SparseMatrix")
         {
            std::vector<SparseMatrix *> A;
            dRdU->Assemble(A);

            REQUIRE(A.size() == 2);
            REQUIRE(A[0] != nullptr);
            REQUIRE(A[1] == nullptr);

            TestSameMatrices(*A[0], blf_fa.SpMat());
            TestSameMatrices(blf_fa.SpMat(), *A[0]);

            delete A[0];
            MPI_Barrier(MPI_COMM_WORLD);
         }
      }
   }
}

#endif // MFEM_USE_MPI
