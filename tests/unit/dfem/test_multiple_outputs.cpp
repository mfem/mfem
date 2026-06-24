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
         for (int i = 0; i < y.Size(); i++)
         {
            y(i) = x(0);
         }
      }
   };

   class Btimpl : public Operator
   {
      void Mult(const Vector &x, Vector &y) const override
      {
         y(0) = x(0);
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
      mfem::forall(u.size(), [=] MFEM_HOST_DEVICE (int q)
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
      const real_t &u,
      const tensor<real_t, DIM> &dudxi,
      const tensor<real_t, DIM, DIM> &J,
      const real_t &w,
      real_t &out1,
      tensor<real_t, DIM> &out2) const
   {
      const auto invJ = inv(J);
      const auto detJ = det(J);
      out1 = u * detJ * w;
      out2 = (dudxi * invJ) * transpose(invJ) * (detJ * w);
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

      {
         MultiVector X {xtvec, nodestvec, qdata, dpf};
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
         dop.AddDomainIntegrator(
            mass_diffusion_qfunc,
            Inputs<Value<U>, Gradient<U>, Gradient<COORDINATES>, Identity<S>, Weight, Value<L>> {},
            Outputs<Value<V>, Gradient<V>, Identity<S>> {},
            *ir, all_domain_attr, derivatives);

         fes.GetRestrictionMatrix()->Mult(x, xtvec);
         dop.Mult(X, Z);

         Vector Y0(ytvecmfem);
         Y0 -= Z[0];

         real_t norm_l = Y0.Normlinf();
         real_t norm_g = norm_l;
         MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         REQUIRE(norm_g == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);

         auto ddop = dop.GetDerivative(U, X);

         ddop->Mult(X[0], Z);
         Y0 = ytvecmfem;
         Y0 -= Z[0];

         norm_l = Y0.Normlinf();
         norm_g = norm_l;
         MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         REQUIRE(norm_g == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
      }

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
   }
}

#endif // MFEM_USE_MPI
