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
template<int DIM, typename ValSeq, typename GradSeq>
struct multiply_inputs_qf;

template<int DIM, std::size_t... Vs, std::size_t... Gs>
struct multiply_inputs_qf<DIM, std::index_sequence<Vs...>,
          std::index_sequence<Gs...>>
{
   template<std::size_t>
   using value_t = const dscalar_t &;

   template<std::size_t>
   using gradient_t = const tensor<dscalar_t, DIM> &;

   inline MFEM_HOST_DEVICE void operator()(
      const tensor<real_t, DIM, DIM> &J,
      value_t<Vs>... vs,
      gradient_t<Gs>... gs,
      const real_t &w,
      dscalar_t &v) const
   {
      v = (vs * ...) * (real_t{1} + (w - w) * (gs(0) + ...)) * w * det(J);
   }
};

template<int DIM, std::size_t Nv, std::size_t Ng>
using multiply_inputs_qf_t =
   multiply_inputs_qf<DIM,
   std::make_index_sequence<Nv>,
   std::make_index_sequence<Ng>>;

// ────────────────────────────────────────────────────────────────────────────
template<int I>
using Values = Value<I>;

template<int I>
using Gradients = Gradient<I>;

template<int... Vs, int... Gs>
constexpr auto make_inputs(std::integer_sequence<int, Vs...>,
                           std::integer_sequence<int, Gs...>)
{
   return Inputs<Gradient<0>, Values<1 + Vs>...,
          Gradients<1 + sizeof...(Vs) + Gs>..., Weight> {};
}

// ────────────────────────────────────────────────────────────────────────────
struct InputsTestContext
{
   ParMesh &pmesh;
   ParFiniteElementSpace &pfes;
   ParFiniteElementSpace *mfes;
   const IntegrationRule &ir;
   Array<int> &all_domain_attr;
   const Vector &Y_ref;
   Vector &N_vec;
   Vector &input;
   int tvsize;
};

// ────────────────────────────────────────────────────────────────────────────
template<int DIM, std::size_t Nv, std::size_t Ng>
void test_nv_ng_inputs(const InputsTestContext &ctx)
{
   constexpr std::size_t Ni = Nv + Ng;

   // inputs & output
   std::vector<FieldDescriptor> in_fds;
   in_fds.reserve(Ni + 1);
   in_fds.emplace_back(0, ctx.mfes); // Coords
   for (std::size_t i = 1; i <= Ni; ++i)
   {
      in_fds.emplace_back(i, &ctx.pfes);
   }
   const auto out_fds = std::vector{ FieldDescriptor{ 1, &ctx.pfes } };

   // Prepare the MultiVector inputs
   Array<int> mx_sizes(Ni + 1);
   mx_sizes[0] = ctx.N_vec.Size(); // Coords
   for (std::size_t i = 1; i <= Ni; ++i) { mx_sizes[i] = ctx.tvsize; }

   MultiVector MX;
   MX.SetSizes(mx_sizes);
   MX.MakeRef(0, ctx.N_vec);
   for (std::size_t i = 1; i <= Ni; ++i)
   {
      MX.MakeRef(static_cast<int>(i), ctx.input);
   }

   using IT = decltype(make_inputs(
                          std::make_integer_sequence<int, Nv> {},
                          std::make_integer_sequence<int, Ng> {}));
   using OT = Outputs<Value<1>>;

   multiply_inputs_qf_t<DIM, Nv, Ng> qfn;

   DifferentiableOperator dop(in_fds, out_fds, ctx.pmesh);
   dop.AddDomainIntegrator<LocalQFBackend>(
      qfn, IT{}, OT{}, ctx.ir, ctx.all_domain_attr);

   Vector Z(ctx.tvsize);
   MultiVector MZ{ Z };
   dop.Mult(MX, MZ);

   Vector Y_diff(ctx.Y_ref);
   Y_diff -= Z;

   ParGridFunction y(&ctx.pfes);
   ConstantCoefficient zero(0.0);
   y.SetFromTrueDofs(Y_diff);
   REQUIRE(y.ComputeMaxError(zero) == MFEM_Approx(0.0));
}

// ────────────────────────────────────────────────────────────────────────────
void test_multiple_inputs(int p)
{
   static constexpr int DIM = 2;
   CAPTURE(DIM, p);

   Mesh smesh("../../data/inline-quad.mesh");
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   smesh.Clear();

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   ParFiniteElementSpace *mfes = nodes->ParFESpace();
   const auto &ir = IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   const int tvsize = pfes.GetTrueVSize();

   ParGridFunction x(&pfes), y(&pfes);
   Vector X_ref(tvsize), Y_ref(tvsize);

   X_ref = 1.0;
   x.SetFromTrueDofs(X_ref);

   ConstantCoefficient one(1.0);

   ParBilinearForm blf(&pfes);
   blf.AddDomainIntegrator(new MassIntegrator(one, &ir));
   blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf.Assemble();
   blf.Mult(x, y);
   pfes.GetProlongationMatrix()->MultTranspose(y, Y_ref);

   Vector N_vec;
   nodes->GetTrueDofs(N_vec);

   Vector input(tvsize);
   input = 1.0;

   const InputsTestContext ctx
   {
      pmesh, pfes, mfes, ir, all_domain_attr, Y_ref, N_vec, input, tvsize
   };

   // with current dFEM tuple: max 9 = J + 3 + 3 + weights + output
#ifndef _WIN32
   test_nv_ng_inputs<DIM, 3, 3>(ctx);
#else
   // avoiding 'number of sections exceeded object file format limit' error
   test_nv_ng_inputs<DIM, 2, 2>(ctx);
#endif // _WIN32
}

// ────────────────────────────────────────────────────────────────────────────
TEST_CASE("dFEM Inputs", "[Parallel][dFEM][GPU]")
{
   const auto p = GenAll({1}, {2, 3, 8});
   test_multiple_inputs(p);
}

#endif // MFEM_USE_MPI
