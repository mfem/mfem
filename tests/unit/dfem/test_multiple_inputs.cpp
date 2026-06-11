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

template<std::size_t MaxN, typename F>
constexpr void for_n_range(F &&f)
{
   static_assert(MaxN >= 1, "At least one scalar input is required");
   for_constexpr([&](auto I)
   {
      f(std::integral_constant<std::size_t, 1 + I.value>{});
   }, std::make_integer_sequence<std::size_t, MaxN>{});
}

template<int F, int Coords, int... Is>
constexpr auto make_inputs(std::integer_sequence<int, Is...>)
{
   return Inputs<
      typename std::enable_if<(Is || !Is), Value<F>>::type...,
      Gradient<Coords>,
      Weight
   >{};
}

template<int DIM, typename Seq>
struct multiply_inputs_qf;

template<int DIM, std::size_t... Is>
struct multiply_inputs_qf<DIM, std::index_sequence<Is...>>
{
   template<std::size_t>
   using scalar_arg = const dscalar_t &;

   inline MFEM_HOST_DEVICE void operator()(scalar_arg<Is>... us,
                                           const tensor<real_t, DIM, DIM> &J,
                                           const real_t &w,
                                           dscalar_t &v) const
   {
      static_assert(sizeof...(Is) > 0, "At least one scalar input is required");
      v = (us * ...) * w * det(J);
   }
};

template<int DIM, std::size_t N>
using multiply_inputs_qf_t = multiply_inputs_qf<DIM, std::make_index_sequence<N>>;

struct ScalarInputsContext
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

template<int DIM, int U, int Coords, std::size_t N>
void test_n_scalar_inputs(const ScalarInputsContext &ctx)
{
   CAPTURE(DIM, N);

   std::vector<FieldDescriptor> in_fds;
   in_fds.reserve(N + 1);
   for (std::size_t i = 0; i < N; ++i)
   {
      in_fds.push_back({ U, &ctx.pfes });
   }
   in_fds.push_back({ Coords, ctx.mfes });
   const auto out_fds = std::vector{ FieldDescriptor{ U, &ctx.pfes } };

   Array<int> mx_sizes(N + 1);
   for (std::size_t i = 0; i < N; ++i)
   {
      mx_sizes[i] = ctx.tvsize;
   }
   mx_sizes[N] = ctx.N_vec.Size();

   MultiVector MX;
   MX.SetSizes(mx_sizes);
   for (std::size_t i = 0; i < N; ++i)
   {
      MX.MakeRef(static_cast<int>(i), ctx.input);
   }
   MX.MakeRef(static_cast<int>(N), ctx.N_vec);

   using IT = decltype(make_inputs<U, Coords>(
                          std::make_integer_sequence<int, N>{}));
   using OT = Outputs<Value<U>>;

   multiply_inputs_qf_t<DIM, N> qfn;

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

template<int DIM, int U, int Coords, std::size_t MaxN>
void test_scalar_inputs(int p)
{
   CAPTURE(DIM, p);

   Mesh smesh("../../data/inline-quad.mesh");
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   smesh.Clear();

   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());

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
   Vector X(tvsize), Y_ref(tvsize);

   X = 1.0;
   x.SetFromTrueDofs(X);

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

   const ScalarInputsContext ctx
   {
      pmesh, pfes, mfes, ir, all_domain_attr, Y_ref, N_vec, input, tvsize
   };

   for_n_range<MaxN>([&](auto N)
   {
      DYNAMIC_SECTION("N = " << N.value)
      {
         test_n_scalar_inputs<DIM, U, Coords, N.value>(ctx);
      }
   });
}

TEST_CASE("dFEM Inputs", "[Parallel][dFEM][GPU][INPUTS]")
{
   static constexpr int DIM = 2, U = 0, Coords = 1;
   static constexpr std::size_t MaxN = 4;
   const auto p = GenAll({ 1 }, { 2, 3 });

   test_scalar_inputs<DIM, U, Coords, MaxN>(p);
}

#endif // MFEM_USE_MPI
