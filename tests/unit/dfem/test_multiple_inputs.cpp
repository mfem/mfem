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

template<int DIM, typename ValSeq, typename GradSeq>
struct multiply_inputs_qf;

template<int DIM, std::size_t... Vs, std::size_t... Gs>
struct multiply_inputs_qf<DIM, std::index_sequence<Vs...>,
          std::index_sequence<Gs...>>
{
   template<std::size_t>
   using scalar_ref = const dscalar_t &;
   template<std::size_t>
   using grad_ref = const tensor<dscalar_t, DIM> &;

   inline MFEM_HOST_DEVICE void operator()(scalar_ref<Vs>... us,
                                           grad_ref<Gs>... gs,
                                           const tensor<real_t, DIM, DIM> &J,
                                           const real_t &w,
                                           dscalar_t &v) const
   {
      v = (us * ...) * (1.0 + (w - w) * (gs(0) + ...)) * w * det(J);
   }
};

template<int DIM, std::size_t Nv, std::size_t Ng>
using multiply_inputs_qf_t = multiply_inputs_qf<
                             DIM, std::make_index_sequence<Nv>, std::make_index_sequence<Ng>>;

template<int U, int /*I*/>
using value_input = Value<U>;

template<int U, int /*I*/>
using gradient_input = Gradient<U>;

template<int U, int Coords, int... Vs, int... Gs>
constexpr auto make_inputs(std::integer_sequence<int, Vs...>,
                           std::integer_sequence<int, Gs...>)
{
   return Inputs<
          value_input<U, Vs>...,
          gradient_input<U, Gs>...,
          Gradient<Coords>,
          Weight
          > {};
}

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

template<int DIM, int U, int Coords, std::size_t Nv, std::size_t Ng>
void test_n_inputs(const InputsTestContext &ctx)
{
   CAPTURE(DIM, Nv, Ng);

   const std::size_t n_u = Nv + Ng;

   std::vector<FieldDescriptor> in_fds;
   in_fds.reserve(n_u + 1);
   for (std::size_t i = 0; i < n_u; ++i)
   {
      in_fds.emplace_back(U, &ctx.pfes);
   }
   in_fds.emplace_back(Coords, ctx.mfes);
   const auto out_fds = std::vector{ FieldDescriptor{ U, &ctx.pfes } };

   Array<int> mx_sizes(n_u + 1);
   for (std::size_t i = 0; i < n_u; ++i)
   {
      mx_sizes[i] = ctx.tvsize;
   }
   mx_sizes[n_u] = ctx.N_vec.Size();

   MultiVector MX;
   MX.SetSizes(mx_sizes);
   for (std::size_t i = 0; i < n_u; ++i)
   {
      MX.MakeRef(static_cast<int>(i), ctx.input);
   }
   MX.MakeRef(static_cast<int>(n_u), ctx.N_vec);

   using IT = decltype(make_inputs<U, Coords>(
                          std::make_integer_sequence<int, Nv> {},
                          std::make_integer_sequence<int, Ng> {}));
   using OT = Outputs<Value<U>>;

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

template<int DIM, int U, int Coords, std::size_t MaxV, std::size_t MaxG>
void test_inputs(int p)
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

   const InputsTestContext ctx
   {
      pmesh, pfes, mfes, ir, all_domain_attr, Y_ref, N_vec, input, tvsize
   };

   for_constexpr([&](auto I)
   {
      for_constexpr([&](auto J)
      {
         DYNAMIC_SECTION("Nv = " << (1 + I.value) << ", Ng = " << (1 + J.value))
         {
            test_n_inputs<DIM, U, Coords, 1 + I.value, 1 + J.value>(ctx);
         }
      }, std::make_integer_sequence<std::size_t, MaxG> {});
   }, std::make_integer_sequence<std::size_t, MaxV> {});
}

TEST_CASE("dFEM Inputs", "[Parallel][dFEM][GPU][INPUTS]")
{
   static constexpr int DIM = 2, U = 0, Coords = 1;
   static constexpr std::size_t MaxV = 3, MaxG = 3;
   const auto p = GenAll({ 1 }, { 2, 3 });

   test_inputs<DIM, U, Coords, MaxV, MaxG>(p);
}

#endif // MFEM_USE_MPI
