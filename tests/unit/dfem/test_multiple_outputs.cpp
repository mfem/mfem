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

// Generic proxy for multiple members
template<typename... Ts>
struct QDataProxy
{
   std::tuple<Ts&...> refs;

   template<typename... Args>
   QDataProxy(Args&... args) : refs(args...) {}

   template<size_t I>
   auto& get() { return std::get<I>(refs); }

   template<size_t I>
   const auto& get() const { return std::get<I>(refs); }
};

// Specialization for single member - allows direct assignment
template<typename T>
struct QDataProxy<T>
{
   T& ref;

   QDataProxy(T& r) : ref(r) {}

   template<size_t I>
   auto& get() { static_assert(I == 0); return ref; }

   template<size_t I>
   const auto& get() const { static_assert(I == 0); return ref; }

   // Allow direct assignment
   template<typename U>
   QDataProxy& operator=(const U& value)
   {
      ref = value;
      return *this;
   }

   // Allow implicit conversion to reference
   operator T&() { return ref; }
   operator const T&() const { return ref; }
};

// Enable structured bindings for QDataProxy
namespace std
{
template<typename... Ts>
struct tuple_size<QDataProxy<Ts...>> :
                                  integral_constant<size_t, sizeof...(Ts)> {};

template<size_t I, typename... Ts>
struct tuple_element<I, QDataProxy<Ts...>>
{
   using type = tuple_element_t<I, tuple<Ts...>>;
};
}

// Generic view for reading SoA data
template<typename... MemberTypes>
struct QDataView
{
   std::tuple<const MemberTypes*...> block_ptrs;

   QDataView(const BlockVector* bv)
   {
      init_blocks(bv, std::index_sequence_for<MemberTypes...> {});
   }

   template<size_t... Is>
   void init_blocks(const BlockVector* bv, std::index_sequence<Is...>)
   {
      ((std::get<Is>(block_ptrs) = reinterpret_cast<const MemberTypes*>(
                                      bv->GetBlock(Is).Read()
                                   )), ...);
   }

   auto operator[](int q) const
   {
      return make_proxy(q, std::index_sequence_for<MemberTypes...> {});
   }

   template<size_t... Is>
   auto make_proxy(int q, std::index_sequence<Is...>) const
   {
      return QDataProxy<const MemberTypes...>(std::get<Is>(block_ptrs)[q]...);
   }
};

// Generic view for writing SoA data
template<typename... MemberTypes>
struct QDataViewMut
{
   std::tuple<MemberTypes*...> block_ptrs;

   QDataViewMut(Vector* v)
   {
      BlockVector* bv = static_cast<BlockVector*>(v);
      init_blocks(bv, std::index_sequence_for<MemberTypes...> {});
   }

   template<size_t... Is>
   void init_blocks(BlockVector* bv, std::index_sequence<Is...>)
   {
      ((std::get<Is>(block_ptrs) = reinterpret_cast<MemberTypes*>(
                                      bv->GetBlock(Is).ReadWrite()
                                   )), ...);
   }

   auto operator[](int q)
   {
      return make_proxy(q, std::index_sequence_for<MemberTypes...> {});
   }

   template<size_t... Is>
   auto make_proxy(int q, std::index_sequence<Is...>)
   {
      return QDataProxy<MemberTypes...>(std::get<Is>(block_ptrs)[q]...);
   }
};

TEST_CASE("dFEM Multiple Outputs", "[Parallel][dFEM]")
{
   const bool all_tests = launch_all_non_regression_tests;

   const auto p = !all_tests ? 2 : GENERATE(1, 2, 3);

   constexpr int DIM = 2;
   const char *filename = "../../data/inline-quad.mesh";
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

   std::cout << "MFEM: M * x = ";
   pretty_print(Y);

   // pretty_print(x);
   // pretty_print(X);

   // static constexpr int U = 0, AUX = 1, COORDINATES = 2;
   static constexpr int U = 0, COORDINATES = 1, AUX = 2;
   const std::vector<FieldDescriptor> in
   {
      {U, &fes},
      {COORDINATES, nodes->ParFESpace()}
   };
   const std::vector<FieldDescriptor> out
   {
      {U, &fes}
   };
   DifferentiableOperator dop(in, out, pmesh);

   const auto mass_globalqf =
      [] MFEM_HOST_DEVICE (const int nq, const BlockVector *qdin, BlockVector *qdout)
   {
      QDataView<real_t, tensor<real_t, DIM, DIM>, real_t> in(qdin);
      QDataViewMut<real_t> out(qdout);

      for (int q = 0; q < nq; q++)
      {
         const auto& [u, J, w] = in[q];
         std::cout << "u: " << u << "\n";
         std::cout << "J: " << J << "\n";
         std::cout << "w: " << w << "\n";

         out[q] = u * det(J) * w;
         std::cout << out[q] << "\n";
      }

      // TODO
      real_t u;
      return tuple{u};
   };

   dop.AddDomainIntegrator(mass_globalqf,
                           tuple{ Value<U>{}, Gradient<COORDINATES>{}, Weight{} },
                           tuple{ Value<U>{} },
                           *ir, all_domain_attr);

   fes.GetRestrictionMatrix()->Mult(x, X.GetBlock(0));
   dop.Mult(X, Z);
   pretty_print(X);
   pretty_print(Z);
   Y -= Z;

   real_t norm_g, norm_l = Y.Normlinf();
   MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
   REQUIRE(norm_g == MFEM_Approx(0.0));
   MPI_Barrier(MPI_COMM_WORLD);
}

#endif // MFEM_USE_MPI
