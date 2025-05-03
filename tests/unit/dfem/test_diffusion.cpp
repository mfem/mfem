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

#include "unit_tests.hpp"
#include "mfem.hpp"
// #include <utility>
#include <type_traits>
#include "fem/dfem/doperator.hpp"
#include "fem/dfem/util.hpp"

#include <fem/integ/bilininteg_diffusion_kernels.hpp>

#undef NVTX_COLOR
#define NVTX_COLOR nvtx::kGold
#include "general/nvtx.hpp"

#ifdef MFEM_USE_MPI

using namespace mfem;
using namespace mfem::future;
using mfem::future::tensor;
using mfem::future::dual;

using DOperator = future::DifferentiableOperator;

namespace dfem_pa_kernels
{

///////////////////////////////////////////////////////////////////////////////
template <typename T, int DIM, int T_MQ1 = 0> struct Diffusion
{
   using dvecd_t = tensor<T, DIM>;
   using matd_t = tensor<real_t, DIM, DIM>;

   struct MFApply
   {
      static constexpr int MQ1 = T_MQ1;
      MFEM_HOST_DEVICE inline auto operator()(const dvecd_t &dudxi,
                                              const real_t &rho,
                                              const matd_t &J,
                                              const real_t &w) const
      {
         const auto invJ = inv(J), TinJ = transpose(invJ);
         return mfem::future::tuple{ (dudxi * invJ) * TinJ * det(J) * w * rho };
      }
   };

   struct PASetup
   {
      MFEM_HOST_DEVICE inline auto operator()(const real_t u,
                                              const real_t &rho,
                                              const matd_t &J,
                                              const real_t &w) const
      {
         return mfem::future::tuple{ inv(J) * transpose(inv(J)) * det(J) * w * rho };
      }
   };

   struct PAApply
   {
      MFEM_HOST_DEVICE inline auto operator()(const dvecd_t &dudxi,
                                              const matd_t &q) const
      {
         return mfem::future::tuple{ q * dudxi };
      };
   };
};

///////////////////////////////////////////////////////////////////////////////
template <typename T, int DIM, std::size_t... MQ1s>
struct MFDiffusionFactory
{
   static auto All()
   {
      return mfem::future::make_tuple(typename Diffusion<T, DIM, MQ1s>::MFApply{}...);
   }
};

template <typename T, int DIM>
using MFDiffusionFactory1to4 = MFDiffusionFactory<T, DIM, 1, 2, 3, 4>;

template <typename T, int DIM>
using TMFApplyVariant = std::variant<
                        typename Diffusion<T, DIM, 1>::MFApply,
                        typename Diffusion<T, DIM, 2>::MFApply,
                        typename Diffusion<T, DIM, 3>::MFApply,
                        typename Diffusion<T, DIM, 4>::MFApply>;

// Declare the map variable with the correct type
std::map<int, TMFApplyVariant<real_t, 3>> map;

// Example class storing the MFApply objects
template <typename T, int DIM>
struct MFDiffusionQFs
{
   using MFApplyTuple = decltype(MFDiffusionFactory1to4<T, DIM>::All());
   MFApplyTuple mf_qfs;

   MFDiffusionQFs(): mf_qfs(MFDiffusionFactory1to4<T, DIM>::All()) {}
};

/*template <typename T, int DIM, std::size_t N>
auto extract(const TMFApplyVariant<T, DIM>& variant)
{
   static_assert(N >= 1 && N <= 4, "N must be between 1 and 4");
   using TargetType = typename Diffusion<real_t, DIM, N>::MFApply;
   if (variant.index() != (N - 1))
   {
      throw std::runtime_error("Variant does not hold the expected type for N");
   }
   return mfem::future::get<TargetType>(variant);
}*/

///////////////////////////////////////////////////////////////////////////////
template <typename... qf_ts, std::size_t... Is>
auto make_qf_map_impl(tuple<qf_ts...> qfs,
                      std::index_sequence<Is...>)
{
   auto make_qf_array = [&](auto i)
   {
      return std::array<bool, sizeof...(qf_ts)>
      {
         (mfem::future::get<i>(qfs).MQ1 == mfem::future::get<Is>(qfs).MQ1)...
      };
   };

   std::unordered_map<int, std::array<bool, sizeof...(qf_ts)>> map;
   for_constexpr<sizeof...(qf_ts)>([&](auto i)
   {
      map[get<i>(qfs).MQ1] = make_qf_array(std::integral_constant<std::size_t, i> {});
   });

   return map;
}
template <typename... qf_ts>
auto make_qf_map(mfem::future::tuple<qf_ts...> qfs)
{
   return make_qf_map_impl(qfs, std::index_sequence_for<qf_ts...> {});
}

///////////////////////////////////////////////////////////////////////////////
template <typename Tuple, typename F, size_t... I>
void runtime_get_impl(Tuple& t, size_t index, F&& f, std::index_sequence<I...>)
{
   using fun_ptr = void (*)(Tuple&, F&&);
   fun_ptr table[] =
   {
      [](Tuple& t, F&& f) { f(mfem::future::get<I>(t)); } ...
   };
   if (index < mfem::future::tuple_size<Tuple>::value)
   {
      table[index](t, std::forward<F>(f));
   }
   else
   {
      throw std::out_of_range("Index out of bounds");
   }
}

template <typename Tuple, typename F>
void runtime_get(Tuple& t, size_t index, F&& f)
{
   runtime_get_impl(t, index, std::forward<F>(f),
                    std::make_index_sequence<mfem::future::tuple_size<Tuple>::value>());
}

///////////////////////////////////////////////////////////////////////////////
// 1️⃣ https://www.foonathan.net/2017/03/tuple-iterator/
template <typename Tup, typename R, typename F, std::size_t... Idxs>
struct tuple_runtime_access_table
{
   using tuple_type = Tup;
   using return_type = R;
   using converter_fun = F;

   template <std::size_t N>
   static return_type access_tuple(tuple_type& t, converter_fun& f)
   {
      return f(mfem::future::get<N>(t));
   }

   using accessor_fun_ptr = return_type(*)(tuple_type&, converter_fun&);
   const static auto table_size = sizeof...(Idxs);

   constexpr static std::array<accessor_fun_ptr, table_size> lookup_table =
   {
      {&access_tuple<Idxs>...}
   };
};

template <typename R, typename Tup, typename F, std::size_t... Idxs>
auto call_access_function(Tup& t, std::size_t i, F f,
                          std::index_sequence<Idxs...>)
{
   auto& table = tuple_runtime_access_table<Tup, R, F, Idxs...>::lookup_table;
   auto* access_function = table[i];
   return access_function(t, f);
}

template <typename Tup> struct common_tuple_access;

template <typename... Ts>
struct common_tuple_access<mfem::future::tuple<Ts...>>
{
   using type = std::variant<std::reference_wrapper<Ts>...>;
};

// template <typename T1, typename T2>
// struct common_tuple_access<std::pair<T1, T2>>
// {
//    using type =
//       std::variant<std::reference_wrapper<T1>, std::reference_wrapper<T2>>;
// };

// template <typename T, auto N>
// struct common_tuple_access<std::array<T, N>>
// {
//    using type = std::variant<std::reference_wrapper<T>>;
// };

template <typename Tup>
using common_tuple_access_t = typename common_tuple_access<Tup>::type;

template <typename Tup>
auto runtime_i(Tup& t, std::size_t i)
{
   return call_access_function<common_tuple_access_t<Tup>>(t, i,
   [](auto & element) { return std::ref(element); },
   std::make_index_sequence<mfem::future::tuple_size<Tup>::value> {}
                                                          );
}

template <class ... Fs>
struct overload : Fs...
{
   overload(Fs&&... fs) : Fs{fs}... {}
      using Fs::operator()...;
};

template <typename Tup> class tuple_iterator
{
   Tup& t;
   size_t i;
public:
   tuple_iterator(Tup& tup, size_t idx): t{tup}, i{idx}
   {}

   tuple_iterator& operator++() { ++i; return *this; }
   bool operator==(tuple_iterator const& other) const
   {
      return std::addressof(other.t) == std::addressof(t)
             && other.i == i;
   }

   bool operator!=(tuple_iterator const& other) const
   {
      return !(*this == other);
   }

   auto operator*() const { return runtime_get(t, i); }
};

template <typename Tup>
class to_range
{
   Tup& t;
public:
   to_range(Tup& tup) : t{tup} {}

   auto begin()
   {
      return tuple_iterator{t, 0};
   }
   auto end()
   {
      return tuple_iterator{t, mfem::future::tuple_size<Tup>::value};
   }

   auto operator[](std::size_t i)
   {
      return runtime_get(t, i);
   }
};

///////////////////////////////////////////////////////////////////////////////
// 2️⃣ https://arne-mertz.de/2017/03/tuple-compile-time-access/
template <std::size_t I>
struct index {};

template <typename T, std::size_t I>
decltype(auto) at(T tuple, index<I>)
{
   return mfem::future::get<I>(tuple);
}

template <char... Digits>
constexpr std::size_t parse()
{
   // convert to array so we can use a loop instead of recursion
   char digits[] = {Digits...};

   // straightforward number parsing code
   auto result = 0u;
   for (auto c : digits)
   {
      result *= 10;
      result += c - '0';
   }
   return result;
}

template <char... Digits>
auto operator ""_i()
{
   return index<parse<Digits...>()> {};
}

///////////////////////////////////////////////////////////////////////////////
template <int DIM>
void DFemDiffusion(const char *filename, int p, const int r)
{
   dbg("DIM:{}", DIM);
   CAPTURE(filename, DIM, p, r);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   smesh.Clear();

   Array<int> all_domain_attr;
   if (pmesh.bdr_attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.bdr_attributes.Max());
      all_domain_attr = 1;
   }

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   const int NE = pfes.GetNE(), d1d(p + 1), q = 2 * p + r;
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), q);
   const int q1d(IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints());
   MFEM_VERIFY(d1d <= q1d, "q1d should be >= d1d");
   MFEM_VERIFY(NE > 0, "Mesh with no elements is not yet supported!");

   ParGridFunction x(&pfes), y(&pfes), z(&pfes);
   Vector X(pfes.GetTrueVSize()), Y(pfes.GetTrueVSize()), Z(pfes.GetTrueVSize());

   X.Randomize(1);
   x.SetFromTrueDofs(X);

   auto rho = [](const Vector &xyz)
   {
      const real_t x = xyz(0), y = xyz(1), z = DIM == 3 ? xyz(2) : 0.0;
      real_t r = M_PI * pow(x, 2);
      if (DIM >= 2) { r += pow(y, 3); }
      if (DIM >= 3) { r += pow(z, 4); }
      return r;
   };
   FunctionCoefficient rho_coeff(rho);

   ParBilinearForm blf_fa(&pfes);
   blf_fa.AddDomainIntegrator(new DiffusionIntegrator(rho_coeff, ir));
   blf_fa.Assemble();
   blf_fa.Finalize();

   /*SECTION("Partial assembly")
   {
      ParBilinearForm blf_pa(&pfes);
      blf_pa.AddDomainIntegrator(new DiffusionIntegrator(rho_coeff, ir));
      blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      blf_pa.Assemble();
      blf_pa.Mult(x, z);

      blf_fa.Mult(x, y);
      y -= z;
      REQUIRE(y.Normlinf() == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }*/

   QuadratureSpace qs(pmesh, *ir);
   CoefficientVector rho_coeff_cv(rho_coeff, qs);
   MFEM_VERIFY(rho_coeff_cv.GetVDim() == 1, "Coefficient should be scalar");
   MFEM_VERIFY(rho_coeff_cv.Size() == q1d * q1d * (DIM == 3 ? q1d : 1) * NE, "");

   const int rho_local_size = 1;
   const int rho_elem_size(rho_local_size * ir->GetNPoints());
   const int rho_total_size(rho_elem_size * NE);
   ParametricSpace rho_ps(DIM, rho_local_size, rho_elem_size, rho_total_size,
                          DIM == 3 ? d1d : d1d * d1d, // 🔥 2D workaround
                          DIM == 3 ? q1d : q1d * q1d);

   static constexpr int U = 0, Coords = 1, Rho = 3;
   const auto sol = std::vector{ FieldDescriptor{ U, &pfes } };

   SECTION("DFEM Matrix free")
   {
      // fields = {solutions, parameters}
      dbg("fields = {{solutions, parameters}} = {{{{U}}, {{Rho, Coords}}}}");
      DOperator dop_mf(sol, {{Rho, &rho_ps}, {Coords, mfes}}, pmesh);
      // typename Diffusion<real_t, DIM, 3>::MFApply mf_apply_qf;

      /*{
         map[1] = Diffusion<real_t, 3, 1>::MFApply{};
         map[2] = Diffusion<real_t, 3, 2>::MFApply{};
         map[3] = Diffusion<real_t, 3, 3>::MFApply{};
         map[4] = Diffusion<real_t, 3, 4>::MFApply{};

         {
            const int i = 2;
            auto qfi = map[2];
            dbg("qfi.MQ1:{}", qfi.MQ1);
            dop_mf.AddDomainIntegrator(qfi,
                                       tuple{ Gradient<U>{}, None<Rho>{},
                                              Gradient<Coords>{}, Weight{} },
                                       tuple{ Gradient<U>{} }, *ir,
                                       all_domain_attr);
         }
      }*/

      dbg("AddDomainIntegrator: {{∇U, Rho, ∇Coords, Weight}} -> {{∇U}}");
      MFDiffusionQFs<real_t, DIM> qfs;

      dbg("AddDomainIntegrator: {{∇U, Rho, ∇Coords, Weight}} -> {{∇U}}");
      runtime_get(qfs.mf_qfs, 3-1, [&](auto &qf)
      {
         dbg("qf.MQ1:{}", qf.MQ1);
         dop_mf.AddDomainIntegrator(qf, //mf_apply_qf,
                                    tuple{ Gradient<U>{}, None<Rho>{},
                                           Gradient<Coords>{}, Weight{} },
                                    tuple{ Gradient<U>{} }, *ir,
                                    all_domain_attr);
      });

      // 1️⃣
      // auto qf2 = runtime_i(qfs.mf_qfs, 2);
      // dbg("🔥🔥🔥 qf2.MQ1:{}",qf2.MQ1);

      /*for (auto const& elem : to_range(qfs.mf_qfs))
      {
         std::visit(
            overload(
         [](int i) { std::cout << "int: " << i << '\n'; },
         [](std::string const& s) { std::cout << "string: " << s << '\n'; },
         [](double d) { std::cout << "double: " << d << '\n'; }
            ),
         elem
         );
      }*/
      // dbg("🔥🔥🔥 qf2.MQ1:{}", qf2.MQ1);

      // 2️⃣ but need to write the 'n'...
      auto qfi = at(qfs.mf_qfs, 3_i);
      dbg("🔥🔥🔥 qfi.MQ1:{}",qfi.MQ1);

      /* auto mf_apply_qf_variant = qf.Get(3);
      for_constexpr<4>([&](auto i)
      {
         dbg("i:{}", i.value);
         return extract<real_t, DIM, 3>(mf_apply_qf_variant);
      });*/

      // auto mf_apply_qf_3 = mfem::future::get<3-1>(qf.mf_qfs);
      // dop_mf.AddDomainIntegrator(mf_apply_qf_3, //mf_apply_qf,
      //                            tuple{ Gradient<U>{}, None<Rho>{},
      //                                   Gradient<Coords>{}, Weight{} },
      //                            tuple{ Gradient<U>{} }, *ir,
      //                            all_domain_attr);
      dop_mf.SetParameters({ &rho_coeff_cv, nodes });

      pfes.GetRestrictionMatrix()->Mult(x, X);
      dop_mf.Mult(X, Z);

      blf_fa.Mult(x, y);
      pfes.GetProlongationMatrix()->MultTranspose(y, Y);
      Y -= Z;

      real_t norm_global = 0.0;
      real_t norm_local = Y.Normlinf();
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                    pmesh.GetComm());

      REQUIRE(norm_global == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

   /*SECTION("DFEM Partial assembly")
   {
      static constexpr int QData = 2;
      const int qd_local_size = DIM * DIM;
      const int qd_elem_size(qd_local_size * ir->GetNPoints());
      const int qd_total_size(qd_elem_size * NE);
      ParametricSpace qd_ps(DIM, qd_local_size, qd_elem_size, qd_total_size,
                            DIM == 3 ? d1d : d1d * d1d, // 🔥 2D workaround
                            DIM == 3 ? q1d : q1d * q1d);
      ParametricFunction qdata(qd_ps);
      qdata.UseDevice(true);

      DOperator dSetup(sol, {{Rho, &rho_ps}, {Coords, mfes}, {QData, &qd_ps}}, pmesh);
      typename Diffusion<real_t, DIM>::PASetup pa_setup_qf;
      dSetup.AddDomainIntegrator(
         pa_setup_qf,
         tuple{ Value<U>{}, None<Rho>{}, Gradient<Coords>{}, Weight{} },
         tuple{ None<QData>{} }, *ir, all_domain_attr);
      dSetup.SetParameters({ &rho_coeff_cv, nodes, &qdata });
      pfes.GetRestrictionMatrix()->Mult(x, X);
      dSetup.Mult(X, qdata);

      DOperator dop_pa(sol, { { QData, &qd_ps } }, pmesh);
      typename Diffusion<real_t, DIM>::PAApply pa_apply_qf;
      dop_pa.AddDomainIntegrator(pa_apply_qf,
                                 tuple{ Gradient<U>{}, None<QData>{} },
                                 tuple{ Gradient<U>{} },
                                 *ir, all_domain_attr);
      dop_pa.SetParameters({ &qdata });

      pfes.GetRestrictionMatrix()->Mult(x, X);
      dop_pa.Mult(X, Z);

      blf_fa.Mult(x, y);
      pfes.GetProlongationMatrix()->MultTranspose(y, Y);
      Y -= Z;

      real_t norm_global = 0.0;
      real_t norm_local = Y.Normlinf();
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                    pmesh.GetComm());

      REQUIRE(norm_global == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }*/

   /*SECTION("DFEM Linearization", "[Parallel][DFEM]")
   {
      DOperator dop_mf(sol, {{Rho, &rho_ps}, {Coords, mfes}}, pmesh);
   #ifdef MFEM_USE_ENZYME
      typename Diffusion<real_t, DIM>::MFApply mf_apply_qf;
   #else
      typename Diffusion<dual<real_t, real_t>, DIM>::MFApply mf_apply_qf;
   #endif
      auto derivatives = std::integer_sequence<size_t, U> {};
      dop_mf.AddDomainIntegrator(mf_apply_qf,
                                 tuple{ Gradient<U>{}, None<Rho>{},
                                        Gradient<Coords>{}, Weight{} },
                                 tuple{ Gradient<U>{} }, *ir,
                                 all_domain_attr, derivatives);
      dop_mf.SetParameters({ &rho_coeff_cv, nodes });
      auto dRdU = dop_mf.GetDerivative(U, {&x}, {&rho_coeff_cv, nodes});

      pfes.GetRestrictionMatrix()->Mult(x, X);
      dop_mf.Mult(X, Z);

      blf_fa.Mult(x, y);
      pfes.GetProlongationMatrix()->MultTranspose(y, Y);
      Y -= Z;

      real_t norm_global = 0.0;
      real_t norm_local = Y.Normlinf();
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                    pmesh.GetComm());

      REQUIRE(norm_global == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }*/
}

TEST_CASE("DFEM Diffusion", "[Parallel][DFEM]")
{
   const bool all_tests = launch_all_non_regression_tests;

   const auto p = !all_tests ? 2 : GENERATE(1, 2, 3);
   const auto r = !all_tests ? 1 : GENERATE(0, 1, 2, 3);

   DiffusionIntegrator::AddSpecialization<3,3,3>();

   /*SECTION("2D p=" + std::to_string(p) + " r=" + std::to_string(r))
   {
      const auto filename =
         GENERATE("../../data/star.mesh",
                  "../../data/star-q3.mesh",
                  "../../data/rt-2d-q3.mesh",
                  "../../data/inline-quad.mesh",
                  "../../data/periodic-square.mesh");
      DFemDiffusion<2>(filename, p, r);
   }*/

   SECTION("3D p=" + std::to_string(p) + " r=" + std::to_string(r))
   {
#if 0
      const auto filename =
         GENERATE("../../data/fichera.mesh",
                  "../../data/fichera-q3.mesh",
                  "../../data/inline-hex.mesh",
                  "../../data/toroid-hex.mesh",
                  "../../data/periodic-cube.mesh");
#else
      const auto filename =
         GENERATE("../../data/fichera.mesh");
#endif
      DFemDiffusion<3>(filename, p, r);
   }
}

} // namespace dfem_pa_kernels

#endif
