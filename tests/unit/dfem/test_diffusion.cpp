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
using MFDiffusionFactory_1_4 = MFDiffusionFactory<T, DIM, 1, 2, 3, 4>;

// Example class storing the MFApply objects
template <typename T, int DIM>
struct MFDiffusionQFs
{
   using MFApplyTuple = decltype(MFDiffusionFactory_1_4<T, DIM>::All());
   MFApplyTuple mf_qfs;

   MFDiffusionQFs(): mf_qfs(MFDiffusionFactory_1_4<T, DIM>::All()) {}
};


///////////////////////////////////////////////////////////////////////////////
// ✅✅✅ works with:
/*{
   auto qf_map = make_qf_map(qfs.mf_qfs);
   const int i = 3;
   dbg("qf_map[{}].MQ1:{}", i,
       reinterpret_cast<typename Diffusion<real_t, DIM, 1>::MFApply*>(qf_map[i])->MQ1);
}*/
template <typename... qf_ts, std::size_t... Is>
auto make_qf_map_impl(tuple<qf_ts...> qfs,
                      std::index_sequence<Is...>)
{
   auto get_qf_adrs = [&](auto i)
   {
      return static_cast<void*>(&mfem::future::get<i>(qfs));
   };

   std::map<int, void*> map;
   for_constexpr<sizeof...(qf_ts)>([&](auto i)
   {
      map[get<i>(qfs).MQ1] = get_qf_adrs(std::integral_constant<std::size_t, i> {});
   });

   return map;
}
template <typename... qf_ts>
auto make_qf_map(mfem::future::tuple<qf_ts...> qfs)
{
   return make_qf_map_impl(qfs, std::index_sequence_for<qf_ts...> {});
}

///////////////////////////////////////////////////////////////////////////////
// ✅✅✅ works with:
/*
      runtime_get(qfs.mf_qfs, 3-1, [&](auto &qf)
      {
         dbg("qf.MQ1:{}", qf.MQ1);
         dop_mf.AddDomainIntegrator(qf, //mf_apply_qf,
                                    tuple{ Gradient<U>{}, None<Rho>{},
                                           Gradient<Coords>{}, Weight{} },
                                    tuple{ Gradient<U>{} }, *ir,
                                    all_domain_attr);
      });
*/
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

      dbg("AddDomainIntegrator: {{∇U, Rho, ∇Coords, Weight}} -> {{∇U}}");
      MFDiffusionQFs<real_t, DIM> qfs;

      {
         auto qf_map = make_qf_map(qfs.mf_qfs);
         const int i = 3;
         dbg("qf_map[{}].MQ1:{}", i,
             reinterpret_cast<typename Diffusion<real_t, DIM, 1>::MFApply*>(qf_map[i])->MQ1);
      }

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

      // dop_mf.AddDomainIntegrator(mf_apply_qf,
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
