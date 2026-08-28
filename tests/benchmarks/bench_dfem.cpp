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

#include "bench.hpp" // IWYU pragma: keep 

#ifdef MFEM_USE_BENCHMARK

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "fem/qinterp/det.hpp" // IWYU pragma: keep
#include "fem/qinterp/grad.hpp" // IWYU pragma: keep
#include "fem/qinterp/grad_transpose.hpp" // IWYU pragma: keep
#include "fem/quadinterpolator.hpp" // IWYU pragma: keep
#include "fem/integ/lininteg_domain_kernels.hpp" // IWYU pragma: keep
#include "fem/integ/bilininteg_vecdiffusion_pa.hpp" // IWYU pragma: keep 

#include "fem/dfem/backends/global_qf/prelude.hpp"
using global_backend = mfem::future::GlobalQFBackend;

#include "fem/dfem/backends/local_qf/prelude.hpp"
using local_backend = mfem::future::LocalQFBackend;

#include "fem/dfem/tuple.hpp"
using future::tuple;

#include "fem/dfem/doperator.hpp"
#include "linalg/tensor.hpp"
#include "linalg/tensor_arrays.hpp"

#if defined(__HIP__)
#include "../usr/src/array/tensor_std_array.hpp"
#endif

using namespace mfem;

using future::tensor;
using future::tensor_array;

using future::DifferentiableOperator;
using future::UniformParameterSpace;
using future::ParameterFunction;
using future::FieldDescriptor;
using future::Gradient;
using future::Value;
using future::Weight;
using future::Identity;

// info
void info()
{
   mfem::out << "\x1b[33m";
   mfem::out <<
             "name: BP<n>/B<m>D<c|q>[.ad]/S<f|gk>/<impl>/<order>/<dofs per side>"
             << std::endl;
   mfem::out << "expression: parentheses mark fused launch/stage boundaries"
             << std::endl;
   mfem::out << "\x1b[m" << std::endl;
}

// Version
enum class Version
{
   // MFEM versions
   PA_mfem_std,
   // dFEM global QF versions
   MF_dfem_global,
   PA_dfem_global,
   // dFEM local QF versions
   MF_dfem_local,
   PA_dfem_local,
   // dFEM GetDerivative versions
   MF_dfem_global_get_derivative,
   MF_dfem_local_get_derivative,
   MF_dfem_global_get_derivative_cached,
   MF_dfem_local_get_derivative_cached,
};

// Benchmark naming spec
// BP3/B2Dc.ad/Sg3/dfem/4/160
//  |      |      |    |    |
//  |      |      |    |    +-- args (order, dofs per side), appended by gbench
//  |      |      |    +-- implementation under test
//  |      |      +-- structure: Sf = fused, Sg<k> = staged over global qp arrays,
//  |      |          with k parenthesized expression groups
//  |      +-- class: B<n> = total number of B/G applications in the operator
//  |          action, including transposes; Dc = cached q-data, Dq = q-data
//  |          evaluated at quadrature points, .ad = obtained via get_derivative
//  |          (provenance only)
//  +-- problem
//
// Expression notation:
//   Parentheses mark fused launch/stage boundaries.
//   Square brackets group block quadrature inputs, not staging.
//   Dq evaluates geometry/parameter data and therefore shows (G x); Dc uses
//   cached data and omits (G x).
//
// Expression column examples:
//   BP1/{B2Dc,B2Dc.ad}/Sf/*  -> (Bᵀ Dc B u)
//   BP1/B4Dq.ad/Sg5/*        -> (Bᵀ (Dq [(B u); (G x); (B p)]))
//   BP3/{B2Dc,B2Dc.ad}/Sf/*  -> (Gᵀ Dc G u)
//   BP3/B4Dq.ad/Sg5/*        -> (Gᵀ (Dq [(G u); (G x); (G p)]))

template <int BFI, Version VER>
constexpr const char *BenchmarkPath() noexcept
{
   if constexpr (BFI == 1)
   {
      if constexpr (VER == Version::PA_mfem_std)
      {
         return "BP1/B2Dc/Sf/mfem_std";
      }
      else if constexpr (VER == Version::MF_dfem_global)
      {
         return "BP1/B3Dq/Sg4/dfem";
      }
      else if constexpr (VER == Version::PA_dfem_global)
      {
         return "BP1/B2Dc/Sg2/dfem";
      }
      else if constexpr (VER == Version::MF_dfem_local)
      {
         return "BP1/B3Dq/Sf/dfem";
      }
      else if constexpr (VER == Version::PA_dfem_local)
      {
         return "BP1/B2Dc/Sf/dfem";
      }
      else if constexpr (VER == Version::MF_dfem_global_get_derivative)
      {
         return "BP1/B4Dq.ad/Sg5/dfem";
      }
      else if constexpr (VER == Version::MF_dfem_local_get_derivative)
      {
         return "BP1/B4Dq.ad/Sf/dfem";
      }
      else if constexpr (VER == Version::MF_dfem_global_get_derivative_cached)
      {
         return "BP1/B2Dc.ad/Sg2/dfem";
      }
      else if constexpr (VER == Version::MF_dfem_local_get_derivative_cached)
      {
         return "BP1/B2Dc.ad/Sf/dfem";
      }
   }
   else if constexpr (BFI == 3)
   {
      if constexpr (VER == Version::PA_mfem_std)
      {
         return "BP3/B2Dc/Sf/mfem_std";
      }
      else if constexpr (VER == Version::MF_dfem_global)
      {
         return "BP3/B3Dq/Sg4/dfem";
      }
      else if constexpr (VER == Version::PA_dfem_global)
      {
         return "BP3/B2Dc/Sg2/dfem";
      }
      else if constexpr (VER == Version::MF_dfem_local)
      {
         return "BP3/B3Dq/Sf/dfem";
      }
      else if constexpr (VER == Version::PA_dfem_local)
      {
         return "BP3/B2Dc/Sf/dfem";
      }
      else if constexpr (VER == Version::MF_dfem_global_get_derivative)
      {
         return "BP3/B4Dq.ad/Sg5/dfem";
      }
      else if constexpr (VER == Version::MF_dfem_local_get_derivative)
      {
         return "BP3/B4Dq.ad/Sf/dfem";
      }
      else if constexpr (VER == Version::MF_dfem_global_get_derivative_cached)
      {
         return "BP3/B2Dc.ad/Sg2/dfem";
      }
      else if constexpr (VER == Version::MF_dfem_local_get_derivative_cached)
      {
         return "BP3/B2Dc.ad/Sf/dfem";
      }
   }
   return "invalid";
}

template <int BFI, Version VER>
constexpr const char *BenchmarkExpression() noexcept
{
   if constexpr (BFI == 1)
   {
      if constexpr (VER == Version::PA_mfem_std ||
                    VER == Version::PA_dfem_local ||
                    VER == Version::MF_dfem_local_get_derivative_cached)
      {
         return "(Bᵀ Dc B u)";
      }
      else if constexpr (VER == Version::MF_dfem_local)
      {
         return "(Bᵀ Dq [B u; G x])";
      }
      else if constexpr (VER == Version::MF_dfem_local_get_derivative)
      {
         return "(Bᵀ Dq [B u; G x; B p])";
      }
      else if constexpr (VER == Version::PA_dfem_global ||
                         VER == Version::MF_dfem_global_get_derivative_cached)
      {
         return "(Bᵀ (Dc B p))";
      }
      else if constexpr (VER == Version::MF_dfem_global)
      {
         return "(Bᵀ (Dq [(B u); (G x)]))";
      }
      else if constexpr (VER == Version::MF_dfem_global_get_derivative)
      {
         return "(Bᵀ (Dq [(B u); (G x); (B p)]))";
      }
   }
   else if constexpr (BFI == 3)
   {
      if constexpr (VER == Version::PA_mfem_std ||
                    VER == Version::PA_dfem_local ||
                    VER == Version::MF_dfem_local_get_derivative_cached)
      {
         return "(Gᵀ Dc G u)";
      }
      else if constexpr (VER == Version::MF_dfem_local)
      {
         return "(Gᵀ Dq [(G u); (G x)])";
      }
      else if constexpr (VER == Version::MF_dfem_local_get_derivative)
      {
         return "(Gᵀ Dq [G u; G x; G p])";
      }
      else if constexpr (VER == Version::PA_dfem_global ||
                         VER == Version::MF_dfem_global_get_derivative_cached)
      {
         return "(Gᵀ (Dc G p))";
      }
      else if constexpr (VER == Version::MF_dfem_global)
      {
         return "(Gᵀ (Dq [(G u); (G x)]))";
      }
      else if constexpr (VER == Version::MF_dfem_global_get_derivative)
      {
         return "(Gᵀ (Dq [(G u); (G x); (G p)]))";
      }
   }
   return "";
}

// Console reporter with a string expression column
class ExpressionReporter : public bm::BenchmarkReporter
{
   static constexpr int expression_width = 20;
   std::size_t name_field_width = 0;
   bm::UserCounters prev_counters;
   bool printed_header = false;

   static std::string FormatTime(double time)
   {
      char buffer[32];
      if (time < 1.0)
      {
         std::snprintf(buffer, sizeof(buffer), "%10.3f", time);
      }
      else if (time < 10.0)
      {
         std::snprintf(buffer, sizeof(buffer), "%10.2f", time);
      }
      else if (time < 100.0)
      {
         std::snprintf(buffer, sizeof(buffer), "%10.1f", time);
      }
      else if (time > 9999999999.0)
      {
         std::snprintf(buffer, sizeof(buffer), "%1.4e", time);
      }
      else
      {
         std::snprintf(buffer, sizeof(buffer), "%10.0f", time);
      }
      return buffer;
   }

   static std::string HumanReadableNumber(double value, bm::Counter::OneK oneK)
   {
      static constexpr const char *suffixes[] = {"", "k", "M", "G", "T"};
      double scaled = value;
      int suffix = 0;
      const double base = static_cast<double>(oneK);
      while (std::abs(scaled) >= base && suffix < 4)
      {
         scaled /= base;
         suffix++;
      }
      std::ostringstream os;
      os << std::setprecision(6) << scaled << suffixes[suffix];
      return os.str();
   }

   static std::string CounterValue(const bm::BenchmarkReporter::Run &run,
                                   const bm::UserCounters::value_type &counter,
                                   std::string &unit)
   {
      if (run.run_type == Run::RT_Aggregate &&
          run.aggregate_unit == bm::StatisticUnit::kPercentage)
      {
         std::ostringstream os;
         os << std::fixed << std::setprecision(2)
            << 100.0 * counter.second.value;
         unit = "%";
         return os.str();
      }
      unit = (counter.second.flags & bm::Counter::kIsRate) != 0 ?
             ((counter.second.flags & bm::Counter::kInvert) != 0 ? "s" : "/s") :
             "";
      return HumanReadableNumber(counter.second.value, counter.second.oneK);
   }

   void PrintHeader(const Run &run)
   {
      std::ostringstream os;
      os << std::left << std::setw(static_cast<int>(name_field_width))
         << "Benchmark" << " "
         << std::right << std::setw(13) << "Time" << " "
         << std::setw(15) << "CPU" << " "
         << std::setw(12) << "Iterations";
      for (const auto &counter : run.counters)
      {
         const auto width = std::max<std::size_t>(10, counter.first.length());
         os << " " << std::setw(static_cast<int>(width)) << counter.first;
      }
      os << " " << std::left << std::setw(expression_width) << "expression";

      const auto header = os.str();
      GetOutputStream() << std::string(header.length(), '-') << "\n"
                        << header << "\n"
                        << std::string(header.length(), '-') << "\n";
   }

   void PrintRunData(const Run &run)
   {
      auto &out = GetOutputStream();
      out << std::left << std::setw(static_cast<int>(name_field_width))
          << run.benchmark_name() << " ";

      if (run.skipped != bmi::NotSkipped)
      {
         out << (run.skipped == bmi::SkippedWithError ? "ERROR: " : "SKIPPED: ")
             << run.skip_message << "\n";
         return;
      }

      const char *time_unit = bm::GetTimeUnitString(run.time_unit);
      out << std::right << FormatTime(run.GetAdjustedRealTime()) << " "
          << std::left << std::setw(4) << time_unit
          << std::right << FormatTime(run.GetAdjustedCPUTime()) << " "
          << std::left << std::setw(4) << time_unit;

      if (run.run_type != Run::RT_Aggregate ||
          run.aggregate_unit == bm::StatisticUnit::kTime)
      {
         out << std::right << std::setw(10) << run.iterations;
      }
      else
      {
         out << std::right << std::setw(10) << "";
      }

      for (const auto &counter : run.counters)
      {
         std::string unit;
         const std::string value = CounterValue(run, counter, unit);
         const auto width = std::max<std::size_t>(10, counter.first.length());
         const auto value_width = std::max<int>(1,
                                                static_cast<int>(width - unit.length()));
         out << " " << std::right << std::setw(value_width) << value << unit;
      }

      out << " " << std::left << std::setw(expression_width)
          << run.report_label << "\n";
   }

public:
   bool ReportContext(const Context &context) override
   {
      name_field_width = std::max<std::size_t>(context.name_field_width, 9);
      printed_header = false;
      prev_counters.clear();
      PrintBasicContext(&mfem::err, context);
      return true;
   }

   void ReportRuns(const std::vector<Run> &reports) override
   {
      for (const auto &run : reports)
      {
         const bool print_header = !printed_header ||
                                   !bmi::SameNames(run.counters, prev_counters);
         if (print_header)
         {
            printed_header = true;
            prev_counters = run.counters;
            PrintHeader(run);
         }
         PrintRunData(run);
      }
   }
};

// Custom benchmark arguments generator
static void CustomArguments(bm::Benchmark *b) noexcept
{
   constexpr int MAX_NDOFS = 8 * 1024 * (mfem_use_gpu ? 1024 : 8);

   const auto orders = { 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1 };

   constexpr auto ndofs = [](int n) constexpr noexcept -> int
   {
      return (n + 1) * (n + 1) * (n + 1);
   };

   constexpr auto inc = [](int n) constexpr noexcept -> int
   {
      return n < 160 ?  4 : n < 240 ?  8 : n < 320 ? 16 : 32;
   };

   for (auto p : orders)
   {
      for (int n = 4; ndofs(n) <= MAX_NDOFS; n += inc(n))
      {
         b->Args({p, n});
      }
   }
}

// Register kernel specializations used in the benchmarks
static void AddKernelSpecializations()
{
#ifndef MFEM_DEBUG
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 2, 2>::Add();
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 2, 3>::Add();
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 2, 5>::Add();
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 2, 6>::Add();
   QuadratureInterpolator::DetKernels::Specialization<3, 3, 5, 5>::Add();
   // Others use too much shared data

   using GRAD = QuadratureInterpolator::GradKernels;
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 2>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 7>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 8>::Add();
   GRAD::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 9>::Add();

   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 3>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 4>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 5>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 6>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 7>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 8>::Add();

   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 1, 2, 3>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 1, 4, 5>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 1, 5, 6>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 1, 6, 7>::Add();
   GRAD::Specialization<3, QVectorLayout::byVDIM, false, 1, 7, 8>::Add();

   using GRAD_TRANSPOSE = QuadratureInterpolator::GradTransposeKernels;
   GRAD_TRANSPOSE::Specialization<3, QVectorLayout::byVDIM, false, 1,2,3>::Add();
   GRAD_TRANSPOSE::Specialization<3, QVectorLayout::byVDIM, false, 1,4,5>::Add();
   GRAD_TRANSPOSE::Specialization<3, QVectorLayout::byVDIM, false, 1,5,6>::Add();
   GRAD_TRANSPOSE::Specialization<3, QVectorLayout::byVDIM, false, 1,6,7>::Add();
   GRAD_TRANSPOSE::Specialization<3, QVectorLayout::byVDIM, false, 1,7,8>::Add();

   using LIN = DomainLFIntegrator::AssembleKernels;
   LIN::Specialization<3, 7, 7>::Add();
   LIN::Specialization<3, 6, 6>::Add();
   LIN::Specialization<3, 8, 8>::Add();

   using VDIFF = VectorDiffusionIntegrator::ApplyPAKernels;
   VDIFF::Specialization<3, 3, 3, 3>::Add();
   VDIFF::Specialization<3, 3, 4, 4>::Add();
   VDIFF::Specialization<3, 3, 5, 5>::Add();
   VDIFF::Specialization<3, 3, 6, 6>::Add();
   VDIFF::Specialization<3, 3, 7, 7>::Add();
   VDIFF::Specialization<3, 3, 8, 8>::Add();
#endif // MFEM_DEBUG
}

// Globals
Device *device_ptr = nullptr;

// GLOBAL Mass Q-Functions
template<int DIM>
struct MF_Mass_global_qf
{
   void operator()(tensor_array<const real_t> &u,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &weight,
                   tensor_array<real_t> &v) const
   {
      mfem::forall<UseEnzyme>(v.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         v(q) = weight(q) * det(J(q)) * u(q);
      });
   }
};

template<int DIM>
struct PA_Mass_Setup_global_qf
{
   void operator()(tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &weight,
                   tensor_array<real_t> &D) const
   {
      mfem::forall(D.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         D(q) = weight(q) * det(J(q));
      });
   }
};

template<int>
struct PA_Mass_Apply_global_qf
{
   void operator()(tensor_array<const real_t> &u,
                   tensor_array<const real_t> &D,
                   tensor_array<real_t> &v) const
   {
      mfem::forall(v.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         v(q) = D(q) * u(q);
      });
   }
};

// LOCAL Mass Q-Functions
template<int DIM>
struct MF_Mass_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const real_t &u,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &weight,
                   real_t &v) const
   {
      v =  weight * det(J) * u;
   };
};

template<int DIM>
struct PA_Mass_Setup_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<real_t, DIM, DIM> &J,
                   const tensor<real_t> &weight,
                   real_t &D) const
   {
      D = weight * det(J);
   }
};

template<int>
struct PA_Mass_Apply_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const real_t &u,
                   const real_t &D,
                   real_t &v) const
   {
      v = D * u;
   };
};

template<typename qfunction_t, int DIM>
constexpr bool mass_qf =
   std::disjunction_v<
   std::is_same<qfunction_t, MF_Mass_global_qf<DIM>>,
   std::is_same<qfunction_t, PA_Mass_Setup_global_qf<DIM>>,
   std::is_same<qfunction_t, PA_Mass_Apply_global_qf<DIM>>,
   std::is_same<qfunction_t, MF_Mass_local_qf<DIM>>,
   std::is_same<qfunction_t, PA_Mass_Setup_local_qf<DIM>>,
   std::is_same<qfunction_t, PA_Mass_Apply_local_qf<DIM>>>;

template<typename qfunction_t, int DIM, int U>
constexpr auto GradOrValue() ->
std::conditional_t<mass_qf<qfunction_t, DIM>, Value<U>, Gradient<U>>
{
   if constexpr (mass_qf<qfunction_t, DIM>) { return Value<U> {}; }
   else { return Gradient<U> {}; }
};

// Add dFEM local QFunction action specializations
template<typename backend_t, int DIM, typename QT, typename IT, typename OT>
void AddLocalQFActionSpecializations()
{
   if constexpr (std::is_same_v<backend_t, local_backend>)
   {
      mfem::future::AddAction<DIM, 6, QT, IT, OT>();
      mfem::future::AddAction<DIM, 8, QT, IT, OT>();
   }
}

template<typename backend_t, int DIM, int DID, typename QT, typename IT,
         typename OT>
void AddLocalQFDerivativeSpecializations()
{
   if constexpr (std::is_same_v<backend_t, local_backend>)
   {
      mfem::future::AddDerivativeAction<DIM, 6, DID, QT, IT, OT>();
      mfem::future::AddDerivativeSetup<DIM, 6, DID, QT, IT, OT>();
      mfem::future::AddDerivativeApply<DIM, 6, DID, QT, IT, OT>();
      mfem::future::AddDerivativeAction<DIM, 8, DID, QT, IT, OT>();
      mfem::future::AddDerivativeSetup<DIM, 8, DID, QT, IT, OT>();
      mfem::future::AddDerivativeApply<DIM, 8, DID, QT, IT, OT>();
   }
}

// GLOBAL Diffusion Q-Functions
template<int DIM>
struct MF_Diffusion_global_qf
{
   void operator()(tensor_array<const real_t, DIM> &Gu,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &weight,
                   tensor_array<real_t, DIM> &Gv) const
   {
      mfem::forall<UseEnzyme>(J.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         const auto invJ = inv(J(q));
         const real_t detJ = det(J(q));
         Gv(q) = weight(q) * detJ * (transpose(invJ) * (invJ * Gu(q)));
      });
   }
};

template<int DIM>
struct PA_Diffusion_Setup_global_qf
{
   void operator()(tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &weight,
                   tensor_array<real_t, DIM, DIM> &D) const
   {
      mfem::forall(J.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         const auto invJ = inv(J(q));
         D(q) = weight(q) * det(J(q)) * (invJ * transpose(invJ));
      });
   }
};

template<int DIM>
struct PA_Diffusion_Apply_global_qf
{
   void operator()(tensor_array<const real_t, DIM> &Gu,
                   tensor_array<const real_t, DIM, DIM> &D,
                   tensor_array<real_t, DIM> &Gv) const
   {
      mfem::forall(Gu.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         Gv(q) = D(q) * Gu(q);
      });
   }
};

// LOCAL Diffusion Q-Functions
template<int DIM>
struct MF_Diffusion_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<real_t, DIM> &Gu,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &weight,
                   tensor<real_t, DIM> &Gv) const
   {
      const auto invJ = inv(J);
      Gv =  weight * det(J) * (transpose(invJ) * (invJ * Gu));
   };
};

template<int DIM>
struct PA_Diffusion_Setup_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<real_t, DIM, DIM> &J,
                   const tensor<real_t> &weight,
                   tensor<real_t, DIM, DIM> &D) const
   {
      const auto invJ = inv(J);
      D = weight * det(J) * (invJ * transpose(invJ));
   }
};

template<int DIM>
struct PA_Diffusion_Apply_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<real_t, DIM> &Gu,
                   const tensor<real_t, DIM, DIM> &D,
                   tensor<real_t, DIM> &Gv) const
   {
      Gv = D * Gu;
   };
};

// BakeOff
template <int BFI, Version VER, int VDIM, bool GLL>
struct BakeOff
{
   static constexpr int bfi = BFI;
   static constexpr Version version = VER;
   static constexpr int DIM = 3;
   const int p, c, q, n, nx, ny, nz;
   Mesh smesh;
   ParMesh pmesh;
   H1_FECollection fec;
   ParFiniteElementSpace pfes;
   const Geometry::Type geom_type;
   IntegrationRules irs;
   const IntegrationRule *ir;
   ConstantCoefficient one;
   Vector uvec;
   VectorConstantCoefficient unit_vec;
   const int dofs;
   ParGridFunction &nodes;
   ParFiniteElementSpace& mfes;
   ParGridFunction x;
   ParBilinearForm a;

   Array<int> ess_tdof_list, ess_bdr;
   ParLinearForm b;
   Vector B, X;
   OperatorPtr A;

   static constexpr int U = 0, Ξ = 1, Q = 2;
   std::unique_ptr<DifferentiableOperator> dop;
   std::unique_ptr<DifferentiableOperator> qdata_setup_dop;
   std::shared_ptr<future::DerivativeOperator> ddop;
   QuadratureSpace qspace;
   VectorQuadratureSpace vqspace;
   QuadratureFunction qfct;

   struct WrapOpArg1: public Operator
   {
      const std::unique_ptr<DifferentiableOperator> &dop;
      Vector &arg1;

      WrapOpArg1(const std::unique_ptr<DifferentiableOperator> &dop,
                 const int height, const int width, Vector &arg1):
         Operator(height, width), dop(dop), arg1(arg1) { }

      void Mult(const Vector &xv, Vector &yv) const override
      {
         MultiVector MX{const_cast<Vector&>(xv), arg1}, MY{yv};
         dop->Mult(MX, MY);
      }
   };
   std::unique_ptr<WrapOpArg1> wop;

   struct WrapDerivativeOp: public Operator
   {
      const std::shared_ptr<future::DerivativeOperator> &ddop;

      WrapDerivativeOp(const std::shared_ptr<future::DerivativeOperator> &ddop,
                       const int height, const int width):
         Operator(height, width), ddop(ddop) { }

      void Mult(const Vector &xv, Vector &yv) const override
      {
         MultiVector MY{yv};
         ddop->Mult(xv, MY);
      }
   };
   std::unique_ptr<WrapDerivativeOp> dwop;

   double mdofs{};

   BakeOff(int p, int side):
      p(p), c(side), q(2 * p + (GLL ? -1 : 3)), n((assert(c >= p), c / p)),
      nx(n + (p * (n + 1) * p * n * p * n < c * c * c ? 1 : 0)),
      ny(n + (p * (n + 1) * p * (n + 1) * p * n < c * c * c ? 1 : 0)), nz(n),
      smesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON)),
      pmesh(MPI_COMM_WORLD, (smesh.EnsureNodes(), smesh)),
      fec(p, DIM, BasisType::GaussLobatto),
      pfes(&pmesh, &fec, VDIM),
      geom_type(pmesh.GetTypicalElementGeometry()),
      irs(0, GLL ? Quadrature1D::GaussLobatto : Quadrature1D::GaussLegendre),
      ir(&irs.Get(geom_type, q)), one(1.0), uvec(DIM),
      unit_vec((uvec = 1.0, uvec /= uvec.Norml2(), uvec)),
      dofs(pfes.GetTrueVSize()),
      nodes(*static_cast<ParGridFunction*>(pmesh.GetNodes())),
      mfes(*(nodes.ParFESpace())),
      x(&pfes),
      a(&pfes),
      ess_bdr(pmesh.bdr_attributes.Max()),
      b(&pfes),
      B(pfes.GetVSize()),
      X(x),
      qspace(pmesh, *ir),
      vqspace(qspace, DIM*DIM),
      qfct(vqspace)
   {
      smesh.Clear();
      x.Randomize(0x9e3779b9);
      const int q1d = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
      assert(q1d*q1d*q1d == ir->GetNPoints());

      ess_bdr = 1;
      pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // LinearForm b
      if constexpr (VDIM == 1)
      {
         b.AddDomainIntegrator(new DomainLFIntegrator(one));
      }
      else
      {
         b.AddDomainIntegrator(new VectorDomainLFIntegrator(unit_vec));
      }
      b.UseFastAssembly(true);
      b.Assemble();

      // BilinearForm a
      const int height = pfes.GetVSize(), width = pfes.GetVSize();
      const auto formLinearSystem = [&] (Vector &arg1)
      {
         Operator *A_ptr = nullptr;
         wop = std::make_unique<WrapOpArg1>(dop, height, width, arg1);
         wop->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         A.Reset(A_ptr);
      };
      const auto formLinearSystemDerivative = [&]
      {
         Operator *A_ptr = nullptr;
         dwop = std::make_unique<WrapDerivativeOp>(ddop, height, width);
         dwop->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         A.Reset(A_ptr);
      };
      // PA MFEM Setup
      const auto mPASetup = [&] (auto integrator)
      {
         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         a.AddDomainIntegrator(integrator);
         a.Assemble();
         a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
      };
      // MF ∂FEM setup
      const auto dMFSetup = [&] (auto backend, auto qfunction)
      {
         using backend_t = decltype(backend);
         using qfunction_t = decltype(qfunction);
         const auto ifd = std::vector<FieldDescriptor> {{U, &pfes}, {Ξ, &mfes}};
         const auto ofd = std::vector<FieldDescriptor> {{U, &pfes}};
         dop = std::make_unique<DifferentiableOperator>(ifd, ofd, pmesh);
         dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         constexpr auto GradValU = GradOrValue<qfunction_t, DIM, U>();
         dop->template AddDomainIntegrator<backend_t>(qfunction,
                                                      tuple{GradValU, Gradient<Ξ>{}, Weight{}},
                                                      tuple{GradValU},
                                                      *ir, ess_bdr);
         using QT = decltype(qfunction);
         using IT = decltype(tuple{GradValU, Gradient<Ξ>{}, Weight{}});
         using OT = decltype(tuple{GradValU});
         AddLocalQFActionSpecializations<backend_t, DIM, QT, IT, OT>();
         formLinearSystem(nodes);
      };
      // MF ∂FEM GetDerivative setup
      const auto dMFGetDerivativeSetup = [&] (auto backend, auto qfunction,
                                              bool use_cached_setup)
      {
         using backend_t = decltype(backend);
         using qfunction_t = decltype(qfunction);
         const auto ifd = std::vector<FieldDescriptor> {{U, &pfes}, {Ξ, &mfes}};
         const auto ofd = std::vector<FieldDescriptor> {{U, &pfes}};
         dop = std::make_unique<DifferentiableOperator>(ifd, ofd, pmesh);
         dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         constexpr auto GradValU = GradOrValue<qfunction_t, DIM, U>();
         dop->template AddDomainIntegrator<backend_t>(
            qfunction,
            tuple{GradValU, Gradient<Ξ>{}, Weight{}},
            tuple{GradValU},
            *ir, ess_bdr, future::Derivatives<U> {});
         using QT = decltype(qfunction);
         using IT = decltype(tuple{GradValU, Gradient<Ξ>{}, Weight{}});
         using OT = decltype(tuple{GradValU});
         AddLocalQFDerivativeSpecializations<backend_t, DIM, U, QT, IT, OT>();
         MultiVector state{x, nodes};
         ddop = dop->GetDerivative(U, state, use_cached_setup);
         formLinearSystemDerivative();
      };
      // PA ∂FEM setup
      const auto dPASetup = [&] (auto backend, auto setup_qf, auto apply_qf)
      {
         using backend_t = decltype(backend);
         const auto ifd0 = std::vector<FieldDescriptor> {{Ξ, &mfes}};
         const auto ofd0 = std::vector<FieldDescriptor> {{Q, &vqspace}};
         qdata_setup_dop = std::make_unique<DifferentiableOperator>(ifd0, ofd0, pmesh);
         qdata_setup_dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         qdata_setup_dop->template AddDomainIntegrator<backend_t>(
            setup_qf,
            tuple{Gradient<Ξ>{}, Weight{}},
            tuple{Identity<Q>{}},
            *ir, ess_bdr);
         using SetupQT = decltype(setup_qf);
         using SetupIT = decltype(tuple{Gradient<Ξ>{}, Weight{}});
         using SetupOT = decltype(tuple{Identity<Q>{}});
         AddLocalQFActionSpecializations<backend_t, DIM, SetupQT, SetupIT, SetupOT>();
         MultiVector N{nodes}, D{qfct};
         qdata_setup_dop->Mult(N, D);

         const auto ifd1 = std::vector<FieldDescriptor> {{U, &pfes}, {Q, &vqspace}};
         const auto ofd1 = std::vector<FieldDescriptor> {{U, &pfes}};
         dop = std::make_unique<DifferentiableOperator>(ifd1, ofd1, pmesh);
         dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         constexpr auto GradValU = GradOrValue<decltype(apply_qf), DIM, U>();
         dop->template AddDomainIntegrator<backend_t>(apply_qf,
                                                      tuple{GradValU, Identity<Q>{}},
                                                      tuple{GradValU},
                                                      *ir, ess_bdr);
         using ApplyQT = decltype(apply_qf);
         using ApplyIT = decltype(tuple{GradValU, Identity<Q>{}});
         using ApplyOT = decltype(tuple{GradValU});
         AddLocalQFActionSpecializations<backend_t, DIM, ApplyQT, ApplyIT, ApplyOT>();
         formLinearSystem(qfct);
      };

      if constexpr (BFI == 1)
      {
         if constexpr (VER == Version::PA_mfem_std)
         {
            mPASetup(new MassIntegrator(/*ir*/));
         }
         // dFEM Global versions
         else if constexpr (VER == Version::MF_dfem_global)
         {
            dMFSetup(global_backend{}, MF_Mass_global_qf<DIM> {});
         }
         else if constexpr (VER == Version::MF_dfem_global_get_derivative)
         {
            dMFGetDerivativeSetup(global_backend{}, MF_Mass_global_qf<DIM> {}, false);
         }
         else if constexpr (VER == Version::MF_dfem_global_get_derivative_cached)
         {
            dMFGetDerivativeSetup(global_backend{}, MF_Mass_global_qf<DIM> {}, true);
         }
         else if constexpr (VER == Version::PA_dfem_global)
         {
            dPASetup(global_backend{},
                     PA_Mass_Setup_global_qf<DIM> {},
                     PA_Mass_Apply_global_qf<DIM> {});
         }
         // dFEM Local versions
         else if constexpr (VER == Version::MF_dfem_local)
         {
            dMFSetup(local_backend{}, MF_Mass_local_qf<DIM> {});
         }
         else if constexpr (VER == Version::MF_dfem_local_get_derivative)
         {
            dMFGetDerivativeSetup(local_backend{}, MF_Mass_local_qf<DIM> {}, false);
         }
         else if constexpr (VER == Version::MF_dfem_local_get_derivative_cached)
         {
            dMFGetDerivativeSetup(local_backend{}, MF_Mass_local_qf<DIM> {}, true);
         }
         else if constexpr (VER == Version::PA_dfem_local)
         {
            dPASetup(local_backend{},
                     PA_Mass_Setup_local_qf<DIM> {},
                     PA_Mass_Apply_local_qf<DIM> {});
         }
         else { static_assert(false, "Invalid version"); }
      }
      else if constexpr (BFI == 2)
      {
         mPASetup(new VectorMassIntegrator(one, ir));
      }
      else if constexpr (BFI == 3 || BFI == 5)
      {
         // MFEM PA versions
         if constexpr (VER == Version::PA_mfem_std)
         {
            mPASetup(new DiffusionIntegrator(/*ir*/));
         }
         // dFEM Global versions
         else if constexpr (VER == Version::MF_dfem_global)
         {
            dMFSetup(global_backend{}, MF_Diffusion_global_qf<DIM> {});
         }
         else if constexpr (VER == Version::MF_dfem_global_get_derivative)
         {
            dMFGetDerivativeSetup(global_backend{}, MF_Diffusion_global_qf<DIM> {}, false);
         }
         else if constexpr (VER == Version::MF_dfem_global_get_derivative_cached)
         {
            dMFGetDerivativeSetup(global_backend{}, MF_Diffusion_global_qf<DIM> {}, true);
         }
         else if constexpr (VER == Version::PA_dfem_global)
         {
            dPASetup(global_backend{},
                     PA_Diffusion_Setup_global_qf<DIM> {},
                     PA_Diffusion_Apply_global_qf<DIM> {});
         }
         // dFEM Local versions
         else if constexpr (VER == Version::MF_dfem_local)
         {
            dMFSetup(local_backend{}, MF_Diffusion_local_qf<DIM> {});
         }
         else if constexpr (VER == Version::MF_dfem_local_get_derivative)
         {
            dMFGetDerivativeSetup(local_backend{}, MF_Diffusion_local_qf<DIM> {}, false);
         }
         else if constexpr (VER == Version::MF_dfem_local_get_derivative_cached)
         {
            dMFGetDerivativeSetup(local_backend{}, MF_Diffusion_local_qf<DIM> {}, true);
         }
         else if constexpr (VER == Version::PA_dfem_local)
         {
            dPASetup(local_backend{},
                     PA_Diffusion_Setup_local_qf<DIM> {},
                     PA_Diffusion_Apply_local_qf<DIM> {});
         }
         else { static_assert(false, "Invalid version"); }
      }
      else if constexpr (BFI == 4 || BFI == 6)
      {
         mPASetup(new VectorDiffusionIntegrator(one, ir));
      }
      else
      {
         static_assert(BFI >= 1 && BFI <= 6, "Invalid BilinearFormIntegrator");
      }
   }

   virtual void benchmark() = 0;

   [[nodiscard]] double SumMdofs() const noexcept { return mdofs; }

   [[nodiscard]] double MDofs() const noexcept { return 1e-6 * dofs; }

};

// Bake-off Problems (BPs)
template <int BFI, Version VER, int VDIM=1, bool GLL=false>
struct BP : public BakeOff<BFI, VER, VDIM, GLL>
{
   const int max_it = 32, print_lvl = -1;

   CGSolver cg;

   using base = BakeOff<BFI, VER, VDIM, GLL>;
   using base::A;
   using base::B;
   using base::X;
   using base::dofs;
   using base::mdofs;

   BP(int p, int side) noexcept: base(p, side),
      cg(MPI_COMM_WORLD)
   {
      static_assert(VDIM == 1 && GLL == false);

      cg.SetOperator(*A);
      cg.SetAbsTol(1e-12);
      cg.iterative_mode = false;
      if (dofs < 128 * 1024)
      {
         cg.SetPrintLevel(-1);
         cg.SetMaxIter(200);
         cg.SetRelTol(1e-8);
         cg.Mult(B, X);
         MFEM_VERIFY(cg.GetConverged(), "❌ CG solver did not converge.");
         // mfem::out << (cg.GetConverged() ? "✅" : "❌") << std::endl;
      }
      cg.SetRelTol(1e-8);
      cg.SetMaxIter(max_it);
      cg.SetPrintLevel(print_lvl);

      benchmark();
      mdofs = 0.0;
   }

   void benchmark() override
   {
      cg.Mult(B, X);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs() * cg.GetNumIterations();
   }
};

// Benchmarks Registration
template <typename T>
static void Benchmark(bm::State& state) noexcept
{
   std::unique_ptr<T> run;
   for ([[maybe_unused]] auto _ : state)
   {
      if (!run)
      {
         state.PauseTiming();
         run = std::make_unique<T>(state.range(0), state.range(1));
         state.ResumeTiming();
      }
      run->benchmark();
   }
   state.counters["Dofs"] = bm::Counter(run->dofs);
   state.counters["MDof/s"] = bm::Counter(run->SumMdofs(), bm::Counter::kIsRate);
   state.counters["p"] = bm::Counter(state.range(0));
   state.SetLabel(BenchmarkExpression<T::bfi, T::version>());
}
#define REGISTER(PK, BFI, VER) \
   BENCHMARK_TEMPLATE(Benchmark, PK<BFI, Version::VER>) \
   ->Name(BenchmarkPath<BFI, Version::VER>())->Apply(CustomArguments)->Unit(bm::kMillisecond)

// BP1: (Bᵀ Dc B u)
REGISTER(BP, 1, PA_mfem_std);
REGISTER(BP, 1, MF_dfem_local_get_derivative_cached);
REGISTER(BP, 1, PA_dfem_local);

// BP1: (Bᵀ (Dc B p))
REGISTER(BP, 1, MF_dfem_global_get_derivative_cached);
REGISTER(BP, 1, PA_dfem_global);

// BP1: (Bᵀ Dq [(B u); (G x)])
REGISTER(BP, 1, MF_dfem_local);

// BP1: (Bᵀ Dq [(B u); (G x); (B p)])
REGISTER(BP, 1, MF_dfem_local_get_derivative);

// BP1: (Bᵀ (Dq [(B u); (G x)]))
REGISTER(BP, 1, MF_dfem_global);

// BP1: (Bᵀ (Dq [(B u); (G x); (B p)]))
REGISTER(BP, 1, MF_dfem_global_get_derivative);

// BP3: (Gᵀ Dc G u)
REGISTER(BP, 3, PA_mfem_std);
REGISTER(BP, 3, MF_dfem_local_get_derivative_cached);
REGISTER(BP, 3, PA_dfem_local);

// BP3: (Gᵀ (Dc G p))
REGISTER(BP, 3, MF_dfem_global_get_derivative_cached);
REGISTER(BP, 3, PA_dfem_global);

// BP3: (Gᵀ Dq [(G u); (G x)])
REGISTER(BP, 3, MF_dfem_local);

// BP3: (Gᵀ Dq [(G u); (G x); (G p)])
REGISTER(BP, 3, MF_dfem_local_get_derivative);

// BP3: (Gᵀ (Dq [(G u); (G x)]))
REGISTER(BP, 3, MF_dfem_global);

// BP3: (Gᵀ (Dq [(G u); (G x); (G p)]))
REGISTER(BP, 3, MF_dfem_global_get_derivative);

// main
int main(int argc, char *argv[])
{
   static mfem::MPI_Session mpi(argc, argv);

   ExpressionReporter CR;
   bm::Initialize(&argc, argv);

   AddKernelSpecializations();
   info();

   // Device setup, cpu by default
   std::string device_config = "cpu";
   const auto global_context = bmi::GetGlobalContext();
   if (global_context != nullptr)
   {
      const auto device = global_context->find("device");
      if (device != global_context->end())
      {
         mfem::out << device->first << " : " << device->second << std::endl;
         device_config = device->second;
      }
   }
   Device device(device_config.c_str());
   device_ptr = &device;
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return EXIT_FAILURE; }

   bm::RunSpecifiedBenchmarks((bm::BenchmarkReporter*)&CR);

   return EXIT_SUCCESS;
}

#endif // MFEM_USE_BENCHMARK
