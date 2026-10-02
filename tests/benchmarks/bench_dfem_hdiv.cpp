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

// H(div) counterpart of bench_dfem: CG on the RT mass (u, v).
//
// Notation:
//
//   HDIV_MASS/<impl>/<p>/<n>: 
//   - <impl> is the variant (local/global, cahced/mf, etc...)
//   - p the order given to RT_FECollection
//   - n the elements per side of the unit cube mesh (n^3 affine hexes).
//
// Sample runs:
//
//   CPU:
//     ./bench_dfem_hdiv --benchmark_filter='HDIV_MASS/.*/4/5$'
//     ./bench_dfem_hdiv --benchmark_filter='HDIV_MASS/.*/0/24$'
//
//   GPU (CUDA; use device=hip on AMD):
//     ./bench_dfem_hdiv --benchmark_context=device=cuda --benchmark_filter='HDIV_MASS/.*/3/32$'
//     ./bench_dfem_hdiv --benchmark_context=device=cuda --benchmark_filter='HDIV_MASS/(mfem_std|PA_dfem_local_primal)/3/'
//

#include "bench.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_BENCHMARK

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "fem/dfem/backends/global_qf/prelude.hpp"
using global_backend = mfem::future::GlobalQFBackend;

#include "fem/dfem/backends/local_qf/prelude.hpp"
using local_backend = mfem::future::LocalQFBackend;

#include "fem/dfem/tuple.hpp"
using future::tuple;

#include "fem/dfem/doperator.hpp"
#include "fem/integ/bilininteg_vectorfemass_kernels.hpp"
#include "fem/qinterp/eval_hdiv.hpp" // IWYU pragma: keep
#include "linalg/tensor.hpp"
#include "linalg/tensor_arrays.hpp"

using namespace mfem;

using future::tensor;
using future::tensor_array;

using future::DifferentiableOperator;
using future::FieldDescriptor;
using future::Gradient;
using future::Value;
using future::Weight;
using future::Identity;

// info
void info()
{
   mfem::out << "\x1b[33m";
   mfem::out << "name: HDIV_MASS/<impl>/<order>/<elements per side>" << std::endl;
   mfem::out << "expression: braces mark kernel launch boundaries" << std::endl;
   mfem::out << "            B: RT values" << std::endl;
   mfem::out << "\x1b[m" << std::endl;
}

// Version
enum class Version
{
   // MFEM versions
   PA_mfem_std,
   // dFEM global QF versions
   MF_dfem_global_primal,
   PA_dfem_global_primal,
   // dFEM local QF versions
   MF_dfem_local_primal,
   PA_dfem_local_primal,
   // dFEM GetDerivative versions
   MF_dfem_global_derivative,
   MF_dfem_local_derivative,
   PA_dfem_global_derivative,
   PA_dfem_local_derivative,
};

template <Version VER>
constexpr const char *BenchmarkPath() noexcept
{
   if constexpr (VER == Version::PA_mfem_std)
   {
      return "HDIV_MASS/mfem_std";
   }
   else if constexpr (VER == Version::MF_dfem_global_primal)
   {
      return "HDIV_MASS/MF_dfem_global_primal";
   }
   else if constexpr (VER == Version::PA_dfem_global_primal)
   {
      return "HDIV_MASS/PA_dfem_global_primal";
   }
   else if constexpr (VER == Version::MF_dfem_local_primal)
   {
      return "HDIV_MASS/MF_dfem_local_primal";
   }
   else if constexpr (VER == Version::PA_dfem_local_primal)
   {
      return "HDIV_MASS/PA_dfem_local_primal";
   }
   else if constexpr (VER == Version::MF_dfem_global_derivative)
   {
      return "HDIV_MASS/MF_dfem_global_derivative";
   }
   else if constexpr (VER == Version::MF_dfem_local_derivative)
   {
      return "HDIV_MASS/MF_dfem_local_derivative";
   }
   else if constexpr (VER == Version::PA_dfem_global_derivative)
   {
      return "HDIV_MASS/PA_dfem_global_derivative";
   }
   else if constexpr (VER == Version::PA_dfem_local_derivative)
   {
      return "HDIV_MASS/PA_dfem_local_derivative";
   }
   return "invalid";
}

template <Version VER>
constexpr const char *BenchmarkExpression() noexcept
{
   if constexpr (VER == Version::PA_mfem_std)
   {
      return "{Bᵀ D B u}";
   }
   else if constexpr (VER == Version::PA_dfem_local_primal)
   {
      return "{Bᵀ Ql(B u)}";
   }
   else if constexpr (VER == Version::PA_dfem_local_derivative)
   {
      return "{Bᵀ Ql'(B u) B du}";
   }
   else if constexpr (VER == Version::MF_dfem_local_primal)
   {
      return "{Bᵀ Ql(B u, G x)}";
   }
   else if constexpr (VER == Version::MF_dfem_local_derivative)
   {
      return "{Bᵀ Ql'(B u, G x) B du}";
   }
   else if constexpr (VER == Version::PA_dfem_global_primal)
   {
      return "{Bᵀ {Qg(B u)}}";
   }
   else if constexpr (VER == Version::PA_dfem_global_derivative)
   {
      return "{Bᵀ {Qg'(B u) B du}}";
   }
   else if constexpr (VER == Version::MF_dfem_global_primal)
   {
      return "{Bᵀ {Qg({B u}, {G x})}}";
   }
   else if constexpr (VER == Version::MF_dfem_global_derivative)
   {
      return "{Bᵀ {Qg'({B u}, {G x}) {B du}}}";
   }
   return "";
}

// Console reporter with a string expression column
class ExpressionReporter : public bm::BenchmarkReporter
{
   static constexpr int expression_width = 20;
   static constexpr std::size_t counter_width = 10;
   static constexpr std::size_t order_width = 2;
   std::size_t name_field_width = 0;
   bm::UserCounters prev_counters;
   bool printed_header = false;

   static std::size_t CounterFieldWidth(const std::string &name)
   {
      return name == "p" ? order_width :
             name == "Setup" ? 15 :
             std::max<std::size_t>(counter_width, name.length());
   }

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
      if (counter.first == "Setup")
      {
         unit = " ms";
         return FormatTime(counter.second.value);
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
         << std::right << std::setw(15) << "Time";
      for (const auto &counter : run.counters)
      {
         const auto width = CounterFieldWidth(counter.first);
         os << " " << std::setw(static_cast<int>(width)) << counter.first;
      }
      os << "  " << std::left << std::setw(expression_width) << "expression";

      const auto header = os.str();
      GetOutputStream() << std::string(header.length(), '-') << "\n"
                        << header << "\n"
                        << std::string(header.length(), '-') << "\n";
   }

   void PrintRunData(const Run &run, const Run *cv = nullptr)
   {
      auto &out = GetOutputStream();
      // A median prints under the plain benchmark name, like a single run
      const bool aggregate = run.run_type == Run::RT_Aggregate;
      out << std::left << std::setw(static_cast<int>(name_field_width))
          << (aggregate ? run.run_name.str() : run.benchmark_name()) << " ";

      if (run.skipped != bmi::NotSkipped)
      {
         out << (run.skipped == bmi::SkippedWithError ? "ERROR: " : "SKIPPED: ")
             << run.skip_message << "\n";
         return;
      }

      const char *time_unit = bm::GetTimeUnitString(run.time_unit);
      out << std::right << FormatTime(run.GetAdjustedRealTime()) << " "
          << std::left << std::setw(4) << time_unit;

      for (const auto &counter : run.counters)
      {
         std::string unit;
         const std::string value = CounterValue(run, counter, unit);
         const auto width = CounterFieldWidth(counter.first);
         const auto value_width = std::max<int>(1,
                                                static_cast<int>(width - unit.length()));
         out << " " << std::right << std::setw(value_width) << value << unit;
      }

      out << "  " << std::left << std::setw(expression_width)
          << run.report_label;
      if (cv)
      {
         // The cv of the time is a ratio, stored as the accumulated time
         char buffer[64];
         std::snprintf(buffer, sizeof(buffer), "  median of %lld, cv %.2f%%",
                       static_cast<long long>(run.repetitions),
                       100.0 * cv->real_accumulated_time);
         out << buffer;
      }
      out << "\n";
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
         const Run *cv = nullptr;
         if (run.run_type == Run::RT_Aggregate)
         {
            if (run.aggregate_name != "median") { continue; }
            for (const auto &other : reports)
            {
               if (other.run_type == Run::RT_Aggregate &&
                   other.aggregate_name == "cv" &&
                   other.run_name.str() == run.run_name.str())
               {
                  cv = &other;
               }
            }
         }
         const bool print_header = !printed_header ||
                                   !bmi::SameNames(run.counters, prev_counters);
         if (print_header)
         {
            printed_header = true;
            prev_counters = run.counters;
            PrintHeader(run);
         }
         PrintRunData(run, cv);
      }
   }
};


// RT_p on an n^3 hex mesh, counting every face dof once
static constexpr int HdivDofs(int p, int n) noexcept
{
   return 3 * n * n * (p + 1) * (p + 1) * (n * (p + 1) + 1);
}

static void CustomArguments(bm::Benchmark *b) noexcept
{
   constexpr int MAX_NDOFS = 8 * 1024 * (mfem_use_gpu ? 1024 : 8);

   // p <= 6: the registered MFEM and dFEM kernels. On GPUs, the dFEM GlobalQF
   // variants are limited to p <= 3 (QuadratureInterpolator H(div) fallbacks)
   const auto orders = std::vector<int> { 6, 5, 4, 3, 2, 1, 0 };

   constexpr auto inc = [](int n) constexpr noexcept -> int
   {
      return n < 16 ? 1 : n < 64 ? 4 : 16;
   };

   for (auto p : orders)
   {
      for (int n = 2; HdivDofs(p, n) <= MAX_NDOFS; n += inc(n))
      {
         b->Args({p, n});
      }
   }
}

// Kernel specializations for p in [0, 6]: with rule 2p + 3
using Q1Ds = std::integer_sequence<int, 2, 3, 4, 5, 6, 7, 8>;

// Register kernel specializations for MFEM, Global and Local QF backends
template<int... D1D>
void AddMFEMHdivMassSpecializations(std::integer_sequence<int, D1D...> = {})
{
   (VectorFEMassIntegrator::AddSpecialization<FiniteElement::DIV,
    FiniteElement::DIV, 3, D1D, D1D, D1D>(), ...);
}

template<int... D1D>
void AddQIHdivSpecializations(std::integer_sequence<int, D1D...> = {})
{
   using QI = QuadratureInterpolator;
   constexpr auto L = QVectorLayout::byVDIM;
   (QI::TensorEvalHDivKernels::Specialization<3, L, QI::VALUES, D1D, D1D>::Add(),
    ...);
   (QI::TensorEvalHDivTransposeKernels::Specialization<3, L, QI::VALUES, D1D, D1D>::Add(),
    ...);
}

template<typename backend_t, int DIM, typename QT, typename IT, typename OT,
         int... Q1D>
void AddLocalQFActionSpecializations(std::integer_sequence<int, Q1D...> = {})
{
   if constexpr (std::is_same_v<backend_t, local_backend>)
   {
      (mfem::future::AddAction<DIM, Q1D, QT, IT, OT>(), ...);
   }
}

template<typename backend_t, int DIM, int DID, typename QT, typename IT,
         typename OT, int... Q1D>
void AddLocalQFDerivativeSpecializations(std::integer_sequence<int, Q1D...> = {})
{
   if constexpr (std::is_same_v<backend_t, local_backend>)
   {
      (mfem::future::AddDerivativeAction<DIM, Q1D, DID, QT, IT, OT>(), ...);
      (mfem::future::AddDerivativeSetup<DIM, Q1D, DID, QT, IT, OT>(), ...);
      (mfem::future::AddDerivativeApply<DIM, Q1D, DID, QT, IT, OT>(), ...);
   }
}

//-----------------------------------------------------------------------------------
//<--- Q-Functions for H(div) Mass Operators
//-----------------------------------------------------------------------------------

// Piola transform for H(div), with u in reference coordinates:
//   (u, v) = \int (w / det J) (Jᵀ J u) . v

// GLOBAL H(div) Mass Q-Functions
template<int DIM>
struct MF_HdivMass_global_qf
{
   void operator()(tensor_array<const real_t, DIM> &u,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &weight,
                   tensor_array<real_t, DIM> &v) const
   {
      mfem::forall<UseEnzyme>(weight.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         const tensor<real_t, DIM, DIM> Jq = J(q);
         const tensor<real_t, DIM> uq = u(q);
         const real_t c = weight(q) / det(Jq);
         v(q) = c * (transpose(Jq) * (Jq * uq));
      });
   }
};

template<int DIM>
struct PA_HdivMass_Setup_global_qf
{
   void operator()(tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &weight,
                   tensor_array<real_t, DIM, DIM> &D) const
   {
      mfem::forall(weight.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         const tensor<real_t, DIM, DIM> Jq = J(q);
         const real_t c = weight(q) / det(Jq);
         D(q) = c * (transpose(Jq) * Jq);
      });
   }
};

template<int DIM>
struct PA_HdivMass_Apply_global_qf
{
   void operator()(tensor_array<const real_t, DIM> &u,
                   tensor_array<const real_t, DIM, DIM> &D,
                   tensor_array<real_t, DIM> &v) const
   {
      mfem::forall(u.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         v(q) = D(q) * u(q);
      });
   }
};

// LOCAL H(div) Mass Q-Functions
template<int DIM>
struct MF_HdivMass_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<real_t, DIM> &u,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &weight,
                   tensor<real_t, DIM> &v) const
   {
      v = (weight / det(J)) * (transpose(J) * (J * u));
   }
};

template<int DIM>
struct PA_HdivMass_Setup_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<real_t, DIM, DIM> &J,
                   const real_t &weight,
                   tensor<real_t, DIM, DIM> &D) const
   {
      D = (weight / det(J)) * (transpose(J) * J);
   }
};

template<int DIM>
struct PA_HdivMass_Apply_local_qf
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<real_t, DIM> &u,
                   const tensor<real_t, DIM, DIM> &D,
                   tensor<real_t, DIM> &v) const
   {
      v = D * u;
   }
};


//-----------------------------------------------------------------------------------
//<--- BakeOff problem
//-----------------------------------------------------------------------------------

template <Version VER>
struct HdivBakeOff
{
   static constexpr Version version = VER;
   static constexpr int DIM = 3;
   const int p, n, q;
   Mesh smesh;
   ParMesh pmesh;
   RT_FECollection fec;
   ParFiniteElementSpace pfes;
   const Geometry::Type geom_type;
   IntegrationRules irs;
   const IntegrationRule *ir;
   Vector uvec;
   VectorConstantCoefficient unit_vec;
   const int dofs;
   ParGridFunction &nodes;
   ParFiniteElementSpace& mfes;
   ParGridFunction x;
   ParBilinearForm a;

   Array<int> ess_tdof_list, ess_bdr, dom_attr;
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
   double setup_time_ms{};

   void AddHdivIntegrators(ParBilinearForm &form)
   {
      auto *mass = new VectorFEMassIntegrator();
      mass->SetIntRule(ir);
      form.AddDomainIntegrator(mass);
   }

   // Check a dFEM L-vector operator against MFEM PA on a random vector
   void VerifyAgainstMFEM(const Operator &op)
   {
      ParBilinearForm ref(&pfes);
      AddHdivIntegrators(ref);
      ref.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      ref.Assemble();

      Vector r(pfes.GetVSize()), y_ref(pfes.GetVSize()), y(pfes.GetVSize());
      r.Randomize(0x2545f491);
      ref.Mult(r, y_ref);
      op.Mult(r, y);
      y -= y_ref;
      MFEM_VERIFY(y.Normlinf() <= 1e-8 * y_ref.Normlinf(),
                  "❌ dFEM operator differs from MFEM PA: "
                  << y.Normlinf() << " vs " << y_ref.Normlinf());
   }

   HdivBakeOff(int p, int n):
      p(p), n(n), q(2 * p + 3),
      smesh(Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON)),
      pmesh(MPI_COMM_WORLD, (smesh.EnsureNodes(), smesh)),
      fec(p, DIM),
      pfes(&pmesh, &fec),
      geom_type(pmesh.GetTypicalElementGeometry()),
      irs(0, Quadrature1D::GaussLegendre),
      ir(&irs.Get(geom_type, q)), uvec(DIM),
      unit_vec((uvec = 1.0, uvec /= uvec.Norml2(), uvec)),
      dofs(pfes.GetTrueVSize()),
      nodes(*static_cast<ParGridFunction*>(pmesh.GetNodes())),
      mfes(*(nodes.ParFESpace())),
      x(&pfes),
      a(&pfes),
      ess_bdr(pmesh.bdr_attributes.Max()),
      dom_attr(pmesh.attributes.Max()),
      b(&pfes),
      B(pfes.GetVSize()),
      X(x),
      qspace(pmesh, *ir),
      vqspace(qspace, DIM*DIM),
      qfct(vqspace)
   {
      smesh.Clear();
      x.Randomize(0x9e3779b9);

      ess_bdr = 1;
      dom_attr = 1;
      pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // LinearForm b
      b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(unit_vec));
      b.Assemble();

      // BilinearForm a
      const int height = pfes.GetVSize(), width = pfes.GetVSize();
      const auto timeSetup = [&] (auto &&setup)
      {
         MFEM_DEVICE_SYNC;
         StopWatch timer;
         timer.Start();
         setup();
         MFEM_DEVICE_SYNC;
         timer.Stop();
         setup_time_ms += 1e3 * timer.RealTime();
      };
      const auto formLinearSystem = [&] (Vector &arg1)
      {
         Operator *A_ptr = nullptr;
         wop = std::make_unique<WrapOpArg1>(dop, height, width, arg1);
         VerifyAgainstMFEM(*wop);
         wop->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         A.Reset(A_ptr);
      };
      const auto formLinearSystemDerivative = [&]
      {
         Operator *A_ptr = nullptr;
         dwop = std::make_unique<WrapDerivativeOp>(ddop, height, width);
         VerifyAgainstMFEM(*dwop);
         dwop->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
         A.Reset(A_ptr);
      };
      // PA MFEM Setup
      const auto mPASetup = [&]
      {
         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         AddHdivIntegrators(a);
         timeSetup([&] { a.Assemble(); });
         a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
      };
      // MF ∂FEM setup
      const auto dMFSetup = [&] (auto backend, auto qfunction)
      {
         using backend_t = decltype(backend);
         const auto ifd = std::vector<FieldDescriptor> {{U, &pfes}, {Ξ, &mfes}};
         const auto ofd = std::vector<FieldDescriptor> {{U, &pfes}};
         dop = std::make_unique<DifferentiableOperator>(ifd, ofd, pmesh);
         dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         dop->template AddDomainIntegrator<backend_t>(
            qfunction,
            tuple{Value<U>{}, Gradient<Ξ>{}, Weight{}},
            tuple{Value<U>{}},
            *ir, dom_attr);
         using QT = decltype(qfunction);
         using IT = decltype(tuple{Value<U>{}, Gradient<Ξ>{}, Weight{}});
         using OT = decltype(tuple{Value<U>{}});
         AddLocalQFActionSpecializations<backend_t, DIM, QT, IT, OT>(Q1Ds{});
         formLinearSystem(nodes);
      };
      // MF ∂FEM GetDerivative setup
      const auto dMFGetDerivativeSetup = [&] (auto backend, auto qfunction,
                                              bool use_cached_setup)
      {
         using backend_t = decltype(backend);
         const auto ifd = std::vector<FieldDescriptor> {{U, &pfes}, {Ξ, &mfes}};
         const auto ofd = std::vector<FieldDescriptor> {{U, &pfes}};
         dop = std::make_unique<DifferentiableOperator>(ifd, ofd, pmesh);
         dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         dop->template AddDomainIntegrator<backend_t>(
            qfunction,
            tuple{Value<U>{}, Gradient<Ξ>{}, Weight{}},
            tuple{Value<U>{}},
            *ir, dom_attr, future::Derivatives<U> {});
         using QT = decltype(qfunction);
         using IT = decltype(tuple{Value<U>{}, Gradient<Ξ>{}, Weight{}});
         using OT = decltype(tuple{Value<U>{}});
         AddLocalQFDerivativeSpecializations<backend_t, DIM, U, QT, IT, OT>(Q1Ds{});
         MultiVector state{x, nodes};
         ddop = dop->GetDerivative(U, state, use_cached_setup);
         if (use_cached_setup)
         {
            timeSetup([&] { ddop->SetupQpCache(); });
         }
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
            *ir, dom_attr);
         using SetupQT = decltype(setup_qf);
         using SetupIT = decltype(tuple{Gradient<Ξ>{}, Weight{}});
         using SetupOT = decltype(tuple{Identity<Q>{}});
         AddLocalQFActionSpecializations<backend_t, DIM, SetupQT, SetupIT, SetupOT>
         (Q1Ds{});
         MultiVector N{nodes}, D{qfct};
         timeSetup([&] { qdata_setup_dop->Mult(N, D); });

         const auto ifd1 = std::vector<FieldDescriptor> {{U, &pfes}, {Q, &vqspace}};
         const auto ofd1 = std::vector<FieldDescriptor> {{U, &pfes}};
         dop = std::make_unique<DifferentiableOperator>(ifd1, ofd1, pmesh);
         dop->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);
         dop->template AddDomainIntegrator<backend_t>(
            apply_qf,
            tuple{Value<U>{}, Identity<Q>{}},
            tuple{Value<U>{}},
            *ir, dom_attr);
         using ApplyQT = decltype(apply_qf);
         using ApplyIT = decltype(tuple{Value<U>{}, Identity<Q>{}});
         using ApplyOT = decltype(tuple{Value<U>{}});
         AddLocalQFActionSpecializations<backend_t, DIM, ApplyQT, ApplyIT, ApplyOT>
         (Q1Ds{});
         formLinearSystem(qfct);
      };

      // MFEM PA version
      if constexpr (VER == Version::PA_mfem_std)
      {
         mPASetup();
      }
      // dFEM Global versions
      else if constexpr (VER == Version::MF_dfem_global_primal)
      {
         dMFSetup(global_backend{}, MF_HdivMass_global_qf<DIM> {});
      }
      else if constexpr (VER == Version::MF_dfem_global_derivative)
      {
         dMFGetDerivativeSetup(global_backend{}, MF_HdivMass_global_qf<DIM> {}, false);
      }
      else if constexpr (VER == Version::PA_dfem_global_derivative)
      {
         dMFGetDerivativeSetup(global_backend{}, MF_HdivMass_global_qf<DIM> {}, true);
      }
      else if constexpr (VER == Version::PA_dfem_global_primal)
      {
         dPASetup(global_backend{},
                  PA_HdivMass_Setup_global_qf<DIM> {},
                  PA_HdivMass_Apply_global_qf<DIM> {});
      }
      // dFEM Local versions
      else if constexpr (VER == Version::MF_dfem_local_primal)
      {
         dMFSetup(local_backend{}, MF_HdivMass_local_qf<DIM> {});
      }
      else if constexpr (VER == Version::MF_dfem_local_derivative)
      {
         dMFGetDerivativeSetup(local_backend{}, MF_HdivMass_local_qf<DIM> {}, false);
      }
      else if constexpr (VER == Version::PA_dfem_local_derivative)
      {
         dMFGetDerivativeSetup(local_backend{}, MF_HdivMass_local_qf<DIM> {}, true);
      }
      else if constexpr (VER == Version::PA_dfem_local_primal)
      {
         dPASetup(local_backend{},
                  PA_HdivMass_Setup_local_qf<DIM> {},
                  PA_HdivMass_Apply_local_qf<DIM> {});
      }
      else { static_assert(false, "Invalid version"); }
   }

   virtual ~HdivBakeOff() = default;

   virtual void benchmark() = 0;

   [[nodiscard]] double SumMdofs() const noexcept { return mdofs; }

   [[nodiscard]] double MDofs() const noexcept { return 1e-6 * dofs; }

   [[nodiscard]] double SetupTimeMilliseconds() const noexcept
   {
      return setup_time_ms;
   }
};

// CG on the H(div) mass problem, with a fixed iteration count
template <Version VER>
struct HDIV : public HdivBakeOff<VER>
{
   const int max_it = 32, print_lvl = -1;

   CGSolver cg;

   using base = HdivBakeOff<VER>;
   using base::A;
   using base::B;
   using base::X;
   using base::mdofs;

   HDIV(int p, int n) noexcept: base(p, n), cg(MPI_COMM_WORLD)
   {
      cg.SetOperator(*A);
      cg.SetAbsTol(1e-12);
      cg.SetRelTol(1e-8);
      cg.SetMaxIter(max_it);
      cg.SetPrintLevel(print_lvl);
      cg.iterative_mode = false;

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
   state.counters["Setup"] = bm::Counter(run->SetupTimeMilliseconds());
   state.counters["p"] = bm::Counter(state.range(0));
   state.SetLabel(BenchmarkExpression<T::version>());
}
#define REGISTER(VER) \
   BENCHMARK_TEMPLATE(Benchmark, HDIV<Version::VER>) \
   ->Name(BenchmarkPath<Version::VER>())->Apply(CustomArguments)->Unit(bm::kMillisecond)

// {Bᵀ D B u}
REGISTER(PA_mfem_std);

// {Bᵀ Ql'(B u) B du}
REGISTER(PA_dfem_local_derivative);

// {Bᵀ Ql(B u)}
REGISTER(PA_dfem_local_primal);

// {Bᵀ {Qg'(B u) B du}}
REGISTER(PA_dfem_global_derivative);

// {Bᵀ {Qg(B u)}}
REGISTER(PA_dfem_global_primal);

// {Bᵀ Ql(B u, G x)}
REGISTER(MF_dfem_local_primal);

// {Bᵀ Ql'(B u, G x) B du}
REGISTER(MF_dfem_local_derivative);

// {Bᵀ {Qg({B u}, {G x})}}
REGISTER(MF_dfem_global_primal);

// {Bᵀ {Qg'({B u}, {G x}) {B du}}}
REGISTER(MF_dfem_global_derivative);

// main
int main(int argc, char *argv[])
{
   static mfem::MPI_Session mpi(argc, argv);

   ExpressionReporter CR;
   bm::Initialize(&argc, argv);

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
   device.Print();

   AddMFEMHdivMassSpecializations(Q1Ds{});
   AddQIHdivSpecializations(std::integer_sequence<int, 2, 3, 4, 5> {});

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return EXIT_FAILURE; }

   bm::RunSpecifiedBenchmarks((bm::BenchmarkReporter*)&CR);

   return EXIT_SUCCESS;
}

#endif // MFEM_USE_BENCHMARK
