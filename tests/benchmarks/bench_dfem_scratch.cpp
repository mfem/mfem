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

// Cost of the different scratch strategies for the linearized action of a
// global q-function.
//
// All variants evaluate the same weak residual
//
//    F(u)_i = int_Omega phi_i c u^3 dx,
//
// and its directional derivative, but they differ in where the intermediate
// value s = u^2 lives and in who owns its tangent (shadow) memory:
//
//   local          : LocalQFBackend, per-quadrature-point q-function.  The
//                    temporary is a register, there is no scratch at all.
//                    Reference point for "no scratch machinery".
//   global         : GlobalQFBackend, single fused kernel, temporary is a
//                    register inside the forall body.  Reference point for
//                    "global backend, no scratch".
//   global+local   : GlobalQFBackend, split into two kernels, the temporary is
//                    a Vector allocated inside the q-function.  Enzyme
//                    allocates and propagates the shadow buffer itself, so this
//                    is the "let Enzyme own the memory" variant, and the one to
//                    compare global+bank against.
//
//                    A scratch buffer handed to the q-function through the
//                    signature as a dFEM *field* is deliberately not benchmarked
//                    here, because neither form works today:
//                      - as an output field, outputs reach Enzyme as
//                        enzyme_dupnoneed, so the primal store of the scratch is
//                        eliminated and the tangent loses its s*du term, coming
//                        out at exactly 2/3 of the true value;
//                      - as an input field it would be enzyme_dup, which is the
//                        right shape, but inputs are materialized through
//                        Vector::Read() and the q-function parameter has to be
//                        const, so the scratch cannot be written.
//                    Both need a library change to become expressible.
//
//   global+bank    : GlobalQFBackend, split into two kernels, the temporary
//                    lives in a ScratchBank owned by the q-function.  Enzyme
//                    differentiates with respect to the q-function object
//                    itself (enzyme_dup on &qfunc / &qfunc_shadow), so the
//                    tangent is reached through the bank's indirections.
//   global+bank+gs : same as global+bank plus a global scratch tuple
//                    (bool, real_t, Vector), to isolate the extra cost of the
//                    non-quadrature-point scratch members.
//
// Only the derivative timings are expected to differ significantly: the
// forward action does not touch any shadow memory.  Every variant reports its
// max-norm error relative to the analytic result, so a timing win that comes
// from a dropped derivative term is visible.
//
// Usage (CPU):   ./bench_dfem_scratch -o 3 -nh 8
//        (GPU):  ./bench_dfem_scratch -o 3 -nh 16 -d cuda

#include "mfem.hpp"

#if defined(MFEM_USE_MPI) && defined(MFEM_USE_ENZYME)

#include "fem/dfem/doperator.hpp"
#include "fem/dfem/backends/global_qf/prelude.hpp"
#include "fem/dfem/backends/local_qf/prelude.hpp"
#include "fem/dfem/backends/scratch_bank.hpp"
#include "linalg/tensor_arrays.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <iostream>
#include <string>
#include <vector>

using namespace mfem;
using namespace mfem::future;

// With Enzyme the differentiated scalar type is just real_t; the scratch bank
// shadow mechanism is Enzyme-specific, hence the MFEM_USE_ENZYME guard above.
using dscalar_t = real_t;

static constexpr int U = 0, COEF = 1, COORDS = 2;

/// @brief Quadrature point loop used by every global q-function below.
///
/// mfem::forall<UseEnzyme> is only differentiated correctly on the device
/// path; on the CPU path Enzyme does not differentiate through it (known
/// issue), which silently zeroes the tangent.  Dispatching on the active
/// backend keeps every variant both correct and comparable on CPU and GPU.
template <typename lambda>
inline void QForall(const int N, lambda &&body)
{
   if (Device::Allows(Backend::DEVICE_MASK))
   {
      mfem::forall<UseEnzyme>(N, body);
   }
   else
   {
      for (int q = 0; q < N; ++q) { body(q); }
   }
}

/// Q-functions ///////////////////////////////////////////////////////////////

// y = c * u^3 * det(J) * w, temporary in a register.
template <int DIM>
struct CubicLocalQF
{
   MFEM_HOST_DEVICE inline
   void operator()(const dscalar_t &u,
                   const real_t &c,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   dscalar_t &y) const
   {
      const dscalar_t s = u * u;
      y = c * s * u * det(J) * w;
   }
};

// Same, single fused global kernel, temporary in a register.
template <int DIM>
struct CubicGlobalQF
{
   void operator()(tensor_array<const dscalar_t> &u,
                   tensor_array<const real_t> &c,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<dscalar_t> &y) const
   {
      QForall(u.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         const dscalar_t s = u(q) * u(q);
         y(q) = c(q) * s * u(q) * det(J(q)) * w(q);
      });
   }
};

// Split into two kernels; the temporary is a buffer allocated inside the
// q-function, so Enzyme allocates and propagates the shadow buffer itself.
template <int DIM>
struct CubicGlobalScratchLocalQF
{
   void operator()(tensor_array<const dscalar_t> &u,
                   tensor_array<const real_t> &c,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<dscalar_t> &y) const
   {
      const int NQ = static_cast<int>(u.size());
      Vector s_vec(NQ);
      s_vec.UseDevice(true);
      auto s = make_tensor_array<>(s_vec.ReadWrite(), NQ);

      QForall(NQ, [=] MFEM_HOST_DEVICE (int q)
      {
         s(q) = u(q) * u(q);
      });

      QForall(NQ, [=] MFEM_HOST_DEVICE (int q)
      {
         y(q) = c(q) * s(q) * u(q) * det(J(q)) * w(q);
      });
   }
};

// Split into two kernels; the temporary lives in the q-function's ScratchBank.
template <int DIM>
struct CubicGlobalScratchBankQF : QFWithScratchType
{
   void operator()(tensor_array<const dscalar_t> &u,
                   tensor_array<const real_t> &c,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<dscalar_t> &y) const
   {
      const int NQ = nq;
      auto s = make_tensor_array<>(GetScratchPointer(0), NQ);

      QForall(NQ, [=] MFEM_HOST_DEVICE (int q)
      {
         s(q) = u(q) * u(q);
      });

      QForall(NQ, [=] MFEM_HOST_DEVICE (int q)
      {
         y(q) = c(q) * s(q) * u(q) * det(J(q)) * w(q);
      });
   }
};

// Same as above with an additional global scratch tuple (bool, real_t, Vector).
// The global entries are set to a neutral scaling so that the computed value
// stays identical to the other variants.
template <int DIM>
struct CubicGlobalScratchBankGlobalQF : QFWithGlobalScratchType
{
   void operator()(tensor_array<const dscalar_t> &u,
                   tensor_array<const real_t> &c,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<dscalar_t> &y) const
   {
      const int NQ = nq;
      auto s = make_tensor_array<>(GetScratchPointer(0), NQ);

      auto &has_scale = GetGlobalScratch<0>();
      const auto scale = GetGlobalScratch<1>();
      auto &global_vector = GetGlobalScratch<2>();

      has_scale = global_vector.Size() > 0;
      if (has_scale) { global_vector(0) = 1.0; }
      const real_t global_scale = has_scale ? scale * global_vector(0) : 1.0;

      QForall(NQ, [=] MFEM_HOST_DEVICE (int q)
      {
         s(q) = u(q) * u(q);
      });

      QForall(NQ, [=] MFEM_HOST_DEVICE (int q)
      {
         y(q) = global_scale * c(q) * s(q) * u(q) * det(J(q)) * w(q);
      });
   }
};

/// Which action is timed by one call to RunVariant().  The two phases are run
/// as separate passes so that each row can be printed the moment it is
/// measured, the way a Google Benchmark report streams: one row per finished
/// benchmark.  The cost is that the (untimed) operator setup runs once per
/// phase instead of once per variant.
enum class Phase { Forward, Derivative };

const char *PhaseName(Phase p)
{
   return (p == Phase::Forward) ? "Forward" : "Derivative";
}

/// Variants //////////////////////////////////////////////////////////////////

enum class Variant
{
   Local,             // local backend, no scratch
   Global,            // global backend, no scratch
   GlobalScratchLocal,// global backend, scratch allocated inside the q-function
   GlobalScratchBank, // global backend, ScratchBank
   GlobalScratchBankGlobal // ScratchBank + global scratch tuple
};

const char *VariantName(Variant v)
{
   switch (v)
   {
      case Variant::Local: return "local";
      case Variant::Global: return "global";
      case Variant::GlobalScratchLocal: return "global+local";
      case Variant::GlobalScratchBank: return "global+bank";
      case Variant::GlobalScratchBankGlobal: return "global+bank+gs";
   }
   return "unknown";
}

// Benchmark-style tag used to build the row name, e.g. "Hex_GlobalBank".
const char *VariantTag(Variant v)
{
   switch (v)
   {
      case Variant::Local: return "Local";
      case Variant::Global: return "Global";
      case Variant::GlobalScratchLocal: return "GlobalLocalAlloc";
      case Variant::GlobalScratchBank: return "GlobalBank";
      case Variant::GlobalScratchBankGlobal: return "GlobalBankGS";
   }
   return "Unknown";
}

template <int DIM, Variant V> struct VariantTraits;

template <int DIM> struct VariantTraits<DIM, Variant::Local>
{
   using qfunc_t = CubicLocalQF<DIM>;
   using backend_t = LocalQFBackend;
};
template <int DIM> struct VariantTraits<DIM, Variant::Global>
{
   using qfunc_t = CubicGlobalQF<DIM>;
   using backend_t = GlobalQFBackend;
};
template <int DIM> struct VariantTraits<DIM, Variant::GlobalScratchLocal>
{
   using qfunc_t = CubicGlobalScratchLocalQF<DIM>;
   using backend_t = GlobalQFBackend;
};
template <int DIM> struct VariantTraits<DIM, Variant::GlobalScratchBank>
{
   using qfunc_t = CubicGlobalScratchBankQF<DIM>;
   using backend_t = GlobalQFBackend;
};
template <int DIM> struct VariantTraits<DIM, Variant::GlobalScratchBankGlobal>
{
   using qfunc_t = CubicGlobalScratchBankGlobalQF<DIM>;
   using backend_t = GlobalQFBackend;
};

/// Utilities /////////////////////////////////////////////////////////////////

// Above this relative error a variant is not computing the true tangent.
constexpr real_t wrong_tangent_tol = 1e-10;

struct Timings
{
   double forward = 0.0;
   double derivative = 0.0;
   real_t err_forward = 0.0;     // relative to the analytic reference
   real_t err_derivative = 0.0;  // relative to the analytic reference
   real_t norm_derivative = 0.0; // |dy|_inf, diagnostic
   real_t norm_ref = 0.0;        // |dy_ref|_inf, diagnostic
   int dofs = 0;
   long long nq = 0;
   int order = 0;
   int mesh_n = 0;
   int iterations = 0;
};

template <typename F>
double TimeIt(const int iterations, MPI_Comm comm, F &&f)
{
   MPI_Barrier(comm);
   MFEM_DEVICE_SYNC;

   StopWatch timer;
   timer.Start();
   for (int i = 0; i < iterations; i++) { f(); }
   MFEM_DEVICE_SYNC;
   timer.Stop();

   const double local_time = timer.RealTime();
   double global_time = 0.0;
   MPI_Allreduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, comm);
   return global_time / iterations;
}

template <int DIM>
Mesh MakeTensorMesh(int n)
{
   if constexpr (DIM == 2)
   {
      return Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, true, 1.0, 1.0);
   }
   else
   {
      return Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   }
}

real_t GlobalNormlinf(const Vector &v, MPI_Comm comm)
{
   const real_t local = v.Normlinf();
   real_t global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 comm);
   return global;
}

// State u and coefficient c used by every variant.
inline real_t StateFunction(const Vector &x) { return 1.0 + x(0) + 0.25 * x(1); }
inline real_t CoefFunction(const Vector &x) { return 0.5 + x(0) + 0.125 * x(1); }

// Analytic references: int phi_i c u^3 dx and, for the direction du = 1,
// int phi_i 3 c u^2 dx, both integrated with the same quadrature rule.
void ComputeReference(ParFiniteElementSpace &pfes, const IntegrationRule &ir,
                      Vector &y_ref, Vector &dy_ref)
{
   FunctionCoefficient forward_coeff([](const Vector &x)
   {
      const real_t u = StateFunction(x);
      return CoefFunction(x) * u * u * u;
   });
   FunctionCoefficient derivative_coeff([](const Vector &x)
   {
      const real_t u = StateFunction(x);
      return 3.0 * CoefFunction(x) * u * u;
   });

   ParLinearForm forward_lf(&pfes);
   forward_lf.AddDomainIntegrator(new DomainLFIntegrator(forward_coeff, &ir));
   forward_lf.Assemble();
   y_ref.SetSize(pfes.GetTrueVSize());
   pfes.GetProlongationMatrix()->MultTranspose(forward_lf, y_ref);

   ParLinearForm derivative_lf(&pfes);
   derivative_lf.AddDomainIntegrator(
      new DomainLFIntegrator(derivative_coeff, &ir));
   derivative_lf.Assemble();
   dy_ref.SetSize(pfes.GetTrueVSize());
   pfes.GetProlongationMatrix()->MultTranspose(derivative_lf, dy_ref);
}

// Relative max-norm error of @a v with respect to @a ref.
real_t RelativeError(const Vector &v, const Vector &ref, MPI_Comm comm)
{
   Vector diff(v);
   diff.HostReadWrite();
   diff -= ref;
   const real_t ref_norm = GlobalNormlinf(ref, comm);
   return GlobalNormlinf(diff, comm) / std::max(ref_norm, real_t(1e-300));
}

/// Benchmark case ////////////////////////////////////////////////////////////

template <int DIM, Variant V>
Timings RunVariant(const int order, const int mesh_n, const int warmup,
                   const int iterations, const bool cached_derivative,
                   const int quad_order, const Phase phase)
{
   using qfunc_t = typename VariantTraits<DIM, V>::qfunc_t;
   using backend_t = typename VariantTraits<DIM, V>::backend_t;
   constexpr bool uses_bank = (V == Variant::GlobalScratchBank ||
                               V == Variant::GlobalScratchBankGlobal);

   Mesh smesh = MakeTensorMesh<DIM>(mesh_n);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   smesh.Clear();

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   const int p = std::max(order, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   // quad_order <= 0 selects the default 2*p rule.
   const int q_order = (quad_order > 0) ? quad_order : 2 * p;
   const IntegrationRule *ir =
      &IntRules.Get(pmesh.GetTypicalElementGeometry(), q_order);
   const int nq_total = pmesh.GetNE() * ir->GetNPoints();

   // Coefficient c, stored as quadrature data.
   QuadratureSpace qspace(pmesh, *ir);
   VectorQuadratureSpace coef_qspace(qspace, 1);
   QuadratureFunction coef(coef_qspace);
   coef.UseDevice(true);
   FunctionCoefficient coeff_fc(CoefFunction);
   coeff_fc.Project(coef);

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   const std::vector<FieldDescriptor> in_fds
   {
      { U, &pfes },
      { COEF, &coef_qspace },
      { COORDS, mfes }
   };
   const std::vector<FieldDescriptor> out_fds { FieldDescriptor{ U, &pfes } };

   DifferentiableOperator dop(in_fds, out_fds, pmesh);

   qfunc_t qf;
   if constexpr (uses_bank)
   {
      // One scalar scratch value per quadrature point of the whole domain.
      qf.SetScratch(nq_total, {1});
   }
   Vector global_scratch_vec;
   if constexpr (V == Variant::GlobalScratchBankGlobal)
   {
      global_scratch_vec.SetSize(1);
      global_scratch_vec.UseDevice(true);
      global_scratch_vec = 1.0;
      const bool global_flag = true;
      const real_t global_scalar = 1.0;
      qf.SetGlobalScratch(
         make_tuple(global_flag, global_scalar, global_scratch_vec));
   }

   dop.AddDomainIntegrator<backend_t>(
      qf,
      Inputs<Value<U>, Identity<COEF>, Gradient<COORDS>, Weight> {},
      Outputs<Value<U>> {},
      *ir, all_domain_attr, Derivatives<U> {});

   // State, direction and outputs.
   Vector xtvec(pfes.GetTrueVSize()), ytvec(pfes.GetTrueVSize());
   Vector dxtvec(pfes.GetTrueVSize()), dytvec(pfes.GetTrueVSize());
   xtvec.UseDevice(true);
   ytvec.UseDevice(true);
   dxtvec.UseDevice(true);
   dytvec.UseDevice(true);

   ParGridFunction x_gf(&pfes);
   FunctionCoefficient input_coeff(StateFunction);
   x_gf.ProjectCoefficient(input_coeff);
   x_gf.GetTrueDofs(xtvec);
   dxtvec = 1.0;
   ytvec = 0.0;
   dytvec = 0.0;

   Vector nodestv;
   nodes->GetTrueDofs(nodestv);

   // Move the fixed inputs to the device before warmup/timing so that repeated
   // applies do not pay host-to-device copies.
   xtvec.Read();
   dxtvec.Read();
   nodestv.Read();
   coef.Read();

   MultiVector X{xtvec, coef, nodestv};
   MultiVector Y{ytvec};
   MultiVector DY{dytvec};

   auto ddop = dop.GetDerivative(U, X, cached_derivative);

   for (int i = 0; i < warmup; i++)
   {
      if (phase == Phase::Forward) { dop.Mult(X, Y); }
      else                         { ddop->Mult(dxtvec, DY); }
   }
   MFEM_DEVICE_SYNC;

   Timings timings;
   timings.dofs = pfes.GlobalTrueVSize();
   // Sum over ranks: Dofs is global, so #qp must be too.
   {
      long long nq_local = nq_total, nq_global = 0;
      MPI_Allreduce(&nq_local, &nq_global, 1, MPI_LONG_LONG, MPI_SUM,
                    pmesh.GetComm());
      timings.nq = static_cast<long long>(nq_global);
   }
   timings.order = order;
   timings.mesh_n = mesh_n;
   timings.iterations = iterations;

   Vector y_ref, dy_ref;
   ComputeReference(pfes, *ir, y_ref, dy_ref);

   if (phase == Phase::Forward)
   {
      timings.forward = TimeIt(iterations, pmesh.GetComm(), [&]()
      {
         dop.Mult(X, Y);
      });
      timings.err_forward = RelativeError(ytvec, y_ref, pmesh.GetComm());
   }
   else
   {
      timings.derivative = TimeIt(iterations, pmesh.GetComm(), [&]()
      {
         ddop->Mult(dxtvec, DY);
      });
      timings.err_derivative = RelativeError(dytvec, dy_ref, pmesh.GetComm());
   }

   return timings;
}

/// Reporting /////////////////////////////////////////////////////////////////

// Counts in the SI style used by Google Benchmark reports: 531441 ->
// "531.441k", 4096000 -> "4.096M".  Up to 6 significant digits, trailing zeros
// stripped by the default float format.
std::string FormatCount(const double v)
{
   std::ostringstream os;
   os << std::setprecision(6);
   const double a = std::abs(v);
   if (a >= 1e9)      { os << v * 1e-9 << "G"; }
   else if (a >= 1e6) { os << v * 1e-6 << "M"; }
   else if (a >= 1e3) { os << v * 1e-3 << "k"; }
   else               { os << v; }
   return os.str();
}

void PrintLegend()
{
   if (Mpi::WorldRank() != 0) { return; }
   mfem::out <<
             "\nAll variants evaluate F(u) = int phi c u^3 dx and its linearization.\n"
             "They differ only in where the intermediate s = u^2 lives and who owns\n"
             "its tangent (shadow) memory.\n"
             "\nVariants\n"
             "  Local             local backend, per-quadrature-point q-function.\n"
             "                    s is a register, no scratch machinery at all.\n"
             "  Global            global backend, one fused kernel, s in a register.\n"
             "                    Reference for 'global backend, no scratch'.\n"
             "  GlobalLocalAlloc  global backend, two kernels, s in a Vector allocated\n"
             "                    inside the q-function.  Enzyme allocates and\n"
             "                    propagates the shadow buffer itself.\n"
             "  GlobalBank        global backend, two kernels, s in a ScratchBank owned\n"
             "                    by the q-function.  Enzyme differentiates w.r.t. the\n"
             "                    q-function object (enzyme_dup on &qf/&qf_shadow), so\n"
             "                    the tangent is reached through the bank indirections.\n"
             "  GlobalBankGS      GlobalBank plus a global scratch tuple\n"
             "                    (bool, real_t, Vector), isolating its extra cost.\n"
             "\n  GlobalBank vs GlobalLocalAlloc is the comparison of interest:\n"
             "  bank-owned vs Enzyme-owned scratch memory.\n"
             "\nRows\n"
             "  _Forward          DifferentiableOperator::Mult\n"
             "  _Derivative       GetDerivative(U)->Mult, the linearized action.\n"
             "                    Only this one touches shadow memory.\n"
             "  name/<p>/<n>      polynomial order and elements per direction.\n"
             "\nColumns\n"
             "  Time              wall time of one application (max over ranks).\n"
             "  Dofs              global true dofs;  #qp  total quadrature points,\n"
             "                    both summed over all MPI ranks.\n"
             "  MDof/s            Dofs / Time, so higher is better.\n"
             "  vs-local          Time relative to Local of the same row kind.\n"
             "  rel_err           max-norm error against the analytic result.\n"
             "                    Above 1e-10 the row is flagged: a variant that is\n"
             "                    not computing the true tangent is not doing the same\n"
             "                    work, so its Time is not comparable.\n"
             << std::endl;
}

void PrintTableHeader()
{
   if (Mpi::WorldRank() != 0) { return; }
   mfem::out << "\n"
             << std::left << std::setw(38) << "Benchmark"
             << std::right
             << std::setw(12) << "Time"
             << std::setw(12) << "Iterations"
             << std::setw(12) << "Dofs"
             << std::setw(12) << "MDof/s"
             << std::setw(12) << "#qp"
             << std::setw(6)  << "p"
             << std::setw(11) << "vs-local"
             << std::setw(11) << "rel_err"
             << "\n" << std::string(124, '-') << std::endl;
}

// One row per timed quantity, in the style of a Google Benchmark report:
//   <mesh>_<variant>_<kind>/<order>/<mesh_n>   <time> ms  <iters>  <counters>
void PrintRow(const char *mesh_name, Variant v, const char *kind,
              const double seconds, const double seconds_local,
              const real_t rel_err, const Timings &t)
{
   if (Mpi::WorldRank() != 0) { return; }

   std::ostringstream name;
   name << mesh_name << "_" << VariantTag(v) << "_" << kind
        << "/" << t.order << "/" << t.mesh_n;

   std::ostringstream time_str;
   time_str << std::fixed << std::setprecision(3) << seconds * 1e3 << " ms";

   std::ostringstream rate_str;
   rate_str << std::fixed << std::setprecision(2)
            << (t.dofs / seconds) * 1e-6;

   std::ostringstream vs_local_str;
   vs_local_str << std::fixed << std::setprecision(2)
                << seconds / seconds_local << "x";

   std::ostringstream err_str;
   err_str << std::scientific << std::setprecision(1) << rel_err;

   mfem::out << std::left << std::setw(38) << name.str()
             << std::right
             << std::setw(12) << time_str.str()
             << std::setw(12) << t.iterations
             << std::setw(12) << FormatCount(t.dofs)
             << std::setw(12) << rate_str.str()
             << std::setw(12) << FormatCount(t.nq)
             << std::setw(6)  << t.order
             << std::setw(11) << vs_local_str.str()
             << std::setw(11) << err_str.str();

   // A variant that does not reproduce the analytic tangent is not doing the
   // same work as the others, so its timing must not be compared against them.
   if (!(rel_err < wrong_tangent_tol))
   {
      mfem::out << "   <-- WRONG TANGENT, timing not comparable";
   }
   mfem::out << std::endl;
}

// Run one (variant, phase) and stream its row out as soon as it is measured.
// Returns the measured time so later rows can be scaled against Local.
template <int DIM, Variant V>
double RunAndPrintRow(const char *mesh_name, const Phase phase,
                      const double local_time, const int order,
                      const int mesh_n, const int warmup, const int iterations,
                      const bool cached_derivative, const int quad_order)
{
   const Timings t = RunVariant<DIM, V>(order, mesh_n, warmup, iterations,
                                        cached_derivative, quad_order, phase);
   const bool fwd = (phase == Phase::Forward);
   const double time = fwd ? t.forward : t.derivative;
   const real_t err = fwd ? t.err_forward : t.err_derivative;
   // Local runs first in each pass and is its own baseline.
   const double baseline = (local_time > 0.0) ? local_time : time;

   PrintRow(mesh_name, V, PhaseName(phase), time, baseline, err, t);
   return time;
}

template <int DIM>
void RunAllVariants(const int order, const int mesh_n, const int warmup,
                    const int iterations, const bool cached_derivative,
                    const int quad_order)
{
   const char *mesh_name = (DIM == 2) ? "quad" : "hex";

   PrintTableHeader();

#define RUN_ROW(VAR, PHASE, BASE) \
   RunAndPrintRow<DIM, VAR>(mesh_name, PHASE, BASE, order, mesh_n, warmup, \
                            iterations, cached_derivative, quad_order)

   // Pass 1: forward action. Each row is printed as soon as it is measured.
   const double fwd_local = RUN_ROW(Variant::Local, Phase::Forward, 0.0);
   RUN_ROW(Variant::Global, Phase::Forward, fwd_local);
   RUN_ROW(Variant::GlobalScratchLocal, Phase::Forward, fwd_local);
   RUN_ROW(Variant::GlobalScratchBank, Phase::Forward, fwd_local);
   RUN_ROW(Variant::GlobalScratchBankGlobal, Phase::Forward, fwd_local);

   if (Mpi::WorldRank() == 0) { mfem::out << std::endl; }

   // Pass 2: derivative action.
   const double drv_local = RUN_ROW(Variant::Local, Phase::Derivative, 0.0);
   const double drv_global =
      RUN_ROW(Variant::Global, Phase::Derivative, drv_local);
   const double drv_local_alloc =
      RUN_ROW(Variant::GlobalScratchLocal, Phase::Derivative, drv_local);
   const double drv_bank =
      RUN_ROW(Variant::GlobalScratchBank, Phase::Derivative, drv_local);
   const double drv_bank_gs =
      RUN_ROW(Variant::GlobalScratchBankGlobal, Phase::Derivative, drv_local);

#undef RUN_ROW

   if (Mpi::Root())
   {
      mfem::out << std::string(124, '-') << "\n"
                << "scratch cost, derivative action ("
                << mesh_name << ", p=" << order << "):"
                << std::fixed << std::setprecision(2)
                << "  bank/local-alloc=" << drv_bank / drv_local_alloc << "x"
                << "   bank/no-scratch=" << drv_bank / drv_global << "x"
                << "   bank+gs/bank=" << drv_bank_gs / drv_bank << "x"
                << std::endl;
   }
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int order = 3;
   int quad_mesh_n = 32;
   int hex_mesh_n = 8;
   int warmup = 3;
   int iterations = 25;
   int quad_order = 0;
   bool cached_derivative = false;
   bool run_2d = true;
   bool run_3d = true;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&quad_mesh_n, "-nq", "--quad-mesh-size",
                  "Number of quad elements per direction.");
   args.AddOption(&hex_mesh_n, "-nh", "--hex-mesh-size",
                  "Number of hex elements per direction.");
   args.AddOption(&warmup, "-w", "--warmup", "Number of warmup iterations.");
   args.AddOption(&iterations, "-i", "--iterations",
                  "Number of timed iterations.");
   args.AddOption(&quad_order, "-q", "--quadrature-order",
                  "Integration rule order. Non-positive selects 2*order. "
                  "A 1D Gauss-Legendre rule of order q has q/2+1 points, so "
                  "-q 15 gives 8 points per dimension.");
   args.AddOption(&cached_derivative, "-c", "--cached-derivative", "-no-c",
                  "--no-cached-derivative",
                  "Use the cached DerivativeSetup/DerivativeApply path.");
   args.AddOption(&run_2d, "-2d", "--run-2d", "-no-2d", "--no-run-2d",
                  "Run the 2D (quad) cases.");
   args.AddOption(&run_3d, "-3d", "--run-3d", "-no-3d", "--no-run-3d",
                  "Run the 3D (hex) cases.");
   args.AddOption(&device_config, "-d", "--device",
                  "MFEM device configuration string.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::WorldRank() == 0) { args.PrintUsage(mfem::out); }
      return 1;
   }

   Device device(device_config);
   if (Mpi::WorldRank() == 0)
   {
      args.PrintOptions(mfem::out);
      device.Print(mfem::out);
   }
   PrintLegend();

   if (run_2d)
   {
      RunAllVariants<2>(order, quad_mesh_n, warmup, iterations,
                        cached_derivative, quad_order);
   }
   if (run_3d)
   {
      RunAllVariants<3>(order, hex_mesh_n, warmup, iterations,
                        cached_derivative, quad_order);
   }

   return 0;
}

#else

int main(int, char *[])
{
   mfem::out << "This benchmark requires MFEM_USE_MPI=YES and "
             << "MFEM_USE_ENZYME=YES.\n";
   return MFEM_SKIP_RETURN_VALUE;
}

#endif // MFEM_USE_MPI && MFEM_USE_ENZYME
