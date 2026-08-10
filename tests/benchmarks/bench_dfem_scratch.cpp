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
//   Local            LocalQFBackend, per-quadrature-point q-function.  The
//                    temporary is a register, there is no scratch at all, and
//                    no Q-vector is ever materialized.  Baseline.
//   Global           GlobalQFBackend, single fused kernel, temporary is a
//                    register inside the forall body.  Reference point for
//                    "global backend, no scratch".
//   GlobalLocalAlloc GlobalQFBackend, split into two kernels, the temporary is
//                    a Vector allocated inside the q-function.  Enzyme
//                    allocates and propagates the shadow buffer itself, so this
//                    is the "let Enzyme own the memory" variant.  Note that the
//                    allocation is Q-sized and happens on every apply, which
//                    dominates its cost on the device.
//   GlobalBank       GlobalQFBackend, split into two kernels, the temporary
//                    lives in a ScratchBank owned by the q-function.  Enzyme
//                    differentiates with respect to the q-function object
//                    itself (enzyme_dup on &qfunc / &qfunc_shadow), so the
//                    tangent is reached through the bank's indirections.
//   GlobalBankGS     GlobalBank plus a global scratch tuple (bool, real_t,
//                    Vector), to isolate the extra cost of the
//                    non-quadrature-point scratch members.
//
// GlobalBank vs GlobalLocalAlloc is the comparison of interest: bank-owned
// versus Enzyme-owned scratch memory.
//
// A scratch buffer handed to the q-function through the signature as a dFEM
// *field* is deliberately not benchmarked, because neither form works today:
//   - as an output field, outputs reach Enzyme as enzyme_dupnoneed, so the
//     primal store of the scratch is eliminated and the tangent loses its
//     s*du term, coming out at exactly 2/3 of the true value;
//   - as an input field it would be enzyme_dup, which is the right shape, but
//     inputs are materialized through Vector::Read() and the q-function
//     parameter has to be const, so the scratch cannot be written.
// Both need a library change to become expressible.
//
// Only the Derivative rows touch shadow memory; the Forward rows are there to
// separate scratch cost from allocation cost.  Every row reports rel_err
// against the analytic result, so a timing win that comes from a dropped
// derivative term is visible rather than silent.
//
// Usage:
//   ./bench_dfem_scratch --benchmark_context=device=cuda
//   ./bench_dfem_scratch --benchmark_context=device=cuda --benchmark_context=q=15
//   ./bench_dfem_scratch --benchmark_context=cached=1
//   ./bench_dfem_scratch --benchmark_context=lvector=1
//   ./bench_dfem_scratch --benchmark_filter='Derivative/4/32'
//
// Run on a single rank unless you pass a fixed iteration count.  Google
// Benchmark picks the iteration count per process from its own timings, the
// applies below are MPI-collective, and mismatched counts across ranks will
// hang.  For a multi-rank run force the count to agree:
//   mpirun -np 4 ./bench_dfem_scratch --benchmark_min_time=30x

#include "bench.hpp" // IWYU pragma: keep

#if defined(MFEM_USE_BENCHMARK) && defined(MFEM_USE_MPI) && \
    defined(MFEM_USE_ENZYME)

#include "fem/qinterp/det.hpp" // IWYU pragma: keep
#include "fem/qinterp/grad.hpp" // IWYU pragma: keep
#include "fem/qinterp/grad_transpose.hpp" // IWYU pragma: keep
#include "fem/quadinterpolator.hpp" // IWYU pragma: keep
#include "fem/integ/lininteg_domain_kernels.hpp" // IWYU pragma: keep

#include "fem/dfem/doperator.hpp"
#include "fem/dfem/backends/global_qf/prelude.hpp"
#include "fem/dfem/backends/local_qf/prelude.hpp"
#include "fem/dfem/backends/scratch_bank.hpp"
#include "linalg/tensor_arrays.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

using namespace mfem;
using namespace mfem::future;

// With Enzyme the differentiated scalar type is just real_t; the scratch bank
// shadow mechanism is Enzyme-specific, hence the MFEM_USE_ENZYME guard above.
using dscalar_t = real_t;

static constexpr int U = 0, COEF = 1, COORDS = 2;

// Above this relative error a variant is not computing the true tangent, and
// its timing is not comparable with the others.
constexpr real_t wrong_tangent_tol = 1e-10;

/// Versions //////////////////////////////////////////////////////////////////

void info()
{
   mfem::out << "\x1b[33m";
   mfem::out << "version  0: Local            no scratch, no Q-vector"
             << std::endl;
   mfem::out << "version  1: Global           no scratch" << std::endl;
   mfem::out << "version  2: GlobalLocalAlloc Enzyme-owned scratch, per-apply "
             << "allocation" << std::endl;
   mfem::out << "version  3: GlobalBank       ScratchBank-owned scratch"
             << std::endl;
   mfem::out << "version  4: GlobalBankGS     ScratchBank + global scratch tuple"
             << std::endl;
   mfem::out << "\x1b[m" << std::endl;
}

enum class Variant
{
   Local,
   Global,
   GlobalLocalAlloc,
   GlobalBank,
   GlobalBankGS,
};

constexpr int version_int(Variant v) noexcept
{
   return static_cast<int>(static_cast<std::underlying_type_t<Variant>>(v));
}

/// Which action is timed.  Forward does not touch shadow memory, so the
/// difference between the two isolates the cost of the tangent.
enum class Phase { Forward, Derivative };

// Custom benchmark arguments generator ///////////////////////////////////////
// Same (p, side) convention as bench_dfem: the second argument is the target
// number of dofs per direction, not the number of elements, so a row named
// /4/24 covers the same problem in both benchmarks.
static void CustomArguments(bmi::Benchmark *b) noexcept
{
   // Smaller ceiling than bench_dfem: the Q-vector of the global variants is
   // roughly 25 * #qp * 8 bytes on the derivative rows, so the sweep cannot
   // run as far.
   constexpr int MAX_NDOFS = 2 * 1024 * (mfem_use_gpu ? 1024 : 8);

   constexpr auto ndofs = [](int n) constexpr noexcept -> int
   {
      return (n + 1) * (n + 1) * (n + 1);
   };

   constexpr auto inc = [](int n) constexpr noexcept -> int
   {
      return n < 80 ? 8 : n < 160 ? 16 : 32;
   };

   for (auto p : { 4, 3, 2 })
   {
      // BakeOff asserts side >= p.
      for (int n = 8; ndofs(n) <= MAX_NDOFS; n += inc(n))
      {
         b->Args({ p, n });
      }
   }
}

// Register kernel specializations used in the benchmarks /////////////////////
// Without these the shared E->Q stage falls back to generic kernels, which
// penalizes both backends and makes the Local baseline unrepresentative.
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
   LIN::Specialization<3, 6, 6>::Add();
   LIN::Specialization<3, 7, 7>::Add();
   LIN::Specialization<3, 8, 8>::Add();
#endif // MFEM_DEBUG
}

/// Globals ///////////////////////////////////////////////////////////////////

// Integration rule order, overridden with --benchmark_context=q=<order>.  A
// non-positive value selects the default 2*p rule.  Note this is the rule's
// exactness order, not its point count: a 1D Gauss-Legendre rule of order q
// has q/2+1 points, so q=11 gives 6 points per dimension and q=15 gives 8.
int quadrature_order = 0;

// Selects the cached DerivativeSetup/DerivativeApply path over the matrix-free
// one, with --benchmark_context=cached=1.  This is the PA/MF distinction that
// bench_dfem reports as separate versions, where PA is consistently the faster
// route; it changes the Derivative rows only.
bool cached_derivative = false;

// Drops the T->L and L->T operators from the timed region, with
// --benchmark_context=lvector=1, matching what bench_dfem does through
// SetMultLevel(MultLevel::LVECTOR).
//
// Off by default, because it applies to the Forward rows only:
// DerivativeOperator::Mult calls the three-argument prolongation(), whose
// is_lvector parameter defaults to false, so the derivative always runs T->T
// and cannot be changed without touching the library.  Leaving it off keeps
// Forward and Derivative measured at the same level, so their ratio is the
// cost of differentiating.  Turn it on to compare a Forward row directly
// against bench_dfem, whose dFEM rows are all LVECTOR.
//
// Either way, comparing two variants within one phase is unaffected: the
// prolongation is identical across variants and cancels in the difference.
bool use_lvector = false;

/// Q-functions ///////////////////////////////////////////////////////////////

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
template <int DIM> struct VariantTraits<DIM, Variant::GlobalLocalAlloc>
{
   using qfunc_t = CubicGlobalScratchLocalQF<DIM>;
   using backend_t = GlobalQFBackend;
};
template <int DIM> struct VariantTraits<DIM, Variant::GlobalBank>
{
   using qfunc_t = CubicGlobalScratchBankQF<DIM>;
   using backend_t = GlobalQFBackend;
};
template <int DIM> struct VariantTraits<DIM, Variant::GlobalBankGS>
{
   using qfunc_t = CubicGlobalScratchBankGlobalQF<DIM>;
   using backend_t = GlobalQFBackend;
};

/// Utilities /////////////////////////////////////////////////////////////////

template <int DIM>
Mesh MakeTensorMesh(int nx, int ny, int nz)
{
   if constexpr (DIM == 2)
   {
      return Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, true,
                                   1.0, 1.0);
   }
   else
   {
      return Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON,
                                   1.0, 1.0, 1.0);
   }
}

/// @brief Elements per direction that best approach @a side dofs per
/// direction, copied from bench_dfem's BakeOff so that a /p/side row means
/// the same problem in both benchmarks.  Note the result is not always cubic:
/// nx and ny are bumped by one when that gets closer to side^3 total dofs.
struct MeshDims
{
   int n, nx, ny, nz;
   MeshDims(int p, int side):
      // BakeOff asserts side >= p; clamp instead so a stray small side gives
      // a one-element mesh rather than a zero-element one.
      n(std::max(side / p, 1)),
      nx(n + (p * (n + 1) * p * n * p * n < side * side * side ? 1 : 0)),
      ny(n + (p * (n + 1) * p * (n + 1) * p * n < side * side * side ? 1 : 0)),
      nz(n) {}
};

real_t GlobalNormlinf(const Vector &v, MPI_Comm comm)
{
   const real_t local = v.Normlinf();
   real_t global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 comm);
   return global;
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

// State u and coefficient c used by every variant.
inline real_t StateFunction(const Vector &x) { return 1.0 + x(0) + 0.25 * x(1); }
inline real_t CoefFunction(const Vector &x) { return 0.5 + x(0) + 0.125 * x(1); }

/// @brief Register the local backend's kernels for every q1d in @a Q1Ds.
///
/// q1d is a runtime value, so the set has to cover whatever the sweep can
/// produce: the default 2*p+3 rule gives q1d = p+2, and an explicit q=11 or
/// q=15 gives 6 or 8.  Derivatives<U> means the derivative kernels are needed
/// too, hence the index_sequence naming field U as the differentiated input.
template <int DIM, typename QT, typename IT, typename OT, int... Q1Ds>
void AddLocalSpecializationSet(std::integer_sequence<int, Q1Ds...>)
{
   (AddLocalSpecializations<DIM, Q1Ds, QT, IT, OT,
                            std::index_sequence<size_t(U)>>(), ...);
}

/// Benchmark case ////////////////////////////////////////////////////////////

template <int DIM, Variant V, Phase P>
struct BKS
{
   static constexpr Variant version = V;

   using qfunc_t = typename VariantTraits<DIM, V>::qfunc_t;
   using backend_t = typename VariantTraits<DIM, V>::backend_t;
   static constexpr bool uses_bank = (V == Variant::GlobalBank ||
                                      V == Variant::GlobalBankGS);

   // Only the forward action honors MultLevel; see use_lvector.
   const bool lvec = use_lvector && (P == Phase::Forward);

   const int p, side;
   MeshDims dims;
   Mesh smesh;
   ParMesh pmesh;
   ParGridFunction *nodes;
   H1_FECollection fec;
   ParFiniteElementSpace pfes;
   const IntegrationRule *ir;
   int nqp_elem, q1d, nq_local;
   QuadratureSpace qspace;
   VectorQuadratureSpace coef_qspace;
   QuadratureFunction coef;
   Array<int> all_domain_attr;
   std::vector<FieldDescriptor> in_fds, out_fds;
   DifferentiableOperator dop;
   qfunc_t qf;
   Vector global_scratch_vec;
   Vector xtvec, ytvec, dxtvec, dytvec, nodestv;
   std::unique_ptr<MultiVector> X, Y, DY;
   // Held as the concrete type, not Operator: the templated
   // Mult(const Vector &, vector_t &) that takes a MultiVector is not virtual.
   std::shared_ptr<DerivativeOperator> ddop;

   long long dofs, qpts;
   real_t rel_err;
   double mdofs, mqpts;

   BKS(int p, int side):
      p(p), side(side),
      dims(p, side),
      smesh(MakeTensorMesh<DIM>(dims.nx, dims.ny, dims.nz)),
      pmesh(MPI_COMM_WORLD, smesh),
      nodes((pmesh.EnsureNodes(),
             static_cast<ParGridFunction *>(pmesh.GetNodes()))),
      fec(p, DIM),
      pfes(&pmesh, &fec),
      // Default rule order 2*p+3, the same Gauss-Legendre convention as
      // bench_dfem's BakeOff, so q1d matches without passing q= explicitly.
      ir(&IntRules.Get(pmesh.GetTypicalElementGeometry(),
                       quadrature_order > 0 ? quadrature_order : 2 * p + 3)),
      nqp_elem(ir->GetNPoints()),
      q1d(IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints()),
      nq_local(pmesh.GetNE() * nqp_elem),
      qspace(pmesh, *ir),
      coef_qspace(qspace, 1),
      coef(coef_qspace),
      in_fds({ { U, &pfes },
               { COEF, &coef_qspace },
               { COORDS, nodes->ParFESpace() } }),
      out_fds({ FieldDescriptor{ U, &pfes } }),
      dop(in_fds, out_fds, pmesh),
      // LVECTOR mode hands the operator L-dof vectors directly.
      xtvec(lvec ? pfes.GetVSize() : pfes.GetTrueVSize()),
      ytvec(lvec ? pfes.GetVSize() : pfes.GetTrueVSize()),
      dxtvec(pfes.GetTrueVSize()), dytvec(pfes.GetTrueVSize()),
      mdofs(0.0), mqpts(0.0)
   {
      smesh.Clear();

      if (lvec) { dop.SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR); }

      coef.UseDevice(true);
      FunctionCoefficient coeff_fc(CoefFunction);
      coeff_fc.Project(coef);

      if (pmesh.attributes.Size() > 0)
      {
         all_domain_attr.SetSize(pmesh.attributes.Max());
         all_domain_attr = 1;
      }

      if constexpr (uses_bank)
      {
         // One scalar scratch value per quadrature point of the whole domain.
         qf.SetScratch(nq_local, {1});
      }
      if constexpr (V == Variant::GlobalBankGS)
      {
         global_scratch_vec.SetSize(1);
         global_scratch_vec.UseDevice(true);
         global_scratch_vec = 1.0;
         qf.SetGlobalScratch(make_tuple(true, real_t(1.0), global_scratch_vec));
      }

      using inputs_t =
         Inputs<Value<U>, Identity<COEF>, Gradient<COORDS>, Weight>;
      using outputs_t = Outputs<Value<U>>;

      if constexpr (V == Variant::Local)
      {
         // The local backend dispatches its action, derivative action, setup
         // and apply kernels on (DIM, q1d), falling back to a generic
         // runtime-sized kernel when the pair is unregistered.  Without this
         // the Local rows run unspecialized while the global variants do not,
         // which biases the whole vs-Local comparison.  bench_dfem does the
         // same through AddLocalQFActionSpecializations.
         AddLocalSpecializationSet<DIM, qfunc_t, inputs_t, outputs_t>(
            std::integer_sequence<int, 3, 4, 5, 6, 7, 8> {});
      }

      dop.AddDomainIntegrator<backend_t>(
         qf, inputs_t {}, outputs_t {},
         *ir, all_domain_attr, Derivatives<U> {});

      xtvec.UseDevice(true);
      ytvec.UseDevice(true);
      dxtvec.UseDevice(true);
      dytvec.UseDevice(true);

      ParGridFunction x_gf(&pfes);
      FunctionCoefficient input_coeff(StateFunction);
      x_gf.ProjectCoefficient(input_coeff);
      // A ParGridFunction is already an L-vector; GetTrueDofs restricts it.
      if (lvec) { xtvec = x_gf; }
      else      { x_gf.GetTrueDofs(xtvec); }
      dxtvec = 1.0;
      ytvec = 0.0;
      dytvec = 0.0;

      if (lvec) { nodestv = *nodes; }
      else      { nodes->GetTrueDofs(nodestv); }

      // Move the fixed inputs to the device before timing so that repeated
      // applies do not pay host-to-device copies.
      xtvec.Read();
      dxtvec.Read();
      nodestv.Read();
      coef.Read();

      X = std::make_unique<MultiVector>(MultiVector{xtvec, coef, nodestv});
      Y = std::make_unique<MultiVector>(MultiVector{ytvec});
      DY = std::make_unique<MultiVector>(MultiVector{dytvec});

      ddop = dop.GetDerivative(U, *X, cached_derivative);

      dofs = pfes.GlobalTrueVSize();
      {
         long long send = nq_local, recv = 0;
         MPI_Allreduce(&send, &recv, 1, MPI_LONG_LONG, MPI_SUM, pmesh.GetComm());
         qpts = recv;
      }

      // One untimed apply, both to warm the lazily built internals and to
      // produce the vector checked against the analytic reference below.
      benchmark();
      MFEM_DEVICE_SYNC;
      ComputeError();
      mdofs = mqpts = 0.0;
   }

   // Analytic references: int phi_i c u^3 dx and, for the direction du = 1,
   // int phi_i 3 c u^2 dx, both integrated with the same quadrature rule, so
   // that the quadrature error cancels and rel_err sees only the derivative
   // propagation.
   void ComputeError()
   {
      FunctionCoefficient ref_coeff([](const Vector &x)
      {
         const real_t u = StateFunction(x);
         return (P == Phase::Forward) ? CoefFunction(x) * u * u * u
                : 3.0 * CoefFunction(x) * u * u;
      });

      ParLinearForm lf(&pfes);
      lf.AddDomainIntegrator(new DomainLFIntegrator(ref_coeff, ir));
      lf.Assemble();

      // The assembled linear form is an L-vector, so it is the reference as-is
      // in LVECTOR mode and needs P^T applied otherwise.
      Vector ref;
      if (lvec) { ref = lf; }
      else
      {
         ref.SetSize(pfes.GetTrueVSize());
         pfes.GetProlongationMatrix()->MultTranspose(lf, ref);
      }

      rel_err = RelativeError((P == Phase::Forward) ? ytvec : dytvec, ref,
                              pmesh.GetComm());
   }

   void benchmark()
   {
      if constexpr (P == Phase::Forward) { dop.Mult(*X, *Y); }
      else { ddop->Mult(dxtvec, *DY); }
      MFEM_DEVICE_SYNC;
      mdofs += 1e-6 * static_cast<double>(dofs);
      mqpts += 1e-6 * static_cast<double>(qpts);
   }
};

/// Benchmarks Registration ///////////////////////////////////////////////////

template <typename T>
static void Benchmark(bm::State &state) noexcept
{
   T run(static_cast<int>(state.range(0)), static_cast<int>(state.range(1)));
   while (state.KeepRunning()) { run.benchmark(); }

   state.counters["Dofs"] = bm::Counter(static_cast<double>(run.dofs));
   state.counters["MDof/s"] = bm::Counter(run.mdofs, bm::Counter::kIsRate);
   state.counters["Qpts"] = bm::Counter(static_cast<double>(run.qpts));
   state.counters["MQpt/s"] = bm::Counter(run.mqpts, bm::Counter::kIsRate);
   state.counters["p"] = bm::Counter(state.range(0));
   state.counters["q1d"] = bm::Counter(run.q1d);
   state.counters["rel_err"] = bm::Counter(run.rel_err);
   state.counters["cached"] = bm::Counter(cached_derivative ? 1 : 0);
   state.counters["lvec"] = bm::Counter(run.lvec ? 1 : 0);
   state.counters["version"] = bm::Counter(version_int(T::version));

   // A variant that does not reproduce the analytic result is not doing the
   // same work as the others, so its timing must not be compared with them.
   if (!(run.rel_err < wrong_tangent_tol))
   {
      state.SkipWithError("WRONG TANGENT, timing not comparable");
   }
}

#define REGISTER(DIM, VAR, PH) \
   BENCHMARK_TEMPLATE(Benchmark, BKS<DIM, Variant::VAR, Phase::PH>) \
      ->Name("BKS_" #VAR #PH)->Apply(CustomArguments)->Unit(bm::kMillisecond)

/// Forward: no shadow memory is touched, so this separates the cost of the
/// scratch itself from the cost of differentiating through it.
REGISTER(3, Local, Forward);
REGISTER(3, Global, Forward);
REGISTER(3, GlobalLocalAlloc, Forward);
REGISTER(3, GlobalBank, Forward);
REGISTER(3, GlobalBankGS, Forward);

/// Derivative: the comparison of interest.
REGISTER(3, Local, Derivative);
REGISTER(3, Global, Derivative);
REGISTER(3, GlobalLocalAlloc, Derivative);
REGISTER(3, GlobalBank, Derivative);
REGISTER(3, GlobalBankGS, Derivative);

/// main //////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   AddKernelSpecializations();
   if (Mpi::Root()) { info(); }

   // Context options, cpu and the default 2*p rule unless overridden:
   //   --benchmark_context=device=cuda  --benchmark_context=q=15
   std::string device_config = "cpu";
   if (const auto *ctx = bmi::GetGlobalContext())
   {
      if (const auto device = ctx->find("device"); device != ctx->end())
      {
         device_config = device->second;
      }
      if (const auto q = ctx->find("q"); q != ctx->end())
      {
         quadrature_order = std::stoi(q->second);
      }
      if (const auto c = ctx->find("cached"); c != ctx->end())
      {
         cached_derivative = (std::stoi(c->second) != 0);
      }
      if (const auto l = ctx->find("lvector"); l != ctx->end())
      {
         use_lvector = (std::stoi(l->second) != 0);
      }
   }
   Device device(device_config.c_str());
   if (Mpi::Root()) { device.Print(); }

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return EXIT_FAILURE; }

   // Every rank runs the benchmarks, since the applies are MPI-collective,
   // but only the root reports.
   if (Mpi::Root())
   {
      bm::RunSpecifiedBenchmarks(&CR);
   }
   else
   {
      bm::BenchmarkReporter *quiet = new bm::ConsoleReporter();
      quiet->SetOutputStream(&mfem::err);
      quiet->SetErrorStream(&mfem::err);
      bm::RunSpecifiedBenchmarks(quiet);
      delete quiet;
   }

   return EXIT_SUCCESS;
}

#else

int main(int, char *[])
{
   mfem::out << "This benchmark requires MFEM_USE_BENCHMARK=YES, "
             << "MFEM_USE_MPI=YES and MFEM_USE_ENZYME=YES.\n";
   return MFEM_SKIP_RETURN_VALUE;
}

#endif // MFEM_USE_BENCHMARK && MFEM_USE_MPI && MFEM_USE_ENZYME
