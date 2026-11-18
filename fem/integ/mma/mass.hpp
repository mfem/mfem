// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

/** @file mass.hpp
    Mass / VectorMass PA MMA — shared QFn + simplex/tensor Kernel decls.

    form::Mass (y = d * u). VectorMass uses the same QFn with vdim.
*/

#include "../../bilininteg.hpp"
#include "form/form.hpp"
#include "form/simplex.hpp"
#include "form/tensors.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal::mma::form
{

/** Mass density scale at a quadrature point: y = d * u. */
struct Mass
{
   MFEM_HOST_DEVICE void operator()(const eval_t &u, eval_t &y, real_t d) const
   {
      y = d * u;
   }
};

template <>
struct qfn_traits<Mass> : EvalEvalQFnTraits {};

} // namespace internal::mma::form

namespace internal
{

template <int DIM, int T_D1D, int T_Q1D>
inline void MmaMassApplyTensors(
   const int NE,
   const Array<real_t> &b, const Array<real_t> &bt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::Mass;
   using mma::form::ApplyTensor;
   ApplyTensor<Mass, DIM, T_D1D, T_Q1D>(NE, b, bt, d, x, y, d1d, q1d);
}

inline void MmaMassApplyTensors2D(
   const int NE,
   const Array<real_t> &b, const Array<real_t> &bt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MmaMassApplyTensors<2, 0, 0>(NE, b, bt, d, x, y, d1d, q1d);
}

inline void MmaMassApplyTensors3D(
   const int NE,
   const Array<real_t> &b, const Array<real_t> &bt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MmaMassApplyTensors<3, 0, 0>(NE, b, bt, d, x, y, d1d, q1d);
}

template <int DIM, int T_D1D, int T_Q1D>
inline void MmaGemMassApplyTensors(const int NE, const Array<real_t> &P,
                                   const Vector &d, const Vector &x, Vector &y)
{
   using mma::form::Mass;
   using mma::form::gem::ApplyTensorGemEval;
   using mma::form::gem::TensorGemPow;
   if constexpr (T_D1D > 0 && T_Q1D > 0)
   {
      constexpr int nq = TensorGemPow(T_Q1D, DIM);
      constexpr int ndof = TensorGemPow(T_D1D, DIM);
      ApplyTensorGemEval<Mass, nq, ndof>(NE, nq, ndof, P, d, x, y);
   }
   else
   {
      const int nq = d.Size() / NE;
      const int ndof = P.Size() / nq;
      ApplyTensorGemEval<Mass>(NE, nq, ndof, P, d, x, y);
   }
}

template <int DIM, int D1D, int QND>
inline void MmaVectorMassApplySimplex(
   const int NE, const int vdim,
   const Array<real_t> &P,
   const Vector &d, const Vector &x, Vector &y)
{
   using mma::form::Mass;
   using mma::form::ApplySimplex;
   ApplySimplex<Mass, DIM, D1D, QND>(NE, P, d, x, y, vdim);
}

inline void MmaVectorMassApplySimplex2D(
   const int NE, const int vdim,
   const Array<real_t> &P,
   const Vector &d, const Vector &x, Vector &y)
{
   using mma::form::Mass;
   using mma::form::ApplySimplex;
   ApplySimplex<Mass, 2>(NE, P, d, x, y, vdim);
}

inline void MmaVectorMassApplySimplex3D(
   const int NE, const int vdim,
   const Array<real_t> &P,
   const Vector &d, const Vector &x, Vector &y)
{
   using mma::form::Mass;
   using mma::form::ApplySimplex;
   ApplySimplex<Mass, 3>(NE, P, d, x, y, vdim);
}

template <int DIM, int T_D1D, int T_Q1D>
inline void MmaVectorMassApplyTensors(
   const int NE, const int vdim,
   const Array<real_t> &b, const Array<real_t> &bt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::Mass;
   using mma::form::ApplyTensor;
   ApplyTensor<Mass, DIM, T_D1D, T_Q1D>(
      NE, b, bt, d, x, y, d1d, q1d, vdim);
}

inline void MmaVectorMassApplyTensors2D(
   const int NE, const int vdim,
   const Array<real_t> &b, const Array<real_t> &bt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MmaVectorMassApplyTensors<2, 0, 0>(NE, vdim, b, bt, d, x, y, d1d, q1d);
}

inline void MmaVectorMassApplyTensors3D(
   const int NE, const int vdim,
   const Array<real_t> &b, const Array<real_t> &bt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MmaVectorMassApplyTensors<3, 0, 0>(NE, vdim, b, bt, d, x, y, d1d, q1d);
}

} // namespace internal

template<int DIM, int D1D, int QND>
MassIntegrator::ApplySimplexMmaKernelType
MassIntegrator::ApplySimplexMmaGemPAKernels::Kernel()
{
   using internal::mma::form::ApplySimplex;
   using internal::mma::form::Mass;
   if constexpr (DIM == 2)
   {
      return ApplySimplex<Mass, 2, D1D, QND>;
   }
   else if constexpr (DIM == 3)
   {
      return ApplySimplex<Mass, 3, D1D, QND>;
   }
   else
   {
      MFEM_ABORT("Simplex MMA mass only supports DIM 2 or 3");
      return nullptr;
   }
}

inline MassIntegrator::ApplySimplexMmaKernelType
MassIntegrator::ApplySimplexMmaGemPAKernels::Fallback(int dim, int, int)
{
   using internal::mma::form::ApplySimplex;
   using internal::mma::form::Mass;
   using Fn = ApplySimplexMmaKernelType;
   MFEM_VERIFY(dim == 2 || dim == 3,
               "Simplex MMA mass PA is only implemented for triangles/tets");
   if (dim == 2)
   {
      return static_cast<Fn>(ApplySimplex<Mass, 2>);
   }
   return static_cast<Fn>(ApplySimplex<Mass, 3>);
}

template<int DIM, int T_D1D, int T_Q1D>
MassIntegrator::ApplySimplexMmaKernelType
MassIntegrator::ApplyTensorsMmaGemPAKernels::Kernel()
{
   return internal::MmaGemMassApplyTensors<DIM, T_D1D, T_Q1D>;
}

inline MassIntegrator::ApplySimplexMmaKernelType
MassIntegrator::ApplyTensorsMmaGemPAKernels::Fallback(int dim, int, int)
{
   using Fn = ApplySimplexMmaKernelType;
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   if (dim == 2)
   {
      return static_cast<Fn>(internal::MmaGemMassApplyTensors<2, 0, 0>);
   }
   return static_cast<Fn>(internal::MmaGemMassApplyTensors<3, 0, 0>);
}

template <int DIM, int T_D1D, int T_Q1D>
MassIntegrator::ApplyTensorsMmaKernelType
MassIntegrator::ApplyTensorsMmaSumPAKernels::Kernel()
{
   return internal::MmaMassApplyTensors<DIM, T_D1D, T_Q1D>;
}

template <int DIM, int D1D, int QND>
VectorMassIntegrator::ApplySimplexMmaKernelType
VectorMassIntegrator::ApplySimplexMmaPAKernels::Kernel()
{
   return internal::MmaVectorMassApplySimplex<DIM, D1D, QND>;
}

template <int DIM, int T_D1D, int T_Q1D>
VectorMassIntegrator::ApplyTensorsMmaKernelType
VectorMassIntegrator::ApplyTensorsMmaPAKernels::Kernel()
{
   return internal::MmaVectorMassApplyTensors<DIM, T_D1D, T_Q1D>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
