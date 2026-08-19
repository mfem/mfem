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
#pragma once

/** @file diffusion.hpp
    Diffusion / VectorDiffusion PA MMA — shared QFn + simplex/tensor Kernel decls.

    form::Diffusion (y = A * u). VectorDiffusion uses SYM metric + vdim.
*/

#include "../../bilininteg.hpp"
#include "form/form.hpp"
#include "form/simplex.hpp"
#include "form/tensors.hpp"

namespace mfem
{

/// \cond DO_NOT_DOCUMENT

namespace internal
{

/** Simplex diffusion PA setup from mesh nodes (defined in diffusion.cpp). */
void PADiffusionSetupSimplexFromNodes(const int dim,
                                      const int coeffDim,
                                      const int NE,
                                      const int NQ,
                                      const int ND,
                                      const Array<real_t> &w,
                                      const Array<real_t> &g,
                                      const Vector &nodes_e,
                                      const Vector &c,
                                      Vector &d);

} // namespace internal

namespace internal::mma::form
{

/** Diffusion matvec at a quadrature point: y = A * u. */
template <int DIM, bool SYM = true>
struct Diffusion
{
   static constexpr bool symmetric_pa = SYM;

   MFEM_HOST_DEVICE void operator()(const grad_t<DIM> &u, grad_t<DIM> &y,
                                    const tensor<real_t, DIM, DIM> &A) const
   {
      y = A * u;
   }
};

template <int DIM, bool SYM>
struct qfn_traits<Diffusion<DIM, SYM>> : GradGradQFnTraits<DIM, SYM> {};

/** Dispatch on runtime `symmetric` for DiffusionIntegrator registration. */
template <int DIM, int D1D, int QND>
inline void ApplyDiffusionDispatch(const int NE,
                                   const bool symmetric,
                                   const Array<real_t> &g,
                                   const Vector &d,
                                   const Vector &x,
                                   Vector &y)
{
   if (symmetric)
   {
      ApplySimplex<Diffusion<DIM, true>, DIM, D1D, QND>(NE, g, d, x, y);
   }
   else
   {
      ApplySimplex<Diffusion<DIM, false>, DIM, D1D, QND>(NE, g, d, x, y);
   }
}

template <int DIM>
inline void ApplyDiffusionDispatch(const int NE,
                                   const bool symmetric,
                                   const Array<real_t> &g,
                                   const Vector &d,
                                   const Vector &x,
                                   Vector &y)
{
   if (symmetric)
   {
      ApplySimplex<Diffusion<DIM, true>, DIM>(NE, g, d, x, y);
   }
   else
   {
      ApplySimplex<Diffusion<DIM, false>, DIM>(NE, g, d, x, y);
   }
}

} // namespace internal::mma::form

namespace internal
{

/** Runtime SYM dispatch: pick Diffusion SYM and call ApplyTensor. */
template <int DIM, int T_D1D, int T_Q1D>
inline void MmaDiffusionApplyTensors(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::ApplyTensor;
   using mma::form::Diffusion;
   if (symmetric)
   {
      ApplyTensor<Diffusion<DIM, true>, DIM, T_D1D, T_Q1D>(
         NE, b, g, bt, gt, d, x, y, d1d, q1d);
   }
   else
   {
      ApplyTensor<Diffusion<DIM, false>, DIM, T_D1D, T_Q1D>(
         NE, b, g, bt, gt, d, x, y, d1d, q1d);
   }
}

inline void MmaDiffusionApplyTensors2D(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MmaDiffusionApplyTensors<2, 0, 0>(
      NE, symmetric, b, g, bt, gt, d, x, y, d1d, q1d);
}

inline void MmaDiffusionApplyTensors3D(
   const int NE, const bool symmetric,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MmaDiffusionApplyTensors<3, 0, 0>(
      NE, symmetric, b, g, bt, gt, d, x, y, d1d, q1d);
}

template <int DIM, int D1D, int QND>
inline void MmaVectorDiffusionApplySimplex(
   const int NE, const int vdim,
   const Array<real_t> &G,
   const Vector &d, const Vector &x, Vector &y)
{
   using mma::form::ApplySimplex;
   using mma::form::Diffusion;
   ApplySimplex<Diffusion<DIM, true>, DIM, D1D, QND>(NE, G, d, x, y, vdim);
}

inline void MmaVectorDiffusionApplySimplex2D(
   const int NE, const int vdim,
   const Array<real_t> &G,
   const Vector &d, const Vector &x, Vector &y)
{
   using mma::form::ApplySimplex;
   using mma::form::Diffusion;
   ApplySimplex<Diffusion<2, true>, 2>(NE, G, d, x, y, vdim);
}

inline void MmaVectorDiffusionApplySimplex3D(
   const int NE, const int vdim,
   const Array<real_t> &G,
   const Vector &d, const Vector &x, Vector &y)
{
   using mma::form::ApplySimplex;
   using mma::form::Diffusion;
   ApplySimplex<Diffusion<3, true>, 3>(NE, G, d, x, y, vdim);
}

template <int DIM, int T_D1D, int T_Q1D>
inline void MmaVectorDiffusionApplyTensors(
   const int NE, const int vdim,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   using mma::form::ApplyTensor;
   using mma::form::Diffusion;
   ApplyTensor<Diffusion<DIM, true>, DIM, T_D1D, T_Q1D>(
      NE, b, g, bt, gt, d, x, y, d1d, q1d, vdim);
}

inline void MmaVectorDiffusionApplyTensors2D(
   const int NE, const int vdim,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MmaVectorDiffusionApplyTensors<2, 0, 0>(
      NE, vdim, b, g, bt, gt, d, x, y, d1d, q1d);
}

inline void MmaVectorDiffusionApplyTensors3D(
   const int NE, const int vdim,
   const Array<real_t> &b, const Array<real_t> &g,
   const Array<real_t> &bt, const Array<real_t> &gt,
   const Vector &d, const Vector &x, Vector &y,
   const int d1d, const int q1d)
{
   MmaVectorDiffusionApplyTensors<3, 0, 0>(
      NE, vdim, b, g, bt, gt, d, x, y, d1d, q1d);
}

} // namespace internal

template<int DIM, int D1D, int QND>
DiffusionIntegrator::ApplySimplexMmaKernelType
DiffusionIntegrator::ApplySimplexMmaPAKernels::Kernel()
{
   using internal::mma::form::ApplyDiffusionDispatch;
   if constexpr (DIM == 2)
   {
      return ApplyDiffusionDispatch<2, D1D, QND>;
   }
   else if constexpr (DIM == 3)
   {
      return ApplyDiffusionDispatch<3, D1D, QND>;
   }
   else
   {
      MFEM_ABORT("Simplex MMA diffusion only supports DIM 2 or 3");
      return nullptr;
   }
}

inline DiffusionIntegrator::ApplySimplexMmaKernelType
DiffusionIntegrator::ApplySimplexMmaPAKernels::Fallback(int dim, int, int)
{
   using internal::mma::form::ApplyDiffusionDispatch;
   using Fn = ApplySimplexMmaKernelType;
   MFEM_VERIFY(dim == 2 || dim == 3,
               "Simplex MMA diffusion PA is only implemented for triangles/tets");
   if (dim == 2)
   {
      return static_cast<Fn>(ApplyDiffusionDispatch<2>);
   }
   return static_cast<Fn>(ApplyDiffusionDispatch<3>);
}

template <int DIM, int T_D1D, int T_Q1D>
DiffusionIntegrator::ApplyTensorsMmaKernelType
DiffusionIntegrator::ApplyTensorsMmaPAKernels::Kernel()
{
   return internal::MmaDiffusionApplyTensors<DIM, T_D1D, T_Q1D>;
}

template <int DIM, int D1D, int QND>
VectorDiffusionIntegrator::ApplySimplexMmaKernelType
VectorDiffusionIntegrator::ApplySimplexMmaPAKernels::Kernel()
{
   return internal::MmaVectorDiffusionApplySimplex<DIM, D1D, QND>;
}

template <int DIM, int T_D1D, int T_Q1D>
VectorDiffusionIntegrator::ApplyTensorsMmaKernelType
VectorDiffusionIntegrator::ApplyTensorsMmaPAKernels::Kernel()
{
   return internal::MmaVectorDiffusionApplyTensors<DIM, T_D1D, T_Q1D>;
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
