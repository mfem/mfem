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

#include "quadinterpolator.hpp"
#include "qinterp/grad.hpp"
#include "qinterp/eval.hpp"
#include "qspace.hpp"
#include "../general/forall.hpp"
#include "../linalg/dtensor.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

namespace internal
{
namespace quadrature_interpolator
{
void InitEvalByNodesKernels();
void InitEvalByVDimKernels();
void InitEvalKernels();
void InitDetKernels();
template <bool P> void InitGradByNodesKernels();
template <bool P> void InitGradByVDimKernels();
void InitTensorEvalHDivKernels();
struct Kernels
{
   Kernels()
   {
      using namespace internal::quadrature_interpolator;

      InitEvalByNodesKernels();
      InitEvalByVDimKernels();
      // Non-phys grad kernels
      InitGradByNodesKernels<false>();
      InitGradByVDimKernels<false>();
      // Phys grad kernels
      InitGradByNodesKernels<true>();
      InitGradByVDimKernels<true>();
      // Determinants
      InitDetKernels();
      // Non-tensor
      InitEvalKernels();
      // Tensor (quad,hex) H(div)
      InitTensorEvalHDivKernels();
   }
};
}
}

QuadratureInterpolator::QuadratureInterpolator(const FiniteElementSpace &fes,
                                               const IntegrationRule &ir):

   fespace(&fes),
   qspace(nullptr),
   IntRule(&ir),
   q_layout(QVectorLayout::byNODES),
   use_tensor_products(UsesTensorBasis(fes))
{
   static internal::quadrature_interpolator::Kernels kernels;

   d_buffer.UseDevice(true);
   if (fespace->GetNE() == 0) { return; }
   MFEM_VERIFY(
      SupportsFESpace(fes),
      "Only elements with MapType VALUE, INTEGRAL, or H_DIV are supported!");
}

QuadratureInterpolator::QuadratureInterpolator(const FiniteElementSpace &fes,
                                               const QuadratureSpace &qs):

   fespace(&fes),
   qspace(&qs),
   IntRule(nullptr),
   q_layout(QVectorLayout::byNODES),
   use_tensor_products(UsesTensorBasis(fes))
{
   d_buffer.UseDevice(true);
   if (fespace->GetNE() == 0) { return; }
   MFEM_VERIFY(
      SupportsFESpace(fes),
      "Only elements with MapType VALUE, INTEGRAL, or H_DIV are supported!");
}

bool QuadratureInterpolator::SupportsFESpace(const FiniteElementSpace &fespace)
{
   const FiniteElement *fe = fespace.GetTypicalFE();
   const Mesh &mesh = *fespace.GetMesh();
   return (fe->GetMapType() == FiniteElement::MapType::VALUE ||
           fe->GetMapType() == FiniteElement::MapType::INTEGRAL ||
           fe->GetMapType() == FiniteElement::MapType::H_DIV) &&
          (!fespace.IsVariableOrder()) && (!mesh.IsMixedMesh());
}

namespace internal
{

namespace quadrature_interpolator
{

// Compute kernel for 1D quadrature interpolation:
// * non-tensor product version,
// * assumes 'e_vec' is using ElementDofOrdering::NATIVE,
// * assumes 'maps.mode == FULL'.
void Eval1D(const int NE, const int vdim, const QVectorLayout q_layout,
            const GeometricFactors *geom, const DofToQuad &maps,
            const Vector &e_vec, Vector &q_val, Vector &q_der, Vector &q_det,
            const int eval_flags)
{
   using QI = QuadratureInterpolator;

   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   MFEM_ASSERT(maps.mode == DofToQuad::FULL, "internal error");
   MFEM_ASSERT(!geom || geom->mesh->SpaceDimension() == 1, "");
   MFEM_VERIFY(vdim == 1 || !(eval_flags & QI::DETERMINANTS), "");
   MFEM_VERIFY(bool(geom) == bool(eval_flags & QI::PHYSICAL_DERIVATIVES),
               "'geom' must be given (non-null) only when evaluating physical"
               " derivatives");
   const auto B = Reshape(maps.B.Read(), nq, nd);
   const auto G = Reshape(maps.G.Read(), nq, nd);
   const auto J = Reshape(geom ? geom->J.Read() : nullptr, nq, NE);
   const auto E = Reshape(e_vec.Read(), nd, vdim, NE);
   auto val = q_layout == QVectorLayout::byNODES ?
              Reshape(q_val.Write(), nq, vdim, NE):
              Reshape(q_val.Write(), vdim, nq, NE);
   auto der = q_layout == QVectorLayout::byNODES ?
              Reshape(q_der.Write(), nq, vdim, NE):
              Reshape(q_der.Write(), vdim, nq, NE);
   auto det = Reshape(q_det.Write(), nq, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < nq; ++q)
      {
         if (eval_flags & (QI::VALUES | QI::PHYSICAL_VALUES))
         {
            for (int c = 0; c < vdim; c++)
            {
               real_t q_val = 0.0;
               for (int d = 0; d < nd; ++d)
               {
                  q_val += B(q,d)*E(d,c,e);
               }
               if (q_layout == QVectorLayout::byVDIM)  { val(c,q,e) = q_val; }
               if (q_layout == QVectorLayout::byNODES) { val(q,c,e) = q_val; }
            }
         }
         if ((eval_flags & QI::DERIVATIVES) ||
             (eval_flags & QI::PHYSICAL_DERIVATIVES) ||
             (eval_flags & QI::DETERMINANTS))
         {
            for (int c = 0; c < vdim; c++)
            {
               real_t q_d = 0.0;
               for (int d = 0; d < nd; ++d)
               {
                  q_d += G(q,d)*E(d,c,e);
               }
               if (eval_flags & QI::PHYSICAL_DERIVATIVES)
               {
                  q_d /= J(q,e);
               }
               if (eval_flags & QI::DERIVATIVES || eval_flags & QI::PHYSICAL_DERIVATIVES)
               {
                  if (q_layout == QVectorLayout::byVDIM) { der(c,q,e) = q_d; }
                  if (q_layout == QVectorLayout::byNODES) { der(q,c,e) = q_d; }
               }
               if (vdim == 1 && (eval_flags & QI::DETERMINANTS))
               {
                  det(q,e) = q_d;
               }
            }
         }
      }
   });
}

// Compute kernel for 1D quadrature interpolation:
// * non-tensor product version,
// * assumes 'e_vec' is using ElementDofOrdering::NATIVE,
// * assumes 'maps.mode == FULL'.
void IntEval1D(const int NE, const int vdim, const QVectorLayout q_layout,
               const GeometricFactors *geom, const DofToQuad &maps,
               const Vector &e_vec, Vector &q_val, Vector &q_der, Vector &q_det,
               const int eval_flags)
{
   // TODO
}

} // namespace quadrature_interpolator

} // namespace internal

void QuadratureInterpolator::Mult(const Vector &e_vec,
                                  unsigned eval_flags,
                                  Vector &q_val,
                                  Vector &q_der,
                                  Vector &q_det) const
{
   using namespace internal::quadrature_interpolator;

   const int ne = fespace->GetNE();
   if (ne == 0) { return; }
   const FiniteElement *fe = fespace->GetFE(0);

   if (fe->GetMapType() == FiniteElement::MapType::H_DIV)
   {
      // q_der == q_div
      return MultHDiv(e_vec, eval_flags, q_val, q_der);
   }

   const int vdim = fespace->GetVDim();
   const int sdim = fespace->GetMesh()->SpaceDimension();
   const bool use_tensor_eval =
      use_tensor_products &&
      dynamic_cast<const TensorBasisElement*>(fe) != nullptr;
   const IntegrationRule *ir =
      IntRule ? IntRule : &qspace->GetElementIntRule(0);
   const DofToQuad::Mode mode =
      use_tensor_eval ? DofToQuad::TENSOR : DofToQuad::FULL;
   const DofToQuad &maps = fe->GetDofToQuad(*ir, mode);
   const int dim = maps.FE->GetDim();
   const int nd = maps.ndof;
   const int nq = maps.nqpt;
   const GeometricFactors *geom = nullptr;
   // evaluated at DOFs for INTEGRAL spaces
   const GeometricFactors *detJgeom = nullptr;
   if (eval_flags & PHYSICAL_DERIVATIVES)
   {
      geom = fespace->GetMesh()->GetGeometricFactors(
                *ir, GeometricFactors::JACOBIANS);
   }
   if (fe->GetMapType() == FiniteElement::MapType::INTEGRAL)
   {
      detJgeom = fespace->GetMesh()->GetGeometricFactors(
                    fe->GetNodes(), GeometricFactors::DETERMINANTS);
   }

   MFEM_ASSERT(!(eval_flags & DETERMINANTS) || dim == vdim ||
               (dim == 2 && vdim == 3) || (dim == 1 && vdim == 2) ||
               (dim == 1 && vdim == 3), "Invalid dimensions for determinants.");
   MFEM_ASSERT(fespace->GetMesh()->GetNumGeometries(
                  fespace->GetMesh()->Dimension()) == 1,
               "mixed meshes are not supported");

   if (use_tensor_eval)
   {
      if (eval_flags & (VALUES | PHYSICAL_VALUES))
      {
         if (fe->GetMapType() == FiniteElement::MapType::INTEGRAL)
         {
            IntTensorEvalKernels::Run(dim, q_layout, vdim, nd, nq, ne,
                                      maps.B.Read(), detJgeom->detJ.Read(),
                                      e_vec.Read(), q_val.Write(), vdim, nd,
                                      nq);
         }
         else
         {
            TensorEvalKernels::Run(dim, q_layout, vdim, nd, nq, ne,
                                   maps.B.Read(), e_vec.Read(), q_val.Write(),
                                   vdim, nd, nq);
         }
      }
      if (eval_flags & (DERIVATIVES | PHYSICAL_DERIVATIVES))
      {
         const bool phys = (eval_flags & PHYSICAL_DERIVATIVES);
         const real_t *J = phys ? geom->J.Read() : nullptr;
         const int s_dim = phys ? sdim : dim;
         if (fe->GetMapType() == FiniteElement::MapType::INTEGRAL)
         {
            IntGradKernels::Run(dim, q_layout, phys, vdim, nd, nq, ne,
                                maps.B.Read(), maps.G.Read(),
                                detJgeom->detJ.Read(), J, e_vec.Read(),
                                q_der.Write(), s_dim, vdim, nd, nq);
         }
         else
         {
            GradKernels::Run(dim, q_layout, phys, vdim, nd, nq, ne,
                             maps.B.Read(), maps.G.Read(), J, e_vec.Read(),
                             q_der.Write(), s_dim, vdim, nd, nq);
         }
      }
      if (eval_flags & DETERMINANTS)
      {
         if (fe->GetMapType() == FiniteElement::MapType::INTEGRAL)
         {
            IntDetKernels::Run(dim, vdim, nd, nq, ne, maps.B.Read(),
                               maps.G.Read(), geom->detJ.Read(), e_vec.Read(),
                               q_det.Write(), nd, nq, &d_buffer);
         }
         else
         {
            DetKernels::Run(dim, vdim, nd, nq, ne, maps.B.Read(), maps.G.Read(),
                            e_vec.Read(), q_det.Write(), nd, nq, &d_buffer);
         }
      }
   }
   else // use_tensor_eval == false
   {
      if (fe->GetMapType() == FiniteElement::MapType::INTEGRAL)
      {
         IntEvalKernels::Run(dim, vdim, maps.ndof, maps.nqpt, ne, vdim,
                             q_layout, detJgeom, geom, maps, e_vec, q_val,
                             q_der, q_det, eval_flags);
      }
      else
      {
         EvalKernels::Run(dim, vdim, maps.ndof, maps.nqpt, ne, vdim, q_layout,
                          geom, maps, e_vec, q_val, q_der, q_det, eval_flags);
      }
   }
}

void QuadratureInterpolator::MultHDiv(const Vector &e_vec,
                                      unsigned eval_flags,
                                      Vector &q_val,
                                      Vector &q_div) const
{
   const int ne = fespace->GetNE();
   if (ne == 0) { return; }
   MFEM_VERIFY(fespace->IsVariableOrder() == false,
               "variable order spaces are not supported yet!");
   const FiniteElement *fe = fespace->GetFE(0);
   MFEM_VERIFY(fe->GetMapType() == FiniteElement::MapType::H_DIV,
               "this method can be used only for H(div) spaces");
   MFEM_VERIFY((eval_flags &
                ~(VALUES | PHYSICAL_VALUES | PHYSICAL_MAGNITUDES)) == 0,
               "only VALUES, PHYSICAL_VALUES, and PHYSICAL_MAGNITUDES"
               " evaluations are implemented!");
   const int dim = fespace->GetMesh()->Dimension();
   const int sdim = fespace->GetMesh()->SpaceDimension();
   MFEM_VERIFY((dim == 2 || dim == 3) && dim == sdim,
               "dim = " << dim << ", sdim = " << sdim
               << " is not supported yet!");
   MFEM_VERIFY(fespace->GetMesh()->GetNumGeometries(dim) <= 1,
               "mixed meshes are not supported yet!");
   const int vdim = fespace->GetVDim();
   MFEM_VERIFY(vdim == 1, "vdim != 1 is not supported yet!");
   auto tfe = dynamic_cast<const VectorTensorFiniteElement *>(fe);
   MFEM_VERIFY(tfe != nullptr, "only quad and hex elements are supported!");
   MFEM_VERIFY(use_tensor_products,
               "non-tensor-product evaluation are not supported yet!");
   const bool use_tensor_eval = use_tensor_products && (tfe != nullptr);
   const IntegrationRule *ir =
      IntRule ? IntRule : &qspace->GetElementIntRule(0);
   const DofToQuad::Mode mode =
      use_tensor_eval ? DofToQuad::TENSOR : DofToQuad::FULL;
   const DofToQuad &maps_c = tfe->GetDofToQuad(*ir, mode);
   const DofToQuad &maps_o = tfe->GetDofToQuadOpen(*ir, mode);
   const int nd = maps_c.ndof;
   const int nq = maps_c.nqpt;
   const GeometricFactors *geom = nullptr;
   if (eval_flags & (PHYSICAL_VALUES | PHYSICAL_MAGNITUDES))
   {
      const int jacobians = GeometricFactors::JACOBIANS;
      geom = fespace->GetMesh()->GetGeometricFactors(*ir, jacobians);
   }
   // Check that at most one of VALUES, PHYSICAL_VALUES, and PHYSICAL_MAGNITUDES
   // is specified:
   MFEM_VERIFY(bool(eval_flags & VALUES) + bool(eval_flags & PHYSICAL_VALUES) +
               bool(eval_flags & PHYSICAL_MAGNITUDES) <= 1,
               "only one of VALUES, PHYSICAL_VALUES, and PHYSICAL_MAGNITUDES"
               " can be requested at a time!");
   const unsigned value_eval_mode =
      eval_flags & (VALUES | PHYSICAL_VALUES | PHYSICAL_MAGNITUDES);
   if (value_eval_mode)
   {
      // For PHYSICAL_MAGNITUDES the QVectorLayouts are the same and we
      // instantiate only QVectorLayout::byNODES:
      const auto q_l = (eval_flags & PHYSICAL_MAGNITUDES) ?
                       QVectorLayout::byNODES : q_layout;
      TensorEvalHDivKernels::Run(
         // dispatch params: dim + the template params of EvalHDiv2D/3D:
         dim, q_l, value_eval_mode, nd, nq,
         // runtime params, see the arguments of EvalHDiv2D/3D:
         ne, maps_o.B.Read(), maps_c.B.Read(), geom ? geom->J.Read() : nullptr,
         e_vec.Read(), q_val.Write(), nd, nq);
   }
   MFEM_CONTRACT_VAR(q_div);
}

void QuadratureInterpolator::MultTranspose(unsigned eval_flags,
                                           const Vector &q_val,
                                           const Vector &q_der,
                                           Vector &e_vec) const
{
   MFEM_CONTRACT_VAR(eval_flags);
   MFEM_CONTRACT_VAR(q_val);
   MFEM_CONTRACT_VAR(q_der);
   MFEM_CONTRACT_VAR(e_vec);
   MFEM_ABORT("this method is not implemented yet");
}

void QuadratureInterpolator::Values(const Vector &e_vec,
                                    Vector &q_val) const
{
   Vector empty;
   Mult(e_vec, VALUES, q_val, empty, empty);
}

void QuadratureInterpolator::PhysValues(const Vector &e_vec,
                                        Vector &q_val) const
{
   Vector empty;
   Mult(e_vec, PHYSICAL_VALUES, q_val, empty, empty);
}

void QuadratureInterpolator::Derivatives(const Vector &e_vec,
                                         Vector &q_der) const
{
   Vector empty;
   Mult(e_vec, DERIVATIVES, empty, q_der, empty);
}

void QuadratureInterpolator::PhysDerivatives(const Vector &e_vec,
                                             Vector &q_der) const
{
   Vector empty;
   Mult(e_vec, PHYSICAL_DERIVATIVES, empty, q_der, empty);
}

void QuadratureInterpolator::Determinants(const Vector &e_vec,
                                          Vector &q_det) const
{
   Vector empty;
   Mult(e_vec, DETERMINANTS, empty, empty, q_det);
}

/// @cond Suppress_Doxygen_warnings

namespace
{

using namespace internal::quadrature_interpolator;

using EvalKernel = QuadratureInterpolator::EvalKernelType;
using IntEvalKernel = QuadratureInterpolator::IntEvalKernelType;
using TensorEvalKernel = QuadratureInterpolator::TensorEvalKernelType;
using IntTensorEvalKernel = QuadratureInterpolator::IntTensorEvalKernelType;
using GradKernel = QuadratureInterpolator::GradKernelType;
using IntGradKernel = QuadratureInterpolator::IntGradKernelType;
using CollocatedGradKernel = QuadratureInterpolator::CollocatedGradKernelType;

template <QVectorLayout Q_LAYOUT>
TensorEvalKernel FallbackTensorEvalKernel(int DIM)
{
   if (DIM == 1) { return Values1D<Q_LAYOUT>; }
   else if (DIM == 2) { return Values2D<Q_LAYOUT>; }
   else if (DIM == 3) { return Values3D<Q_LAYOUT>; }
   else { MFEM_ABORT(""); }
}

template <QVectorLayout Q_LAYOUT>
IntTensorEvalKernel FallbackIntTensorEvalKernel(int DIM)
{
   if (DIM == 1) { return IntValues1D<Q_LAYOUT>; }
   else if (DIM == 2) { return IntValues2D<Q_LAYOUT>; }
   else if (DIM == 3) { return IntValues3D<Q_LAYOUT>; }
   else { MFEM_ABORT(""); }
}

template<QVectorLayout Q_LAYOUT, bool GRAD_PHYS>
GradKernel GetGradKernel(int DIM)
{
   if (DIM == 1) { return Derivatives1D<Q_LAYOUT, GRAD_PHYS>; }
   else if (DIM == 2) { return Derivatives2D<Q_LAYOUT, GRAD_PHYS>; }
   else if (DIM == 3) { return Derivatives3D<Q_LAYOUT, GRAD_PHYS>; }
   else { MFEM_ABORT(""); }
}


template<QVectorLayout Q_LAYOUT>
GradKernel GetGradKernel(int DIM, bool GRAD_PHYS)
{
   if (GRAD_PHYS) { return GetGradKernel<Q_LAYOUT, true>(DIM); }
   else { return GetGradKernel<Q_LAYOUT, false>(DIM); }
}

template <QVectorLayout Q_LAYOUT, bool GRAD_PHYS>
IntGradKernel GetIntGradKernel(int DIM)
{
   if (DIM == 1) { return IntDerivatives1D<Q_LAYOUT, GRAD_PHYS>; }
   else if (DIM == 2) { return IntDerivatives2D<Q_LAYOUT, GRAD_PHYS>; }
   else if (DIM == 3) { return IntDerivatives3D<Q_LAYOUT, GRAD_PHYS>; }
   else { MFEM_ABORT(""); }
}


template<QVectorLayout Q_LAYOUT>
IntGradKernel GetIntGradKernel(int DIM, bool GRAD_PHYS)
{
   if (GRAD_PHYS) { return GetIntGradKernel<Q_LAYOUT, true>(DIM); }
   else { return GetIntGradKernel<Q_LAYOUT, false>(DIM); }
}

template<QVectorLayout Q_LAYOUT, bool GRAD_PHYS>
CollocatedGradKernel GetCollocatedGradKernel(int DIM)
{
   if (DIM == 1) { return CollocatedDerivatives1D<Q_LAYOUT, GRAD_PHYS>; }
   else if (DIM == 2) { return CollocatedDerivatives2D<Q_LAYOUT, GRAD_PHYS>; }
   else if (DIM == 3) { return CollocatedDerivatives3D<Q_LAYOUT, GRAD_PHYS>; }
   else { MFEM_ABORT(""); }
}

template<QVectorLayout Q_LAYOUT>
CollocatedGradKernel GetCollocatedGradKernel(int DIM, bool GRAD_PHYS)
{
   if (GRAD_PHYS) { return GetCollocatedGradKernel<Q_LAYOUT, true>(DIM); }
   else { return GetCollocatedGradKernel<Q_LAYOUT, false>(DIM); }
}
} // namespace

template <int DIM>
EvalKernel GetEvalKernelVDimFallback(int VDIM)
{
   using EvalKernels = QuadratureInterpolator::EvalKernels;
   if (VDIM == 1) { return EvalKernels::Kernel<DIM,1,0,0>(); }
   else if (VDIM == 2) { return EvalKernels::Kernel<DIM,2,0,0>(); }
   else if (VDIM == 3) { return EvalKernels::Kernel<DIM,3,0,0>(); }
   else { MFEM_ABORT(""); }
}

EvalKernel QuadratureInterpolator::EvalKernels::Fallback(
   int DIM, int VDIM, int ND, int NQ)
{
   if (DIM == 1) { return GetEvalKernelVDimFallback<1>(VDIM); }
   else if (DIM == 2) { return GetEvalKernelVDimFallback<2>(VDIM); }
   else if (DIM == 3) { return GetEvalKernelVDimFallback<3>(VDIM); }
   else { MFEM_ABORT(""); }
}

template <int DIM>
IntEvalKernel GetIntEvalKernelVDimFallback(int VDIM)
{
   using IntEvalKernels = QuadratureInterpolator::IntEvalKernels;
   if (VDIM == 1) { return IntEvalKernels::Kernel<DIM,1,0,0>(); }
   else if (VDIM == 2) { return IntEvalKernels::Kernel<DIM,2,0,0>(); }
   else if (VDIM == 3) { return IntEvalKernels::Kernel<DIM,3,0,0>(); }
   else { MFEM_ABORT(""); }
}

IntEvalKernel QuadratureInterpolator::IntEvalKernels::Fallback(int DIM,
                                                               int VDIM, int ND,
                                                               int NQ)
{
   if (DIM == 1) { return GetIntEvalKernelVDimFallback<1>(VDIM); }
   else if (DIM == 2) { return GetIntEvalKernelVDimFallback<2>(VDIM); }
   else if (DIM == 3) { return GetIntEvalKernelVDimFallback<3>(VDIM); }
   else { MFEM_ABORT(""); }
}

TensorEvalKernel QuadratureInterpolator::TensorEvalKernels::Fallback(
   int DIM, QVectorLayout Q_LAYOUT, int, int, int)
{
   if (Q_LAYOUT == QVectorLayout::byNODES) { return FallbackTensorEvalKernel<QVectorLayout::byNODES>(DIM); }
   else { return FallbackTensorEvalKernel<QVectorLayout::byVDIM>(DIM); }
}

IntTensorEvalKernel QuadratureInterpolator::IntTensorEvalKernels::Fallback(
   int DIM, QVectorLayout Q_LAYOUT, int, int, int)
{
   if (Q_LAYOUT == QVectorLayout::byNODES) { return FallbackIntTensorEvalKernel<QVectorLayout::byNODES>(DIM); }
   else { return FallbackIntTensorEvalKernel<QVectorLayout::byVDIM>(DIM); }
}

GradKernel QuadratureInterpolator::GradKernels::Fallback(
   int DIM, QVectorLayout Q_LAYOUT, bool GRAD_PHYS, int, int, int)
{
   if (Q_LAYOUT == QVectorLayout::byNODES) { return GetGradKernel<QVectorLayout::byNODES>(DIM, GRAD_PHYS); }
   else { return GetGradKernel<QVectorLayout::byVDIM>(DIM, GRAD_PHYS); }
}

IntGradKernel QuadratureInterpolator::IntGradKernels::Fallback(
   int DIM, QVectorLayout Q_LAYOUT, bool GRAD_PHYS, int, int, int)
{
   if (Q_LAYOUT == QVectorLayout::byNODES)
   {
      return GetIntGradKernel<QVectorLayout::byNODES>(DIM, GRAD_PHYS);
   }
   else
   {
      return GetIntGradKernel<QVectorLayout::byVDIM>(DIM, GRAD_PHYS);
   }
}

CollocatedGradKernel QuadratureInterpolator::CollocatedGradKernels::Fallback(
   int DIM, QVectorLayout Q_LAYOUT, bool GRAD_PHYS, int, int)
{
   if (Q_LAYOUT == QVectorLayout::byNODES) { return GetCollocatedGradKernel<QVectorLayout::byNODES>(DIM, GRAD_PHYS); }
   else { return GetCollocatedGradKernel<QVectorLayout::byVDIM>(DIM, GRAD_PHYS); }
}

/// @endcond

namespace internal
{
namespace quadrature_interpolator
{
void InitEvalKernels()
{
   // 2D, VDIM = 1
   QuadratureInterpolator::AddEvalSpecializations<2,1,1,1>();
   QuadratureInterpolator::AddEvalSpecializations<2,1,1,4>();
   // Q1
   QuadratureInterpolator::AddEvalSpecializations<2,1,4,4>();
   QuadratureInterpolator::AddEvalSpecializations<2,1,4,9>();
   // Q2
   QuadratureInterpolator::AddEvalSpecializations<2,1,9,9>();
   QuadratureInterpolator::AddEvalSpecializations<2,1,9,16>();
   // Q3
   QuadratureInterpolator::AddEvalSpecializations<2,1,16,16>();
   QuadratureInterpolator::AddEvalSpecializations<2,1,16,25>();
   QuadratureInterpolator::AddEvalSpecializations<2,1,16,36>();
   // Q4
   QuadratureInterpolator::AddEvalSpecializations<2,1,25,25>();
   QuadratureInterpolator::AddEvalSpecializations<2,1,25,36>();
   QuadratureInterpolator::AddEvalSpecializations<2,1,25,49>();
   QuadratureInterpolator::AddEvalSpecializations<2,1,25,64>();

   // 3D, VDIM = 1
   // Q0
   QuadratureInterpolator::AddEvalSpecializations<3,1,1,1>();
   QuadratureInterpolator::AddEvalSpecializations<3,1,1,8>();
   // Q1
   QuadratureInterpolator::AddEvalSpecializations<3,1,8,8>();
   QuadratureInterpolator::AddEvalSpecializations<3,1,8,27>();
   // Q2
   QuadratureInterpolator::AddEvalSpecializations<3,1,27,27>();
   QuadratureInterpolator::AddEvalSpecializations<3,1,27,64>();
   // Q3
   QuadratureInterpolator::AddEvalSpecializations<3,1,64,64>();
   QuadratureInterpolator::AddEvalSpecializations<3,1,64,125>();
   QuadratureInterpolator::AddEvalSpecializations<3,1,64,216>();
   // Q4
   QuadratureInterpolator::AddEvalSpecializations<3,1,125,125>();
   QuadratureInterpolator::AddEvalSpecializations<3,1,125,216>();

   // 2D, VDIM = 3
   // Q0
   QuadratureInterpolator::AddEvalSpecializations<2,3,1,1>();
   QuadratureInterpolator::AddEvalSpecializations<2,3,1,4>();
   // Q1
   QuadratureInterpolator::AddEvalSpecializations<2,3,4,4>();
   QuadratureInterpolator::AddEvalSpecializations<2,3,4,9>();
   // Q2
   QuadratureInterpolator::AddEvalSpecializations<2,3,9,4>();
   QuadratureInterpolator::AddEvalSpecializations<2,3,9,9>();
   QuadratureInterpolator::AddEvalSpecializations<2,3,9,16>();
   QuadratureInterpolator::AddEvalSpecializations<2,3,9,25>();
   // Q3
   QuadratureInterpolator::AddEvalSpecializations<2,3,16,16>();
   QuadratureInterpolator::AddEvalSpecializations<2,3,16,25>();
   QuadratureInterpolator::AddEvalSpecializations<2,3,16,36>();
   // Q4
   QuadratureInterpolator::AddEvalSpecializations<2,3,25,25>();
   QuadratureInterpolator::AddEvalSpecializations<2,3,25,36>();
   QuadratureInterpolator::AddEvalSpecializations<2,3,25,49>();
   QuadratureInterpolator::AddEvalSpecializations<2,3,25,64>();

   // 2D, VDIM = 2
   // Q1
   QuadratureInterpolator::AddEvalSpecializations<2,2,4,4>();
   QuadratureInterpolator::AddEvalSpecializations<2,2,4,9>();
   // Q2
   QuadratureInterpolator::AddEvalSpecializations<2,2,9,9>();
   QuadratureInterpolator::AddEvalSpecializations<2,2,9,16>();
   // Q3
   QuadratureInterpolator::AddEvalSpecializations<2,2,16,16>();
   QuadratureInterpolator::AddEvalSpecializations<2,2,16,25>();
   QuadratureInterpolator::AddEvalSpecializations<2,2,16,36>();
   // Q4
   QuadratureInterpolator::AddEvalSpecializations<2,2,25,25>();
   QuadratureInterpolator::AddEvalSpecializations<2,2,25,36>();
   QuadratureInterpolator::AddEvalSpecializations<2,2,25,49>();
   QuadratureInterpolator::AddEvalSpecializations<2,2,25,64>();

   // 3D, VDIM = 3
   // Q1
   QuadratureInterpolator::AddEvalSpecializations<3,3,8,8>();
   QuadratureInterpolator::AddEvalSpecializations<3,3,8,27>();
   // Q2
   QuadratureInterpolator::AddEvalSpecializations<3,3,27,27>();
   QuadratureInterpolator::AddEvalSpecializations<3,3,27,64>();
   QuadratureInterpolator::AddEvalSpecializations<3,3,27,125>();
   // Q3
   QuadratureInterpolator::AddEvalSpecializations<3,3,64,64>();
   QuadratureInterpolator::AddEvalSpecializations<3,3,64,125>();
   QuadratureInterpolator::AddEvalSpecializations<3,3,64,216>();
   // Q4
   QuadratureInterpolator::AddEvalSpecializations<3,3,125,125>();
   QuadratureInterpolator::AddEvalSpecializations<3,3,125,216>();
}

} // namespace quadrature_Interpolator
} // namespace internal

} // namespace mfem
