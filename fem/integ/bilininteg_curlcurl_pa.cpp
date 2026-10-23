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

#include "../qfunction.hpp"
#include "bilininteg_hcurl_kernels.hpp"
#include "mma/mma.hpp"
#include "mma/hcurl.hpp"

namespace mfem
{

CurlCurlIntegrator::CurlCurlIntegrator() : Q(nullptr), DQ(nullptr), MQ(nullptr)
{
   static Kernels kernels;
}

CurlCurlIntegrator::CurlCurlIntegrator(Coefficient &q,
                                       const IntegrationRule *ir)
   : BilinearFormIntegrator(ir), Q(&q), DQ(nullptr), MQ(nullptr)
{
   static Kernels kernels;
}

CurlCurlIntegrator::CurlCurlIntegrator(DiagonalMatrixCoefficient &dq,
                                       const IntegrationRule *ir)
   : BilinearFormIntegrator(ir), Q(nullptr), DQ(&dq), MQ(nullptr)
{
   static Kernels kernels;
}

CurlCurlIntegrator::CurlCurlIntegrator(MatrixCoefficient &mq,
                                       const IntegrationRule *ir)
   : BilinearFormIntegrator(ir), Q(nullptr), DQ(nullptr), MQ(&mq)
{
   static Kernels kernels;
}

/// \cond DO_NOT_DOCUMENT

CurlCurlIntegrator::Kernels::Kernels()
{
   // Tensor GL q=p+2: (D1D,Q1D)=(p+1,p+2) for p = 1…6
   CurlCurlIntegrator::AddSpecialization<2, 2, 3>();
   CurlCurlIntegrator::AddSpecialization<2, 3, 4>();
   CurlCurlIntegrator::AddSpecialization<2, 4, 5>();
   CurlCurlIntegrator::AddSpecialization<2, 5, 6>();
   CurlCurlIntegrator::AddSpecialization<2, 6, 7>();
   CurlCurlIntegrator::AddSpecialization<2, 7, 8>();

   CurlCurlIntegrator::AddSpecialization<3, 2, 3>();
   CurlCurlIntegrator::AddSpecialization<3, 3, 4>();
   CurlCurlIntegrator::AddSpecialization<3, 4, 5>();
   CurlCurlIntegrator::AddSpecialization<3, 5, 6>();
   CurlCurlIntegrator::AddSpecialization<3, 6, 7>();
   CurlCurlIntegrator::AddSpecialization<3, 7, 8>();
}

CurlCurlIntegrator::ApplyKernelType
CurlCurlIntegrator::ApplyPAKernels::Fallback(int DIM, int, int)
{
   if (DIM == 2) { return internal::PACurlCurlApply2D; }
   else if (DIM == 3)
   {
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         return internal::SmemPACurlCurlApply3D;
      }
      else
      {
         return internal::PACurlCurlApply3D;
      }
   }
   else { MFEM_ABORT(""); }
}

CurlCurlIntegrator::DiagonalKernelType
CurlCurlIntegrator::DiagonalPAKernels::Fallback(int DIM, int, int)
{
   if (DIM == 2)
   {
      return internal::PACurlCurlAssembleDiagonal2D;
   }
   else if (DIM == 3)
   {
      if (Device::Allows(Backend::DEVICE_MASK))
      {
         return internal::SmemPACurlCurlAssembleDiagonal3D;
      }
      else
      {
         return internal::PACurlCurlAssembleDiagonal3D;
      }
   }
   else
   {
      MFEM_ABORT("");
   }
}

/// \endcond DO_NOT_DOCUMENT

void CurlCurlIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   use_tensors_mma_sum = false;
   use_simplex_mma = false;

   if (UsesSimplexMmaHcurl(fes))
   {
      AssembleSimplexMmaPA(fes);
      return;
   }

   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *fel = fes.GetTypicalFE();

   const VectorTensorFiniteElement *el =
      dynamic_cast<const VectorTensorFiniteElement*>(fel);
   MFEM_VERIFY(el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*el, *el,
                                                     *mesh->GetTypicalElementTransformation());

   const int dims = el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::SYMMETRIC);
   if (Q) { coeff.Project(*Q); }
   else if (MQ) { coeff.ProjectTranspose(*MQ); }
   else if (DQ) { coeff.Project(*DQ); }
   else { coeff.SetConstant(1.0); }

   const int coeff_dim = coeff.GetVDim();
   symmetric = (coeff_dim != dim*dim);
   const int sym_dims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int ndata = (dim == 2) ? 1 : (symmetric ? sym_dims : dim*dim);
   pa_data.SetSize(ndata * nq * ne, Device::GetMemoryType());

   if (el->GetDerivType() != mfem::FiniteElement::CURL)
   {
      MFEM_ABORT("Unknown kernel.");
   }

   if (dim == 3)
   {
      internal::PACurlCurlSetup3D(quad1D, coeff_dim, ne, ir->GetWeights(), geom->J,
                                  coeff, pa_data);
   }
   else
   {
      internal::PACurlCurlSetup2D(quad1D, ne, ir->GetWeights(), geom->J, coeff,
                                  pa_data);
   }

   if (UsesTensorMmaHcurl(fes)) { use_tensors_mma_sum = true; }
}

static void ApplySimplexNdFacePairs(const Array<int> &fo, int dim, int p,
                                    int ne, int e, int mode, Vector &v)
{
   if (fo.Size() == 0 || p < 2) { return; }
   const int ntdofs = p * (p - 1);
   if (ntdofs < 2) { return; }
   const int nedges = (dim == 2) ? 3 : 6;
   const int face_base = nedges * p;
   const int npairs = ntdofs / 2;
   const int nfaces = fo.Size() / ne;
   const auto Fo = Reshape(fo.HostRead(), nfaces, ne);
   for (int f = 0; f < nfaces; ++f)
   {
      const int ori = Fo(f, e);
      for (int j = 0; j < npairs; ++j)
      {
         const int idx = face_base + f * ntdofs + 2 * j;
         real_t a = v(idx);
         real_t b = v(idx + 1);
         internal::NdApplyFacePair(mode, ori, a, b);
         v(idx) = a;
         v(idx + 1) = b;
      }
   }
}

void CurlCurlIntegrator::AssembleDiagonalPA(Vector& diag)
{
   if (use_simplex_mma)
   {
      const int nd = simplex_nd;
      MFEM_VERIFY(nd > 0 && ne > 0 && simplex_fes, "");
      MFEM_VERIFY(diag.Size() == nd * ne, "");
      MFEM_VERIFY(pa_data.Size() % ne == 0, "");
      const int stride = pa_data.Size() / ne;
      const real_t *pa = pa_data.HostRead();
      const int p = simplex_fes->GetTypicalFE()->GetOrder();
      const int inv = static_cast<int>(internal::NdDofTransOp::InvPrimal);
      const int dual = static_cast<int>(internal::NdDofTransOp::Dual);
      Vector x(nd), y(nd), slice(stride);
      real_t *d = diag.HostReadWrite();
      for (int e = 0; e < ne; ++e)
      {
         const real_t *src = pa + static_cast<size_t>(stride) * e;
         for (int k = 0; k < stride; ++k) { slice(k) = src[k]; }
         for (int i = 0; i < nd; ++i)
         {
            x = 0.0;
            x(i) = 1.0;
            ApplySimplexNdFacePairs(simplex_nd_fo, dim, p, ne, e, inv, x);
            y = 0.0;
            internal::MmaCurlCurlApplySimplex(
               dim, 1, nd, nq, dofs1D, simplex_curl_dim, symmetric,
               simplex_B, slice, x, y);
            ApplySimplexNdFacePairs(simplex_nd_fo, dim, p, ne, e, dual, y);
            d[i + nd * e] += y(i);
         }
      }
      return;
   }
   DiagonalPAKernels::Run(dim, dofs1D, quad1D, dofs1D, quad1D, symmetric, ne,
                          mapsO->B, mapsC->B, mapsO->G, mapsC->G, pa_data,
                          diag);
}

void CurlCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (use_simplex_mma)
   {
      auto apply = [&](const Vector &xh, Vector &yh, internal::NdDualCtx dual)
      {
         internal::MmaCurlCurlApplySimplex(dim, ne, simplex_nd, nq, dofs1D,
                                           simplex_curl_dim, symmetric,
                                           simplex_B, pa_data, xh, yh, dual);
      };
      if (simplex_fes &&
          simplex_fes->GetTypicalFE()->GetDofTransformation() != nullptr)
      {
         internal::AddMultSimplexNdDual(*simplex_fes, simplex_nd_fo,
                                        simplex_xhat, simplex_yinc,
                                        x, y, apply);
      }
      else
      {
         apply(x, y, internal::NdDualCtx{});
      }
      return;
   }
   if (use_tensors_mma_sum)
   {
      if (dim == 2)
      {
         internal::MmaCurlCurlApplyTensors2D(
            dofs1D, quad1D, symmetric, ne, mapsO->B, mapsC->B, mapsO->Bt,
            mapsC->Bt, mapsC->G, mapsC->Gt, pa_data, x, y, false);
      }
      else
      {
         internal::MmaCurlCurlApplyTensors3D(
            dofs1D, quad1D, symmetric, ne, mapsO->B, mapsC->B, mapsO->Bt,
            mapsC->Bt, mapsC->G, mapsC->Gt, pa_data, x, y, false);
      }
      return;
   }
   ApplyPAKernels::Run(dim, dofs1D, quad1D, dofs1D, quad1D, symmetric, ne,
                       mapsO->B, mapsC->B, mapsO->Bt, mapsC->Bt, mapsC->G,
                       mapsC->Gt, pa_data, x, y, false);
}

void CurlCurlIntegrator::AddAbsMultPA(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(!use_simplex_mma,
               "AbsMultPA not implemented for simplex MMA PA");
   Vector abs_pa_data(pa_data);
   abs_pa_data.Abs();
   auto absO = mapsO->Abs();
   auto absC = mapsC->Abs();

   ApplyPAKernels::Run(dim, dofs1D, quad1D, dofs1D, quad1D, symmetric, ne,
                       absO.B, absC.B, absO.Bt, absC.Bt, absC.G, absC.Gt,
                       abs_pa_data, x, y, true);
}

} // namespace mfem
