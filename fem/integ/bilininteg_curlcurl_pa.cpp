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

#include "../qfunction.hpp"
#include "bilininteg_hcurl_kernels.hpp"

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
   CurlCurlIntegrator::AddSpecialization<3, 2, 3>();
   CurlCurlIntegrator::AddSpecialization<3, 3, 4>();
   CurlCurlIntegrator::AddSpecialization<3, 4, 5>();
   CurlCurlIntegrator::AddSpecialization<3, 5, 6>();
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
}

void CurlCurlIntegrator::AssembleDiagonalPA(Vector& diag)
{
   DiagonalPAKernels::Run(dim, dofs1D, quad1D, dofs1D, quad1D, symmetric, ne,
                          mapsO->B, mapsC->B, mapsO->G, mapsC->G, pa_data,
                          diag);
}

void CurlCurlIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   ApplyPAKernels::Run(dim, dofs1D, quad1D, dofs1D, quad1D, symmetric, ne,
                       mapsO->B, mapsC->B, mapsO->Bt, mapsC->Bt, mapsC->G,
                       mapsC->Gt, pa_data, x, y, false);
}

void CurlCurlIntegrator::AddAbsMultPA(const Vector &x, Vector &y) const
{
   Vector abs_pa_data(pa_data);
   abs_pa_data.Abs();
   auto absO = mapsO->Abs();
   auto absC = mapsC->Abs();

   ApplyPAKernels::Run(dim, dofs1D, quad1D, dofs1D, quad1D, symmetric, ne,
                       absO.B, absC.B, absO.Bt, absC.Bt, absC.G, absC.Gt,
                       abs_pa_data, x, y, true);
}

} // namespace mfem
