// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "bilininteg.hpp"

namespace mfem
{

void PAHcurlSetup2D(const int Q1D,
                    const int NE,
                    const Array<double> &w,
                    const Vector &j,
                    Vector &_coeff,
                    Vector &op);

void PAHcurlSetup3D(const int Q1D,
                    const int NE,
                    const Array<double> &w,
                    const Vector &j,
                    Vector &_coeff,
                    Vector &op);

void PAHcurlMassAssembleDiagonal2D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const Array<double> &_Bo,
                                   const Array<double> &_Bc,
                                   const Vector &_op,
                                   Vector &_diag);

void PAHcurlMassAssembleDiagonal3D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const Array<double> &_Bo,
                                   const Array<double> &_Bc,
                                   const Vector &_op,
                                   Vector &_diag);

void PAHcurlMassApply2D(const int D1D,
                        const int Q1D,
                        const int NE,
                        const Array<double> &_Bo,
                        const Array<double> &_Bc,
                        const Array<double> &_Bot,
                        const Array<double> &_Bct,
                        const Vector &_op,
                        const Vector &_x,
                        Vector &_y);

void PAHcurlMassApply3D(const int D1D,
                        const int Q1D,
                        const int NE,
                        const Array<double> &_Bo,
                        const Array<double> &_Bc,
                        const Array<double> &_Bot,
                        const Array<double> &_Bct,
                        const Vector &_op,
                        const Vector &_x,
                        Vector &_y);

void PAHdivSetup2D(const int Q1D,
                   const int NE,
                   const Array<double> &w,
                   const Vector &j,
                   Vector &_coeff,
                   Vector &op);

void PAHdivSetup3D(const int Q1D,
                   const int NE,
                   const Array<double> &w,
                   const Vector &j,
                   Vector &_coeff,
                   Vector &op);

void PAHcurlH1Apply2D(const int D1D,
                      const int Q1D,
                      const int NE,
                      const Array<double> &_Bc,
                      const Array<double> &_Gc,
                      const Array<double> &_Bot,
                      const Array<double> &_Bct,
                      const Vector &_op,
                      const Vector &_x,
                      Vector &_y);

void PAHcurlH1Apply3D(const int D1D,
                      const int Q1D,
                      const int NE,
                      const Array<double> &_Bc,
                      const Array<double> &_Gc,
                      const Array<double> &_Bot,
                      const Array<double> &_Bct,
                      const Vector &_op,
                      const Vector &_x,
                      Vector &_y);

void PAHdivMassAssembleDiagonal2D(const int D1D,
                                  const int Q1D,
                                  const int NE,
                                  const Array<double> &_Bo,
                                  const Array<double> &_Bc,
                                  const Vector &_op,
                                  Vector &_diag);

void PAHdivMassAssembleDiagonal3D(const int D1D,
                                  const int Q1D,
                                  const int NE,
                                  const Array<double> &_Bo,
                                  const Array<double> &_Bc,
                                  const Vector &_op,
                                  Vector &_diag);

void PAHdivMassApply2D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<double> &_Bo,
                       const Array<double> &_Bc,
                       const Array<double> &_Bot,
                       const Array<double> &_Bct,
                       const Vector &_op,
                       const Vector &_x,
                       Vector &_y);

void PAHdivMassApply3D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<double> &_Bo,
                       const Array<double> &_Bc,
                       const Array<double> &_Bot,
                       const Array<double> &_Bct,
                       const Vector &_op,
                       const Vector &_x,
                       Vector &_y);

void VectorFEMassIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement *fel = fes.GetFE(0);

   const VectorTensorFiniteElement *el =
      dynamic_cast<const VectorTensorFiniteElement*>(fel);
   MFEM_VERIFY(el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*el, *el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   pa_data.SetSize(symmDims * nq * ne, Device::GetMemoryType());

   Vector coeff(ne * nq);
   coeff = 1.0;
   if (Q)
   {
      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);
         for (int p=0; p<nq; ++p)
         {
            coeff[p + (e * nq)] = Q->Eval(*tr, ir->IntPoint(p));
         }
      }
   }

   fetype = el->GetDerivType();

   if (el->GetDerivType() == mfem::FiniteElement::CURL && dim == 3)
   {
      PAHcurlSetup3D(quad1D, ne, ir->GetWeights(), geom->J,
                     coeff, pa_data);
   }
   else if (el->GetDerivType() == mfem::FiniteElement::CURL && dim == 2)
   {
      PAHcurlSetup2D(quad1D, ne, ir->GetWeights(), geom->J,
                     coeff, pa_data);
   }
   else if (el->GetDerivType() == mfem::FiniteElement::DIV && dim == 3)
   {
      PAHdivSetup3D(quad1D, ne, ir->GetWeights(), geom->J,
                    coeff, pa_data);
   }
   else if (el->GetDerivType() == mfem::FiniteElement::DIV && dim == 2)
   {
      PAHdivSetup2D(quad1D, ne, ir->GetWeights(), geom->J,
                    coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

void VectorFEMassIntegrator::AssembleDiagonalPA(Vector& diag)
{
   if (dim == 3)
   {
      if (fetype == mfem::FiniteElement::CURL)
      {
         PAHcurlMassAssembleDiagonal3D(dofs1D, quad1D, ne,
                                       mapsO->B, mapsC->B, pa_data, diag);
      }
      else if (fetype == mfem::FiniteElement::DIV)
      {
         PAHdivMassAssembleDiagonal3D(dofs1D, quad1D, ne,
                                      mapsO->B, mapsC->B, pa_data, diag);
      }
      else
      {
         MFEM_ABORT("Unknown kernel.");
      }
   }
   else
   {
      if (fetype == mfem::FiniteElement::CURL)
      {
         PAHcurlMassAssembleDiagonal2D(dofs1D, quad1D, ne,
                                       mapsO->B, mapsC->B, pa_data, diag);
      }
      else if (fetype == mfem::FiniteElement::DIV)
      {
         PAHdivMassAssembleDiagonal2D(dofs1D, quad1D, ne,
                                      mapsO->B, mapsC->B, pa_data, diag);
      }
      else
      {
         MFEM_ABORT("Unknown kernel.");
      }
   }
}

void VectorFEMassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
   {
      if (fetype == mfem::FiniteElement::CURL)
      {
         PAHcurlMassApply3D(dofs1D, quad1D, ne, mapsO->B, mapsC->B, mapsO->Bt,
                            mapsC->Bt, pa_data, x, y);
      }
      else if (fetype == mfem::FiniteElement::DIV)
      {
         PAHdivMassApply3D(dofs1D, quad1D, ne, mapsO->B, mapsC->B, mapsO->Bt,
                           mapsC->Bt, pa_data, x, y);
      }
      else
      {
         MFEM_ABORT("Unknown kernel.");
      }
   }
   else
   {
      if (fetype == mfem::FiniteElement::CURL)
      {
         PAHcurlMassApply2D(dofs1D, quad1D, ne, mapsO->B, mapsC->B, mapsO->Bt,
                            mapsC->Bt, pa_data, x, y);
      }
      else if (fetype == mfem::FiniteElement::DIV)
      {
         PAHdivMassApply2D(dofs1D, quad1D, ne, mapsO->B, mapsC->B, mapsO->Bt,
                           mapsC->Bt, pa_data, x, y);
      }
      else
      {
         MFEM_ABORT("Unknown kernel.");
      }
   }
}

void MixedVectorGradientIntegrator::AssemblePA(const FiniteElementSpace
                                               &trial_fes,
                                               const FiniteElementSpace &test_fes)
{
   // Assumes tensor-product elements, with a vector test space and H^1 trial space.
   Mesh *mesh = trial_fes.GetMesh();
   const FiniteElement *trial_fel = trial_fes.GetFE(0);
   const FiniteElement *test_fel = test_fes.GetFE(0);

   const NodalTensorFiniteElement *trial_el =
      dynamic_cast<const NodalTensorFiniteElement*>(trial_fel);
   MFEM_VERIFY(trial_el != NULL, "Only NodalTensorFiniteElement is supported!");

   const VectorTensorFiniteElement *test_el =
      dynamic_cast<const VectorTensorFiniteElement*>(test_fel);
   MFEM_VERIFY(test_el != NULL, "Only VectorTensorFiniteElement is supported!");

   const IntegrationRule *ir
      = IntRule ? IntRule : &MassIntegrator::GetRule(*trial_el, *trial_el,
                                                     *mesh->GetElementTransformation(0));
   const int dims = trial_el->GetDim();
   MFEM_VERIFY(dims == 2 || dims == 3, "");

   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");

   MFEM_VERIFY(trial_el->GetOrder() == test_el->GetOrder(), "");

   ne = trial_fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   mapsC = &test_el->GetDofToQuad(*ir, DofToQuad::TENSOR);
   mapsO = &test_el->GetDofToQuadOpen(*ir, DofToQuad::TENSOR);
   dofs1D = mapsC->ndof;
   quad1D = mapsC->nqpt;

   MFEM_VERIFY(dofs1D == mapsO->ndof + 1 && quad1D == mapsO->nqpt, "");

   pa_data.SetSize(symmDims * nq * ne, Device::GetMemoryType());

   Vector coeff(ne * nq);
   coeff = 1.0;
   if (Q)
   {
      for (int e=0; e<ne; ++e)
      {
         ElementTransformation *tr = mesh->GetElementTransformation(e);
         for (int p=0; p<nq; ++p)
         {
            coeff[p + (e * nq)] = Q->Eval(*tr, ir->IntPoint(p));
         }
      }
   }

   // Use the same setup functions as VectorFEMassIntegrator.
   if (test_el->GetDerivType() == mfem::FiniteElement::CURL && dim == 3)
   {
      PAHcurlSetup3D(quad1D, ne, ir->GetWeights(), geom->J,
                     coeff, pa_data);
   }
   else if (test_el->GetDerivType() == mfem::FiniteElement::CURL && dim == 2)
   {
      PAHcurlSetup2D(quad1D, ne, ir->GetWeights(), geom->J,
                     coeff, pa_data);
   }
   else
   {
      MFEM_ABORT("Unknown kernel.");
   }
}

void MixedVectorGradientIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 3)
      PAHcurlH1Apply3D(dofs1D, quad1D, ne, mapsC->B, mapsC->G,
                       mapsO->Bt, mapsC->Bt, pa_data, x, y);
   else if (dim == 2)
      PAHcurlH1Apply2D(dofs1D, quad1D, ne, mapsC->B, mapsC->G,
                       mapsO->Bt, mapsC->Bt, pa_data, x, y);
   else
   {
      MFEM_ABORT("Unsupported dimension!");
   }
}

} // namespace mfem
