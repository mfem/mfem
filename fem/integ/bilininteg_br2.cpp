// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../bilininteg.hpp"
#include "../pfespace.hpp"
#include <algorithm>

namespace mfem
{

DGDiffusionBR2Integrator::DGDiffusionBR2Integrator(
   FiniteElementSpace &fes, double e) : eta(e), Q(NULL)
{
   PrecomputeMassInverse(fes);
}

DGDiffusionBR2Integrator::DGDiffusionBR2Integrator(
   FiniteElementSpace &fes, Coefficient &Q_, double e) : eta(e), Q(&Q_)
{
   PrecomputeMassInverse(fes);
}

DGDiffusionBR2Integrator::DGDiffusionBR2Integrator(
   FiniteElementSpace *fes, double e) : eta(e), Q(NULL)
{
   PrecomputeMassInverse(*fes);
}

void DGDiffusionBR2Integrator::PrecomputeMassInverse(FiniteElementSpace &fes)
{
   MFEM_VERIFY(fes.IsDGSpace(),
               "The BR2 integrator is only defined for DG spaces.");
   // Precompute local mass matrix inverses needed for the lifting operators
   // First compute offsets and total size needed (e.g. for mixed meshes or
   // p-refinement)
   int nel = fes.GetNE();
   Minv_offsets.SetSize(nel+1);
   ipiv_offsets.SetSize(nel+1);
   ipiv_offsets[0] = 0;
   Minv_offsets[0] = 0;
   for (int i=0; i<nel; ++i)
   {
      int dof = fes.GetFE(i)->GetDof();
      ipiv_offsets[i+1] = ipiv_offsets[i] + dof;
      Minv_offsets[i+1] = Minv_offsets[i] + dof*dof;
   }

#ifdef MFEM_USE_MPI
   // When running in parallel, we also need to compute the local mass matrices
   // of face neighbor elements
   ParFiniteElementSpace *pfes = dynamic_cast<ParFiniteElementSpace *>(&fes);
   if (pfes != NULL)
   {
      ParMesh *pmesh = pfes->GetParMesh();
      pfes->ExchangeFaceNbrData();
      int nel_nbr = pmesh->GetNFaceNeighborElements();
      Minv_offsets.SetSize(nel+nel_nbr+1);
      ipiv_offsets.SetSize(nel+nel_nbr+1);
      for (int i=0; i<nel_nbr; ++i)
      {
         int dof = pfes->GetFaceNbrFE(i)->GetDof();
         ipiv_offsets[nel+i+1] = ipiv_offsets[nel+i] + dof;
         Minv_offsets[nel+i+1] = Minv_offsets[nel+i] + dof*dof;
      }
      nel += nel_nbr;
   }
#endif
   // The final "offset" is the total size of all the blocks
   Minv.SetSize(Minv_offsets[nel]);
   ipiv.SetSize(ipiv_offsets[nel]);

   // Assemble the local mass matrices and compute LU factorization
   MassIntegrator mi;
   for (int i=0; i<nel; ++i)
   {
      const FiniteElement *fe = NULL;
      ElementTransformation *tr = NULL;
      if (i < fes.GetNE())
      {
         fe = fes.GetFE(i);
         tr = fes.GetElementTransformation(i);
      }
      else
      {
#ifdef MFEM_USE_MPI
         int inbr = i - fes.GetNE();
         fe = pfes->GetFaceNbrFE(inbr);
         tr = pfes->GetParMesh()->GetFaceNbrElementTransformation(inbr);
#endif
      }
      int dof = fe->GetDof();
      double *Minv_el = &Minv[Minv_offsets[i]];
      int *ipiv_el = &ipiv[ipiv_offsets[i]];
      DenseMatrix Me(Minv_el, dof, dof);
      mi.AssembleElementMatrix(*fe, *tr, Me);
      LUFactors lu(Minv_el, ipiv_el);
      lu.Factor(dof);
   }
}

void DGDiffusionBR2Integrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int ndof1 = el1.GetDof();
   shape1.SetSize(ndof1);

   R11.SetSize(ndof1, ndof1);
   R11 = 0.0;
   LUFactors M1inv(&Minv[Minv_offsets[Trans.Elem1No]],
                   &ipiv[ipiv_offsets[Trans.Elem1No]]);
   LUFactors M2inv;

   double factor = Geometries.NumBdr(Trans.Elem1->GetGeometryType());

   int ndof2;
   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
      shape2.SetSize(ndof2);
      R12.SetSize(ndof1, ndof2);
      R21.SetSize(ndof2, ndof1);
      R22.SetSize(ndof2, ndof2);
      M2inv.data = &Minv[Minv_offsets[Trans.Elem2No]];
      M2inv.ipiv = &ipiv[ipiv_offsets[Trans.Elem2No]];

      R12 = 0.0;
      R21 = 0.0;
      R22 = 0.0;

      Geometry::Type geom2 = Trans.Elem2->GetGeometryType();
      factor = std::max(factor, double(Geometries.NumBdr(geom2)));
   }
   else
   {
      ndof2 = 0;
   }

   int ndofs = ndof1 + ndof2;

   Re.SetSize(ndofs, ndofs);
   MinvRe.SetSize(ndofs, ndofs);

   elmat.SetSize(ndofs);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      if (ndof2)
      {
         order = 2*std::max(el1.GetOrder(), el2.GetOrder());
      }
      else
      {
         order = 2*el1.GetOrder();
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      Trans.SetAllIntPoints(&ip);

      const IntegrationPoint &eip1 = Trans.Elem1->GetIntPoint();
      el1.CalcShape(eip1, shape1);
      double q = Q ? Q->Eval(*Trans.Elem1, eip1) : 1.0;
      if (ndof2)
      {
         const IntegrationPoint &eip2 = Trans.Elem2->GetIntPoint();
         el2.CalcShape(eip2, shape2);
         // Set coefficient value q to the average of the values on either side
         if (Q) { q = 0.5*(q + Q->Eval(*Trans.Elem2, eip2)); }
      }
      // Take sqrt here because
      //    eta (r_e([u]), r_e([v])) = (sqrt(eta) r_e([u]), sqrt(eta) r_e([v]))
      double w = sqrt((factor + 1)*eta*q)*ip.weight*Trans.Face->Weight();
      // r_e is defined by, (r_e([u]), tau) = <[u], {tau}>, so we pick up a
      // factor of 0.5 on interior faces from the average term.
      if (ndof2) { w *= 0.5; }

      for (int i = 0; i < ndof1; i++)
      {
         const double wsi = w*shape1(i);
         for (int j = 0; j < ndof1; j++)
         {
            R11(i, j) += wsi*shape1(j);
         }
      }

      if (ndof2)
      {
         for (int i = 0; i < ndof2; i++)
         {
            const double wsi = w*shape2(i);
            for (int j = 0; j < ndof1; j++)
            {
               R21(i, j) += wsi*shape1(j);
               R12(j, i) -= wsi*shape1(j);
            }
            for (int j = 0; j < ndof2; j++)
            {
               R22(i, j) -= wsi*shape2(j);
            }
         }
      }
   }

   MinvR11 = R11;
   M1inv.Solve(ndof1, ndof1, MinvR11.Data());
   for (int i = 0; i < ndof1; i++)
   {
      for (int j = 0; j < ndof1; j++)
      {
         Re(i, j) = R11(i, j);
         MinvRe(i, j) = MinvR11(i, j);
      }
   }

   if (ndof2)
   {
      MinvR12 = R12;
      MinvR21 = R21;
      MinvR22 = R22;
      M1inv.Solve(ndof1, ndof2, MinvR12.Data());
      M2inv.Solve(ndof2, ndof1, MinvR21.Data());
      M2inv.Solve(ndof2, ndof2, MinvR22.Data());

      for (int i = 0; i < ndof2; i++)
      {
         for (int j = 0; j < ndof1; j++)
         {
            Re(ndof1 + i, j) = R21(i, j);
            MinvRe(ndof1 + i, j) = MinvR21(i, j);

            Re(j, ndof1 + i) = R12(j, i);
            MinvRe(j, ndof1 + i) = MinvR12(j, i);
         }
         for (int j = 0; j < ndof2; j++)
         {
            Re(ndof1 + i, ndof1 + j) = R22(i, j);
            MinvRe(ndof1 + i, ndof1 + j) = MinvR22(i, j);
         }
      }
   }

   // Compute the matrix associated with (r_e([u]), r_e([u])).
   // The matrix for r_e([u]) is `MinvRe`, and so we need to form the product
   // `(MinvRe)^T M MinvRe`. Using `Minv^T M = Minv M = I`, we obtain
   // `Re^T MinvRe`.
   MultAtB(Re, MinvRe, elmat);
}

}
