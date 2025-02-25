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

#include "integrator.hpp"
#include "fem.hpp"
#include "intrules.hpp"
#include "../mesh/nurbs.hpp"

namespace mfem
{
const IntegrationRule* Integrator::GetIntegrationRule(
   const FiniteElement& trial_fe, const FiniteElement& test_fe,
   const ElementTransformation& trans) const
{
   const IntegrationRule* result;
   const NURBSFiniteElement *NURBSFE;
   if (patchRules &&
       (NURBSFE = dynamic_cast<const NURBSFiniteElement *>(&test_fe)))
   {
      const int patch = NURBSFE->GetPatch();
      const int* ijk = NURBSFE->GetIJK();
      Array<const KnotVector*>& kv = NURBSFE->KnotVectors();
      result = &patchRules->GetElementRule(NURBSFE->GetElement(), patch, ijk,
                                           kv);
   }
   else if (IntRule)
   {
      result = IntRule;
   }
   else
   {
      result = GetDefaultIntegrationRule(trial_fe, test_fe, trans);
   }
   return result;
}

const IntegrationRule* Integrator::GetIntegrationRule(
   const FiniteElement& el,
   const ElementTransformation& trans) const
{
   return GetIntegrationRule(el, el, trans);
}

PatchBasisInfo::PatchBasisInfo(int vdim, Mesh *mesh, unsigned int patch, NURBSMeshRules *patchRules)
   : patch(patch), B(vdim), G(vdim), ir1d(vdim), Q1D(vdim), D1D(vdim),
      minD(vdim), maxD(vdim), minQ(vdim), maxQ(vdim), minDD(vdim), maxDD(vdim)
{
   Array<const KnotVector*> pkv;
   mesh->NURBSext->GetPatchKnotVectors(patch, pkv);
   MFEM_VERIFY(pkv.Size() == vdim, "");
   Array<int> orders(vdim);

   for (int d=0; d<vdim; ++d)
   {
      ir1d[d] = patchRules->GetPatchRule1D(patch, d);

      Q1D[d] = ir1d[d]->GetNPoints();

      orders[d] = pkv[d]->GetOrder();
      D1D[d] = pkv[d]->GetNCP();

      Vector shapeKV(orders[d]+1);
      Vector dshapeKV(orders[d]+1);

      B[d].SetSize(Q1D[d], D1D[d]);
      G[d].SetSize(Q1D[d], D1D[d]);

      minD[d].assign(D1D[d], Q1D[d]);
      maxD[d].assign(D1D[d], 0);

      minQ[d].assign(Q1D[d], D1D[d]);
      maxQ[d].assign(Q1D[d], 0);

      B[d] = 0.0;
      G[d] = 0.0;

      const Array<int>& knotSpan1D = patchRules->GetPatchRule1D_KnotSpan(patch, d);
      MFEM_VERIFY(knotSpan1D.Size() == Q1D[d], "");

      for (int i = 0; i < Q1D[d]; i++)
      {
         const IntegrationPoint &ip = ir1d[d]->IntPoint(i);
         const int ijk = knotSpan1D[i];
         const real_t kv0 = (*pkv[d])[orders[d] + ijk];
         real_t kv1 = (*pkv[d])[0];
         for (int j = orders[d] + ijk + 1; j < pkv[d]->Size(); ++j)
         {
            if ((*pkv[d])[j] > kv0)
            {
               kv1 = (*pkv[d])[j];
               break;
            }
         }

         MFEM_VERIFY(kv1 > kv0, "");

         pkv[d]->CalcShape(shapeKV, ijk, (ip.x - kv0) / (kv1 - kv0));
         pkv[d]->CalcDShape(dshapeKV, ijk, (ip.x - kv0) / (kv1 - kv0));

         // Put shapeKV into array B storing shapes for all points.
         // TODO: This should be based on NURBS3DFiniteElement::CalcShape and CalcDShape.
         // For now, it works under the assumption that all NURBS weights are 1.
         for (int j=0; j<orders[d]+1; ++j)
         {
            B[d](i,ijk + j) = shapeKV[j];
            G[d](i,ijk + j) = dshapeKV[j];

            minD[d][ijk + j] = std::min(minD[d][ijk + j], i);
            maxD[d][ijk + j] = std::max(maxD[d][ijk + j], i);
         }

         minQ[d][i] = std::min(minQ[d][i], ijk);
         maxQ[d][i] = std::max(maxQ[d][i], ijk + orders[d]);
      }

      // Determine which DOFs each DOF interacts with, in 1D.
      minDD[d].resize(D1D[d]);
      maxDD[d].resize(D1D[d]);
      for (int i=0; i<D1D[d]; ++i)
      {
         const int qmin = minD[d][i];
         minDD[d][i] = minQ[d][qmin];

         const int qmax = maxD[d][i];
         maxDD[d][i] = maxQ[d][qmax];
      }
   }

   // Total quadrature points
   NQ = Q1D[0];
   ND = D1D[0];
   for (int i=1; i<vdim; ++i)
   {
      NQ *= Q1D[i];
      ND *= D1D[i];
   }
}

}
