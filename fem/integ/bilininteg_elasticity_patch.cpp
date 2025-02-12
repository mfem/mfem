// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


#include "../fem.hpp"
#include "../../mesh/nurbs.hpp"

#include "../../linalg/dtensor.hpp"  // For Reshape
#include "../../general/forall.hpp"
#include "fem/bilininteg.hpp"
namespace mfem
{

void PatchElasticitySetup3D(const int Q1Dx,
                            const int Q1Dy,
                            const int Q1Dz,
                            const Array<real_t> &w,
                            const Vector &j,
                            Vector &d)
{
   // computes d=[J^{-T}(xq), W(xq)*det(J(xq))] at quadrature points
   const auto W = Reshape(w.Read(), Q1Dx,Q1Dy,Q1Dz);
   const auto J = Reshape(j.Read(), Q1Dx,Q1Dy,Q1Dz,3,3);
   // nq * [9 (J^{-T}) + 1 WdetJ]
   d.SetSize(Q1Dx * Q1Dy * Q1Dz * 10);
   auto D = Reshape(d.Write(), Q1Dx,Q1Dy,Q1Dz, 10);
   const int NE = 1;  // TODO: MFEM_FORALL_3D without e?
   MFEM_FORALL_3D(e, NE, Q1Dx, Q1Dy, Q1Dz,
   {
      MFEM_FOREACH_THREAD(qx,x,Q1Dx)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1Dy)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1Dz)
            {
               const real_t J11 = J(qx,qy,qz,0,0);
               const real_t J21 = J(qx,qy,qz,1,0);
               const real_t J31 = J(qx,qy,qz,2,0);
               const real_t J12 = J(qx,qy,qz,0,1);
               const real_t J22 = J(qx,qy,qz,1,1);
               const real_t J32 = J(qx,qy,qz,2,1);
               const real_t J13 = J(qx,qy,qz,0,2);
               const real_t J23 = J(qx,qy,qz,1,2);
               const real_t J33 = J(qx,qy,qz,2,2);
               const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
               /* */               J21 * (J12 * J33 - J32 * J13) +
               /* */               J31 * (J12 * J23 - J22 * J13);
               // adj(J)
               const real_t A11 = (J22 * J33) - (J23 * J32);
               const real_t A12 = (J32 * J13) - (J12 * J33);
               const real_t A13 = (J12 * J23) - (J22 * J13);
               const real_t A21 = (J31 * J23) - (J21 * J33);
               const real_t A22 = (J11 * J33) - (J13 * J31);
               const real_t A23 = (J21 * J13) - (J11 * J23);
               const real_t A31 = (J21 * J32) - (J31 * J22);
               const real_t A32 = (J31 * J12) - (J11 * J32);
               const real_t A33 = (J11 * J22) - (J12 * J21);

               // store J^{-T} = adj(J)^T / detJ
               D(qx,qy,qz,0) = A11 / detJ;
               D(qx,qy,qz,1) = A21 / detJ;
               D(qx,qy,qz,2) = A31 / detJ;
               D(qx,qy,qz,3) = A12 / detJ;
               D(qx,qy,qz,4) = A22 / detJ;
               D(qx,qy,qz,5) = A32 / detJ;
               D(qx,qy,qz,6) = A13 / detJ;
               D(qx,qy,qz,7) = A23 / detJ;
               D(qx,qy,qz,8) = A33 / detJ;
               // store w_detJ
               // TODO: Small efficiency to multiply by sqrt(W/detJ)? Might not work for negative weights
               D(qx,qy,qz,9) = W(qx,qy,qz) / detJ;
            }
         }
      }
   });
}

// TODO: maybe move this into a base class?
void ElasticityIntegrator::SetupPatchBasisData(Mesh *mesh, unsigned int patch)
{
   mfem::out << "SetupPatchBasisData() " << patch << std::endl;
   MFEM_VERIFY(pB.size() == patch && pG.size() == patch, "");
   MFEM_VERIFY(pQ1D.size() == patch && pD1D.size() == patch, "");
   MFEM_VERIFY(pminQ.size() == patch && pmaxQ.size() == patch, "");
   MFEM_VERIFY(pminD.size() == patch && pmaxD.size() == patch, "");
   MFEM_VERIFY(pminDD.size() == patch && pmaxDD.size() == patch, "");
   MFEM_VERIFY(pir1d.size() == patch, "");

   // Set basis functions and gradients for this patch
   Array<const KnotVector*> pkv;
   mesh->NURBSext->GetPatchKnotVectors(patch, pkv);
   MFEM_VERIFY(pkv.Size() == vdim, "");

   Array<int> Q1D(vdim);
   Array<int> orders(vdim);
   Array<int> D1D(vdim);
   std::vector<Array2D<real_t>> B(vdim);
   std::vector<Array2D<real_t>> G(vdim);
   Array<const IntegrationRule*> ir1d(vdim);

   IntArrayVar2D minD(vdim);
   IntArrayVar2D maxD(vdim);
   IntArrayVar2D minQ(vdim);
   IntArrayVar2D maxQ(vdim);

   IntArrayVar2D minDD(vdim);
   IntArrayVar2D maxDD(vdim);

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

   // Push patch data to global data structures
   pB.push_back(B);
   pG.push_back(G);

   pQ1D.push_back(Q1D);
   pD1D.push_back(D1D);

   pminQ.push_back(minQ);
   pmaxQ.push_back(maxQ);

   pminD.push_back(minD);
   pmaxD.push_back(maxD);

   pminDD.push_back(minDD);
   pmaxDD.push_back(maxDD);

   pir1d.push_back(ir1d);
}

// Computes mu, lambda, J^{-T}, and det(J) at quadrature points
void ElasticityIntegrator::SetupPatchPA(const int patch, Mesh *mesh,
                                        bool unitWeights)
{
   mfem::out << "SetupPatchPA() " << patch << std::endl;

   // Quadrature points in each dimension for this patch
   const Array<int>& Q1D = pQ1D[patch];
   MFEM_VERIFY(Q1D.Size() == vdim, "");

   // Total quadrature points
   int nq = Q1D[0];
   for (int i=1; i<vdim; ++i)
   {
      nq *= Q1D[i];
   }

   // for reduced rules
   // const Array<int>& D1D = pD1D[patch];
   // const std::vector<Array2D<real_t>>& B = pB[patch];
   // const std::vector<Array2D<real_t>>& G = pG[patch];
   // const IntArrayVar2D& minD = pminD[patch];
   // const IntArrayVar2D& maxD = pmaxD[patch];
   // const IntArrayVar2D& minQ = pminQ[patch];
   // const IntArrayVar2D& maxQ = pmaxQ[patch];
   // const IntArrayVar2D& minDD = pminDD[patch];
   // const IntArrayVar2D& maxDD = pmaxDD[patch];
   // const Array<const IntegrationRule*>& ir1d = pir1d[patch];

   Array<real_t> weights(nq);
   IntegrationPoint ip;

   Vector jac(vdim * vdim * nq);  // Computed as in GeometricFactors::Compute

   // TODO: use QuadratureInterpolator instead of ElementTransformation?
   for (int qz=0; qz<Q1D[2]; ++qz)
   {
      for (int qy=0; qy<Q1D[1]; ++qy)
      {
         for (int qx=0; qx<Q1D[0]; ++qx)
         {
            const int p = qx + (qy * Q1D[0]) + (qz * Q1D[0] * Q1D[1]);
            patchRules->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);
            const int e = patchRules->GetPointElement(patch, qx, qy, qz);
            ElementTransformation *tr = mesh->GetElementTransformation(e);

            weights[p] = ip.weight;

            tr->SetIntPoint(&ip);

            const DenseMatrix& Jp = tr->Jacobian();
            for (int i=0; i<vdim; ++i)
               for (int j=0; j<vdim; ++j)
               {
                  jac[p + ((i + (j * vdim)) * nq)] = Jp(i,j);
               }
         }
      }
   }

   // TODO: Compute coefficient at quadrature points
   const FiniteElementSpace *fes = mesh->GetNodalFESpace();
   SetUpQuadratureSpaceAndCoefficients(*fes);

   if (unitWeights)
   {
      weights = 1.0;
   }
   // Computes "D" matrix
   PatchElasticitySetup3D(Q1D[0], Q1D[1], Q1D[2], weights, jac, pa_data);

   mfem::out << "Finished computing D " << patch << std::endl;


   if (integrationMode != PATCHWISE_REDUCED)
   {
      return;
   }
   else
   {
      MFEM_ABORT("Not implemented yet.");
   }


   // numPatches = mesh->NURBSext->GetNP();
   // // Solve for reduced 1D quadrature rules
   // const int totalDim = numPatches * dim * numTypes;
   // reducedWeights.resize(totalDim);
   // reducedIDs.resize(totalDim);

   // auto rw = Reshape(reducedWeights.data(), numTypes, dim, numPatches);
   // auto rid = Reshape(reducedIDs.data(), numTypes, dim, numPatches);

   // for (int d=0; d<dim; ++d)
   // {
   //    // The reduced rules could be cached to avoid repeated computation, but
   //    // the cost of this setup seems low.
   //    PatchDiffusionGetReducedRule(Q1D[d], D1D[d], B[d], G[d],
   //                   minQ[d], maxQ[d],
   //                   minD[d], maxD[d],
   //                   minDD[d], maxDD[d], ir1d[d], true,
   //                   rw(0,d,patch), rid(0,d,patch));
   //    PatchDiffusionGetReducedRule(Q1D[d], D1D[d], B[d], G[d],
   //                   minQ[d], maxQ[d],
   //                   minD[d], maxD[d],
   //                   minDD[d], maxDD[d], ir1d[d], false,
   //                   rw(1,d,patch), rid(1,d,patch));
   // }
}

}