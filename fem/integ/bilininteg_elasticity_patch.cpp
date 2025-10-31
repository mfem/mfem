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


#include "../fem.hpp"
#include "../../mesh/nurbs.hpp"

#include "../../linalg/dtensor.hpp"  // For Reshape
#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../integrator.hpp"
namespace mfem
{

void PatchElasticitySetup3D(const int Q1Dx,
                            const int Q1Dy,
                            const int Q1Dz,
                            const Vector &w,
                            const Vector &j,
                            const Vector &c,
                            Vector &d)
{
   // computes [J^{-T}, lambda*W*detJ, mu*W*detJ] at quadrature points
   const auto W = Reshape(w.Read(), Q1Dx,Q1Dy,Q1Dz);
   const auto J = Reshape(j.Read(), Q1Dx,Q1Dy,Q1Dz,3,3);
   const auto C = Reshape(c.Read(), Q1Dx,Q1Dy,Q1Dz,2);
   // nq * [9 (J^{-T}) + 1 (WdetJ) + 1 (lambda) + 1 (mu)]
   d.SetSize(Q1Dx * Q1Dy * Q1Dz * 11);
   auto D = Reshape(d.Write(), Q1Dx,Q1Dy,Q1Dz, 11);
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
               const real_t wdetj = W(qx,qy,qz) * detJ;
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
               // Coefficients
               D(qx,qy,qz,9) = C(qx,qy,qz,0) * wdetj; // lambda * w * detj
               D(qx,qy,qz,10) = C(qx,qy,qz,1) * wdetj; // mu * w * detj
            }
         }
      }
   });
}

void ElasticityIntegrator::SetupPatchBasisData(Mesh *mesh, unsigned int patch)
{
   // Push patch data to global data structures
   PatchBasisInfo pb(mesh, patch, patchRules);
   pbinfo.push_back(pb);
}

// Computes mu, lambda, J^{-T}, and W*det(J) at quadrature points
void ElasticityIntegrator::SetupPatchPA(const int patch, Mesh *mesh,
                                        bool unitWeights)
{
   // Quadrature points in each dimension for this patch
   const Array<int>& Q1D = pbinfo[patch].Q1D;
   // Total quadrature points
   const int nq = pbinfo[patch].NQ;

   Vector weightsv(nq);
   auto weights = Reshape(weightsv.HostReadWrite(), Q1D[0], Q1D[1], Q1D[2]);
   IntegrationPoint ip;

   Vector jacv(nq * vdim * vdim);  // Computed as in GeometricFactors::Compute
   auto jac = Reshape(jacv.HostReadWrite(), Q1D[0], Q1D[1], Q1D[2], vdim, vdim);
   Vector coeffsv(nq * 2);        // lambda, mu at quad points
   auto coeffs = Reshape(coeffsv.HostReadWrite(), Q1D[0], Q1D[1], Q1D[2], 2);

   MFEM_VERIFY(Q1D.Size() == 3, "Only 3D for now");

   for (int qz=0; qz<Q1D[2]; ++qz)
   {
      for (int qy=0; qy<Q1D[1]; ++qy)
      {
         for (int qx=0; qx<Q1D[0]; ++qx)
         {
            const int e = patchRules->GetPointElement(patch, qx, qy, qz);
            ElementTransformation *tr = mesh->GetElementTransformation(e);
            patchRules->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);

            weights(qx,qy,qz) = ip.weight;

            coeffs(qx,qy,qz,0) = lambda->Eval(*tr, ip);
            coeffs(qx,qy,qz,1) = mu->Eval(*tr, ip);

            tr->SetIntPoint(&ip);

            const DenseMatrix& Jp = tr->Jacobian();
            for (int i=0; i<vdim; ++i)
            {
               for (int j=0; j<vdim; ++j)
               {
                  jac(qx,qy,qz,i,j) = Jp(i,j);
               }
            }
         }
      }
   }

   // For reduced rules
   if (unitWeights)
   {
      weightsv = 1.0;
   }

   // Computes values at quadrature points
   PatchElasticitySetup3D(Q1D[0], Q1D[1], Q1D[2], weightsv, jacv, coeffsv,
                          ppa_data[patch]);

   if (integrationMode != PATCHWISE_REDUCED)
   {
      return;
   }
   else
   {
      MFEM_ABORT("Not implemented yet.");
   }

}

}