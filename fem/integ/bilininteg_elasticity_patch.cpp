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
#include "bilininteg_patch.hpp"
namespace mfem
{

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