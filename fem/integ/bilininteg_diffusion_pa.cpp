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
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "../ceed/integrators/diffusion/diffusion.hpp"
#include "bilininteg_diffusion_kernels.hpp"

using namespace std;

namespace mfem
{

void DiffusionIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;
   // Assuming the same element type
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      MFEM_VERIFY(!VQ && !MQ,
                  "Only scalar coefficient supported for DiffusionIntegrator"
                  " with libCEED");
      const bool mixed = mesh->GetNumGeometries(mesh->Dimension()) > 1 ||
                         fes.IsVariableOrder();
      if (mixed)
      {
         ceedOp = new ceed::MixedPADiffusionIntegrator(*this, fes, Q);
      }
      else
      {
         ceedOp = new ceed::PADiffusionIntegrator(fes, *ir, Q);
      }
      return;
   }
   const int dims = el.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt);
   const int sdim = mesh->SpaceDimension();
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::COMPRESSED);

   if (MQ) { coeff.ProjectTranspose(*MQ); }
   else if (VQ) { coeff.Project(*VQ); }
   else if (Q) { coeff.Project(*Q); }
   else { coeff.SetConstant(1.0); }

   const int coeff_dim = coeff.GetVDim();
   symmetric = (coeff_dim != dims*dims);
   const int pa_size = symmetric ? symmDims : dims*dims;

   pa_data.SetSize(pa_size * nq * ne, mt);
   internal::PADiffusionSetup(dim, sdim, dofs1D, quad1D, coeff_dim, ne,
                              ir->GetWeights(),
                              geom->J, coeff, pa_data);
}

// PA Diffusion Apply kernel
void DiffusionIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      internal::PADiffusionApply(dim, dofs1D, quad1D, ne, symmetric,
                                 maps->B, maps->G, maps->Bt, maps->Gt,
                                 pa_data, x, y);
   }
}

void DiffusionIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   if (symmetric)
   {
      AddMultPA(x, y);
   }
   else
   {
      MFEM_ABORT("DiffusionIntegrator::AddMultTransposePA only implemented in "
                 "the symmetric case.")
   }
}

void DiffusionIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      ceedOp->GetDiagonal(diag);
   }
   else
   {
      if (pa_data.Size()==0) { AssemblePA(*fespace); }
      internal::PADiffusionAssembleDiagonal(dim, dofs1D, quad1D, ne, symmetric,
                                            maps->B, maps->G, pa_data, diag);
   }
}

} // namespace mfem
