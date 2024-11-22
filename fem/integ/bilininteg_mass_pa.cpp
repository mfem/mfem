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

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "../ceed/integrators/mass/mass.hpp"
#include "bilininteg_mass_kernels.hpp"

namespace mfem
{

// PA Mass Integrator

void MassIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   // Assuming the same element type
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0) { return; }
   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation *T0 = mesh->GetElementTransformation(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, *T0);
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      const bool mixed = mesh->GetNumGeometries(mesh->Dimension()) > 1 ||
                         fes.IsVariableOrder();
      if (mixed)
      {
         ceedOp = new ceed::MixedPAMassIntegrator(*this, fes, Q);
      }
      else
      {
         ceedOp = new ceed::PAMassIntegrator(fes, *ir, Q);
      }
      return;
   }
   int map_type = el.GetMapType();
   dim = mesh->Dimension();
   ne = fes.GetMesh()->GetNE();
   nq = ir->GetNPoints();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::DETERMINANTS, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(ne*nq, mt);

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);
   {
      const int NE = ne;
      const int NQ = nq;
      const bool const_c = coeff.Size() == 1;
      const bool by_val = map_type == FiniteElement::VALUE;
      const auto W = Reshape(ir->GetWeights().Read(), NQ);
      const auto J = Reshape(geom->detJ.Read(), NQ, NE);
      const auto C =
          const_c ? Reshape(coeff.Read(), 1, 1) : Reshape(coeff.Read(), NQ, NE);
      auto v = Reshape(pa_data.Write(), NQ, NE);
      mfem::forall(NE * NQ, [=] MFEM_HOST_DEVICE(int idx) {
         int e = idx / NQ;
         int q = idx % NQ;
         const real_t detJ = J(q, e);
         const real_t coeff = const_c ? C(0, 0) : C(q, e);
         v(q, e) = W(q) * coeff * (by_val ? detJ : 1.0 / detJ);
      });
   }
}

void MassIntegrator::AssemblePABoundary(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   // Assuming the same element type
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNBE() == 0) { return; }
   const FiniteElement &el = *fes.GetBE(0);
   ElementTransformation *T0 = mesh->GetBdrElementTransformation(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, *T0);

   int map_type = el.GetMapType();
   dim = el.GetDim(); // Dimension of the boundary element, *not* the mesh
   ne = fes.GetMesh()->GetNFbyType(FaceType::Boundary);
   nq = ir->GetNPoints();
   face_geom = mesh->GetFaceGeometricFactors(*ir, GeometricFactors::DETERMINANTS,
                                             FaceType::Boundary, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(ne*nq, mt);

   FaceQuadratureSpace qs(*mesh, *ir, FaceType::Boundary);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);

   const int NE = ne;
   const int NQ = nq;
   const int Q1D = quad1D;
   const bool const_c = coeff.Size() == 1;
   const bool by_val = map_type == FiniteElement::VALUE;
   {
      const auto W = Reshape(ir->GetWeights().Read(), NQ);
      const auto J = Reshape(face_geom->detJ.Read(), NQ, NE);
      const auto C = const_c ? Reshape(coeff.Read(), 1, 1)
                             : Reshape(coeff.Read(), NQ, NE);
      auto v = Reshape(pa_data.Write(), NQ, NE);
      mfem::forall(NE * NQ, [=] MFEM_HOST_DEVICE(int idx) {
         int e = idx / NQ;
         int q = idx % NQ;
         const real_t detJ = J(q, e);
         const real_t coeff = const_c ? C(0, 0) : C(q, e);
         v(q, e) = W(q) * coeff * (by_val ? detJ : 1.0 / detJ);
      });
   }
}

void MassIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      ceedOp->GetDiagonal(diag);
   }
   else
   {
      DiagonalPAKernels::Run(dim, dofs1D, quad1D, ne, maps->B, pa_data,
                             diag, dofs1D, quad1D);
   }
}

void MassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      const int D1D = dofs1D;
      const int Q1D = quad1D;
      const Array<real_t> &B = maps->B;
      const Array<real_t> &Bt = maps->Bt;
      const Vector &D = pa_data;
#ifdef MFEM_USE_OCCA
      if (DeviceCanUseOcca())
      {
         if (dim == 2)
         {
            return internal::OccaPAMassApply2D(D1D,Q1D,ne,B,Bt,D,x,y);
         }
         if (dim == 3)
         {
            return internal::OccaPAMassApply3D(D1D,Q1D,ne,B,Bt,D,x,y);
         }
         MFEM_ABORT("OCCA PA Mass Apply unknown kernel!");
      }
#endif // MFEM_USE_OCCA
      ApplyPAKernels::Run(dim, D1D, Q1D, ne, B, Bt, D, x, y, D1D, Q1D);
   }
}

void MassIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   // Mass integrator is symmetric
   AddMultPA(x, y);
}

} // namespace mfem
