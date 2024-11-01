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

   if (dim==1) { MFEM_ABORT("Not supported yet... stay tuned!"); }
   if (dim==2)
   {
      const int NE = ne;
      const int Q1D = quad1D;
      const bool const_c = coeff.Size() == 1;
      const bool by_val = map_type == FiniteElement::VALUE;
      const auto W = Reshape(ir->GetWeights().Read(), Q1D,Q1D);
      const auto J = Reshape(geom->detJ.Read(), Q1D,Q1D,NE);
      const auto C = const_c ? Reshape(coeff.Read(), 1,1,1) :
                     Reshape(coeff.Read(), Q1D,Q1D,NE);
      auto v = Reshape(pa_data.Write(), Q1D,Q1D, NE);
      mfem::forall_2D(NE,Q1D,Q1D, [=] MFEM_HOST_DEVICE (int e)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               const real_t detJ = J(qx,qy,e);
               const real_t coeff = const_c ? C(0,0,0) : C(qx,qy,e);
               v(qx,qy,e) =  W(qx,qy) * coeff * (by_val ? detJ : 1.0/detJ);
            }
         }
      });
   }
   if (dim==3)
   {
      const int NE = ne;
      const int Q1D = quad1D;
      const bool const_c = coeff.Size() == 1;
      const bool by_val = map_type == FiniteElement::VALUE;
      const auto W = Reshape(ir->GetWeights().Read(), Q1D,Q1D,Q1D);
      const auto J = Reshape(geom->detJ.Read(), Q1D,Q1D,Q1D,NE);
      const auto C = const_c ? Reshape(coeff.Read(), 1,1,1,1) :
                     Reshape(coeff.Read(), Q1D,Q1D,Q1D,NE);
      auto v = Reshape(pa_data.Write(), Q1D,Q1D,Q1D,NE);
      mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qz,z,Q1D)
               {
                  const real_t detJ = J(qx,qy,qz,e);
                  const real_t coeff = const_c ? C(0,0,0,0) : C(qx,qy,qz,e);
                  v(qx,qy,qz,e) = W(qx,qy,qz) * coeff * (by_val ? detJ : 1.0/detJ);
               }
            }
         }
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
   const int Q1D = quad1D;
   const bool const_c = coeff.Size() == 1;
   const bool by_val = map_type == FiniteElement::VALUE;
   if (dim==1)
   {
      const auto W = Reshape(ir->GetWeights().Read(), Q1D);
      const auto J = Reshape(face_geom->detJ.Read(), Q1D, NE);
      const auto C = const_c ? Reshape(coeff.Read(), 1, 1) :
                     Reshape(coeff.Read(), Q1D, NE);
      auto v = Reshape(pa_data.Write(), Q1D, NE);
      mfem::forall_2D(NE, Q1D, 1, [=] MFEM_HOST_DEVICE (int e)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const real_t detJ = J(qx,e);
            const real_t coeff = const_c ? C(0,0) : C(qx,e);
            v(qx,e) =  W(qx) * coeff * (by_val ? detJ : 1.0/detJ);
         }
      });
   }
   else if (dim==2)
   {
      const auto W = Reshape(ir->GetWeights().Read(), Q1D,Q1D);
      const auto J = Reshape(face_geom->detJ.Read(), Q1D,Q1D,NE);
      const auto C = const_c ? Reshape(coeff.Read(), 1,1,1) :
                     Reshape(coeff.Read(), Q1D,Q1D,NE);
      auto v = Reshape(pa_data.Write(), Q1D,Q1D, NE);
      mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               const real_t detJ = J(qx,qy,e);
               const real_t coeff = const_c ? C(0,0,0) : C(qx,qy,e);
               v(qx,qy,e) =  W(qx,qy) * coeff * (by_val ? detJ : 1.0/detJ);
            }
         }
      });
   }
   else
   {
      MFEM_ABORT("Not supported.");
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
