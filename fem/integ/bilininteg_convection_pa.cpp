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

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "../ceed/integrators/convection/convection.hpp"

#include "bilininteg_convection_kernels.hpp"

/// \cond DO_NOT_DOCUMENT
namespace mfem
{

ConvectionIntegrator::ConvectionIntegrator(VectorCoefficient &q, real_t a)
   : Q(&q), alpha(a)
{
   static Kernels kernels;
}

ConvectionIntegrator::Kernels::Kernels()
{
   // 2D
   // Q = P + 1
   ConvectionIntegrator::AddSpecialization<2, 2, 2>();
   ConvectionIntegrator::AddSpecialization<2, 3, 3>();
   ConvectionIntegrator::AddSpecialization<2, 4, 4>();
   ConvectionIntegrator::AddSpecialization<2, 5, 5>();
   ConvectionIntegrator::AddSpecialization<2, 6, 6>();
   // Q = P + 2
   ConvectionIntegrator::AddSpecialization<2, 2, 3>();
   ConvectionIntegrator::AddSpecialization<2, 3, 4>();
   ConvectionIntegrator::AddSpecialization<2, 4, 5>();
   ConvectionIntegrator::AddSpecialization<2, 5, 6>();
   ConvectionIntegrator::AddSpecialization<2, 6, 7>();
   // 3D
   // Q = P + 1
   ConvectionIntegrator::AddSpecialization<3, 2, 2>();
   ConvectionIntegrator::AddSpecialization<3, 3, 3>();
   ConvectionIntegrator::AddSpecialization<3, 4, 4>();
   ConvectionIntegrator::AddSpecialization<3, 5, 5>();
   ConvectionIntegrator::AddSpecialization<3, 6, 6>();
   // Q = P + 2
   ConvectionIntegrator::AddSpecialization<3, 2, 3>();
   ConvectionIntegrator::AddSpecialization<3, 3, 4>();
   ConvectionIntegrator::AddSpecialization<3, 4, 5>();
   ConvectionIntegrator::AddSpecialization<3, 5, 6>();
   ConvectionIntegrator::AddSpecialization<3, 6, 7>();
}

// PA Convection Assemble 2D kernel
static void PAConvectionSetup2D(const int NQ,
                                const int NE,
                                const Array<real_t> &w,
                                const Vector &j,
                                const Vector &vel,
                                const real_t alpha,
                                Vector &op)
{
   constexpr int DIM = 2;

   const bool const_v = vel.Size() == DIM;

   const auto W = w.Read();
   const auto J = Reshape(j.Read(), NQ,DIM,DIM,NE);
   const auto V = const_v ?
                  Reshape(vel.Read(), DIM,1,1) :
                  Reshape(vel.Read(), DIM,NQ,NE);
   auto y = Reshape(op.Write(), NQ,DIM,NE);

   mfem::forall(NE*NQ, [=] MFEM_HOST_DEVICE (int q_global)
   {
      const int e = q_global / NQ;
      const int q = q_global % NQ;
      const real_t J11 = J(q,0,0,e);
      const real_t J21 = J(q,1,0,e);
      const real_t J12 = J(q,0,1,e);
      const real_t J22 = J(q,1,1,e);
      const real_t w = alpha * W[q];
      const real_t v0 = const_v ? V(0,0,0) : V(0,q,e);
      const real_t v1 = const_v ? V(1,0,0) : V(1,q,e);
      const real_t wx = w * v0;
      const real_t wy = w * v1;
      // y = alpha * W * det(J) * J^{-1} . v = adj(J) . { wx, wy }
      y(q,0,e) =  wx * J22 - wy * J12; // 1
      y(q,1,e) = -wx * J21 + wy * J11; // 2
   });
}

// PA Convection Assemble 3D kernel
static void PAConvectionSetup3D(const int NQ,
                                const int NE,
                                const Array<real_t> &w,
                                const Vector &j,
                                const Vector &vel,
                                const real_t alpha,
                                Vector &op)
{
   constexpr int DIM = 3;
   constexpr int SDIM = DIM;
   const auto W = Reshape(w.Read(), NQ);
   const auto J = Reshape(j.Read(), NQ,SDIM,DIM,NE);
   const bool const_v = vel.Size() == DIM;
   const auto V = const_v ?
                  Reshape(vel.Read(), 3,1,1) :
                  Reshape(vel.Read(), 3,NQ,NE);
   auto y = Reshape(op.Write(), NQ,3,NE);
   mfem::forall(NE*NQ, [=] MFEM_HOST_DEVICE (int q_global)
   {
      const int e = q_global / NQ;
      const int q = q_global % NQ;
      const real_t J11 = J(q,0,0,e);
      const real_t J12 = J(q,0,1,e);
      const real_t J13 = J(q,0,2,e);
      const real_t J21 = J(q,1,0,e);
      const real_t J22 = J(q,1,1,e);
      const real_t J23 = J(q,1,2,e);
      const real_t J31 = J(q,2,0,e);
      const real_t J32 = J(q,2,1,e);
      const real_t J33 = J(q,2,2,e);
      const real_t w = alpha * W(q);
      const real_t v0 = const_v ? V(0,0,0) : V(0,q,e);
      const real_t v1 = const_v ? V(1,0,0) : V(1,q,e);
      const real_t v2 = const_v ? V(2,0,0) : V(2,q,e);
      const real_t wx = w * v0;
      const real_t wy = w * v1;
      const real_t wz = w * v2;
      // A = adj(J)
      const real_t A11 = (J22 * J33) - (J23 * J32);
      const real_t A12 = (J32 * J13) - (J12 * J33);
      const real_t A13 = (J12 * J23) - (J22 * J13);
      const real_t A21 = (J31 * J23) - (J21 * J33);
      const real_t A22 = (J11 * J33) - (J13 * J31);
      const real_t A23 = (J21 * J13) - (J11 * J23);
      const real_t A31 = (J21 * J32) - (J31 * J22);
      const real_t A32 = (J31 * J12) - (J11 * J32);
      const real_t A33 = (J11 * J22) - (J12 * J21);
      // y = alpha * W * det(J) * J^{-1} . v = adj(J) . { wx, wy, wz }
      y(q,0,e) = wx * A11 + wy * A12 + wz * A13;
      y(q,1,e) = wx * A21 + wy * A22 + wz * A23;
      y(q,2,e) = wx * A31 + wy * A32 + wz * A33;
   });
}

static void PAConvectionSetup(const int dim,
                              const int NQ,
                              const int NE,
                              const Array<real_t> &W,
                              const Vector &J,
                              const Vector &coeff,
                              const real_t alpha,
                              Vector &op)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PAConvectionSetup"); }
   if (dim == 2)
   {
      PAConvectionSetup2D(NQ, NE, W, J, coeff, alpha, op);
   }
   if (dim == 3)
   {
      PAConvectionSetup3D(NQ, NE, W, J, coeff, alpha, op);
   }
}

void ConvectionIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalFE();
   ElementTransformation &Trans = *mesh->GetTypicalElementTransformation();
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, Trans);
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      const bool mixed = mesh->GetNumGeometries(mesh->Dimension()) > 1 ||
                         fes.IsVariableOrder();
      if (mixed)
      {
         ceedOp = new ceed::MixedPAConvectionIntegrator(*this, fes, Q, alpha);
      }
      else
      {
         ceedOp = new ceed::PAConvectionIntegrator(fes, *ir, Q, alpha);
      }
      return;
   }
   const int dims = el.GetDim();
   const int symmDims = dims;
   nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(symmDims * nq * ne, mt);

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector vel(*Q, qs, CoefficientStorage::COMPRESSED);

   PAConvectionSetup(dim, nq, ne, ir->GetWeights(), geom->J,
                     vel, alpha, pa_data);
}

void ConvectionIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      ceedOp->GetDiagonal(diag);
   }
   else
   {
      MFEM_ABORT("AssembleDiagonalPA not yet implemented for"
                 " ConvectionIntegrator.");
   }
}

inline ConvectionIntegrator::ApplyKernelType
ConvectionIntegrator::ApplyPAKernels::Fallback(int DIM, int, int)
{
   if (DIM == 2)
   {
      return PAConvectionApply2D;
   }
   else if (DIM == 3)
   {
      return PAConvectionApply3D;
   }
   else
   {
      MFEM_ABORT("");
   }
}

inline ConvectionIntegrator::ApplyKernelType
ConvectionIntegrator::ApplyPATKernels::Fallback(int DIM, int, int)
{
   if (DIM == 2)
   {
      return PAConvectionApplyT2D;
   }
   else if (DIM == 3)
   {
      return PAConvectionApplyT3D;
   }
   else
   {
      MFEM_ABORT("");
   }
}

void ConvectionIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else
   {
      ApplyPAKernels::Run(dim, dofs1D, quad1D, ne, maps->B, maps->G, maps->Bt,
                          maps->Gt, pa_data, x, y, dofs1D, quad1D);
   }
}

void ConvectionIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      MFEM_ABORT("AddMultPA not yet implemented with libCEED for"
                 " ConvectionIntegrator.");
   }
   else
   {
      ApplyPATKernels::Run(dim, dofs1D, quad1D, ne, maps->B, maps->G, maps->Bt,
                           maps->Gt, pa_data, x, y, dofs1D, quad1D);
   }
}

} // namespace mfem
/// \endcond DO_NOT_DOCUMENT
