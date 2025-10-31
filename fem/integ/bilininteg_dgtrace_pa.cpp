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
#include "../restriction.hpp"

#include "bilininteg_dgtrace_kernels.hpp"

namespace mfem
{

// PA DG Trace Integrator
static void PADGTraceSetup2D(const int Q1D, const int NF,
                             const Array<real_t> &w, const Vector &det,
                             const Vector &nor, const Vector &rho,
                             const Vector &vel, const real_t alpha,
                             const real_t beta, Vector &op)
{
   const int VDIM = 2;

   auto d = Reshape(det.Read(), Q1D, NF);
   auto n = Reshape(nor.Read(), Q1D, VDIM, NF);
   const bool const_r = rho.Size() == 1;
   auto R = const_r ? Reshape(rho.Read(), 1, 1) : Reshape(rho.Read(), Q1D, NF);
   const bool const_v = vel.Size() == 2;
   auto V =
      const_v ? Reshape(vel.Read(), 2, 1, 1) : Reshape(vel.Read(), 2, Q1D, NF);
   auto W = w.Read();
   auto qd = Reshape(op.Write(), Q1D, 2, 2, NF);

   mfem::forall(Q1D * NF, [=] MFEM_HOST_DEVICE(int tid)
   {
      const int f = tid / Q1D;
      const int q = tid % Q1D;
      {
         const real_t r = const_r ? R(0, 0) : R(q, f);
         const real_t v0 = const_v ? V(0, 0, 0) : V(0, q, f);
         const real_t v1 = const_v ? V(1, 0, 0) : V(1, q, f);
         const real_t dot = n(q, 0, f) * v0 + n(q, 1, f) * v1;
         const real_t abs = dot > 0_r ? dot : -dot;
         const real_t w = W[q] * r * d(q, f);
         qd(q, 0, 0, f) = w * (alpha / 2 * dot + beta * abs);
         qd(q, 1, 0, f) = w * (alpha / 2 * dot - beta * abs);
         qd(q, 0, 1, f) = w * (-alpha / 2 * dot - beta * abs);
         qd(q, 1, 1, f) = w * (-alpha / 2 * dot + beta * abs);
      }
   });
}

static void PADGTraceSetup3D(const int Q1D, const int NF,
                             const Array<real_t> &w, const Vector &det,
                             const Vector &nor, const Vector &rho,
                             const Vector &vel, const real_t alpha,
                             const real_t beta, Vector &op)
{
   const int VDIM = 3;

   auto d = Reshape(det.Read(), Q1D, Q1D, NF);
   auto n = Reshape(nor.Read(), Q1D, Q1D, VDIM, NF);
   const bool const_r = rho.Size() == 1;
   auto R = const_r ? Reshape(rho.Read(), 1, 1, 1)
            : Reshape(rho.Read(), Q1D, Q1D, NF);
   const bool const_v = vel.Size() == 3;
   auto V = const_v ? Reshape(vel.Read(), 3, 1, 1, 1)
            : Reshape(vel.Read(), 3, Q1D, Q1D, NF);
   auto W = w.Read();
   auto qd = Reshape(op.Write(), Q1D, Q1D, 2, 2, NF);

   mfem::forall(Q1D * Q1D * NF, [=] MFEM_HOST_DEVICE(int tid)
   {
      int f = tid / (Q1D * Q1D);
      int q2 = (tid / Q1D) % Q1D;
      int q1 = tid % Q1D;
      {
         {
            const real_t r = const_r ? R(0, 0, 0) : R(q1, q2, f);
            const real_t v0 = const_v ? V(0, 0, 0, 0) : V(0, q1, q2, f);
            const real_t v1 = const_v ? V(1, 0, 0, 0) : V(1, q1, q2, f);
            const real_t v2 = const_v ? V(2, 0, 0, 0) : V(2, q1, q2, f);
            const real_t dot = n(q1, q2, 0, f) * v0 + n(q1, q2, 1, f) * v1 +
                               n(q1, q2, 2, f) * v2;
            const real_t abs = dot > 0.0 ? dot : -dot;
            const real_t w = W[q1 + q2 * Q1D] * r * d(q1, q2, f);
            qd(q1, q2, 0, 0, f) = w * (alpha / 2 * dot + beta * abs);
            qd(q1, q2, 1, 0, f) = w * (alpha / 2 * dot - beta * abs);
            qd(q1, q2, 0, 1, f) = w * (-alpha / 2 * dot - beta * abs);
            qd(q1, q2, 1, 1, f) = w * (-alpha / 2 * dot + beta * abs);
         }
      }
   });
}

static void PADGTraceSetup(const int dim, const int D1D, const int Q1D,
                           const int NF, const Array<real_t> &W,
                           const Vector &det, const Vector &nor,
                           const Vector &rho, const Vector &u,
                           const real_t alpha, const real_t beta, Vector &op)
{
   if (dim == 1)
   {
      MFEM_ABORT("dim==1 not supported in PADGTraceSetup");
   }
   if (dim == 2)
   {
      PADGTraceSetup2D(Q1D, NF, W, det, nor, rho, u, alpha, beta, op);
   }
   if (dim == 3)
   {
      PADGTraceSetup3D(Q1D, NF, W, det, nor, rho, u, alpha, beta, op);
   }
}

void DGTraceIntegrator::SetupPA(const FiniteElementSpace &fes, FaceType type)
{
   const MemoryType mt =
      (pa_mt == MemoryType::DEFAULT) ? Device::GetDeviceMemoryType() : pa_mt;

   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalTraceElement();
   const IntegrationRule *ir = IntRule?
                               IntRule:
                               &GetRule(el.GetGeomType(), el.GetOrder(),
                                        *mesh->GetTypicalElementTransformation());


   FaceQuadratureSpace qs(*mesh, *ir, type);
   nf = qs.GetNumFaces();
   if (nf==0) { return; }
   const int symmDims = 4;
   nq = ir->GetNPoints();
   dim = mesh->Dimension();
   geom = mesh->GetFaceGeometricFactors(
             *ir, FaceGeometricFactors::DETERMINANTS | FaceGeometricFactors::NORMALS,
             type, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(symmDims * nq * nf, Device::GetMemoryType());
   CoefficientVector vel(*u, qs, CoefficientStorage::COMPRESSED);

   CoefficientVector r(qs, CoefficientStorage::COMPRESSED);
   if (rho == nullptr)
   {
      r.SetConstant(1.0);
   }
   else if (ConstantCoefficient *const_rho =
               dynamic_cast<ConstantCoefficient *>(rho))
   {
      r.SetConstant(const_rho->constant);
   }
   else if (QuadratureFunctionCoefficient *qf_rho =
               dynamic_cast<QuadratureFunctionCoefficient *>(rho))
   {
      r.MakeRef(qf_rho->GetQuadFunction());
   }
   else
   {
      r.SetSize(nq * nf);
      auto C_vel = Reshape(vel.HostRead(), dim, nq, nf);
      auto n = Reshape(geom->normal.HostRead(), nq, dim, nf);
      auto C = Reshape(r.HostWrite(), nq, nf);
      int f_ind = 0;
      for (int f = 0; f < mesh->GetNumFacesWithGhost(); ++f)
      {
         Mesh::FaceInformation face = mesh->GetFaceInformation(f);
         if (face.IsNonconformingCoarse() || !face.IsOfFaceType(type))
         {
            // We skip nonconforming coarse faces as they are treated
            // by the corresponding nonconforming fine faces.
            continue;
         }
         FaceElementTransformations &T =
            *fes.GetMesh()->GetFaceElementTransformations(f);
         for (int q = 0; q < nq; ++q)
         {
            // Convert to lexicographic ordering
            int iq =
               ToLexOrdering(dim, face.element[0].local_face_id, quad1D, q);

            T.SetAllIntPoints(&ir->IntPoint(q));
            const IntegrationPoint &eip1 = T.GetElement1IntPoint();
            const IntegrationPoint &eip2 = T.GetElement2IntPoint();
            real_t rq;

            if (face.IsBoundary())
            {
               rq = rho->Eval(*T.Elem1, eip1);
            }
            else
            {
               real_t udotn = 0.0;
               for (int d = 0; d < dim; ++d)
               {
                  udotn += C_vel(d, iq, f_ind) * n(iq, d, f_ind);
               }
               if (udotn >= 0.0)
               {
                  rq = rho->Eval(*T.Elem2, eip2);
               }
               else
               {
                  rq = rho->Eval(*T.Elem1, eip1);
               }
            }
            C(iq, f_ind) = rq;
         }
         f_ind++;
      }
      MFEM_VERIFY(f_ind == nf, "Incorrect number of faces.");
   }
   PADGTraceSetup(dim, dofs1D, quad1D, nf, ir->GetWeights(), geom->detJ,
                  geom->normal, r, vel, alpha, beta, pa_data);
}

void DGTraceIntegrator::AssemblePAInteriorFaces(const FiniteElementSpace &fes)
{
   SetupPA(fes, FaceType::Interior);
}

void DGTraceIntegrator::AssemblePABoundaryFaces(const FiniteElementSpace &fes)
{
   SetupPA(fes, FaceType::Boundary);
}

// PA DGTraceIntegrator Apply kernel
void DGTraceIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   ApplyPAKernels::Run(dim, dofs1D, quad1D, nf, maps->B, maps->Bt, pa_data, x,
                       y, dofs1D, quad1D);
}

void DGTraceIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   ApplyPATKernels::Run(dim, dofs1D, quad1D, nf, maps->B, maps->Bt, pa_data, x,
                        y, dofs1D, quad1D);
}

DGTraceIntegrator::DGTraceIntegrator(real_t a, real_t b) : alpha(a), beta(b)
{
   static Kernels kernels;
}

DGTraceIntegrator::DGTraceIntegrator(VectorCoefficient &u_, real_t a)
   : DGTraceIntegrator(a, 0.5 * a)
{
   u = &u_;
}

DGTraceIntegrator::DGTraceIntegrator(VectorCoefficient &u_, real_t a, real_t b)
   : DGTraceIntegrator(a, b)
{
   u = &u_;
}

DGTraceIntegrator::DGTraceIntegrator(Coefficient &rho_, VectorCoefficient &u_,
                                     real_t a, real_t b)
   : DGTraceIntegrator(a, b)
{
   rho = &rho_;
   u = &u_;
}

/// \cond DO_NOT_DOCUMENT

DGTraceIntegrator::Kernels::Kernels()
{
   // 2D
   DGTraceIntegrator::AddSpecialization<2, 2, 2>();
   DGTraceIntegrator::AddSpecialization<2, 3, 3>();
   DGTraceIntegrator::AddSpecialization<2, 4, 4>();
   DGTraceIntegrator::AddSpecialization<2, 5, 5>();
   DGTraceIntegrator::AddSpecialization<2, 6, 6>();
   DGTraceIntegrator::AddSpecialization<2, 7, 7>();
   DGTraceIntegrator::AddSpecialization<2, 8, 8>();
   DGTraceIntegrator::AddSpecialization<2, 9, 9>();
   // 3D
   DGTraceIntegrator::AddSpecialization<3, 2, 3>();
   DGTraceIntegrator::AddSpecialization<3, 3, 4>();
   DGTraceIntegrator::AddSpecialization<3, 4, 5>();
   DGTraceIntegrator::AddSpecialization<3, 5, 6>();
   DGTraceIntegrator::AddSpecialization<3, 6, 7>();
   DGTraceIntegrator::AddSpecialization<3, 7, 8>();
   DGTraceIntegrator::AddSpecialization<3, 8, 9>();
}

DGTraceIntegrator::ApplyKernelType
DGTraceIntegrator::ApplyPAKernels::Fallback(int dim, int, int)
{
   if (dim == 2)
   {
      return internal::PADGTraceApply2D;
   }
   else if (dim == 3)
   {
      return internal::PADGTraceApply3D;
   }
   else
   {
      MFEM_ABORT("");
   }
}

DGTraceIntegrator::ApplyKernelType
DGTraceIntegrator::ApplyPATKernels::Fallback(int dim, int, int)
{
   if (dim == 2)
   {
      return internal::PADGTraceApplyTranspose2D;
   }
   else if (dim == 3)
   {
      return internal::PADGTraceApplyTranspose3D;
   }
   else
   {
      MFEM_ABORT("");
   }
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
