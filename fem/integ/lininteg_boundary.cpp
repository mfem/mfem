// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
#include "../../fem/kernels.hpp"
#include "../fem.hpp"

#include "lininteg_boundary_kernels.hpp"

namespace mfem
{

/// Boundary load multiplying the test function at a quadrature point.
enum class BdrLoad
{
   Value,     ///< f v, scalar f           (BoundaryLFIntegrator)
   NormalDot, ///< (g.n) v, vector g       (BoundaryNormalLFIntegrator)
   NormalFlux ///< f (n.v), vdim = dim     (VectorBoundaryFluxLFIntegrator)
};

template<int T_D1D = 0, int T_Q1D = 0>
static void BLFEvalAssemble2D(const int vdim, const int nbe, const int d,
                              const int q,
                              const BdrLoad load, const real_t scale,
                              const int *markers, const real_t *b,
                              const real_t *detj, const real_t *n, const real_t *weights,
                              const Vector &coeff, real_t *y)
{
   const auto F = coeff.Read();
   const auto M = Reshape(markers, nbe);
   const auto B = Reshape(b, q, d);
   const auto detJ = Reshape(detj, q, nbe);
   const auto N = Reshape(n, q, 2, nbe);
   const auto W = Reshape(weights, q);
   const bool normals = load == BdrLoad::NormalDot;
   const int cvdim = normals ? 2 : 1;
   const bool cst = coeff.Size() == cvdim;
   const auto C = cst ? Reshape(F,cvdim,1,1) : Reshape(F,cvdim,q,nbe);
   auto Y = Reshape(y, d, vdim, nbe);

   mfem::forall(nbe, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t QQ[Q];

      for (int c = 0; c < vdim; ++c)
      {
         for (int qx = 0; qx < q; ++qx)
         {
            real_t coeff_val = 0.0;
            if (normals)
            {
               for (int cd = 0; cd < 2; ++cd)
               {
                  const real_t cval = cst ? C(cd,0,0) : C(cd,qx,e);
                  coeff_val += cval * N(qx, cd, e);
               }
            }
            else
            {
               coeff_val = cst ? C(0,0,0) : C(0,qx,e);
            }
            QQ[qx] = scale * W(qx) * coeff_val * detJ(qx,e);
         }
         for (int dx = 0; dx < d; ++dx)
         {
            real_t u = 0;
            for (int qx = 0; qx < q; ++qx) { u += QQ[qx] * B(qx,dx); }
            Y(dx,c,e) += u;
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void BLFEvalAssemble3D(const int vdim, const int nbe, const int d,
                              const int q,
                              const BdrLoad load, const real_t scale,
                              const int *markers, const real_t *b,
                              const real_t *detj, const real_t *n, const real_t *weights,
                              const Vector &coeff, real_t *y)
{
   const auto F = coeff.Read();
   const auto M = Reshape(markers, nbe);
   const auto B = Reshape(b, q, d);
   const auto detJ = Reshape(detj, q, q, nbe);
   const auto N = Reshape(n, q, q, 3, nbe);
   const auto W = Reshape(weights, q, q);
   const bool normals = load == BdrLoad::NormalDot;
   const int cvdim = normals ? 3 : 1;
   const bool cst = coeff.Size() == cvdim;
   const auto C = cst ? Reshape(F,cvdim,1,1,1) : Reshape(F,cvdim,q,q,nbe);
   auto Y = Reshape(y, d, d, vdim, nbe);

   mfem::forall_2D(nbe, q, q, [=] MFEM_HOST_DEVICE (int e)
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;

      MFEM_SHARED real_t sBt[Q*D];
      MFEM_SHARED real_t sQQ[Q*Q];
      MFEM_SHARED real_t sQD[Q*D];

      const DeviceMatrix Bt(sBt, d, q);
      kernels::internal::LoadB<D,Q>(d, q, B, sBt);

      const DeviceMatrix QQ(sQQ, q, q);
      const DeviceMatrix QD(sQD, q, d);

      for (int c = 0; c < vdim; ++c)
      {
         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               real_t coeff_val = 0.0;
               if (normals)
               {
                  for (int cd = 0; cd < 3; ++cd)
                  {
                     real_t cval = cst ? C(cd,0,0,0) : C(cd,x,y,e);
                     coeff_val += cval * N(x,y,cd,e);
                  }
               }
               else
               {
                  coeff_val = cst ? C(0,0,0,0) : C(0,x,y,e);
               }
               QQ(y,x) = scale * W(x,y) * coeff_val * detJ(x,y,e);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,q)
         {
            MFEM_FOREACH_THREAD(dx,x,d)
            {
               real_t u = 0.0;
               for (int qx = 0; qx < q; ++qx) { u += QQ(qy,qx) * Bt(dx,qx); }
               QD(qy,dx) = u;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(dx,x,d)
            {
               real_t u = 0.0;
               for (int qy = 0; qy < q; ++qy) { u += QD(qy,dx) * Bt(dy,qy); }
               Y(dx,dy,c,e) += u;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

/** Simplex (triangle/tetrahedron) boundary faces, i.e. segments/triangles.

    The face E-vector @a y is in NATIVE order (ConformingFaceRestriction),
    @a b is the DofToQuad::FULL basis of the face element, and the geometric
    factors and the coefficient follow the native IntegrationRule order (see
    FaceGeometricFactors). Boundary loads are assembled once and touch only
    the boundary DOFs, so a flat kernel (one thread per face DOF) is used
    rather than the simplex MMA engine of the domain integrators. */
static void BLFEvalAssembleSimplex(const int vdim, const int nbf,
                                   const int d, const int q, const int sdim,
                                   const BdrLoad load, const real_t scale,
                                   const int *markers, const real_t *b,
                                   const real_t *detj, const real_t *n,
                                   const real_t *weights,
                                   const Vector &coeff, real_t *y)
{
   const bool normals = load == BdrLoad::NormalDot;
   const bool flux = load == BdrLoad::NormalFlux;
   const int cvdim = normals ? sdim : 1;
   const bool cst = coeff.Size() == cvdim;
   const auto F = coeff.Read();
   const auto C = cst ? Reshape(F, cvdim, 1, 1) : Reshape(F, cvdim, q, nbf);
   const auto M = Reshape(markers, nbf);
   const auto B = Reshape(b, q, d);
   const auto detJ = Reshape(detj, q, nbf);
   const auto N = Reshape(n, q, sdim, nbf);
   const auto W = Reshape(weights, q);
   auto Y = Reshape(y, d, vdim, nbf);

   mfem::forall(nbf*d, [=] MFEM_HOST_DEVICE (int i)
   {
      const int dof = i % d, f = i / d;
      if (M(f) == 0) { return; } // ignore
      for (int c = 0; c < vdim; ++c)
      {
         real_t u = 0.0;
         for (int k = 0; k < q; ++k)
         {
            real_t coeff_val = 0.0;
            if (normals)
            {
               for (int cd = 0; cd < sdim; ++cd)
               {
                  coeff_val += (cst ? C(cd,0,0) : C(cd,k,f)) * N(k,cd,f);
               }
            }
            else { coeff_val = cst ? C(0,0,0) : C(0,k,f); }
            if (flux) { coeff_val *= N(k,c,f); }
            u += B(k,dof) * W(k) * coeff_val * detJ(k,f);
         }
         Y(dof,c,f) += scale * u;
      }
   });
}

static void BLFEvalAssemble(const FiniteElementSpace &fes,
                            const IntegrationRule &ir,
                            const Array<int> &markers,
                            const Vector &coeff,
                            const BdrLoad load,
                            const real_t scale,
                            Vector &y)
{
   Mesh &mesh = *fes.GetMesh();
   const int dim = mesh.Dimension();
   const FiniteElement &el = *fes.GetTypicalBE();
   const MemoryType mt = Device::GetDeviceMemoryType();
   int flags = FaceGeometricFactors::DETERMINANTS;
   if (load != BdrLoad::Value) { flags |= FaceGeometricFactors::NORMALS; }
   const FaceGeometricFactors *geom = mesh.GetFaceGeometricFactors(
                                         ir, flags, FaceType::Boundary, mt);
   MFEM_VERIFY(load != BdrLoad::NormalFlux || fes.GetVDim() == dim,
               "the normal flux load needs a vector space with vdim = dim");

   const int vdim = fes.GetVDim();
   const int nbf = fes.GetMesh()->GetNFbyType(FaceType::Boundary);
   const int *M = markers.Read();
   const real_t *detJ = geom->detJ.Read();
   const real_t *n = geom->normal.Read();
   const real_t *W = ir.GetWeights().Read();
   real_t *Y = y.ReadWrite();

   if (!UsesTensorBasis(fes))
   {
      // NATIVE face E-vector: full (non-tensor) basis of the face element.
      const DofToQuad &maps = el.GetDofToQuad(ir, DofToQuad::FULL);
      BLFEvalAssembleSimplex(vdim, nbf, maps.ndof, maps.nqpt, dim, load,
                             scale, M, maps.B.Read(), detJ, n, W, coeff, Y);
      return;
   }

   // Tensor normal flux loads go through VectorBoundaryFluxLFIntegrator's
   // AssembleKernels registry, see VBFEvalAssemble.
   MFEM_ASSERT(load != BdrLoad::NormalFlux, "use VBFEvalAssemble");
   const DofToQuad &maps = el.GetDofToQuad(ir, DofToQuad::TENSOR);
   const int d = maps.ndof, q = maps.nqpt;
   auto ker = (dim == 2) ? BLFEvalAssemble2D<> : BLFEvalAssemble3D<>;

   if (dim==2)
   {
      if (d==1 && q==1) { ker=BLFEvalAssemble2D<1,1>; }
      if (d==2 && q==2) { ker=BLFEvalAssemble2D<2,2>; }
      if (d==3 && q==3) { ker=BLFEvalAssemble2D<3,3>; }
      if (d==4 && q==4) { ker=BLFEvalAssemble2D<4,4>; }
      if (d==5 && q==5) { ker=BLFEvalAssemble2D<5,5>; }
      if (d==2 && q==3) { ker=BLFEvalAssemble2D<2,3>; }
      if (d==3 && q==4) { ker=BLFEvalAssemble2D<3,4>; }
      if (d==4 && q==5) { ker=BLFEvalAssemble2D<4,5>; }
      if (d==5 && q==6) { ker=BLFEvalAssemble2D<5,6>; }
   }

   if (dim==3)
   {
      if (d==1 && q==1) { ker=BLFEvalAssemble3D<1,1>; }
      if (d==2 && q==2) { ker=BLFEvalAssemble3D<2,2>; }
      if (d==3 && q==3) { ker=BLFEvalAssemble3D<3,3>; }
      if (d==4 && q==4) { ker=BLFEvalAssemble3D<4,4>; }
      if (d==5 && q==5) { ker=BLFEvalAssemble3D<5,5>; }
      if (d==2 && q==3) { ker=BLFEvalAssemble3D<2,3>; }
      if (d==3 && q==4) { ker=BLFEvalAssemble3D<3,4>; }
      if (d==4 && q==5) { ker=BLFEvalAssemble3D<4,5>; }
      if (d==5 && q==6) { ker=BLFEvalAssemble3D<5,6>; }
   }

   MFEM_VERIFY(ker, "No kernel ndof " << d << " nqpt " << q);
   ker(vdim, nbf, d, q, load, scale, M, maps.B.Read(), detJ, n, W, coeff, Y);
}

void BoundaryLFIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                          const Array<int> &markers,
                                          Vector &b)
{
   const FiniteElement &fe = *fes.GetTypicalBE();
   const int qorder = oa * fe.GetOrder() + ob;
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule &ir = IntRule ? *IntRule : IntRules.Get(gtype, qorder);
   Mesh &mesh = *fes.GetMesh();

   FaceQuadratureSpace qs(mesh, ir, FaceType::Boundary);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);
   BLFEvalAssemble(fes, ir, markers, coeff, BdrLoad::Value, 1.0, b);
}

void BoundaryNormalLFIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                                const Array<int> &markers,
                                                Vector &b)
{
   const FiniteElement &fe = *fes.GetTypicalBE();
   const int qorder = oa * fe.GetOrder() + ob;
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule &ir = IntRule ? *IntRule : IntRules.Get(gtype, qorder);
   Mesh &mesh = *fes.GetMesh();

   FaceQuadratureSpace qs(mesh, ir, FaceType::Boundary);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);
   BLFEvalAssemble(fes, ir, markers, coeff, BdrLoad::NormalDot, 1.0, b);
}

/// Tensor (quadrilateral/hexahedron) VectorBoundaryFluxLFIntegrator assembly,
/// dispatched through the AssembleKernels registry.
static void VBFEvalAssemble(const FiniteElementSpace &fes,
                            const IntegrationRule &ir,
                            const Array<int> &markers, const Vector &coeff,
                            const real_t sign, Vector &y)
{
   Mesh &mesh = *fes.GetMesh();
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "VectorBoundaryFluxLFIntegrator "
               "device assembly is only defined in 2D and 3D");
   MFEM_VERIFY(fes.GetVDim() == dim,
               "the normal flux load needs a vector space with vdim = dim");
   const FiniteElement &el = *fes.GetTypicalBE();
   const MemoryType mt = Device::GetDeviceMemoryType();
   const DofToQuad &maps = el.GetDofToQuad(ir, DofToQuad::TENSOR);
   const int d = maps.ndof, q = maps.nqpt;
   const int flags = FaceGeometricFactors::DETERMINANTS |
                     FaceGeometricFactors::NORMALS;
   const FaceGeometricFactors *geom = mesh.GetFaceGeometricFactors(
                                         ir, flags, FaceType::Boundary, mt);

   const int vdim = fes.GetVDim();
   const int nbf = mesh.GetNFbyType(FaceType::Boundary);
   const int *M = markers.Read();
   const real_t *B = maps.B.Read();
   const real_t *detJ = geom->detJ.Read();
   const real_t *n = geom->normal.Read();
   const real_t *W = ir.GetWeights().Read();
   real_t *Y = y.ReadWrite();
   VectorBoundaryFluxLFIntegrator::AssembleKernels::Run(
      dim, d, q, vdim, nbf, d, q, M, B, detJ, W, n, coeff, sign, Y);
}

void VectorBoundaryFluxLFIntegrator::AssembleDevice(
   const FiniteElementSpace &fes, const Array<int> &markers, Vector &b)
{
   if (fes.GetNBE() == 0) { return; }
   const FiniteElement &fe = *fes.GetTypicalBE();
   // Same default rule as AssembleRHSElementVect().
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule &ir =
      IntRule ? *IntRule : IntRules.Get(gtype, fe.GetOrder() + 1);
   Mesh &mesh = *fes.GetMesh();

   FaceQuadratureSpace qs(mesh, ir, FaceType::Boundary);
   CoefficientVector coeff(*F, qs, CoefficientStorage::COMPRESSED);
   if (UsesTensorBasis(fes))
   {
      VBFEvalAssemble(fes, ir, markers, coeff, Sign, b);
   }
   else
   {
      BLFEvalAssemble(fes, ir, markers, coeff, BdrLoad::NormalFlux, Sign, b);
   }
}

/// \cond DO_NOT_DOCUMENT

VectorBoundaryFluxLFIntegrator::AssembleKernelType
VectorBoundaryFluxLFIntegrator::AssembleKernels::Fallback(int DIM, int, int)
{
   if (DIM == 2) { return VBFEvalAssemble2D<>; }
   if (DIM == 3) { return VBFEvalAssemble3D<>; }
   MFEM_ABORT("VectorBoundaryFluxLFIntegrator: unsupported DIM " << DIM);
   return nullptr;
}

VectorBoundaryFluxLFIntegrator::Kernels::Kernels()
{
   using VBF = VectorBoundaryFluxLFIntegrator;
   // Default LEGACY rule: order p+1, i.e. Q1D = (P+1)/2 + 1 for D1D = P+1
   VBF::AddSpecialization<2, 2, 2>();
   VBF::AddSpecialization<2, 3, 2>();
   VBF::AddSpecialization<2, 4, 3>();
   VBF::AddSpecialization<2, 5, 3>();
   VBF::AddSpecialization<2, 6, 4>();
   VBF::AddSpecialization<3, 2, 2>();
   VBF::AddSpecialization<3, 3, 2>();
   VBF::AddSpecialization<3, 4, 3>();
   VBF::AddSpecialization<3, 5, 3>();
   VBF::AddSpecialization<3, 6, 4>();
   // User rules with Q1D = D1D and Q1D = D1D + 1
   VBF::AddSpecialization<2, 1, 1>();
   VBF::AddSpecialization<2, 3, 3>();
   VBF::AddSpecialization<2, 4, 4>();
   VBF::AddSpecialization<2, 5, 5>();
   VBF::AddSpecialization<2, 2, 3>();
   VBF::AddSpecialization<2, 3, 4>();
   VBF::AddSpecialization<2, 4, 5>();
   VBF::AddSpecialization<2, 5, 6>();
   VBF::AddSpecialization<3, 1, 1>();
   VBF::AddSpecialization<3, 3, 3>();
   VBF::AddSpecialization<3, 4, 4>();
   VBF::AddSpecialization<3, 5, 5>();
   VBF::AddSpecialization<3, 2, 3>();
   VBF::AddSpecialization<3, 3, 4>();
   VBF::AddSpecialization<3, 4, 5>();
   VBF::AddSpecialization<3, 5, 6>();
}

/// \endcond DO_NOT_DOCUMENT

} // namespace mfem
