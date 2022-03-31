// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Internal header, included only by .cpp files

#include "fem.hpp"

#include "../fem/kernels.hpp"
#include "../general/forall.hpp"
#include "../fem/kernels.hpp"
#include "../linalg/kernels.hpp"

namespace mfem
{

namespace internal
{

namespace linearform_extension
{

///
class LinearFormExtensionElementRestriction : public ElementRestriction
{
public:
   LinearFormExtensionElementRestriction(const FiniteElementSpace &fes,
                                         ElementDofOrdering ordering):
      ElementRestriction(fes, ordering) { /**/ }

   /**
    * @brief GatherMap
    * @return the mapping from L dofs to E dofs.
    */
   const Array<int> &GatherMap() const { return gather_map; }
};

/// Signature of the kernels used for linear form extension
using LinearFormExtensionKernel_f = void (*)(const int vdim,
                                             const bool byVDIM,
                                             const int ND,
                                             const int NE,
                                             const int d,
                                             const int q,
                                             const int *markers,
                                             const double *b,
                                             const double *g,
                                             const int *idx,
                                             const double *J,
                                             const double *detJ,
                                             const double *weights,
                                             const Vector &coeff,
                                             double *output);

/// Signature of the function used to compute the quadrature order
using GetOrder_f = std::function<int(int)>;

/// Internal helper function to get the integration rule
inline const IntegrationRule *GetIntRuleFromOrder(const FiniteElementSpace &fes,
                                                  const IntegrationRule *IntRule,
                                                  const GetOrder_f &qorder_fct)
{
   const FiniteElement &fe = *fes.GetFE(0);
   const int qorder = qorder_fct(fe.GetOrder());
   const Geometry::Type geom_type = fe.GetGeomType();
   return IntRule ? IntRule : &IntRules.Get(geom_type, qorder);
}

/// Internal helper function to encode the ID of a LinearFormExtension kernel
/// The ID is equal to: (dim << 8) |
///                     (1D number of degrees of freedom << 4) |
///                     (1D number of quadrature points)
inline int GetKernelId(const FiniteElementSpace &fes,
                       const IntegrationRule *ir)
{
   Mesh *mesh = fes.GetMesh();
   const int dim = mesh->Dimension();
   const FiniteElement &el = *fes.GetFE(0);
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   const int d = maps.ndof;
   const int q = maps.nqpt;
   return (dim << 8) | (d << 4) | q;
}

/// Internal helper function to launch the LinearFormExtension kernel
inline void Launch(const LinearFormExtensionKernel_f &kernel,
                   const FiniteElementSpace &fes,
                   const IntegrationRule *ir,
                   const Vector &coeff,
                   const Array<int> &markers,
                   Vector &y)
{
   Mesh *mesh = fes.GetMesh();
   const int vdim = fes.GetVDim();
   const bool byVDIM = fes.GetOrdering() == Ordering::byVDIM;

   const FiniteElement &el = *fes.GetFE(0);
   constexpr int flags = GeometricFactors::JACOBIANS |
                         GeometricFactors::DETERMINANTS;
   const MemoryType mt = Device::GetDeviceMemoryType();
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, flags, mt);
   const DofToQuad &maps = el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   constexpr ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   //LinearFormExtensionElementRestriction LFEER(fes, ordering);
   const Operator *ERop = fes.GetElementRestriction(ordering);
   const ElementRestriction* ER = dynamic_cast<const ElementRestriction*>(ERop);
   MFEM_ASSERT(ER, "Not supported!");

   const int *M = markers.Read();
   const double *B = maps.B.Read();
   const double *G = maps.G.Read();
   const double *J = geom->J.Read();
   const double *detJ = geom->detJ.Read();
   const int *I = ER->GatherMap().Read();
   const double *W = ir->GetWeights().Read();
   double *Y = y.ReadWrite();

   const int ND = fes.GetNDofs();
   const int NE = fes.GetMesh()->GetNE();

   const int d = maps.ndof;
   const int q = maps.nqpt;

   kernel(vdim, byVDIM, ND, NE, d, q, M, B, G, I, J, detJ, W, coeff, Y);
}

/// Internal assembly kernel for the 2D (Vector)DomainLFIntegrator
template<int D=0, int Q=0> static
void VectorDomainLFIntegratorAssemble2D(const int vdim,
                                        const bool byVDIM,
                                        const int ND,
                                        const int NE,
                                        const int d,
                                        const int q,
                                        const int *markers,
                                        const double *b,
                                        const double *, // g
                                        const int *idx,
                                        const double *, // jacobians
                                        const double *detJ,
                                        const double *weights,
                                        const Vector &coeff,
                                        double *y)
{
   constexpr bool USE_SMEM = D > 0 && Q > 0;

   const bool cst_coeff = coeff.Size() == vdim;

   const auto F = coeff.Read();
   const auto M = Reshape(markers, NE);
   const auto B = Reshape(b, q,d);
   const auto DetJ = Reshape(detJ, q,q, NE);
   const auto W = Reshape(weights, q,q);
   const auto I = Reshape(idx, d,d, NE);
   const auto C = cst_coeff ? Reshape(F,vdim,1,1,1) : Reshape(F,vdim,q,q,NE);

   auto Y = Reshape(y, byVDIM ? vdim : ND, byVDIM ? ND : vdim);

   const int sm_size = 2*q*(d+q);
   constexpr int GRID = USE_SMEM ? 0 : 128;
   double *gmem = kernels::internal::pool::SetSize<GRID>(sm_size);

   MFEM_FORALL_3D_GRID(e, NE, q,q,1, GRID,
   {
      if (M(e) == 0) { /* ignore */ return; }

      const int bid = MFEM_BLOCK_ID(x);
      constexpr int SM_SIZE = 2*Q*(D+Q);
      constexpr bool USE_SMEM = D > 0 && Q > 0;
      MFEM_SHARED double SMEM[USE_SMEM ? SM_SIZE : 1];
      double *sm = USE_SMEM ? SMEM : (gmem + sm_size*bid);
      const DeviceMatrix Bt(kernels::internal::pool::Alloc(sm,q*d), q,d);
      const DeviceMatrix QQ(kernels::internal::pool::Alloc(sm,q*q), q,q);
      const DeviceMatrix QD(kernels::internal::pool::Alloc(sm,q*d), q,d);

      kernels::internal::load::B(d,q,B,Bt);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val = C(c,0,0,0);
         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               const double detJ = DetJ(x,y,e);
               const double coeff_val = cst_coeff ? cst_val : C(c,x,y,e);
               QQ(y,x) = W(x,y) * coeff_val * detJ;
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::eval::fast::Transpose(d,q,Bt,QQ,QD,I,Y,c,e,byVDIM);
      }
   });
}

/// Internal assembly kernel for the 2D (Vector)DomainLFIntegrator
template<int D=0, int Q=0> static
void VectorDomainLFIntegratorAssemble3D(const int vdim,
                                        const bool byVDIM,
                                        const int ND,
                                        const int NE,
                                        const int d,
                                        const int q,
                                        const int *markers,
                                        const double *b,
                                        const double *, // g
                                        const int *idx,
                                        const double *, // jacobians
                                        const double *detJ,
                                        const double *weights,
                                        const Vector &coeff,
                                        double *y)
{
   constexpr bool USE_SMEM = D > 0 && Q > 0;

   const bool cst_coeff = coeff.Size() == vdim;

   const auto F = coeff.Read();
   const auto M = Reshape(markers, NE);
   const auto B = Reshape(b, q,d);
   const auto DetJ = Reshape(detJ, q,q,q, NE);
   const auto W = Reshape(weights, q,q,q);
   const auto I = Reshape(idx, d,d,d, NE);
   const auto C = cst_coeff ? Reshape(F,vdim,1,1,1,1):Reshape(F,vdim,q,q,q,NE);

   auto Y = Reshape(y, byVDIM ? vdim : ND, byVDIM ? ND : vdim);

   const int sm_size = q*d + q*q*q;
   const int GRID = USE_SMEM ? 0 : 128;
   double *gmem = kernels::internal::pool::SetSize<GRID>(sm_size);
   MFEM_VERIFY(q < 32, "Unsupported quadrature order!");

   MFEM_FORALL_3D_GRID(e, NE, q,q,1, GRID,
   {
      if (M(e) == 0) { /* ignore */ return; }

      double u[Q>0?Q:32];

      const int bid = MFEM_BLOCK_ID(x);
      constexpr int SM_SIZE = Q*D + Q*Q*Q;
      constexpr bool USE_SMEM = D > 0 && Q > 0;
      MFEM_SHARED double SMEM[USE_SMEM ? SM_SIZE : 1];
      double *sm = USE_SMEM ? SMEM : (gmem + sm_size*bid);
      const DeviceCube QQQ(kernels::internal::pool::Alloc(sm,q*q*q), q,q,q);
      const DeviceMatrix Bt(kernels::internal::pool::Alloc(sm,q*d), q,d);
      kernels::internal::load::B(d,q,B,Bt);

      for (int c = 0; c < vdim; ++c)
      {
         const double cst_val = C(c,0,0,0,0);
         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               for (int z = 0; z < q; ++z)
               {
                  const double detJ = DetJ(x,y,z,e);
                  const double coeff_val = cst_coeff ? cst_val : C(c,x,y,z,e);
                  QQQ(z,y,x) = W(x,y,z) * coeff_val * detJ;
               }
            }
         }
         MFEM_SYNC_THREAD;
         kernels::internal::eval::fast::Transpose(d,q,u,Bt,QQQ,I,Y,c,e,byVDIM);
      }
   });
}

} // namespace linearform_extension

} // namespace internal

} // namespace mfem
