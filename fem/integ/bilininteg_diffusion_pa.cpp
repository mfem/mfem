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

#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "../../mesh/nurbs.hpp"
#include "../ceed/integrators/diffusion/diffusion.hpp"
#include "bilininteg_diffusion_kernels.hpp"

namespace mfem
{

void DiffusionIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      ceedOp->GetDiagonal(diag);
   }
   else
   {
      if (pa_data.Size() == 0) { AssemblePA(*fespace); }
      const Array<real_t> &B = maps->B;
      const Array<real_t> &G = maps->G;
      const Vector &Dv = pa_data;
      DiagonalPAKernels::Run(dim, dofs1D, quad1D, ne, symmetric, B, G, Dv,
                             diag, dofs1D, quad1D);
   }
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
      const Array<real_t> &B = maps->B;
      const Array<real_t> &G = maps->G;
      const Array<real_t> &Bt = maps->Bt;
      const Array<real_t> &Gt = maps->Gt;
      const Vector &Dv = pa_data;

#ifdef MFEM_USE_OCCA
      if (DeviceCanUseOcca())
      {
         if (dim == 2)
         {
            internal::OccaPADiffusionApply2D(dofs1D,quad1D,ne,B,G,Bt,Gt,Dv,x,y);
            return;
         }
         if (dim == 3)
         {
            internal::OccaPADiffusionApply3D(dofs1D,quad1D,ne,B,G,Bt,Gt,Dv,x,y);
            return;
         }
         MFEM_ABORT("OCCA PADiffusionApply unknown kernel!");
      }
#endif // MFEM_USE_OCCA

      ApplyPAKernels::Run(dim, dofs1D, quad1D, ne, symmetric, B, G, Bt,
                          Gt, Dv, x, y, dofs1D, quad1D);
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

void DiffusionIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;
   // Assuming the same element type
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalFE();
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
                              ir->GetWeights(), geom->J, coeff, pa_data);
}

void DiffusionIntegrator::AssembleNURBSPA(const FiniteElementSpace &fes)
{
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(3 == dim, "Only 3D so far");

   numPatches = mesh->NURBSext->GetNP();
   ppa_data.resize(numPatches);
   for (int p=0; p<numPatches; ++p)
   {
      AssemblePatchPA(p, fes);
   }
}

void DiffusionIntegrator::AssemblePatchPA(const int patch,
                                          const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   SetupPatchBasisData(mesh, patch);

   SetupPatchPA(patch, mesh);  // For full quadrature, unitWeights = false
}

// void DiffusionIntegrator::AddMultPatchPA3D(const Vector &pa_data,
//                                            const PatchBasisInfo &pb,
//                                            const Vector &x,
//                                            Vector &y) const
// {
//    MFEM_VERIFY(3 == dim, "");

//    // Unpack patch basis info
//    const Array<int>& Q1D = pb.Q1D;
//    const Array<int>& D1D = pb.D1D;
//    const std::vector<Array2D<real_t>>& B = pb.B;
//    const std::vector<Array2D<real_t>>& G = pb.G;

//    const IntArrayVar2D& minD = pb.minD;
//    const IntArrayVar2D& maxD = pb.maxD;
//    const IntArrayVar2D& minQ = pb.minQ;
//    const IntArrayVar2D& maxQ = pb.maxQ;

//    const int NQ = pb.NQ;
//    const std::vector<int> acc = pb.accsize;

//    const auto X = Reshape(x.HostRead(), D1D[0], D1D[1], D1D[2]);
//    auto Y = Reshape(y.HostReadWrite(), D1D[0], D1D[1], D1D[2]);

//    Vector gradv(3*NQ);
//    gradv = 0.0;
//    auto grad = Reshape(gradv.HostReadWrite(), 3, Q1D[0], Q1D[1], Q1D[2]);

//    const auto qd = Reshape(pa_data.HostRead(), NQ, (symmetric ? 6 : 9));

//    // Accumulators; these are shared between grad_u interpolation and grad_v_T
//    // application, so their size is the max of qpts/dofs
//    Vector sumXYv(3*acc[0]*acc[1]);
//    Vector sumXv(3*acc[0]);
//    auto sumXY = Reshape(sumXYv.HostReadWrite(), 3, acc[0], acc[1]);
//    auto sumX = Reshape(sumXv.HostReadWrite(), 3, acc[0]);

//    // Interpolate grad_u
//    for (int dz = 0; dz < D1D[2]; ++dz)
//    {
//       sumXYv = 0.0;
//       for (int dy = 0; dy < D1D[1]; ++dy)
//       {
//          sumXv = 0.0;
//          for (int dx = 0; dx < D1D[0]; ++dx)
//          {
//             const real_t u = X(dx,dy,dz);
//             for (int qx = minD[0][dx]; qx <= maxD[0][dx]; ++qx)
//             {
//                sumX(0,qx) += u * B[0](qx,dx);
//                sumX(1,qx) += u * G[0](qx,dx);
//             }
//          } // dx
//          for (int qy = minD[1][dy]; qy <= maxD[1][dy]; ++qy)
//          {
//             const real_t wy  = B[1](qy,dy);
//             const real_t wDy = G[1](qy,dy);
//             // This full range of qx values is generally necessary.
//             for (int qx = 0; qx < Q1D[0]; ++qx)
//             {
//                const real_t wx  = sumX(0,qx);
//                const real_t wDx = sumX(1,qx);
//                sumXY(0,qx,qy) += wDx * wy;
//                sumXY(1,qx,qy) += wx  * wDy;
//                sumXY(2,qx,qy) += wx  * wy;
//             }
//          } // qy
//       } // dy

//       for (int qz = minD[2][dz]; qz <= maxD[2][dz]; ++qz)
//       {
//          const real_t wz  = B[2](qz,dz);
//          const real_t wDz = G[2](qz,dz);
//          for (int qy = 0; qy < Q1D[1]; ++qy)
//          {
//             for (int qx = 0; qx < Q1D[0]; ++qx)
//             {
//                grad(0,qx,qy,qz) += sumXY(0,qx,qy) * wz;
//                grad(1,qx,qy,qz) += sumXY(1,qx,qy) * wz;
//                grad(2,qx,qy,qz) += sumXY(2,qx,qy) * wDz;
//             }
//          }
//       }
//    } // dz

//    // Apply kernel
//    for (int qz = 0; qz < Q1D[2]; ++qz)
//    {
//       for (int qy = 0; qy < Q1D[1]; ++qy)
//       {
//          for (int qx = 0; qx < Q1D[0]; ++qx)
//          {
//             const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
//             const real_t O00 = qd(q,0);
//             const real_t O01 = qd(q,1);
//             const real_t O02 = qd(q,2);
//             const real_t O10 = symmetric ? O01 : qd(q,3);
//             const real_t O11 = symmetric ? qd(q,3) : qd(q,4);
//             const real_t O12 = symmetric ? qd(q,4) : qd(q,5);
//             const real_t O20 = symmetric ? O02 : qd(q,6);
//             const real_t O21 = symmetric ? O12 : qd(q,7);
//             const real_t O22 = symmetric ? qd(q,5) : qd(q,8);

//             const real_t grad0 = grad(0,qx,qy,qz);
//             const real_t grad1 = grad(1,qx,qy,qz);
//             const real_t grad2 = grad(2,qx,qy,qz);

//             grad(0,qx,qy,qz) = (O00*grad0)+(O01*grad1)+(O02*grad2);
//             grad(1,qx,qy,qz) = (O10*grad0)+(O11*grad1)+(O12*grad2);
//             grad(2,qx,qy,qz) = (O20*grad0)+(O21*grad1)+(O22*grad2);
//          } // qx
//       } // qy
//    } // qz

//    // Apply gradv^T
//    for (int qz = 0; qz < Q1D[2]; ++qz)
//    {
//       sumXYv = 0.0;
//       for (int qy = 0; qy < Q1D[1]; ++qy)
//       {
//          sumXv = 0.0;
//          for (int qx = 0; qx < Q1D[0]; ++qx)
//          {
//             const real_t gX = grad(0,qx,qy,qz);
//             const real_t gY = grad(1,qx,qy,qz);
//             const real_t gZ = grad(2,qx,qy,qz);
//             for (int dx = minQ[0][qx]; dx <= maxQ[0][qx]; ++dx)
//             {
//                const real_t wx  = B[0](qx,dx);
//                const real_t wDx = G[0](qx,dx);
//                sumX(0,dx) += gX * wDx;
//                sumX(1,dx) += gY * wx;
//                sumX(2,dx) += gZ * wx;
//             }
//          }

//          for (int dy = minQ[1][qy]; dy <= maxQ[1][qy]; ++dy)
//          {
//             const real_t wy  = B[1](qy,dy);
//             const real_t wDy = G[1](qy,dy);
//             for (int dx = 0; dx < D1D[0]; ++dx)
//             {
//                sumXY(0,dx,dy) += sumX(0,dx) * wy;
//                sumXY(1,dx,dy) += sumX(1,dx) * wDy;
//                sumXY(2,dx,dy) += sumX(2,dx) * wy;
//             }
//          }
//       }

//       for (int dz = minQ[2][qz]; dz <= maxQ[2][qz]; ++dz)
//       {
//          const real_t wz  = B[2](qz,dz);
//          const real_t wDz = G[2](qz,dz);
//          for (int dy = 0; dy < D1D[1]; ++dy)
//          {
//             for (int dx = 0; dx < D1D[0]; ++dx)
//             {
//                Y(dx,dy,dz) +=
//                   ((sumXY(0,dx,dy) * wz) +
//                    (sumXY(1,dx,dy) * wz) +
//                    (sumXY(2,dx,dy) * wDz));
//             }
//          }
//       } // dz
//    } // qz
// }

// template <int dim, int d1d, int q1d>
// static inline MFEM_HOST_DEVICE void
// CalcGrad(const tensor<real_t, q1d, d1d> &B,
//          const tensor<real_t, q1d, d1d> &G,
//          tensor<real_t,2,3,q1d,q1d,q1d> &smem,
//          const DeviceTensor<4, const real_t> &U,
//          tensor<real_t, q1d, q1d, q1d, dim, dim> &dUdxi)

// template <int q1dx, int q1dy, int q1dz, int d1dx, int d1dy, int d1dz>
// void PatchInterpolateGradient(const PatchBasisInfo &pb,
//                               const DeviceTensor<3, const real_t> &U,
//                               // tensor<real_t,2,3,q1d,q1d,q1d> &smem,
//                               const DeviceTensor<4, const real_t> smem,
//                               DeviceTensor<5, real_t> &grad_uhat)
// {
//    const int D1DX = pb.D1D[0];
//    const int D1DY = pb.D1D[1];
//    const int D1DZ = pb.D1D[2];
//    const int Q1DX = pb.Q1D[0];
//    const int Q1DY = pb.Q1D[1];
//    const int Q1DZ = pb.Q1D[2];

//    MFEM_FOREACH_THREAD(dz,z,D1DZ)
//    {
//       MFEM_FOREACH_THREAD(dy,y,D1DY)
//       {
//          MFEM_FOREACH_THREAD(dx,x,D1DX)
//          {
//             smem(0,0,dx,dy,dz) = U(dx,dy,dz);
//          }
//       }
//    }
//    MFEM_SYNC_THREAD;
//    MFEM_FOREACH_THREAD(dz,z,d1d)
//    {
//       MFEM_FOREACH_THREAD(dy,y,d1d)
//       {
//          MFEM_FOREACH_THREAD(qx,x,q1d)
//          {
//             real_t u = 0.0;
//             real_t v = 0.0;
//             for (int dx = 0; dx < d1d; ++dx)
//             {
//                const real_t input = smem(0,0,dx,dy,dz);
//                u += input * B(qx,dx);
//                v += input * G(qx,dx);
//             }
//             smem(0,1,dz,dy,qx) = u;
//             smem(0,2,dz,dy,qx) = v;
//          }
//       }
//    }
//    MFEM_SYNC_THREAD;
//    MFEM_FOREACH_THREAD(dz,z,d1d)
//    {
//       MFEM_FOREACH_THREAD(qy,y,q1d)
//       {
//          MFEM_FOREACH_THREAD(qx,x,q1d)
//          {
//             real_t u = 0.0;
//             real_t v = 0.0;
//             real_t w = 0.0;
//             for (int dy = 0; dy < d1d; ++dy)
//             {
//                u += smem(0,2,dz,dy,qx) * B(qy,dy);
//                v += smem(0,1,dz,dy,qx) * G(qy,dy);
//                w += smem(0,1,dz,dy,qx) * B(qy,dy);
//             }
//             smem(1,0,dz,qy,qx) = u;
//             smem(1,1,dz,qy,qx) = v;
//             smem(1,2,dz,qy,qx) = w;
//          }
//       }
//    }
//    MFEM_SYNC_THREAD;
//    MFEM_FOREACH_THREAD(qz,z,q1d)
//    {
//       MFEM_FOREACH_THREAD(qy,y,q1d)
//       {
//          MFEM_FOREACH_THREAD(qx,x,q1d)
//          {
//             real_t u = 0.0;
//             real_t v = 0.0;
//             real_t w = 0.0;
//             for (int dz = 0; dz < d1d; ++dz)
//             {
//                u += smem(1,0,dz,qy,qx) * B(qz,dz);
//                v += smem(1,1,dz,qy,qx) * B(qz,dz);
//                w += smem(1,2,dz,qy,qx) * G(qz,dz);
//             }
//             dUdxi(qz,qy,qx,c,0) += u;
//             dUdxi(qz,qy,qx,c,1) += v;
//             dUdxi(qz,qy,qx,c,2) += w;
//          }
//       }
//    }
//    MFEM_SYNC_THREAD;
// }

// // Attempting to speed this up
// void DiffusionIntegrator::AddMultPatchPA3D(const Vector &pa_data,
//                                            const PatchBasisInfo &pb,
//                                            const Vector &x,
//                                            Vector &y) const
// {
//    // Unpack patch basis info
//    const Array<int>& Q1D = pb.Q1D;
//    const Array<int>& D1D = pb.D1D;
//    const std::vector<Array2D<real_t>>& B = pb.B;
//    const std::vector<Array2D<real_t>>& G = pb.G;
//    const Array<int>& orders = pb.orders;

//    const auto Btx = Reshape(pb.Btx.Read(), (orders[0]+1), Q1D[0]);
//    const auto Gtx = Reshape(pb.Gtx.Read(), (orders[0]+1), Q1D[0]);
//    const auto Bty = Reshape(pb.Bty.Read(), (orders[1]+1), Q1D[1]);
//    const auto Gty = Reshape(pb.Gty.Read(), (orders[1]+1), Q1D[1]);
//    const auto Btz = Reshape(pb.Btz.Read(), (orders[2]+1), Q1D[2]);
//    const auto Gtz = Reshape(pb.Gtz.Read(), (orders[2]+1), Q1D[2]);

//    const IntArrayVar2D& minD = pb.minD;
//    const IntArrayVar2D& maxD = pb.maxD;
//    const IntArrayVar2D& minQ = pb.minQ;
//    const IntArrayVar2D& maxQ = pb.maxQ;

//    const int NQ = pb.NQ;
//    const std::vector<int> acc = pb.accsize;

//    const auto X = Reshape(x.HostRead(), D1D[0], D1D[1], D1D[2]);
//    auto Y = Reshape(y.HostReadWrite(), D1D[0], D1D[1], D1D[2]);

//    Vector gradv(3*NQ);
//    gradv = 0.0;
//    auto grad = Reshape(gradv.HostReadWrite(), 3, Q1D[0], Q1D[1], Q1D[2]);

//    const auto qd = Reshape(pa_data.HostRead(), NQ, (symmetric ? 6 : 9));

//    // Accumulators; these are shared between grad_u interpolation and grad_v_T
//    // application, so their size is the max of qpts/dofs
//    // MFEM_SHARED tensor<real_t, 2, 3, Q1D[2], acc[1], acc[0]> smem;
//    // Vector smemv(2*3*NQ);

//    // Vector sumXYv(3*acc[0]*acc[1]);
//    // Vector sumXv(3*acc[0]);
//    // auto sumXY = Reshape(sumXYv.HostReadWrite(), 3, acc[0], acc[1]);
//    // auto sumX = Reshape(sumXv.HostReadWrite(), 3, acc[0]);

//    // Interpolate grad_u
//    for (int dz = 0; dz < D1D[2]; ++dz)
//    {
//       sumXYv = 0.0;
//       for (int dy = 0; dy < D1D[1]; ++dy)
//       {
//          sumXv = 0.0;
//          for (int dx = 0; dx < D1D[0]; ++dx)
//          {
//             const real_t u = X(dx,dy,dz);
//             for (int qx = minD[0][dx]; qx <= maxD[0][dx]; ++qx)
//             {
//                sumX(0,qx) += u * B[0](qx,dx);
//                sumX(1,qx) += u * G[0](qx,dx);
//             }
//          } // dx
//          for (int qy = minD[1][dy]; qy <= maxD[1][dy]; ++qy)
//          {
//             const real_t wy  = B[1](qy,dy);
//             const real_t wDy = G[1](qy,dy);
//             // This full range of qx values is generally necessary.
//             for (int qx = 0; qx < Q1D[0]; ++qx)
//             {
//                const real_t wx  = sumX(0,qx);
//                const real_t wDx = sumX(1,qx);
//                sumXY(0,qx,qy) += wDx * wy;
//                sumXY(1,qx,qy) += wx  * wDy;
//                sumXY(2,qx,qy) += wx  * wy;
//             }
//          } // qy
//       } // dy

//       for (int qz = minD[2][dz]; qz <= maxD[2][dz]; ++qz)
//       {
//          const real_t wz  = B[2](qz,dz);
//          const real_t wDz = G[2](qz,dz);
//          for (int qy = 0; qy < Q1D[1]; ++qy)
//          {
//             for (int qx = 0; qx < Q1D[0]; ++qx)
//             {
//                grad(0,qx,qy,qz) += sumXY(0,qx,qy) * wz;
//                grad(1,qx,qy,qz) += sumXY(1,qx,qy) * wz;
//                grad(2,qx,qy,qz) += sumXY(2,qx,qy) * wDz;
//             }
//          }
//       }
//    } // dz

//    // faster?
//    for (int dz = 0; dz < D1D[2]; ++dz)
//    {
//       sumXYv = 0.0;
//       for (int dy = 0; dy < D1D[1]; ++dy)
//       {
//          sumXv = 0.0;
//          for (int qx = 0; qx < Q1D[0]; ++qx)
//          {
//             const int dx0 = minQ[0][qx];
//             for (int sx = 0; sx <= orders[0]; ++sx)
//             {
//                const int dx = sx + dx0;
//                const real_t u = X(dx,dy,dz);
//                sumX(0,qx) += u * Btx(sx,qx);
//                sumX(1,qx) += u * Gtx(sx,qx);
//             }
//          } // dx
//          for (int qy = minD[1][dy]; qy <= maxD[1][dy]; ++qy)
//          {
//             const real_t wy  = B[1](qy,dy);
//             const real_t wDy = G[1](qy,dy);
//             // This full range of qx values is generally necessary.
//             for (int qx = 0; qx < Q1D[0]; ++qx)
//             {
//                const real_t wx  = sumX(0,qx);
//                const real_t wDx = sumX(1,qx);
//                sumXY(0,qx,qy) += wDx * wy;
//                sumXY(1,qx,qy) += wx  * wDy;
//                sumXY(2,qx,qy) += wx  * wy;
//             }
//          } // qy
//       } // dy

//       for (int qz = minD[2][dz]; qz <= maxD[2][dz]; ++qz)
//       {
//          const real_t wz  = B[2](qz,dz);
//          const real_t wDz = G[2](qz,dz);
//          for (int qy = 0; qy < Q1D[1]; ++qy)
//          {
//             for (int qx = 0; qx < Q1D[0]; ++qx)
//             {
//                grad(0,qx,qy,qz) += sumXY(0,qx,qy) * wz;
//                grad(1,qx,qy,qz) += sumXY(1,qx,qy) * wz;
//                grad(2,qx,qy,qz) += sumXY(2,qx,qy) * wDz;
//             }
//          }
//       }
//    } // dz

//    // Apply kernel
//    for (int qz = 0; qz < Q1D[2]; ++qz)
//    {
//       for (int qy = 0; qy < Q1D[1]; ++qy)
//       {
//          for (int qx = 0; qx < Q1D[0]; ++qx)
//          {
//             const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
//             const real_t O00 = qd(q,0);
//             const real_t O01 = qd(q,1);
//             const real_t O02 = qd(q,2);
//             const real_t O10 = symmetric ? O01 : qd(q,3);
//             const real_t O11 = symmetric ? qd(q,3) : qd(q,4);
//             const real_t O12 = symmetric ? qd(q,4) : qd(q,5);
//             const real_t O20 = symmetric ? O02 : qd(q,6);
//             const real_t O21 = symmetric ? O12 : qd(q,7);
//             const real_t O22 = symmetric ? qd(q,5) : qd(q,8);

//             const real_t grad0 = grad(0,qx,qy,qz);
//             const real_t grad1 = grad(1,qx,qy,qz);
//             const real_t grad2 = grad(2,qx,qy,qz);

//             grad(0,qx,qy,qz) = (O00*grad0)+(O01*grad1)+(O02*grad2);
//             grad(1,qx,qy,qz) = (O10*grad0)+(O11*grad1)+(O12*grad2);
//             grad(2,qx,qy,qz) = (O20*grad0)+(O21*grad1)+(O22*grad2);
//          } // qx
//       } // qy
//    } // qz

//    // Apply gradv^T
//    for (int qz = 0; qz < Q1D[2]; ++qz)
//    {
//       // sumXYv = 0.0;
//       for (int dy = 0; dy <= D1D[1]; ++dy)
//       {
//          for (int dx = 0; dx < D1D[0]; ++dx)
//          {
//             sumXY(0,dx,dy) = 0.0;
//             sumXY(1,dx,dy) = 0.0;
//             sumXY(2,dx,dy) = 0.0;
//          }
//       }

//       for (int qy = 0; qy < Q1D[1]; ++qy)
//       {
//          // sumXv = 0.0;
//          for (int dx = 0; dx < D1D[0]; ++dx)
//          {
//             sumX(0,dx) = 0.0;
//             sumX(1,dx) = 0.0;
//             sumX(2,dx) = 0.0;
//          }

//          for (int qx = 0; qx < Q1D[0]; ++qx)
//          {
//             const real_t gX = grad(0,qx,qy,qz);
//             const real_t gY = grad(1,qx,qy,qz);
//             const real_t gZ = grad(2,qx,qy,qz);

//             const int dx0 = minQ[0][qx];
//              // for (int dx = minQ[0][qx]; dx <= maxQ[0][qx]; ++dx)
//             for (int sx = 0; sx <= orders[0]; ++sx)
//             {
//                const int dx = sx + dx0;
//                const real_t wx  = Btx(sx,qx);
//                const real_t wDx = Gtx(sx,qx);
//                sumX(0,dx) += gX * wDx;
//                sumX(1,dx) += gY * wx;
//                sumX(2,dx) += gZ * wx;
//             }
//          }

//          const int dy0 = minQ[1][qy];
//          // for (int dy = minQ[1][qy]; dy <= maxQ[1][qy]; ++dy)
//          for (int sy = 0; sy <= orders[1]; ++sy)
//          {
//             const int dy = sy + dy0;
//             const real_t wy  = Bty(sy,qy);
//             const real_t wDy = Gty(sy,qy);
//             for (int dx = 0; dx < D1D[0]; ++dx)
//             {
//                sumXY(0,dx,dy) += sumX(0,dx) * wy;
//                sumXY(1,dx,dy) += sumX(1,dx) * wDy;
//                sumXY(2,dx,dy) += sumX(2,dx) * wy;
//             }
//          }
//       }

//       const int dz0 = minQ[2][qz];
//       // for (int dz = minQ[2][qz]; dz <= maxQ[2][qz]; ++dz)
//       for (int sz = 0; sz <= orders[2]; ++sz)
//       {
//          const int dz = sz + dz0;
//          const real_t wz  = Btz(sz,qz);
//          const real_t wDz = Gtz(sz,qz);
//          for (int dy = 0; dy < D1D[1]; ++dy)
//          {
//             for (int dx = 0; dx < D1D[0]; ++dx)
//             {
//                Y(dx,dy,dz) +=
//                   ((sumXY(0,dx,dy) * wz) +
//                    (sumXY(1,dx,dy) * wz) +
//                    (sumXY(2,dx,dy) * wDz));
//             }
//          }
//       } // dz
//    } // qz
// }

// Attempting to speed this up

void DiffusionIntegrator::AddMultPatchPA3D(const Vector &pa_data,
                                           const PatchBasisInfo &pb,
                                           const Vector &x,
                                           Vector &y) const
{
   // Unpack patch basis info
   const Array<int>& Q1D = pb.Q1D;
   const Array<int>& D1D = pb.D1D;
   const Array<int>& orders = pb.orders;

   const auto Btx = Reshape(pb.Btx.Read(), (orders[0]+1), Q1D[0]);
   const auto Gtx = Reshape(pb.Gtx.Read(), (orders[0]+1), Q1D[0]);
   const auto Bty = Reshape(pb.Bty.Read(), (orders[1]+1), Q1D[1]);
   const auto Gty = Reshape(pb.Gty.Read(), (orders[1]+1), Q1D[1]);
   const auto Btz = Reshape(pb.Btz.Read(), (orders[2]+1), Q1D[2]);
   const auto Gtz = Reshape(pb.Gtz.Read(), (orders[2]+1), Q1D[2]);

   const IntArrayVar2D& minQ = pb.minQ;

   const int NQ = pb.NQ;

   const auto X = Reshape(x.HostRead(), D1D[0], D1D[1], D1D[2]);
   auto Y = Reshape(y.HostReadWrite(), D1D[0], D1D[1], D1D[2]);

   Vector gradv(3*NQ);
   gradv = 0.0;
   auto grad = Reshape(gradv.HostReadWrite(), 3, Q1D[0], Q1D[1], Q1D[2]);

   const auto qd = Reshape(pa_data.HostRead(), NQ, (symmetric ? 6 : 9));

   // Accumulators; these are shared between grad_u interpolation and grad_v_T
   // application, so their size is the max of qpts/dofs
   Vector smemv(2*3*D1D[2]*Q1D[1]*Q1D[0]);
   auto smem = Reshape(smemv.HostReadWrite(), 2, 3, D1D[2], Q1D[1], Q1D[0]);

   // Interpolate grad_u
   // (adopted from hooke/kernel_helpers/CalcGrad)
   for (int dz = 0; dz < D1D[2]; ++dz)
   {
      for (int dy = 0; dy < D1D[1]; ++dy)
      {
         for (int dx = 0; dx < D1D[0]; ++dx)
         {
            smem(0,0,dx,dy,dz) = X(dx,dy,dz);
         }
      }
   }

   // dz dy qx dx
   for (int dz = 0; dz < D1D[2]; ++dz)
   {
      for (int dy = 0; dy < D1D[1]; ++dy)
      {
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            real_t u = 0.0;
            real_t v = 0.0;

            const int dx0 = minQ[0][qx];
            for (int sx = 0; sx <= orders[0]; ++sx)
            {
               const int dx = sx + dx0;
               const real_t input = smem(0,0,dx,dy,dz);
               u += input * Btx(sx,qx);
               v += input * Gtx(sx,qx);
            }
            smem(0,1,dz,dy,qx) = u;
            smem(0,2,dz,dy,qx) = v;
         }
      }
   }


   // dz qy qx dy
   for (int dz = 0; dz < D1D[2]; ++dz)
   {
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         const int dy0 = minQ[1][qy];
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            real_t u = 0.0;
            real_t v = 0.0;
            real_t w = 0.0;
            for (int sy = 0; sy <= orders[1]; ++sy)
            {
               const int dy = sy + dy0;
               const real_t wy  = Bty(sy,qy);
               const real_t wDy = Gty(sy,qy);
               u += smem(0,2,dz,dy,qx) * wy;
               v += smem(0,1,dz,dy,qx) * wDy;
               w += smem(0,1,dz,dy,qx) * wy;
            }
            smem(1,0,dz,qy,qx) = u;
            smem(1,1,dz,qy,qx) = v;
            smem(1,2,dz,qy,qx) = w;
         }
      }
   }

   // qz qy qx dz
   for (int qz = 0; qz < Q1D[2]; ++qz)
   {
      const int dz0 = minQ[2][qz];
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            real_t u = 0.0;
            real_t v = 0.0;
            real_t w = 0.0;
            for (int sz = 0; sz <= orders[2]; ++sz)
            {
               const int dz = sz + dz0;
               const real_t wz  = Btz(sz,qz);
               const real_t wDz = Gtz(sz,qz);
               u += smem(1,0,dz,qy,qx) * wz;
               v += smem(1,1,dz,qy,qx) * wz;
               w += smem(1,2,dz,qy,qx) * wDz;
            }
            grad(0,qx,qy,qz) += u;
            grad(1,qx,qy,qz) += v;
            grad(2,qx,qy,qz) += w;
         }
      }
   }

   // Apply kernel
   for (int qz = 0; qz < Q1D[2]; ++qz)
   {
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
            const real_t O00 = qd(q,0);
            const real_t O01 = qd(q,1);
            const real_t O02 = qd(q,2);
            const real_t O10 = symmetric ? O01 : qd(q,3);
            const real_t O11 = symmetric ? qd(q,3) : qd(q,4);
            const real_t O12 = symmetric ? qd(q,4) : qd(q,5);
            const real_t O20 = symmetric ? O02 : qd(q,6);
            const real_t O21 = symmetric ? O12 : qd(q,7);
            const real_t O22 = symmetric ? qd(q,5) : qd(q,8);

            const real_t grad0 = grad(0,qx,qy,qz);
            const real_t grad1 = grad(1,qx,qy,qz);
            const real_t grad2 = grad(2,qx,qy,qz);

            grad(0,qx,qy,qz) = (O00*grad0)+(O01*grad1)+(O02*grad2);
            grad(1,qx,qy,qz) = (O10*grad0)+(O11*grad1)+(O12*grad2);
            grad(2,qx,qy,qz) = (O20*grad0)+(O21*grad1)+(O22*grad2);
         } // qx
      } // qy
   } // qz

   // Apply gradv^T
   // TODO: This can be combined with smem
   Vector sumXYv(3*D1D[0]*D1D[1]);
   Vector sumXv(3*D1D[0]);
   auto sumXY = Reshape(sumXYv.HostReadWrite(), 3, D1D[0], D1D[1]);
   auto sumX = Reshape(sumXv.HostReadWrite(), 3, D1D[0]);
   for (int qz = 0; qz < Q1D[2]; ++qz)
   {
      sumXYv = 0.0;
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         sumXv = 0.0;
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            const real_t gX = grad(0,qx,qy,qz);
            const real_t gY = grad(1,qx,qy,qz);
            const real_t gZ = grad(2,qx,qy,qz);

            const int dx0 = minQ[0][qx];
            for (int sx = 0; sx <= orders[0]; ++sx)
            {
               const int dx = sx + dx0;
               const real_t wx  = Btx(sx,qx);
               const real_t wDx = Gtx(sx,qx);
               sumX(0,dx) += gX * wDx;
               sumX(1,dx) += gY * wx;
               sumX(2,dx) += gZ * wx;
            }
         }

         const int dy0 = minQ[1][qy];
         for (int sy = 0; sy <= orders[1]; ++sy)
         {
            const int dy = sy + dy0;
            const real_t wy  = Bty(sy,qy);
            const real_t wDy = Gty(sy,qy);
            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               sumXY(0,dx,dy) += sumX(0,dx) * wy;
               sumXY(1,dx,dy) += sumX(1,dx) * wDy;
               sumXY(2,dx,dy) += sumX(2,dx) * wy;
            }
         }
      }

      const int dz0 = minQ[2][qz];
      for (int sz = 0; sz <= orders[2]; ++sz)
      {
         const int dz = sz + dz0;
         const real_t wz  = Btz(sz,qz);
         const real_t wDz = Gtz(sz,qz);
         for (int dy = 0; dy < D1D[1]; ++dy)
         {
            for (int dx = 0; dx < D1D[0]; ++dx)
            {
               Y(dx,dy,dz) +=
                  ((sumXY(0,dx,dy) * wz) +
                   (sumXY(1,dx,dy) * wz) +
                   (sumXY(2,dx,dy) * wDz));
            }
         }
      } // dz
   } // qz

}


// This version uses full 1D quadrature rules, taking into account the
// minimum interaction between basis functions and integration points.
void DiffusionIntegrator::AddMultPatchPA(const int patch, const Vector &x,
                                         Vector &y) const
{
   if (dim == 3)
   {
      AddMultPatchPA3D(ppa_data[patch], pbinfo[patch], x, y);
   }
   else
   {
      MFEM_ABORT("Only 3D is supported.");
   }
}

void DiffusionIntegrator::AddMultNURBSPA(const Vector &x, Vector &y) const
{
   Vector xp, yp;

   for (int p=0; p<numPatches; ++p)
   {
      Array<int> vdofs;
      fespace->GetPatchVDofs(p, vdofs);

      x.GetSubVector(vdofs, xp);
      yp.SetSize(vdofs.Size());
      yp = 0.0;

      AddMultPatchPA(p, xp, yp);

      y.AddElementVector(vdofs, yp);
   }
}

} // namespace mfem
