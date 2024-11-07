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

#include "../quadinterpolator.hpp"
#include "grad.hpp"

#include "../../general/forall.hpp"

namespace mfem
{
namespace internal
{
namespace quadrature_interpolator
{

static void DynamicSmemDerivatives3D(const int NE,
                                     const double *b_,
                                     const double *g_,
                                     const double *x_,
                                     double *y_,
                                     const int vdim,
                                     const int d,
                                     const int q)
{
   const auto b = Reshape(b_, q, d);
   const auto g = Reshape(g_, q, d);
   const auto x = Reshape(x_, d, d, d, vdim, NE);

   auto y = Reshape(y_, q, q, q, vdim, 3, NE);

   const size_t smem_size = 2*q*d + d*d*d + 2*d*d*q + 3*d*q*q;

   mfem::forall_3D(NE, q,q,q, smem_size,
                   [=] MFEM_HOST_DEVICE (int e, double *sm)
   {
      auto sB = GetSmem(sm, q*d), sG = GetSmem(sm, q*d);
      DeviceMatrix B(sB, q, d), G(sG, q, d);
      kernels::internal::LoadBG(d,q,b,g,B,G);

      auto sm0 = GetSmem(sm, d*d*q), sm1 = GetSmem(sm, d*d*q);
      auto sm2 = GetSmem(sm, d*d*d), sm3 = GetSmem(sm, d*q*q);
      auto sm4 = GetSmem(sm, d*q*q), sm5 = GetSmem(sm, d*q*q);

      DeviceCube DDQ0(sm0, d,d,q), DDQ1(sm1, d,d,q), X(sm2, d,d,d);
      DeviceCube DQQ0(sm3, d,q,q), DQQ1(sm4, d,q,q), DQQ2(sm5, d,q,q);

      for (int c = 0; c < vdim; ++c)
      {
         kernels::internal::LoadX(e,d,c,x,X);
         MFEM_FOREACH_THREAD(dz,z,d)
         {
            MFEM_FOREACH_THREAD(dy,y,d)
            {
               MFEM_FOREACH_THREAD(qx,x,q)
               {
                  double u = 0.0;
                  double v = 0.0;
                  for (int dx = 0; dx < d; ++dx)
                  {
                     const double input = X(dx,dy,dz);
                     u += input * B(qx,dx);
                     v += input * G(qx,dx);
                  }
                  DDQ0(dz,dy,qx) = u;
                  DDQ1(dz,dy,qx) = v;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz,z,d)
         {
            MFEM_FOREACH_THREAD(qy,y,q)
            {
               MFEM_FOREACH_THREAD(qx,x,q)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dy = 0; dy < d; ++dy)
                  {
                     u += DDQ1(dz,dy,qx) * B(qy,dy);
                     v += DDQ0(dz,dy,qx) * G(qy,dy);
                     w += DDQ0(dz,dy,qx) * B(qy,dy);
                  }
                  DQQ0(dz,qy,qx) = u;
                  DQQ1(dz,qy,qx) = v;
                  DQQ2(dz,qy,qx) = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,q)
         {
            MFEM_FOREACH_THREAD(qy,y,q)
            {
               MFEM_FOREACH_THREAD(qx,x,q)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dz = 0; dz < d; ++dz)
                  {
                     u += DQQ0(dz,qy,qx) * B(qz,dz);
                     v += DQQ1(dz,qy,qx) * B(qz,dz);
                     w += DQQ2(dz,qy,qx) * G(qz,dz);
                  }
                  y(qx,qy,qz,c,0,e) = u;
                  y(qx,qy,qz,c,1,e) = v;
                  y(qx,qy,qz,c,2,e) = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

// Tensor-product evaluation of quadrature point derivatives: dispatch function.
// Instantiation for the case QVectorLayout::byNODES.
template<>
void TensorDerivatives<QVectorLayout::byNODES>(const int NE,
                                               const int vdim,
                                               const DofToQuad &maps,
                                               const Vector &e_vec,
                                               Vector &q_der)
{
   using k = QuadratureInterpolator::GradKernels;
   constexpr auto L = QVectorLayout::byNODES;
   // 2D
   k::Specialization<2,L,P,1,3,3>::template Opt<16>::Add();
   k::Specialization<2,L,P,1,3,4>::template Opt<16>::Add();
   k::Specialization<2,L,P,1,4,3>::template Opt<16>::Add();
   k::Specialization<2,L,P,1,4,4>::template Opt<16>::Add();

   k::Specialization<2,L,P,2,2,2>::template Opt<16>::Add();
   k::Specialization<2,L,P,2,2,3>::template Opt<8>::Add();
   k::Specialization<2,L,P,2,2,4>::template Opt<4>::Add();
   k::Specialization<2,L,P,2,2,5>::template Opt<4>::Add();
   k::Specialization<2,L,P,2,2,6>::template Opt<2>::Add();

   k::Specialization<2,L,P,2,3,3>::template Opt<2>::Add();
   k::Specialization<2,L,P,2,3,4>::template Opt<4>::Add();
   k::Specialization<2,L,P,2,4,3>::template Opt<4>::Add();
   k::Specialization<2,L,P,2,3,6>::template Opt<2>::Add();

   k::Specialization<2,L,P,2,4,4>::template Opt<2>::Add();
   k::Specialization<2,L,P,2,4,5>::template Opt<2>::Add();
   k::Specialization<2,L,P,2,4,6>::template Opt<2>::Add();
   k::Specialization<2,L,P,2,4,7>::template Opt<2>::Add();

   k::Specialization<2,L,P,2,5,6>::template Opt<2>::Add();
   // 3D
   k::Specialization<3,L,P,1,2,4>::Add();
   k::Specialization<3,L,P,1,3,3>::Add();
   k::Specialization<3,L,P,1,3,4>::Add();
   k::Specialization<3,L,P,1,3,6>::Add();
   k::Specialization<3,L,P,1,4,4>::Add();
   k::Specialization<3,L,P,1,4,8>::Add();

   k::Specialization<3,L,P,3,2,3>::Add();
   k::Specialization<3,L,P,3,2,4>::Add();
   k::Specialization<3,L,P,3,2,5>::Add();
   k::Specialization<3,L,P,3,2,6>::Add();

   k::Specialization<3,L,P,3,3,3>::Add();
   k::Specialization<3,L,P,3,3,4>::Add();
   k::Specialization<3,L,P,3,3,5>::Add();
   k::Specialization<3,L,P,3,3,6>::Add();
   k::Specialization<3,L,P,3,4,4>::Add();
   k::Specialization<3,L,P,3,4,6>::Add();
   k::Specialization<3,L,P,3,4,7>::Add();
   k::Specialization<3,L,P,3,4,8>::Add();

   using k2 = QuadratureInterpolator::CollocatedGradKernels;

   // 2D
   k2::Specialization<2,L,P,1,2>::template Opt<16>::Add();
   k2::Specialization<2,L,P,1,3>::template Opt<16>::Add();
   k2::Specialization<2,L,P,1,4>::template Opt<16>::Add();
   k2::Specialization<2,L,P,2,2>::template Opt<16>::Add();
   k2::Specialization<2,L,P,2,3>::template Opt<4>::Add();
   k2::Specialization<2,L,P,2,4>::template Opt<2>::Add();

   k2::Specialization<3,L,P,1,2>::Add();
   k2::Specialization<3,L,P,1,3>::Add();
   k2::Specialization<3,L,P,1,4>::Add();

   k2::Specialization<3,L,P,2,2>::Add();
   k2::Specialization<3,L,P,2,3>::Add();
   k2::Specialization<3,L,P,2,4>::Add();

   k2::Specialization<3,L,P,3,2>::Add();
   k2::Specialization<3,L,P,3,3>::Add();
   k2::Specialization<3,L,P,3,4>::Add();
}

template void InitGradByNodesKernels<true>();
template void InitGradByNodesKernels<false>();
if (NE == 0) { return; }
const int dim = maps.FE->GetDim();
const int D1D = maps.ndof;
const int Q1D = maps.nqpt;
const real_t *B = maps.B.Read();
const real_t *G = maps.G.Read();
const real_t *J = nullptr; // not used in DERIVATIVES (non-GRAD_PHYS) mode
const real_t *X = e_vec.Read();
real_t *Y = q_der.Write();

constexpr QVectorLayout L = QVectorLayout::byNODES;
constexpr bool P = false; // GRAD_PHYS

const int id = (vdim<<8) | (D1D<<4) | Q1D;

if (dim == 1)
{
   return Derivatives1D<L,P>(NE,G,J,X,Y,dim,vdim,D1D,Q1D);
}
if (dim == 2)
{
   switch (id)
   {
      case 0x133: return Derivatives2D<L,P,1,3,3,16>(NE,B,G,J,X,Y);
      case 0x134: return Derivatives2D<L,P,1,3,4,16>(NE,B,G,J,X,Y);
      case 0x143: return Derivatives2D<L,P,1,4,3,16>(NE,B,G,J,X,Y);
      case 0x144: return Derivatives2D<L,P,1,4,4,16>(NE,B,G,J,X,Y);

      case 0x222: return Derivatives2D<L,P,2,2,2,16>(NE,B,G,J,X,Y);
      case 0x223: return Derivatives2D<L,P,2,2,3,8>(NE,B,G,J,X,Y);
      case 0x224: return Derivatives2D<L,P,2,2,4,4>(NE,B,G,J,X,Y);
      case 0x225: return Derivatives2D<L,P,2,2,5,4>(NE,B,G,J,X,Y);
      case 0x226: return Derivatives2D<L,P,2,2,6,2>(NE,B,G,J,X,Y);

      case 0x233: return Derivatives2D<L,P,2,3,3,2>(NE,B,G,J,X,Y);
      case 0x234: return Derivatives2D<L,P,2,3,4,4>(NE,B,G,J,X,Y);
      case 0x243: return Derivatives2D<L,P,2,4,3,4>(NE,B,G,J,X,Y);
      case 0x236: return Derivatives2D<L,P,2,3,6,2>(NE,B,G,J,X,Y);

      case 0x244: return Derivatives2D<L,P,2,4,4,2>(NE,B,G,J,X,Y);
      case 0x245: return Derivatives2D<L,P,2,4,5,2>(NE,B,G,J,X,Y);
      case 0x246: return Derivatives2D<L,P,2,4,6,2>(NE,B,G,J,X,Y);
      case 0x247: return Derivatives2D<L,P,2,4,7,2>(NE,B,G,J,X,Y);

      case 0x256: return Derivatives2D<L,P,2,5,6,2>(NE,B,G,J,X,Y);
      default:
      {
         const int MD = DeviceDofQuadLimits::Get().MAX_D1D;
         const int MQ = DeviceDofQuadLimits::Get().MAX_Q1D;
         if (D1D > MD || Q1D > MQ)
         {
            MFEM_ABORT("");
         }
         Derivatives2D<L,P>(NE,B,G,J,X,Y,dim,vdim,D1D,Q1D);
         return;
      }
   }
}
if (dim == 3)
{
   switch (id)
   {
      case 0x124: return Derivatives3D<L,P,1,2,4>(NE,B,G,J,X,Y);
      case 0x133: return Derivatives3D<L,P,1,3,3>(NE,B,G,J,X,Y);
      case 0x134: return Derivatives3D<L,P,1,3,4>(NE,B,G,J,X,Y);
      case 0x136: return Derivatives3D<L,P,1,3,6>(NE,B,G,J,X,Y);
      case 0x144: return Derivatives3D<L,P,1,4,4>(NE,B,G,J,X,Y);
      case 0x148: return Derivatives3D<L,P,1,4,8>(NE,B,G,J,X,Y);

      case 0x323: return Derivatives3D<L,P,3,2,3>(NE,B,G,J,X,Y);
      case 0x324: return Derivatives3D<L,P,3,2,4>(NE,B,G,J,X,Y);
      case 0x325: return Derivatives3D<L,P,3,2,5>(NE,B,G,J,X,Y);
      case 0x326: return Derivatives3D<L,P,3,2,6>(NE,B,G,J,X,Y);

      case 0x333: return Derivatives3D<L,P,3,3,3>(NE,B,G,J,X,Y);
      case 0x334: return Derivatives3D<L,P,3,3,4>(NE,B,G,J,X,Y);
      case 0x335: return Derivatives3D<L,P,3,3,5>(NE,B,G,J,X,Y);
      case 0x336: return Derivatives3D<L,P,3,3,6>(NE,B,G,J,X,Y);
      case 0x344: return Derivatives3D<L,P,3,4,4>(NE,B,G,J,X,Y);
      case 0x346: return Derivatives3D<L,P,3,4,6>(NE,B,G,J,X,Y);
      case 0x347: return Derivatives3D<L,P,3,4,7>(NE,B,G,J,X,Y);
      case 0x348: return Derivatives3D<L,P,3,4,8>(NE,B,G,J,X,Y);
      default:
#if 0
      {
         const int MD = DeviceDofQuadLimits::Get().MAX_INTERP_1D;
         const int MQ = DeviceDofQuadLimits::Get().MAX_INTERP_1D;
         MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                     << " are not supported!");
         MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than "
                     << MQ << " 1D points are not supported!");
         Derivatives3D<L,P>(NE,B,G,J,X,Y,vdim,D1D,Q1D);
         return;
      }
#else
      return DynamicSmemDerivatives3D(NE,B,G,X,Y,vdim,D1D,Q1D);
#endif
   }
}
mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
MFEM_ABORT("Kernel not supported yet");
}

} // namespace quadrature_interpolator
} // namespace internal
} // namespace mfem
