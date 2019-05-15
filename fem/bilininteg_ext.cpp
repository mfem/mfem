// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"

#include <map>
#include <cmath>
#include <algorithm>
#include <unordered_map>

using namespace std;

namespace mfem
{

#ifdef MFEM_USE_OCCA
typedef std::pair<int,int> id_t;
typedef std::map<id_t, occa::kernel> occa_kernel_t;
#endif // MFEM_USE_OCCA

static const IntegrationRule &DefaultGetRule(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe)
{
   int order;
   if (trial_fe.Space() == FunctionSpace::Pk)
   {
      order = trial_fe.GetOrder() + test_fe.GetOrder() - 2;
   }
   else
   {
      // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
      order = trial_fe.GetOrder() + test_fe.GetOrder() + trial_fe.GetDim() - 1;
   }
   if (trial_fe.Space() == FunctionSpace::rQk)
   {
      return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   }
   return IntRules.Get(trial_fe.GetGeomType(), order);
}

// PA Diffusion Integrator

// OCCA 2D Assemble kernel
#ifdef MFEM_USE_OCCA
static void OccaPADiffusionSetup2D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const double *W,
                                   const double *J,
                                   const double COEFF,
                                   double *op)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_W = mfem::OccaPtr(W);
   const occa::memory o_J = mfem::OccaPtr(J);
   occa::memory o_op = mfem::OccaPtr(op);
   const id_t id = std::make_pair(D1D,Q1D);
   static occa_kernel_t OccaDiffSetup2D_ker;
   if (OccaDiffSetup2D_ker.find(id) == OccaDiffSetup2D_ker.end())
   {
      const occa::kernel DiffusionSetup2D =
         mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                     "DiffusionSetup2D", props);
      OccaDiffSetup2D_ker.emplace(id, DiffusionSetup2D);
   }
   OccaDiffSetup2D_ker.at(id)(NE, o_W, o_J, COEFF, o_op);
}

static void OccaPADiffusionSetup3D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const double *W,
                                   const double *J,
                                   const double COEFF,
                                   double *op)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_W = mfem::OccaPtr(W);
   const occa::memory o_J = mfem::OccaPtr(J);
   occa::memory o_op = mfem::OccaPtr(op);
   const id_t id = std::make_pair(D1D,Q1D);
   static occa_kernel_t OccaDiffSetup3D_ker;
   if (OccaDiffSetup3D_ker.find(id) == OccaDiffSetup3D_ker.end())
   {
      const occa::kernel DiffusionSetup3D =
         mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                     "DiffusionSetup3D", props);
      OccaDiffSetup3D_ker.emplace(id, DiffusionSetup3D);
   }
   OccaDiffSetup3D_ker.at(id)(NE, o_W, o_J, COEFF, o_op);
}
#endif // MFEM_USE_OCCA

// PA Diffusion Assemble 2D kernel
static void PADiffusionSetup2D(const int Q1D,
                               const int NE,
                               const double* w,
                               const double* j,
                               const double COEFF,
                               double* op)
{
   const int NQ = Q1D*Q1D;
   const DeviceVector W(w, NQ);
   const DeviceTensor<4> J(j, 2, 2, NQ, NE);
   DeviceTensor<3> y(op, 3, NQ, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(0,0,q,e);
         const double J12 = J(1,0,q,e);
         const double J21 = J(0,1,q,e);
         const double J22 = J(1,1,q,e);
         const double c_detJ = W(q) * COEFF / ((J11*J22)-(J21*J12));
         y(0,q,e) =  c_detJ * (J21*J21 + J22*J22);
         y(1,q,e) = -c_detJ * (J21*J11 + J22*J12);
         y(2,q,e) =  c_detJ * (J11*J11 + J12*J12);
      }
   });
}

// PA Diffusion Assemble 3D kernel
static void PADiffusionSetup3D(const int Q1D,
                               const int NE,
                               const double* w,
                               const double* j,
                               const double COEFF,
                               double* op)
{
   const int NQ = Q1D*Q1D*Q1D;
   const DeviceVector W(w, NQ);
   const DeviceTensor<4> J(j, 3, 3, NQ, NE);
   DeviceTensor<3> y(op, 6, NQ, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(0,0,q,e);
         const double J12 = J(1,0,q,e);
         const double J13 = J(2,0,q,e);
         const double J21 = J(0,1,q,e);
         const double J22 = J(1,1,q,e);
         const double J23 = J(2,1,q,e);
         const double J31 = J(0,2,q,e);
         const double J32 = J(1,2,q,e);
         const double J33 = J(2,2,q,e);
         const double detJ =
         ((J11 * J22 * J33) + (J12 * J23 * J31) +
         (J13 * J21 * J32) - (J13 * J22 * J31) -
         (J12 * J21 * J33) - (J11 * J23 * J32));
         const double c_detJ = W(q) * COEFF / detJ;
         // adj(J)
         const double A11 = (J22 * J33) - (J23 * J32);
         const double A12 = (J23 * J31) - (J21 * J33);
         const double A13 = (J21 * J32) - (J22 * J31);
         const double A21 = (J13 * J32) - (J12 * J33);
         const double A22 = (J11 * J33) - (J13 * J31);
         const double A23 = (J12 * J31) - (J11 * J32);
         const double A31 = (J12 * J23) - (J13 * J22);
         const double A32 = (J13 * J21) - (J11 * J23);
         const double A33 = (J11 * J22) - (J12 * J21);
         // adj(J)^Tadj(J)
         y(0,q,e) = c_detJ * (A11*A11 + A21*A21 + A31*A31);
         y(1,q,e) = c_detJ * (A11*A12 + A21*A22 + A31*A32);
         y(2,q,e) = c_detJ * (A11*A13 + A21*A23 + A31*A33);
         y(3,q,e) = c_detJ * (A12*A12 + A22*A22 + A32*A32);
         y(4,q,e) = c_detJ * (A12*A13 + A22*A23 + A32*A33);
         y(5,q,e) = c_detJ * (A13*A13 + A23*A23 + A33*A33);
      }
   });
}

namespace internal
{

#ifdef MFEM_USE_OCCA
// This function is currently used to determine if an OCCA kernel should be
// used.
static bool DeviceUseOcca()
{
   return Device::Allows(Backend::OCCA_CUDA) ||
          (Device::Allows(Backend::OCCA_OMP) &&
           !Device::Allows(Backend::DEVICE_MASK)) ||
          (Device::Allows(Backend::OCCA_CPU) &&
           !Device::Allows(Backend::DEVICE_MASK|Backend::OMP_MASK));
}
#endif

}

static void PADiffusionSetup(const int dim,
                             const int D1D,
                             const int Q1D,
                             const int NE,
                             const double* W,
                             const double* J,
                             const double COEFF,
                             double* op)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADiffusionSetup"); }
   if (dim == 2)
   {
#ifdef MFEM_USE_OCCA
      if (internal::DeviceUseOcca())
      {
         OccaPADiffusionSetup2D(D1D, Q1D, NE, W, J, COEFF, op);
         return;
      }
#endif // MFEM_USE_OCCA
      PADiffusionSetup2D(Q1D, NE, W, J, COEFF, op);
   }
   if (dim == 3)
   {
#ifdef MFEM_USE_OCCA
      if (internal::DeviceUseOcca())
      {
         OccaPADiffusionSetup3D(D1D, Q1D, NE, W, J, COEFF, op);
         return;
      }
#endif // MFEM_USE_OCCA
      PADiffusionSetup3D(Q1D, NE, W, J, COEFF, op);
   }
}

void DiffusionIntegrator::Assemble(const FiniteElementSpace &fes)
{
   const Mesh *mesh = fes.GetMesh();
   const IntegrationRule *rule = IntRule;
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = rule?rule:&DefaultGetRule(el,el);
   const int dims = el.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes.GetNE();
   dofs1D = el.GetOrder() + 1;
   quad1D = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
   geom = GeometryExtension::Get(fes,*ir);
   maps = DofToQuad::Get(fes, fes, *ir);
   vec.SetSize(symmDims * nq * ne);
   const double coeff = static_cast<ConstantCoefficient*>(Q)->constant;
   PADiffusionSetup(dim, dofs1D, quad1D, ne, maps->W, geom->J, coeff, vec);
}

#ifdef MFEM_USE_OCCA
// OCCA PA Diffusion Apply 2D kernel
static void OccaPADiffusionApply2D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const double* B,
                                   const double* G,
                                   const double* Bt,
                                   const double* Gt,
                                   const double* op,
                                   const double* x,
                                   double* y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = mfem::OccaPtr(B);
   const occa::memory o_G = mfem::OccaPtr(G);
   const occa::memory o_Bt = mfem::OccaPtr(Bt);
   const occa::memory o_Gt = mfem::OccaPtr(Gt);
   const occa::memory o_op = mfem::OccaPtr(op);
   const occa::memory o_x = mfem::OccaPtr(x);
   occa::memory o_y = mfem::OccaPtr(y);
   const id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaDiffApply2D_cpu;
      if (OccaDiffApply2D_cpu.find(id) == OccaDiffApply2D_cpu.end())
      {
         const occa::kernel DiffusionApply2D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply2D_CPU", props);
         OccaDiffApply2D_cpu.emplace(id, DiffusionApply2D_CPU);
      }
      OccaDiffApply2D_cpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_op, o_x, o_y);
   }
   else
   {
      static occa_kernel_t OccaDiffApply2D_gpu;
      if (OccaDiffApply2D_gpu.find(id) == OccaDiffApply2D_gpu.end())
      {
         const occa::kernel DiffusionApply2D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply2D_GPU", props);
         OccaDiffApply2D_gpu.emplace(id, DiffusionApply2D_GPU);
      }
      OccaDiffApply2D_gpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_op, o_x, o_y);
   }
}

// OCCA PA Diffusion Apply 3D kernel
static void OccaPADiffusionApply3D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const double* B,
                                   const double* G,
                                   const double* Bt,
                                   const double* Gt,
                                   const double* op,
                                   const double* x,
                                   double* y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = mfem::OccaPtr(B);
   const occa::memory o_G = mfem::OccaPtr(G);
   const occa::memory o_Bt = mfem::OccaPtr(Bt);
   const occa::memory o_Gt = mfem::OccaPtr(Gt);
   const occa::memory o_op = mfem::OccaPtr(op);
   const occa::memory o_x = mfem::OccaPtr(x);
   occa::memory o_y = mfem::OccaPtr(y);
   const id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaDiffApply3D_cpu;
      if (OccaDiffApply3D_cpu.find(id) == OccaDiffApply3D_cpu.end())
      {
         const occa::kernel DiffusionApply3D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply3D_CPU", props);
         OccaDiffApply3D_cpu.emplace(id, DiffusionApply3D_CPU);
      }
      OccaDiffApply3D_cpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_op, o_x, o_y);
   }
   else
   {
      static occa_kernel_t OccaDiffApply3D_gpu;
      if (OccaDiffApply3D_gpu.find(id) == OccaDiffApply3D_gpu.end())
      {
         const occa::kernel DiffusionApply3D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply3D_GPU", props);
         OccaDiffApply3D_gpu.emplace(id, DiffusionApply3D_GPU);
      }
      OccaDiffApply3D_gpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_op, o_x, o_y);
   }
}
#endif // MFEM_USE_OCCA

// Shared memory PA Diffusion Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0, const int T_NBZ = 0> static
void PADiffusionApply2D(const int NE,
                        const double* _b,
                        const double* _g,
                        const double* _bt,
                        const double* _gt,
                        const double* _op,
                        const double* _x,
                        double* _y,
                        const int d1d = 0,
                        const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int NBZ = T_NBZ ? T_NBZ : 1;
   const int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   const int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(MQ1 == MD1, "");
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   const DeviceMatrix b(_b, Q1D, D1D);
   const DeviceMatrix g(_g, Q1D, D1D);
   const DeviceMatrix bt(_bt, D1D, Q1D);
   const DeviceMatrix gt(_gt, D1D, Q1D);
   const DeviceTensor<3> op(_op, 3, Q1D*Q1D, NE);
   const DeviceTensor<3> x(_x, D1D, D1D, NE);
   DeviceTensor<3> y(_y, D1D, D1D, NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int NBZ = T_NBZ ? T_NBZ : 1;
      const int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      const int MD1 = T_D1D ? T_D1D : MAX_D1D;
      MFEM_SHARED double B[MQ1][MD1];
      MFEM_SHARED double G[MQ1][MD1];
      MFEM_SHARED double Bt[MD1][MQ1];
      MFEM_SHARED double Gt[MD1][MQ1];
      MFEM_SHARED double Xz[NBZ][MD1][MD1];
      MFEM_SHARED double GD[2][NBZ][MD1][MQ1];
      MFEM_SHARED double GQ[2][NBZ][MD1][MQ1];
      double (*X)[MD1] = (double (*)[MD1])(Xz + threadIdx(z));
      double (*DQ0)[MD1] = (double (*)[MD1])(GD[0] + threadIdx(z));
      double (*DQ1)[MD1] = (double (*)[MD1])(GD[1] + threadIdx(z));
      double (*QQ0)[MD1] = (double (*)[MD1])(GQ[0] + threadIdx(z));
      double (*QQ1)[MD1] = (double (*)[MD1])(GQ[1] + threadIdx(z));
      for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
      {
         for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
         {
            X[dy][dx] = x(dx,dy,e);
         }
      }
      if (threadIdx(z) == 0)
      {
         for (int dx = threadIdx(y); dx < D1D; dx += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               B[qx][dx] = b(qx,dx);
               G[qx][dx] = g(qx,dx);
               Bt[dx][qx] = bt(dx,qx);
               Gt[dx][qx] = gt(dx,qx);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            double u = 0.0;
            double v = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double coords = X[dy][dx];
               u += B[qx][dx] * coords;
               v += G[qx][dx] * coords;
            }
            DQ0[dy][qx] = u;
            DQ1[dy][qx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            double u = 0.0;
            double v = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               u += DQ1[dy][qx] * B[qy][dy];
               v += DQ0[dy][qx] * G[qy][dy];
            }
            QQ0[qy][qx] = u;
            QQ1[qy][qx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            const int q = (qx + ((qy) * Q1D));
            const double O11 = op(0,q,e);
            const double O12 = op(1,q,e);
            const double O22 = op(2,q,e);
            const double gX = QQ0[qy][qx];
            const double gY = QQ1[qy][qx];
            QQ0[qy][qx] = (O11 * gX) + (O12 * gY);
            QQ1[qy][qx] = (O12 * gX) + (O22 * gY);
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
         {
            double u = 0.0;
            double v = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               u += Gt[dx][qx] * QQ0[qy][qx];
               v += Bt[dx][qx] * QQ1[qy][qx];
            }
            DQ0[qy][dx] = u;
            DQ1[qy][dx] = v;
         }
      }
      MFEM_SYNC_THREAD;
      for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
      {
         for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
         {
            double u = 0.0;
            double v = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               u += DQ0[qy][dx] * Bt[dy][qy];
               v += DQ1[qy][dx] * Gt[dy][qy];
            }
            y(dx,dy,e) += (u + v);
         }
      }
   });
}

// Shared memory PA Diffusion Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0> static
void PADiffusionApply3D(const int NE,
                        const double* _b,
                        const double* _g,
                        const double* _bt,
                        const double* _gt,
                        const double* _op,
                        const double* _x,
                        double* _y,
                        const int d1d = 0,
                        const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   const int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   const DeviceMatrix b(_b, Q1D, D1D);
   const DeviceMatrix g(_g, Q1D, D1D);
   const DeviceMatrix bt(_bt, D1D, Q1D);
   const DeviceMatrix gt(_gt, D1D, Q1D);
   const DeviceTensor<3> op(_op, 6, Q1D*Q1D*Q1D, NE);
   const DeviceTensor<4> x(_x, D1D, D1D, D1D, NE);
   DeviceTensor<4> y(_y, D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      const int MD1 = T_D1D ? T_D1D : MAX_D1D;
      const int MDQ = MQ1 > MD1 ? MQ1 : MD1;
      MFEM_SHARED double sBG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (sBG+0);
      double (*G)[MD1] = (double (*)[MD1]) (sBG+1);
      double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
      double (*Gt)[MQ1] = (double (*)[MQ1]) (sBG+1);
      MFEM_SHARED double sm0[3][MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[3][MDQ*MDQ*MDQ];
      double (*X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0+2);
      double (*DDQ0)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+0);
      double (*DDQ1)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+1);
      double (*DQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+0);
      double (*DQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+1);
      double (*DQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+2);
      double (*QQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+0);
      double (*QQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+1);
      double (*QQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+2);
      double (*QQD0)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+0);
      double (*QQD1)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+1);
      double (*QQD2)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+2);
      double (*QDD0)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+0);
      double (*QDD1)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+1);
      double (*QDD2)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+2);
      for (int dz = threadIdx(z); dz < D1D; dz += blockDim(z))
      {
         for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               X[dz][dy][dx] = x(dx,dy,dz,e);
            }
         }
      }
      if (threadIdx(z) == 0)
      {
         for (int dx = threadIdx(y); dx < D1D; dx += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               B[qx][dx] = b(qx,dx);
               G[qx][dx] = g(qx,dx);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int dz = threadIdx(z); dz < D1D; dz += blockDim(z))
      {
         for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               double u = 0.0;
               double v = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double coords = X[dz][dy][dx];
                  u += coords * B[qx][dx];
                  v += coords * G[qx][dx];
               }
               DDQ0[dz][dy][qx] = u;
               DDQ1[dz][dy][qx] = v;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int dz = threadIdx(z); dz < D1D; dz += blockDim(z))
      {
         for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  u += DDQ1[dz][dy][qx] * B[qy][dy];
                  v += DDQ0[dz][dy][qx] * G[qy][dy];
                  w += DDQ0[dz][dy][qx] * B[qy][dy];
               }
               DQQ0[dz][qy][qx] = u;
               DQQ1[dz][qy][qx] = v;
               DQQ2[dz][qy][qx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int qz = threadIdx(z); qz < Q1D; qz += blockDim(z))
      {
         for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int dz = 0; dz < D1D; ++dz)
               {
                  u += DQQ0[dz][qy][qx] * B[qz][dz];
                  v += DQQ1[dz][qy][qx] * B[qz][dz];
                  w += DQQ2[dz][qy][qx] * G[qz][dz];
               }
               QQQ0[qz][qy][qx] = u;
               QQQ1[qz][qy][qx] = v;
               QQQ2[qz][qy][qx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int qz = threadIdx(z); qz < Q1D; qz += blockDim(z))
      {
         for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               const int q = qx + ((qy*Q1D) + (qz*Q1D*Q1D));
               const double O11 = op(0,q,e);
               const double O12 = op(1,q,e);
               const double O13 = op(2,q,e);
               const double O22 = op(3,q,e);
               const double O23 = op(4,q,e);
               const double O33 = op(5,q,e);
               const double gX = QQQ0[qz][qy][qx];
               const double gY = QQQ1[qz][qy][qx];
               const double gZ = QQQ2[qz][qy][qx];
               QQQ0[qz][qy][qx] = (O11*gX) + (O12*gY) + (O13*gZ);
               QQQ1[qz][qy][qx] = (O12*gX) + (O22*gY) + (O23*gZ);
               QQQ2[qz][qy][qx] = (O13*gX) + (O23*gY) + (O33*gZ);
            }
         }
      }
      MFEM_SYNC_THREAD;
      if (threadIdx(z) == 0)
      {
         for (int dx = threadIdx(y); dx < D1D; dx += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               Bt[dx][qx] = bt(dx,qx);
               Gt[dx][qx] = gt(dx,qx);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int qz = threadIdx(z); qz < Q1D; qz += blockDim(z))
      {
         for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += QQQ0[qz][qy][qx] * Gt[dx][qx];
                  v += QQQ1[qz][qy][qx] * Bt[dx][qx];
                  w += QQQ2[qz][qy][qx] * Bt[dx][qx];
               }
               QQD0[qz][qy][dx] = u;
               QQD1[qz][qy][dx] = v;
               QQD2[qz][qy][dx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int qz = threadIdx(z); qz < Q1D; qz += blockDim(z))
      {
         for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += QQD0[qz][qy][dx] * Bt[dy][qy];
                  v += QQD1[qz][qy][dx] * Gt[dy][qy];
                  w += QQD2[qz][qy][dx] * Bt[dy][qy];
               }
               QDD0[qz][dy][dx] = u;
               QDD1[qz][dy][dx] = v;
               QDD2[qz][dy][dx] = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int dz = threadIdx(z); dz < D1D; dz += blockDim(z))
      {
         for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               double u = 0.0;
               double v = 0.0;
               double w = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  u += QDD0[qz][dy][dx] * Bt[dz][qz];
                  v += QDD1[qz][dy][dx] * Bt[dz][qz];
                  w += QDD2[qz][dy][dx] * Gt[dz][qz];
               }
               y(dx,dy,dz,e) += (u + v + w);
            }
         }
      }
   });
}

static void PADiffusionApply(const int dim,
                             const int D1D,
                             const int Q1D,
                             const int NE,
                             const double* B,
                             const double* G,
                             const double* Bt,
                             const double* Gt,
                             const double* op,
                             const double* x,
                             double* y)
{
#ifdef MFEM_USE_OCCA
   if (internal::DeviceUseOcca())
   {
      if (dim == 2)
      {
         OccaPADiffusionApply2D(D1D, Q1D, NE, B, G, Bt, Gt, op, x, y);
         return;
      }
      if (dim == 3)
      {
         OccaPADiffusionApply3D(D1D, Q1D, NE, B, G, Bt, Gt, op, x, y);
         return;
      }
      MFEM_ABORT("OCCA PADiffusionApply unknown kernel!");
   }
#endif // MFEM_USE_OCCA
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return PADiffusionApply2D<2,2,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x33: return PADiffusionApply2D<3,3,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x44: return PADiffusionApply2D<4,4,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x55: return PADiffusionApply2D<5,5,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x66: return PADiffusionApply2D<6,6,2>(NE,B,G,Bt,Gt,op,x,y);
         case 0x77: return PADiffusionApply2D<7,7,2>(NE,B,G,Bt,Gt,op,x,y);
         case 0x88: return PADiffusionApply2D<8,8,1>(NE,B,G,Bt,Gt,op,x,y);
         case 0x99: return PADiffusionApply2D<9,9,1>(NE,B,G,Bt,Gt,op,x,y);
         default:   return PADiffusionApply2D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
      }
   }
   if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return PADiffusionApply3D<2,3>(NE,B,G,Bt,Gt,op,x,y);
         case 0x34: return PADiffusionApply3D<3,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x45: return PADiffusionApply3D<4,5>(NE,B,G,Bt,Gt,op,x,y);
         case 0x56: return PADiffusionApply3D<5,6>(NE,B,G,Bt,Gt,op,x,y);
         case 0x67: return PADiffusionApply3D<6,7>(NE,B,G,Bt,Gt,op,x,y);
         case 0x78: return PADiffusionApply3D<7,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x89: return PADiffusionApply3D<8,9>(NE,B,G,Bt,Gt,op,x,y);
         default:   return PADiffusionApply3D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

// PA Diffusion Apply kernel
void DiffusionIntegrator::MultAssembled(Vector &x, Vector &y)
{
   PADiffusionApply(dim, dofs1D, quad1D, ne,
                    maps->B, maps->G, maps->Bt, maps->Gt,
                    vec, x, y);
}

DiffusionIntegrator::~DiffusionIntegrator()
{
   delete geom;
   delete maps;
}

// PA Mass Assemble kernel
void MassIntegrator::Assemble(const FiniteElementSpace &fes)
{
   const Mesh *mesh = fes.GetMesh();
   const IntegrationRule *rule = IntRule;
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = rule?rule:&DefaultGetRule(el,el);
   dim = mesh->Dimension();
   ne = fes.GetMesh()->GetNE();
   nq = ir->GetNPoints();
   dofs1D = el.GetOrder() + 1;
   quad1D = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
   geom = GeometryExtension::Get(fes,*ir);
   maps = DofToQuad::Get(fes, fes, *ir);
   vec.SetSize(ne*nq);
   ConstantCoefficient *const_coeff = dynamic_cast<ConstantCoefficient*>(Q);
   FunctionCoefficient *function_coeff = dynamic_cast<FunctionCoefficient*>(Q);
   // TODO: other types of coefficients ...
   if (dim==1) { MFEM_ABORT("Not supported yet... stay tuned!"); }
   if (dim==2)
   {
      double constant = 0.0;
      double (*function)(const Vector3&) = NULL;
      if (const_coeff)
      {
         constant = const_coeff->constant;
      }
      else if (function_coeff)
      {
         function = function_coeff->GetDeviceFunction();
      }
      else
      {
         MFEM_ABORT("Coefficient type not supported");
      }
      const int NE = ne;
      const int NQ = nq;
      const DeviceVector w(maps->W.GetData(), NQ);
      const DeviceTensor<3> x(geom->X.GetData(), 2,NQ,NE);
      const DeviceTensor<4> J(geom->J.GetData(), 2,2,NQ,NE);
      DeviceMatrix v(vec.GetData(), NQ, NE);
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double J11 = J(0,0,q,e);
            const double J12 = J(1,0,q,e);
            const double J21 = J(0,1,q,e);
            const double J22 = J(1,1,q,e);
            const double detJ = (J11*J22)-(J21*J12);
            const Vector3 Xq(x(0,q,e), x(1,q,e));
            const double coeff =
            const_coeff ? constant
            : function_coeff ? function(Xq)
            : 0.0;
            v(q,e) =  w[q] * coeff * detJ;
         }
      });
   }
   if (dim==3)
   {
      double constant = 0.0;
      double (*function)(const Vector3&) = NULL;
      if (const_coeff)
      {
         constant = const_coeff->constant;
      }
      else if (function_coeff)
      {
         function = function_coeff->GetDeviceFunction();
      }
      else
      {
         MFEM_ABORT("Coefficient type not supported");
      }
      const int NE = ne;
      const int NQ = nq;
      const DeviceVector W(maps->W.GetData(), NQ);
      const DeviceTensor<3> x(geom->X.GetData(), 3,NQ,NE);
      const DeviceTensor<4> J(geom->J.GetData(), 3,3,NQ,NE);
      DeviceMatrix v(vec.GetData(), NQ,NE);
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double J11 = J(0,0,q,e),J12 = J(1,0,q,e),J13 = J(2,0,q,e);
            const double J21 = J(0,1,q,e),J22 = J(1,1,q,e),J23 = J(2,1,q,e);
            const double J31 = J(0,2,q,e),J32 = J(1,2,q,e),J33 = J(2,2,q,e);
            const double detJ =
            ((J11 * J22 * J33) + (J12 * J23 * J31) + (J13 * J21 * J32) -
            (J13 * J22 * J31) - (J12 * J21 * J33) - (J11 * J23 * J32));
            const Vector3 Xq(x(0,q,e), x(1,q,e), x(2,q,e));
            const double coeff =
            const_coeff ? constant
            : function_coeff ? function(Xq)
            : 0.0;
            v(q,e) = W(q) * coeff * detJ;
         }
      });
   }
}

#ifdef MFEM_USE_OCCA
// OCCA PA Mass Apply 2D kernel
static void OccaPAMassApply2D(const int D1D,
                              const int Q1D,
                              const int NE,
                              const double* B,
                              const double* Bt,
                              const double* op,
                              const double* x,
                              double* y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = mfem::OccaPtr(B);
   const occa::memory o_Bt = mfem::OccaPtr(Bt);
   const occa::memory o_op = mfem::OccaPtr(op);
   const occa::memory o_x = mfem::OccaPtr(x);
   occa::memory o_y = mfem::OccaPtr(y);
   const id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaMassApply2D_cpu;
      if (OccaMassApply2D_cpu.find(id) == OccaMassApply2D_cpu.end())
      {
         const occa::kernel MassApply2D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply2D_CPU", props);
         OccaMassApply2D_cpu.emplace(id, MassApply2D_CPU);
      }
      OccaMassApply2D_cpu.at(id)(NE, o_B, o_Bt, o_op, o_x, o_y);
   }
   else
   {
      static occa_kernel_t OccaMassApply2D_gpu;
      if (OccaMassApply2D_gpu.find(id) == OccaMassApply2D_gpu.end())
      {
         const occa::kernel MassApply2D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply2D_GPU", props);
         OccaMassApply2D_gpu.emplace(id, MassApply2D_GPU);
      }
      OccaMassApply2D_gpu.at(id)(NE, o_B, o_Bt, o_op, o_x, o_y);
   }
}

// OCCA PA Mass Apply 3D kernel
static void OccaPAMassApply3D(const int D1D,
                              const int Q1D,
                              const int NE,
                              const double* B,
                              const double* Bt,
                              const double* op,
                              const double* x,
                              double* y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = mfem::OccaPtr(B);
   const occa::memory o_Bt = mfem::OccaPtr(Bt);
   const occa::memory o_op = mfem::OccaPtr(op);
   const occa::memory o_x = mfem::OccaPtr(x);
   occa::memory o_y = mfem::OccaPtr(y);
   const id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaMassApply3D_cpu;
      if (OccaMassApply3D_cpu.find(id) == OccaMassApply3D_cpu.end())
      {
         const occa::kernel MassApply3D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply3D_CPU", props);
         OccaMassApply3D_cpu.emplace(id, MassApply3D_CPU);
      }
      OccaMassApply3D_cpu.at(id)(NE, o_B, o_Bt, o_op, o_x, o_y);
   }
   else
   {
      static occa_kernel_t OccaMassApply3D_gpu;
      if (OccaMassApply3D_gpu.find(id) == OccaMassApply3D_gpu.end())
      {
         const occa::kernel MassApply3D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply3D_GPU", props);
         OccaMassApply3D_gpu.emplace(id, MassApply3D_GPU);
      }
      OccaMassApply3D_gpu.at(id)(NE, o_B, o_Bt, o_op, o_x, o_y);
   }
}
#endif // MFEM_USE_OCCA

template<const int T_D1D = 0, const int T_Q1D = 0, const int T_NBZ = 0> static
void PAMassApply2D(const int NE,
                   const double* _b,
                   const double* _bt,
                   const double* _op,
                   const double* _x,
                   double* _y,
                   const int d1d = 0,
                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int NBZ = T_NBZ ? T_NBZ : 1;
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   const DeviceMatrix B(_b, Q1D, D1D);
   const DeviceMatrix Bt(_bt, D1D, Q1D);
   const DeviceTensor<3> op(_op, Q1D, Q1D, NE);
   const DeviceTensor<3> x(_x, D1D, D1D, NE);
   DeviceTensor<3> y(_y, D1D, D1D, NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int NBZ = T_NBZ ? T_NBZ : 1;
      const int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      MFEM_SHARED double buf1[NBZ][MQ1][MQ1];
      MFEM_SHARED double buf2[NBZ][MQ1][MQ1];
      MFEM_SHARED double matrix[MQ1][MQ1];
      double (*sol_x)[MQ1] = (double (*)[MQ1])(buf2 + threadIdx(z));
      double (*sol_xy)[MQ1] = (double (*)[MQ1])(buf1 + threadIdx(z));
      double (*input)[MQ1] = (double (*)[MQ1])(buf1 + threadIdx(z));
      for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
      {
         for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
         {
            input[dy][dx] = x(dx,dy,e);
         }
      }
      if (threadIdx(z) == 0)
      {
         for (int dx = threadIdx(y); dx < D1D; dx += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               matrix[dx][qx] = B(qx,dx);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            double t = 0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               t += matrix[dx][qx]*input[dy][dx];
            }
            sol_x[dy][qx] = t;
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            double t = 0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               t += matrix[dy][qy]*sol_x[dy][qx];
            }
            sol_xy[qy][qx] = t;
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
         {
            sol_xy[qy][qx] *= op(qx,qy,e);
         }
      }
      if (threadIdx(z) == 0)
      {
         for (int qx = threadIdx(y); qx < Q1D; qx += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               matrix[qx][dx] = Bt(dx,qx);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
      {
         for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
         {
            double t = 0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               t += matrix[qx][dx] * sol_xy[qy][qx];
            }
            sol_x[qy][dx] = t;
         }
      }
      MFEM_SYNC_THREAD;
      for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
      {
         for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
         {
            double t = 0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               t += matrix[qy][dy] * sol_x[qy][dx];
            }
            y(dx,dy,e) = t;
         }
      }
   });
}

template<const int T_D1D = 0, const int T_Q1D =0> static
void PAMassApply3D(const int NE,
                   const double* _b,
                   const double* _bt,
                   const double* _op,
                   const double* _x,
                   double *_y,
                   const int d1d = 0,
                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   const DeviceMatrix B(_b, Q1D, D1D);
   const DeviceMatrix Bt(_bt, D1D, Q1D);
   const DeviceTensor<4> op(_op, Q1D, Q1D, Q1D, NE);
   const DeviceTensor<4> X(_x, D1D, D1D, D1D, NE);
   DeviceTensor<4> Y(_y, D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      MFEM_SHARED double buf1[MQ1][MQ1][MQ1];
      MFEM_SHARED double buf2[MQ1][MQ1][MQ1];
      MFEM_SHARED double matrix[MQ1][MQ1];
      double (*sol_xyz)[MQ1][MQ1] = buf1;
      double (*sol_xy)[MQ1][MQ1] = buf2;
      double (*sol_x)[MQ1][MQ1] = buf1;
      double (*input)[MQ1][MQ1] = buf2;
      for (int dz = threadIdx(z); dz < D1D; dz += blockDim(z))
      {
         for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               input[dz][dy][dx] = X(dx,dy,dz,e);
            }
         }
      }
      if (threadIdx(z) == 0)
      {
         for (int dx = threadIdx(y); dx < D1D; dx += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               matrix[dx][qx] = B(qx,dx);
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int dz = threadIdx(z); dz < D1D; dz += blockDim(z))
      {
         for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               double t = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  t += matrix[dx][qx] * input[dz][dy][dx];
               }
               sol_x[dz][dy][qx] = t;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int dz = threadIdx(z); dz < D1D; dz += blockDim(z))
      {
         for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               double t = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  t += matrix[dy][qy] * sol_x[dz][dy][qx];
               }
               sol_xy[dz][qy][qx] = t;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int qz = threadIdx(z); qz < Q1D; qz += blockDim(z))
      {
         for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               double t = 0.0;
               for (int dz = 0; dz < D1D; ++dz)
               {
                  t += matrix[dz][qz] * sol_xy[dz][qy][qx];
               }
               sol_xyz[qz][qy][qx] = t;
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int qz = threadIdx(z); qz < Q1D; qz += blockDim(z))
      {
         for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
         {
            for (int qx = threadIdx(x); qx < Q1D; qx += blockDim(x))
            {
               sol_xyz[qz][qy][qx] *= op(qx,qy,qz,e);
            }
         }
      }
      if (threadIdx(z) == 0)
      {
         for (int qx = threadIdx(y); qx < Q1D; qx += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               matrix[qx][dx] = Bt(dx,qx);
            }
         }
      }
      MFEM_SYNC_THREAD;
      sol_x = buf2;
      sol_xy = buf1;
      for (int qz = threadIdx(z); qz < Q1D; qz += blockDim(z))
      {
         for (int qy = threadIdx(y); qy < Q1D; qy += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               double t = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  t += matrix[qx][dx] * sol_xyz[qz][qy][qx];
               }
               sol_x[qz][qy][dx] = t;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int qz = threadIdx(z); qz < Q1D; qz += blockDim(z))
      {
         for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               double t = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  t += matrix[qy][dy] * sol_x[qz][qy][dx];
               }
               sol_xy[qz][dy][dx] = t;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int dz = threadIdx(z); dz < D1D; dz += blockDim(z))
      {
         for (int dy = threadIdx(y); dy < D1D; dy += blockDim(y))
         {
            for (int dx = threadIdx(x); dx < D1D; dx += blockDim(x))
            {
               double t = Y(dx,dy,dz,e);
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  t += matrix[qz][dz] * sol_xy[qz][dy][dx];
               }
               Y(dx,dy,dz,e) = t;
            }
         }
      }
   });
}

static void PAMassApply(const int dim,
                        const int D1D,
                        const int Q1D,
                        const int NE,
                        const double* B,
                        const double* Bt,
                        const double* op,
                        const double* x,
                        double* y)
{
#ifdef MFEM_USE_OCCA
   if (internal::DeviceUseOcca())
   {
      if (dim == 2)
      {
         OccaPAMassApply2D(D1D, Q1D, NE, B, Bt, op, x, y);
         return;
      }
      if (dim == 3)
      {
         OccaPAMassApply3D(D1D, Q1D, NE, B, Bt, op, x, y);
         return;
      }
      MFEM_ABORT("OCCA PA Mass Apply unknown kernel!");
   }
#endif // MFEM_USE_OCCA
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return PAMassApply2D<2,2,8>(NE, B, Bt, op, x, y);
         case 0x33: return PAMassApply2D<3,3,8>(NE, B, Bt, op, x, y);
         case 0x44: return PAMassApply2D<4,4,2>(NE, B, Bt, op, x, y);
         case 0x55: return PAMassApply2D<5,5,2>(NE, B, Bt, op, x, y);
         case 0x66: return PAMassApply2D<6,6,2>(NE, B, Bt, op, x, y);
         case 0x77: return PAMassApply2D<7,7,2>(NE, B, Bt, op, x, y);
         case 0x88: return PAMassApply2D<8,8,1>(NE, B, Bt, op, x, y);
         case 0x99: return PAMassApply2D<9,9,1>(NE, B, Bt, op, x, y);
         default:   return PAMassApply2D(NE, B, Bt, op, x, y, D1D, Q1D);
      }
   }
   if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return PAMassApply3D<2,3>(NE, B, Bt, op, x, y);
         case 0x34: return PAMassApply3D<3,4>(NE, B, Bt, op, x, y);
         case 0x45: return PAMassApply3D<4,5>(NE, B, Bt, op, x, y);
         case 0x56: return PAMassApply3D<5,6>(NE, B, Bt, op, x, y);
         case 0x67: return PAMassApply3D<6,7>(NE, B, Bt, op, x, y);
         case 0x78: return PAMassApply3D<7,8>(NE, B, Bt, op, x, y);
         case 0x89: return PAMassApply3D<8,9>(NE, B, Bt, op, x, y);
         default:   return PAMassApply3D(NE, B, Bt, op, x, y, D1D, Q1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

void MassIntegrator::MultAssembled(Vector &x, Vector &y)
{
   PAMassApply(dim, dofs1D, quad1D, ne, maps->B, maps->Bt, vec, x, y);
}

MassIntegrator::~MassIntegrator()
{
   delete geom;
   delete maps;
}

// DofToQuad
static std::map<std::string, DofToQuad* > AllDofQuadMaps;

DofToQuad::~DofToQuad()
{
   MFEM_ASSERT(AllDofQuadMaps.at(hash),"");
   AllDofQuadMaps.erase(hash);
}

DofToQuad* DofToQuad::Get(const FiniteElementSpace& fes,
                          const IntegrationRule& ir,
                          const bool transpose)
{
   return Get(*fes.GetFE(0), *fes.GetFE(0), ir, transpose);
}

DofToQuad* DofToQuad::Get(const FiniteElementSpace& trialFES,
                          const FiniteElementSpace& testFES,
                          const IntegrationRule& ir,
                          const bool transpose)
{
   return Get(*trialFES.GetFE(0), *testFES.GetFE(0), ir, transpose);
}

DofToQuad* DofToQuad::Get(const FiniteElement& trialFE,
                          const FiniteElement& testFE,
                          const IntegrationRule& ir,
                          const bool transpose)
{
   return GetTensorMaps(trialFE, testFE, ir, transpose);
}

DofToQuad* DofToQuad::GetTensorMaps(const FiniteElement& trialFE,
                                    const FiniteElement& testFE,
                                    const IntegrationRule& ir,
                                    const bool transpose)
{
   const TensorBasisElement& trialTFE =
      dynamic_cast<const TensorBasisElement&>(trialFE);
   const TensorBasisElement& testTFE =
      dynamic_cast<const TensorBasisElement&>(testFE);
   std::stringstream ss;
   ss << "TensorMap:"
      << " O1:"  << trialFE.GetOrder()
      << " O2:"  << testFE.GetOrder()
      << " BT1:" << trialTFE.GetBasisType()
      << " BT2:" << testTFE.GetBasisType()
      << " Q:"   << ir.GetNPoints();
   std::string hash = ss.str();
   // If we've already made the dof-quad maps, reuse them
   if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
   {
      return AllDofQuadMaps[hash];
   }
   // Otherwise, build them
   DofToQuad *maps = new DofToQuad();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;
   const DofToQuad* trialMaps = GetD2QTensorMaps(trialFE, ir);
   const DofToQuad* testMaps  = GetD2QTensorMaps(testFE, ir, true);
   maps->B = trialMaps->B;
   maps->G = trialMaps->G;
   maps->Bt = testMaps->B;
   maps->Gt = testMaps->G;
   maps->W = testMaps->W;
   delete trialMaps;
   delete testMaps;
   return maps;
}

DofToQuad* DofToQuad::GetD2QTensorMaps(const FiniteElement& fe,
                                       const IntegrationRule& ir,
                                       const bool transpose)
{
   const IntegrationRule& ir1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder());

   const int dims = fe.GetDim();
   const int order = fe.GetOrder();
   const int numDofs = order + 1;
   const int numQuad1D = ir1D.GetNPoints();
   const int numQuad2D = numQuad1D * numQuad1D;
   const int numQuad3D = numQuad2D * numQuad1D;
   const int numQuad =
      (dims == 1) ? numQuad1D :
      (dims == 2) ? numQuad2D :
      (dims == 3) ? numQuad3D : 0;
   std::stringstream ss;
   ss << "D2QTensorMap:"
      << " dims:" << dims
      << " order:" << order
      << " numDofs:" << numDofs
      << " numQuad1D:" << numQuad1D
      << " transpose:"  << (transpose?"true":"false");
   std::string hash = ss.str();
   if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
   {
      return AllDofQuadMaps[hash];
   }
   DofToQuad *maps = new DofToQuad();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;
   maps->B.SetSize(numQuad1D*numDofs);
   maps->G.SetSize(numQuad1D*numDofs);
   const int dim0 = (!transpose)?1:numDofs;
   const int dim1 = (!transpose)?numQuad1D:1;
   if (transpose) // Initialize quad weights only for transpose
   {
      maps->W.SetSize(numQuad);
   }
   mfem::Vector d2q(numDofs);
   mfem::Vector d2qD(numDofs);
   mfem::Array<double> W1d(numQuad1D);
   mfem::Array<double> B1d(numQuad1D*numDofs);
   mfem::Array<double> G1d(numQuad1D*numDofs);
   const TensorBasisElement& tbe = dynamic_cast<const TensorBasisElement&>(fe);
   const Poly_1D::Basis& basis = tbe.GetBasis1D();
   for (int q = 0; q < numQuad1D; ++q)
   {
      const IntegrationPoint& ip = ir1D.IntPoint(q);
      if (transpose)
      {
         W1d[q] = ip.weight;
      }
      basis.Eval(ip.x, d2q, d2qD);
      for (int d = 0; d < numDofs; ++d)
      {
         const double w = d2q[d];
         const double wD = d2qD[d];
         const int idx = dim0*q + dim1*d;
         B1d[idx] = w;
         G1d[idx] = wD;
      }
   }
   if (transpose)
   {
      mfem::Array<double> W(numQuad);
      for (int q = 0; q < numQuad; ++q)
      {
         const int qx = q % numQuad1D;
         const int qz = q / numQuad2D;
         const int qy = (q - qz*numQuad2D) / numQuad1D;
         double w = W1d[qx];
         if (dims > 1) { w *= W1d[qy]; }
         if (dims > 2) { w *= W1d[qz]; }
         W[q] = w;
      }
      maps->W = W;
   }
   mfem::Memcpy(maps->B, B1d, numQuad1D*numDofs*sizeof(double));
   mfem::Memcpy(maps->G, G1d, numQuad1D*numDofs*sizeof(double));
   return maps;
}

DofToQuad* DofToQuad::GetSimplexMaps(const FiniteElement& fe,
                                     const IntegrationRule& ir,
                                     const bool transpose)
{
   return GetSimplexMaps(fe, fe, ir, transpose);
}

DofToQuad* DofToQuad::GetSimplexMaps(const FiniteElement& trialFE,
                                     const FiniteElement& testFE,
                                     const IntegrationRule& ir,
                                     const bool transpose)
{
   std::stringstream ss;
   ss << "SimplexMap:"
      << " O1:" << trialFE.GetOrder()
      << " O2:" << testFE.GetOrder()
      << " Q:"  << ir.GetNPoints();
   std::string hash = ss.str();
   // If we've already made the dof-quad maps, reuse them
   if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
   {
      return AllDofQuadMaps[hash];
   }
   DofToQuad *maps = new DofToQuad();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;
   const DofToQuad* trialMaps = GetD2QSimplexMaps(trialFE, ir);
   const DofToQuad* testMaps  = GetD2QSimplexMaps(testFE, ir, true);
   maps->B = trialMaps->B;
   maps->G = trialMaps->G;
   maps->Bt = testMaps->B;
   maps->Gt = testMaps->G;
   maps->W = testMaps->W;
   delete trialMaps;
   delete testMaps;
   return maps;
}

DofToQuad* DofToQuad::GetD2QSimplexMaps(const FiniteElement& fe,
                                        const IntegrationRule& ir,
                                        const bool transpose)
{
   const int dims = fe.GetDim();
   const int numDofs = fe.GetDof();
   const int numQuad = ir.GetNPoints();
   std::stringstream ss ;
   ss << "D2QSimplexMap:"
      << " Dim:" << dims
      << " numDofs:" << numDofs
      << " numQuad:" << numQuad
      << " transpose:" << (transpose?"true":"false");
   std::string hash = ss.str();
   if (AllDofQuadMaps.find(hash)!=AllDofQuadMaps.end())
   {
      return AllDofQuadMaps[hash];
   }
   DofToQuad* maps = new DofToQuad();
   AllDofQuadMaps[hash]=maps;
   maps->hash = hash;
   maps->B.SetSize(numQuad*numDofs);
   maps->G.SetSize(dims*numQuad*numDofs);
   const int dim0 = (!transpose)?1:numDofs;
   const int dim1 = (!transpose)?numQuad:1;
   const int dim0D = (!transpose)?1:numQuad;
   const int dim1D = (!transpose)?dims:1;
   const int dim2D = dims*numQuad;
   if (transpose) // Initialize quad weights only for transpose
   {
      maps->W.SetSize(numQuad);
   }
   mfem::Vector d2q(numDofs);
   mfem::DenseMatrix d2qD(numDofs, dims);
   mfem::Array<double> W(numQuad);
   mfem::Array<double> B(numQuad*numDofs);
   mfem::Array<double> G(dims*numQuad*numDofs);
   for (int q = 0; q < numQuad; ++q)
   {
      const IntegrationPoint& ip = ir.IntPoint(q);
      if (transpose)
      {
         W[q] = ip.weight;
      }
      fe.CalcShape(ip, d2q);
      fe.CalcDShape(ip, d2qD);
      for (int d = 0; d < numDofs; ++d)
      {
         const double w = d2q[d];
         const int idx = dim0*q + dim1*d;
         B[idx] = w;
         for (int dim = 0; dim < dims; ++dim)
         {
            const double wD = d2qD(d, dim);
            const int idxD = dim0D*dim + dim1D*q + dim2D*d;
            G[idxD] = wD;
         }
      }
   }
   if (transpose)
   {
      mfem::Memcpy(maps->W, W, numQuad*sizeof(double));
   }
   mfem::Memcpy(maps->B, B, numQuad*numDofs*sizeof(double));
   mfem::Memcpy(maps->G, G, dims*numQuad*numDofs*sizeof(double));
   return maps;
}


static long sequence = -1;
static GeometryExtension *geom = NULL;

static void GeomFill(const int vdim,
                     const int NE, const int ND, const int NX,
                     const int* elementMap, int* eMap,
                     const double *_X, double *meshNodes)
{
   const DeviceArray d_elementMap(elementMap, ND*NE);
   DeviceArray d_eMap(eMap, ND*NE);
   const DeviceVector X(_X, NX);
   DeviceVector d_meshNodes(meshNodes, vdim*ND*NE);
   MFEM_FORALL(e, NE,
   {
      for (int d = 0; d < ND; ++d)
      {
         const int lid = d+ND*e;
         const int gid = d_elementMap[lid];
         d_eMap[lid] = gid;
         for (int v = 0; v < vdim; ++v)
         {
            const int moffset = v+vdim*lid;
            const int xoffset = v+vdim*gid;
            d_meshNodes[moffset] = X[xoffset];
         }
      }
   });
}

static void NodeCopyByVDim(const int elements,
                           const int numDofs,
                           const int ndofs,
                           const int dims,
                           const int* eMap,
                           const double* Sx,
                           double* nodes)
{
   MFEM_FORALL(e,elements,
   {
      for (int dof = 0; dof < numDofs; ++dof)
      {
         const int lid = dof+numDofs*e;
         const int gid = eMap[lid];
         for (int v = 0; v < dims; ++v)
         {
            const int moffset = v+dims*lid;
            const int voffset = gid+v*ndofs;
            nodes[moffset] = Sx[voffset];
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0> static
void PAGeom2D(const int NE,
              const double* _B,
              const double* _G,
              const double* _X,
              double* _Xq,
              double* _J,
              double* _invJ,
              double* _detJ,
              const int d1d = 0,
              const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   const int ND = D1D*D1D;
   const int NQ = Q1D*Q1D;

   const DeviceTensor<2> B(_B, NQ, ND);
   const DeviceTensor<3> G(_G, 2,NQ, ND);
   const DeviceTensor<3> X(_X, 2,ND, NE);
   DeviceTensor<3> Xq(_Xq, 2, NQ, NE);
   DeviceTensor<4> J(_J, 2, 2, NQ, NE);
   DeviceTensor<4> invJ(_invJ, 2, 2, NQ, NE);
   DeviceMatrix detJ(_detJ, NQ, NE);

   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int ND = D1D*D1D;
      const int NQ = Q1D*Q1D;

      double s_X[2*MAX_D1D*MAX_D1D];
      for (int q = 0; q < NQ; ++q)
      {
         for (int d = q; d < ND; d +=NQ)
         {
            s_X[0+d*2] = X(0,d,e);
            s_X[1+d*2] = X(1,d,e);
         }
      }
      for (int q = 0; q < NQ; ++q)
      {
         double X0  = 0; double X1  = 0;
         double J11 = 0; double J12 = 0;
         double J21 = 0; double J22 = 0;
         for (int d = 0; d < ND; ++d)
         {
            const double b = B(q,d);
            const double wx = G(0,q,d);
            const double wy = G(1,q,d);
            const double x = s_X[0+d*2];
            const double y = s_X[1+d*2];
            J11 += (wx * x); J12 += (wx * y);
            J21 += (wy * x); J22 += (wy * y);
            X0 += b*x; X1 += b*y;
         }
         Xq(0,q,e) = X0; Xq(1,q,e) = X1;
         const double r_detJ = (J11 * J22)-(J12 * J21);
         J(0,0,q,e) = J11;
         J(1,0,q,e) = J12;
         J(0,1,q,e) = J21;
         J(1,1,q,e) = J22;
         const double r_idetJ = 1.0 / r_detJ;
         invJ(0,0,q,e) =  J22 * r_idetJ;
         invJ(1,0,q,e) = -J12 * r_idetJ;
         invJ(0,1,q,e) = -J21 * r_idetJ;
         invJ(1,1,q,e) =  J11 * r_idetJ;
         detJ(q,e) = r_detJ;
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0> static
void PAGeom3D(const int NE,
              const double* _B,
              const double* _G,
              const double* _X,
              double* _Xq,
              double* _J,
              double* _invJ,
              double* _detJ,
              const int d1d = 0,
              const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   const int ND = D1D*D1D*D1D;
   const int NQ = Q1D*Q1D*Q1D;

   const DeviceTensor<2> B(_B, NQ, NE);
   const DeviceTensor<3> G(_G, 3, NQ, NE);
   const DeviceTensor<3> X(_X, 3, ND, NE);
   DeviceTensor<3> Xq(_Xq, 3, NQ, NE);
   DeviceTensor<4> J(_J, 3, 3, NQ, NE);
   DeviceTensor<4> invJ(_invJ, 3, 3, NQ, NE);
   DeviceMatrix detJ(_detJ, NQ, NE);

   MFEM_FORALL(e,NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int ND = D1D*D1D*D1D;
      const int NQ = Q1D*Q1D*Q1D;

      double s_nodes[3*MAX_D1D*MAX_D1D*MAX_D1D];
      for (int q = 0; q < NQ; ++q)
      {
         for (int d = q; d < ND; d += NQ)
         {
            s_nodes[0+d*3] = X(0,d,e);
            s_nodes[1+d*3] = X(1,d,e);
            s_nodes[2+d*3] = X(2,d,e);
         }
      }
      for (int q = 0; q < NQ; ++q)
      {
         double J11 = 0; double J12 = 0; double J13 = 0;
         double J21 = 0; double J22 = 0; double J23 = 0;
         double J31 = 0; double J32 = 0; double J33 = 0;
         for (int d = 0; d < ND; ++d)
         {
            const double b = B(q,d);
            const double wx = G(0,q,d);
            const double wy = G(1,q,d);
            const double wz = G(2,q,d);
            const double x = s_nodes[0+d*3];
            const double y = s_nodes[1+d*3];
            const double z = s_nodes[2+d*3];
            J11 += (wx * x); J12 += (wx * y); J13 += (wx * z);
            J21 += (wy * x); J22 += (wy * y); J23 += (wy * z);
            J31 += (wz * x); J32 += (wz * y); J33 += (wz * z);
            Xq(0,q,e) = b*x; Xq(1,q,e) = b*y; Xq(2,q,e) = b*z;
         }
         const double r_detJ = ((J11 * J22 * J33) + (J12 * J23 * J31) +
                                (J13 * J21 * J32) - (J13 * J22 * J31) -
                                (J12 * J21 * J33) - (J11 * J23 * J32));
         J(0,0,q,e) = J11;
         J(1,0,q,e) = J12;
         J(2,0,q,e) = J13;
         J(0,1,q,e) = J21;
         J(1,1,q,e) = J22;
         J(2,1,q,e) = J23;
         J(0,2,q,e) = J31;
         J(1,2,q,e) = J32;
         J(2,2,q,e) = J33;
         const double r_idetJ = 1.0 / r_detJ;
         invJ(0,0,q,e) = r_idetJ * ((J22 * J33)-(J23 * J32));
         invJ(1,0,q,e) = r_idetJ * ((J32 * J13)-(J33 * J12));
         invJ(2,0,q,e) = r_idetJ * ((J12 * J23)-(J13 * J22));
         invJ(0,1,q,e) = r_idetJ * ((J23 * J31)-(J21 * J33));
         invJ(1,1,q,e) = r_idetJ * ((J33 * J11)-(J31 * J13));
         invJ(2,1,q,e) = r_idetJ * ((J13 * J21)-(J11 * J23));
         invJ(0,2,q,e) = r_idetJ * ((J21 * J32)-(J22 * J31));
         invJ(1,2,q,e) = r_idetJ * ((J31 * J12)-(J32 * J11));
         invJ(2,2,q,e) = r_idetJ * ((J11 * J22)-(J12 * J21));
         detJ(q,e) = r_detJ;
      }
   });
}

static void PAGeom(const int dim,
                   const int D1D,
                   const int Q1D,
                   const int NE,
                   const double* B,
                   const double* G,
                   const double* X,
                   double* Xq,
                   double* J,
                   double* invJ,
                   double* detJ)
{
   if (dim == 2)
   {
      switch ((D1D << 4) | Q1D)
      {
         case 0x22: PAGeom2D<2,2>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x23: PAGeom2D<2,3>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x24: PAGeom2D<2,4>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x25: PAGeom2D<2,5>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x32: PAGeom2D<3,2>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x34: PAGeom2D<3,4>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x42: PAGeom2D<4,2>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x44: PAGeom2D<4,4>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x45: PAGeom2D<4,5>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x46: PAGeom2D<4,6>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x58: PAGeom2D<5,8>(NE, B, G, X, Xq, J, invJ, detJ); break;
         default: PAGeom2D(NE, B, G, X, Xq, J, invJ, detJ, D1D, Q1D); break;
      }
      return;
   }
   if (dim == 3)
   {
      switch ((D1D << 4) | Q1D)
      {
         case 0x23: PAGeom3D<2,3>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x24: PAGeom3D<2,4>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x25: PAGeom3D<2,5>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x26: PAGeom3D<2,6>(NE, B, G, X, Xq, J, invJ, detJ); break;
         case 0x34: PAGeom3D<3,4>(NE, B, G, X, Xq, J, invJ, detJ); break;
         default: PAGeom3D(NE, B, G, X, Xq, J, invJ, detJ, D1D, Q1D); break;
      }
      return;
   }
   MFEM_ABORT("Unknown kernel.");
}

GeometryExtension* GeometryExtension::Get(const FiniteElementSpace& fes,
                                          const IntegrationRule& ir,
                                          const Vector& Sx)
{
   const Mesh *mesh = fes.GetMesh();
   const GridFunction *nodes = mesh->GetNodes();
   const FiniteElementSpace *fespace = nodes->FESpace();
   const FiniteElement *fe = fespace->GetFE(0);
   const IntegrationRule& ir1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder());
   const int dims     = fe->GetDim();
   const int numDofs  = fe->GetDof();
   const int D1D      = fe->GetOrder() + 1;
   const int Q1D      = ir1D.GetNPoints();
   const int elements = fespace->GetNE();
   const int ndofs    = fespace->GetNDofs();
   const DofToQuad* maps = DofToQuad::GetSimplexMaps(*fe, ir);
   NodeCopyByVDim(elements,numDofs,ndofs,dims,geom->eMap,Sx,geom->nodes);
   PAGeom(dims, D1D, Q1D, elements,
          maps->B, maps->G, geom->nodes,
          geom->X, geom->J, geom->invJ, geom->detJ);
   return geom;
}

GeometryExtension* GeometryExtension::Get(const FiniteElementSpace& fes,
                                          const IntegrationRule& ir)
{
   Mesh *mesh = fes.GetMesh();
   const bool geom_to_allocate = sequence < fes.GetSequence();
   sequence = fes.GetSequence();
   if (geom_to_allocate)
   {
      if (geom) { delete geom; }
      geom = new GeometryExtension();
   }

   const bool dev_enabled = Device::IsEnabled();
   if (dev_enabled) { Device::Disable(); }
   mesh->EnsureNodes();
   if (dev_enabled) { Device::Enable(); }

   const GridFunction *nodes = mesh->GetNodes();
   const mfem::FiniteElementSpace *fespace = nodes->FESpace();
   const mfem::FiniteElement *fe = fespace->GetFE(0);
   const IntegrationRule& ir1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder());
   const int dims     = fe->GetDim();
   const int elements = fespace->GetNE();
   const int numDofs  = fe->GetDof();
   const int D1D      = fe->GetOrder() + 1;
   const int Q1D      = ir1D.GetNPoints();
   const int numQuad  = ir.GetNPoints();
   const bool orderedByNODES = (fespace->GetOrdering() == Ordering::byNODES);
   if (orderedByNODES) { ReorderByVDim(nodes); }
   const int asize = dims*numDofs*elements;
   mfem::Array<double> meshNodes(asize);
   const Table& e2dTable = fespace->GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   mfem::Array<int> eMap(numDofs*elements);
   GeomFill(dims,
            elements,
            numDofs,
            nodes->Size(),
            elementMap,
            eMap,
            nodes->GetData(),
            meshNodes);
   if (geom_to_allocate)
   {
      geom->nodes.SetSize(dims*numDofs*elements);
      geom->eMap.SetSize(numDofs*elements);
   }
   geom->nodes = meshNodes;
   geom->eMap = eMap;
   // Reorder the original gf back
   if (orderedByNODES) { ReorderByNodes(nodes); }
   if (geom_to_allocate)
   {
      geom->X.SetSize(dims*numQuad*elements);
      geom->J.SetSize(dims*dims*numQuad*elements);
      geom->invJ.SetSize(dims*dims*numQuad*elements);
      geom->detJ.SetSize(numQuad*elements);
   }
   const DofToQuad* maps = DofToQuad::GetSimplexMaps(*fe, ir);
   PAGeom(dims, D1D, Q1D, elements,
          maps->B, maps->G, geom->nodes,
          geom->X, geom->J, geom->invJ, geom->detJ);
   delete maps;
   return geom;
}

void GeometryExtension::ReorderByVDim(const GridFunction *nodes)
{
   const mfem::FiniteElementSpace *fes = nodes->FESpace();
   const int size = nodes->Size();
   const int vdim = fes->GetVDim();
   const int ndofs = fes->GetNDofs();
   double *data = nodes->GetData();
   double *temp = new double[size];
   int k=0;
   for (int d = 0; d < ndofs; d++)
      for (int v = 0; v < vdim; v++)
      {
         temp[k++] = data[d+v*ndofs];
      }
   for (int i = 0; i < size; i++)
   {
      data[i] = temp[i];
   }
   delete [] temp;
}

void GeometryExtension::ReorderByNodes(const GridFunction *nodes)
{
   const mfem::FiniteElementSpace *fes = nodes->FESpace();
   const int size = nodes->Size();
   const int vdim = fes->GetVDim();
   const int ndofs = fes->GetNDofs();
   double *data = nodes->GetData();
   double *temp = new double[size];
   int k = 0;
   for (int j = 0; j < ndofs; j++)
      for (int i = 0; i < vdim; i++)
      {
         temp[j+i*ndofs] = data[k++];
      }
   for (int i = 0; i < size; i++)
   {
      data[i] = temp[i];
   }
   delete [] temp;
}

} // namespace mfem
