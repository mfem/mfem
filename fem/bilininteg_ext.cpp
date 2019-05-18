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

// FIXME: the default rules depend on the integrator -- they should be made
// member functions in the integrator classes. This rule corresponds to the
// diffusion integrator, however, it is used for both the mass and the diffusion
// integrator below.
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
                                   const Array<double> &W,
                                   const Array<double> &J,
                                   const double COEFF,
                                   Vector &op)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_W = OccaMemoryRead(W.GetMemory(), W.Size());
   const occa::memory o_J = OccaMemoryRead(J.GetMemory(), J.Size());
   occa::memory o_op = OccaMemoryWrite(op.GetMemory(), op.Size());
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
                                   const Array<double> &W,
                                   const Array<double> &J,
                                   const double COEFF,
                                   Vector &op)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_W = OccaMemoryRead(W.GetMemory(), W.Size());
   const occa::memory o_J = OccaMemoryRead(J.GetMemory(), J.Size());
   occa::memory o_op = OccaMemoryWrite(op.GetMemory(), op.Size());
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
                               const Array<double> &w,
                               const Array<double> &j,
                               const double COEFF,
                               Vector &op)
{
   const int NQ = Q1D*Q1D;
   auto W = w.ReadAccess();
   auto J = Reshape(j.ReadAccess(), 2, 2, NQ, NE);
   auto y = Reshape(op.WriteAccess(), 3, NQ, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(0,0,q,e);
         const double J21 = J(1,0,q,e);
         const double J12 = J(0,1,q,e);
         const double J22 = J(1,1,q,e);
         const double c_detJ = W[q] * COEFF / ((J11*J22)-(J21*J12));
         y(0,q,e) =  c_detJ * (J12*J12 + J22*J22); // 1,1
         y(1,q,e) = -c_detJ * (J12*J11 + J22*J21); // 1,2
         y(2,q,e) =  c_detJ * (J11*J11 + J21*J21); // 2,2
      }
   });
}

// PA Diffusion Assemble 3D kernel
static void PADiffusionSetup3D(const int Q1D,
                               const int NE,
                               const Array<double> &w,
                               const Array<double> &j,
                               const double COEFF,
                               Vector &op)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.ReadAccess();
   auto J = Reshape(j.ReadAccess(), 3, 3, NQ, NE);
   auto y = Reshape(op.WriteAccess(), 6, NQ, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(0,0,q,e);
         const double J21 = J(1,0,q,e);
         const double J31 = J(2,0,q,e);
         const double J12 = J(0,1,q,e);
         const double J22 = J(1,1,q,e);
         const double J32 = J(2,1,q,e);
         const double J13 = J(0,2,q,e);
         const double J23 = J(1,2,q,e);
         const double J33 = J(2,2,q,e);
         const double detJ = J11 * (J22 * J33 - J32 * J23) -
         /* */               J21 * (J12 * J33 - J32 * J13) +
         /* */               J31 * (J12 * J23 - J22 * J13);
         const double c_detJ = W[q] * COEFF / detJ;
         // adj(J)
         const double A11 = (J22 * J33) - (J23 * J32);
         const double A12 = (J32 * J13) - (J12 * J33);
         const double A13 = (J12 * J23) - (J22 * J13);
         const double A21 = (J31 * J23) - (J21 * J33);
         const double A22 = (J11 * J33) - (J13 * J31);
         const double A23 = (J21 * J13) - (J11 * J23);
         const double A31 = (J21 * J32) - (J31 * J22);
         const double A32 = (J31 * J12) - (J11 * J32);
         const double A33 = (J11 * J22) - (J12 * J21);
         // detJ J^{-1} J^{-T} = (1/detJ) adj(J) adj(J)^T
         y(0,q,e) = c_detJ * (A11*A11 + A12*A12 + A13*A13); // 1,1
         y(1,q,e) = c_detJ * (A11*A21 + A12*A22 + A13*A23); // 2,1
         y(2,q,e) = c_detJ * (A11*A31 + A12*A32 + A13*A33); // 3,1
         y(3,q,e) = c_detJ * (A21*A21 + A22*A22 + A23*A23); // 2,2
         y(4,q,e) = c_detJ * (A21*A31 + A22*A32 + A23*A33); // 3,2
         y(5,q,e) = c_detJ * (A31*A31 + A32*A32 + A33*A33); // 3,3
      }
   });
}

static void PADiffusionSetup(const int dim,
                             const int D1D,
                             const int Q1D,
                             const int NE,
                             const Array<double> &W,
                             const Array<double> &J,
                             const double COEFF,
                             Vector &op)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADiffusionSetup"); }
   if (dim == 2)
   {
#ifdef MFEM_USE_OCCA
      if (DeviceUseOcca())
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
      if (DeviceUseOcca())
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
   // FIXME: use default quadrature rule defined as a member function.
   const IntegrationRule *ir = rule?rule:&DefaultGetRule(el,el);
   const int dims = el.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes.GetNE();
   dofs1D = el.GetOrder() + 1;
   quad1D = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
   geom = GeometryExtension::Get(fes,*ir); // FIXME: returns the global 'geom'
   maps = DofToQuad::Get(fes, fes, *ir);
   pa_data.SetSize(symmDims * nq * ne, Device::GetMemoryType());
   ConstantCoefficient *cQ = dynamic_cast<ConstantCoefficient*>(Q);
   MFEM_VERIFY(cQ != NULL, "only ConstantCoefficient is supported!");
   const double coeff = cQ->constant;
   PADiffusionSetup(dim, dofs1D, quad1D, ne, maps->W, geom->J, coeff, pa_data);
}

#ifdef MFEM_USE_OCCA
// OCCA PA Diffusion Apply 2D kernel
static void OccaPADiffusionApply2D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const Array<double> &B,
                                   const Array<double> &G,
                                   const Array<double> &Bt,
                                   const Array<double> &Gt,
                                   const Vector &op,
                                   const Vector &x,
                                   Vector &y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = OccaMemoryRead(B.GetMemory(), B.Size());
   const occa::memory o_G = OccaMemoryRead(G.GetMemory(), G.Size());
   const occa::memory o_Bt = OccaMemoryRead(Bt.GetMemory(), Bt.Size());
   const occa::memory o_Gt = OccaMemoryRead(Gt.GetMemory(), Gt.Size());
   const occa::memory o_op = OccaMemoryRead(op.GetMemory(), op.Size());
   const occa::memory o_x = OccaMemoryRead(x.GetMemory(), x.Size());
   occa::memory o_y = OccaMemoryWrite(y.GetMemory(), y.Size());
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
                                   const Array<double> &B,
                                   const Array<double> &G,
                                   const Array<double> &Bt,
                                   const Array<double> &Gt,
                                   const Vector &op,
                                   const Vector &x,
                                   Vector &y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = OccaMemoryRead(B.GetMemory(), B.Size());
   const occa::memory o_G = OccaMemoryRead(G.GetMemory(), G.Size());
   const occa::memory o_Bt = OccaMemoryRead(Bt.GetMemory(), Bt.Size());
   const occa::memory o_Gt = OccaMemoryRead(Gt.GetMemory(), Gt.Size());
   const occa::memory o_op = OccaMemoryRead(op.GetMemory(), op.Size());
   const occa::memory o_x = OccaMemoryRead(x.GetMemory(), x.Size());
   occa::memory o_y = OccaMemoryWrite(y.GetMemory(), y.Size());
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


const int MAX_Q1D = 10;
const int MAX_D1D = 10;

// PA Diffusion Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0> static
void PADiffusionApply2D(const int NE,
                        const Array<double> &b,
                        const Array<double> &g,
                        const Array<double> &bt,
                        const Array<double> &gt,
                        const Vector &_op,
                        const Vector &_x,
                        Vector &_y,
                        const int d1d = 0,
                        const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");

   auto B = Reshape(b.ReadAccess(), Q1D, D1D);
   auto G = Reshape(g.ReadAccess(), Q1D, D1D);
   auto Bt = Reshape(bt.ReadAccess(), D1D, Q1D);
   auto Gt = Reshape(gt.ReadAccess(), D1D, Q1D);
   auto op = Reshape(_op.ReadAccess(), 3, Q1D*Q1D, NE);
   auto x = Reshape(_x.ReadAccess(), D1D, D1D, NE);
   auto y = Reshape(_y.WriteAccess(), D1D, D1D, NE);

   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int m_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int m_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double grad[m_Q1D][m_Q1D][2];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            grad[qy][qx][0] = 0.0;
            grad[qy][qx][1] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         double gradX[m_Q1D][2];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = x(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] += s * B(qx,dx);
               gradX[qx][1] += s * G(qx,dx);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double wy  = B(qy,dy);
            const double wDy = G(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qy][qx][0] += gradX[qx][1] * wy;
               grad[qy][qx][1] += gradX[qx][0] * wDy;
            }
         }
      }
      // Calculate Dxy, xDy in plane
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const int q = qx + qy * Q1D;

            const double O11 = op(0,q,e);
            const double O12 = op(1,q,e);
            const double O22 = op(2,q,e);

            const double gradX = grad[qy][qx][0];
            const double gradY = grad[qy][qx][1];

            grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
            grad[qy][qx][1] = (O12 * gradX) + (O22 * gradY);
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double gradX[m_D1D][2];
         for (int dx = 0; dx < D1D; ++dx)
         {
            gradX[dx][0] = 0;
            gradX[dx][1] = 0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double gX = grad[qy][qx][0];
            const double gY = grad[qy][qx][1];
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double wx  = Bt(dx,qx);
               const double wDx = Gt(dx,qx);
               gradX[dx][0] += gX * wDx;
               gradX[dx][1] += gY * wx;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double wy  = Bt(dy,qy);
            const double wDy = Gt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               y(dx,dy,e) += ((gradX[dx][0] * wy) + (gradX[dx][1] * wDy));
            }
         }
      }
   });
}

// PA Diffusion Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0> static
void PADiffusionApply3D(const int NE,
                        const Array<double> &b,
                        const Array<double> &g,
                        const Array<double> &bt,
                        const Array<double> &gt,
                        const Vector &_op,
                        const Vector &_x,
                        Vector &_y,
                        int d1d = 0, int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");

   auto B = Reshape(b.ReadAccess(), Q1D, D1D);
   auto G = Reshape(g.ReadAccess(), Q1D, D1D);
   auto Bt = Reshape(bt.ReadAccess(), D1D, Q1D);
   auto Gt = Reshape(gt.ReadAccess(), D1D, Q1D);
   auto op = Reshape(_op.ReadAccess(), 6, Q1D*Q1D*Q1D, NE);
   auto x = Reshape(_x.ReadAccess(), D1D, D1D, D1D, NE);
   auto y = Reshape(_y.WriteAccess(), D1D, D1D, D1D, NE);

   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int m_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int m_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double grad[m_Q1D][m_Q1D][m_Q1D][4];
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qz][qy][qx][0] = 0.0;
               grad[qz][qy][qx][1] = 0.0;
               grad[qz][qy][qx][2] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         double gradXY[m_Q1D][m_Q1D][4];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradXY[qy][qx][0] = 0.0;
               gradXY[qy][qx][1] = 0.0;
               gradXY[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            double gradX[m_Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double s = x(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] += s * B(qx,dx);
                  gradX[qx][1] += s * G(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy  = B(qy,dy);
               const double wDy = G(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double wx  = gradX[qx][0];
                  const double wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx  * wDy;
                  gradXY[qy][qx][2] += wx  * wy;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double wz  = B(qz,dz);
            const double wDz = G(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                  grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                  grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
               }
            }
         }
      }
      // Calculate Dxyz, xDyz, xyDz in plane
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + (qy + qz * Q1D) * Q1D;
               const double O11 = op(0,q,e);
               const double O12 = op(1,q,e);
               const double O13 = op(2,q,e);
               const double O22 = op(3,q,e);
               const double O23 = op(4,q,e);
               const double O33 = op(5,q,e);
               const double gradX = grad[qz][qy][qx][0];
               const double gradY = grad[qz][qy][qx][1];
               const double gradZ = grad[qz][qy][qx][2];
               grad[qz][qy][qx][0] = (O11*gradX)+(O12*gradY)+(O13*gradZ);
               grad[qz][qy][qx][1] = (O12*gradX)+(O22*gradY)+(O23*gradZ);
               grad[qz][qy][qx][2] = (O13*gradX)+(O23*gradY)+(O33*gradZ);
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         double gradXY[m_D1D][m_D1D][4];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradXY[dy][dx][0] = 0;
               gradXY[dy][dx][1] = 0;
               gradXY[dy][dx][2] = 0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double gradX[m_D1D][4];
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradX[dx][0] = 0;
               gradX[dx][1] = 0;
               gradX[dx][2] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double gX = grad[qz][qy][qx][0];
               const double gY = grad[qz][qy][qx][1];
               const double gZ = grad[qz][qy][qx][2];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double wx  = Bt(dx,qx);
                  const double wDx = Gt(dx,qx);
                  gradX[dx][0] += gX * wDx;
                  gradX[dx][1] += gY * wx;
                  gradX[dx][2] += gZ * wx;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double wy  = Bt(dy,qy);
               const double wDy = Gt(dy,qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  gradXY[dy][dx][0] += gradX[dx][0] * wy;
                  gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                  gradXY[dy][dx][2] += gradX[dx][2] * wy;
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            const double wz  = Bt(dz,qz);
            const double wDz = Gt(dz,qz);
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  y(dx,dy,dz,e) +=
                     ((gradXY[dy][dx][0] * wz) +
                      (gradXY[dy][dx][1] * wz) +
                      (gradXY[dy][dx][2] * wDz));
               }
            }
         }
      }
   });
}

static void PADiffusionApply(const int dim,
                             const int D1D,
                             const int Q1D,
                             const int NE,
                             const Array<double> &B,
                             const Array<double> &G,
                             const Array<double> &Bt,
                             const Array<double> &Gt,
                             const Vector &op,
                             const Vector &x,
                             Vector &y)
{
#ifdef MFEM_USE_OCCA
   if (DeviceUseOcca())
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
      switch ((D1D << 4) | Q1D)
      {
         case 0x22: PADiffusionApply2D<2,2>(NE, B, G, Bt, Gt, op, x, y); break;
         case 0x33: PADiffusionApply2D<3,3>(NE, B, G, Bt, Gt, op, x, y); break;
         case 0x44: PADiffusionApply2D<4,4>(NE, B, G, Bt, Gt, op, x, y); break;
         case 0x55: PADiffusionApply2D<5,5>(NE, B, G, Bt, Gt, op, x, y); break;
         default: PADiffusionApply2D(NE, B, G, Bt, Gt, op, x, y, D1D, Q1D);
      }
      return;
   }
   if (dim == 3)
   {
      switch ((D1D << 4) | Q1D)
      {
         case 0x23: PADiffusionApply3D<2,3>(NE, B, G, Bt, Gt, op, x, y); break;
         case 0x34: PADiffusionApply3D<3,4>(NE, B, G, Bt, Gt, op, x, y); break;
         case 0x45: PADiffusionApply3D<4,5>(NE, B, G, Bt, Gt, op, x, y); break;
         case 0x56: PADiffusionApply3D<5,6>(NE, B, G, Bt, Gt, op, x, y); break;
         default: PADiffusionApply3D(NE, B, G, Bt, Gt, op, x, y, D1D, Q1D);
      }
      return;
   }
   MFEM_ABORT("Unknown kernel.");
}

// PA Diffusion Apply kernel
void DiffusionIntegrator::MultAssembled(Vector &x, Vector &y)
{
   // FIXME: this function actually performs '+=', not '='
   PADiffusionApply(dim, dofs1D, quad1D, ne,
                    maps->B, maps->G, maps->Bt, maps->Gt,
                    pa_data, x, y);
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
   // FIXME: use default quadrature rule defined as a member function.
   const IntegrationRule *ir = rule?rule:&DefaultGetRule(el,el);
   dim = mesh->Dimension();
   ne = fes.GetMesh()->GetNE();
   nq = ir->GetNPoints();
   dofs1D = el.GetOrder() + 1;
   quad1D = IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints();
   geom = GeometryExtension::Get(fes,*ir); // FIXME: returns the global 'geom'
   maps = DofToQuad::Get(fes, fes, *ir);
   pa_data.SetSize(ne*nq, Device::GetMemoryType());
   ConstantCoefficient *const_coeff = dynamic_cast<ConstantCoefficient*>(Q);
   // FunctionCoefficient *function_coeff = dynamic_cast<FunctionCoefficient*>(Q);
   // TODO: other types of coefficients ...
   if (dim==1) { MFEM_ABORT("Not supported yet... stay tuned!"); }
   if (dim==2)
   {
      double constant = 0.0;
      // double (*function)(const Vector3&) = NULL;
      if (const_coeff)
      {
         constant = const_coeff->constant;
      }
#if 0
      else if (function_coeff)
      {
         function = function_coeff->GetDeviceFunction();
      }
#endif
      else
      {
         MFEM_ABORT("Coefficient type not supported");
      }
      const int NE = ne;
      const int NQ = nq;
      auto w = maps->W.ReadAccess();
      // auto x = Reshape(geom->X.ReadAccess(), 2, NQ, NE);
      auto J = Reshape(geom->J.ReadAccess(), 2, 2, NQ, NE);
      auto v = Reshape(pa_data.WriteAccess(), NQ, NE);
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double J11 = J(0,0,q,e);
            const double J21 = J(1,0,q,e);
            const double J12 = J(0,1,q,e);
            const double J22 = J(1,1,q,e);
            const double detJ = (J11*J22)-(J21*J12);
            // FIXME: this code seems to break the cuda kernel
            //
            // const Vector3 Xq(x(0,q,e), x(1,q,e));
            // const double coeff =
            // const_coeff ? constant
            // : function_coeff ? function(Xq)
            // : 0.0;
            // v(q,e) =  w[q] * coeff * detJ;
            //
            v(q,e) =  w[q] * constant * detJ;
         }
      });
   }
   if (dim==3)
   {
      double constant = 0.0;
      // double (*function)(const Vector3&) = NULL;
      if (const_coeff)
      {
         constant = const_coeff->constant;
      }
#if 0
      else if (function_coeff)
      {
         function = function_coeff->GetDeviceFunction();
      }
#endif
      else
      {
         MFEM_ABORT("Coefficient type not supported");
      }
      const int NE = ne;
      const int NQ = nq;
      auto W = maps->W.ReadAccess();
      // auto x = Reshape(geom->X.ReadAccess(), 3, NQ, NE);
      auto J = Reshape(geom->J.ReadAccess(), 3, 3, NQ, NE);
      auto v = Reshape(pa_data.WriteAccess(), NQ, NE);
      MFEM_FORALL(e, NE,
      {
         for (int q = 0; q < NQ; ++q)
         {
            const double J11 = J(0,0,q,e), J12 = J(0,1,q,e), J13 = J(0,2,q,e);
            const double J21 = J(1,0,q,e), J22 = J(1,1,q,e), J23 = J(1,2,q,e);
            const double J31 = J(2,0,q,e), J32 = J(2,1,q,e), J33 = J(2,2,q,e);
            const double detJ = J11 * (J22 * J33 - J32 * J23) -
            /* */               J21 * (J12 * J33 - J32 * J13) +
            /* */               J31 * (J12 * J23 - J22 * J13);
            // FIXME: this code seems to break the cuda kernel
            //
            // const Vector3 Xq(x(0,q,e), x(1,q,e), x(2,q,e));
            // const double coeff =
            // const_coeff ? constant
            // : function_coeff ? function(Xq)
            // : 0.0;
            // v(q,e) = W[q] * coeff * detJ;
            //
            v(q,e) = W[q] * constant * detJ;
         }
      });
   }
}

#ifdef MFEM_USE_OCCA
// OCCA PA Mass Apply 2D kernel
static void OccaPAMassApply2D(const int D1D,
                              const int Q1D,
                              const int NE,
                              const Array<double> &B,
                              const Array<double> &Bt,
                              const Vector &op,
                              const Vector &x,
                              Vector &y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = OccaMemoryRead(B.GetMemory(), B.Size());
   const occa::memory o_Bt = OccaMemoryRead(Bt.GetMemory(), Bt.Size());
   const occa::memory o_op = OccaMemoryRead(op.GetMemory(), op.Size());
   const occa::memory o_x = OccaMemoryRead(x.GetMemory(), x.Size());
   occa::memory o_y = OccaMemoryWrite(y.GetMemory(), y.Size());
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
                              const Array<double> &B,
                              const Array<double> &Bt,
                              const Vector &op,
                              const Vector &x,
                              Vector &y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = OccaMemoryRead(B.GetMemory(), B.Size());
   const occa::memory o_Bt = OccaMemoryRead(Bt.GetMemory(), Bt.Size());
   const occa::memory o_op = OccaMemoryRead(op.GetMemory(), op.Size());
   const occa::memory o_x = OccaMemoryRead(x.GetMemory(), x.Size());
   occa::memory o_y = OccaMemoryWrite(y.GetMemory(), y.Size());
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

template<const int T_D1D = 0, const int T_Q1D = 0> static
void PAMassApply2D(const int NE,
                   const Array<double> &_B,
                   const Array<double> &_Bt,
                   const Vector &_op,
                   const Vector &_x,
                   Vector &_y,
                   const int d1d = 0,
                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");

   auto B = Reshape(_B.ReadAccess(), Q1D, D1D);
   auto Bt = Reshape(_Bt.ReadAccess(), D1D, Q1D);
   auto op = Reshape(_op.ReadAccess(), Q1D, Q1D, NE);
   auto x = Reshape(_x.ReadAccess(), D1D, D1D, NE);
   auto y = Reshape(_y.WriteAccess(), D1D, D1D, NE);

   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int m_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int m_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double sol_xy[m_Q1D][m_Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         double sol_x[m_Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            sol_x[qy] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double s = x(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] += B(qx,dx)* s;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double d2q = B(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] += d2q * sol_x[qx];
            }
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            sol_xy[qy][qx] *= op(qx,qy,e);
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double sol_x[m_D1D];
         for (int dx = 0; dx < D1D; ++dx)
         {
            sol_x[dx] = 0.0;
         }
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double s = sol_xy[qy][qx];
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] += Bt(dx,qx) * s;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double q2d = Bt(dy,qy);
            for (int dx = 0; dx < D1D; ++dx)
            {
               y(dx,dy,e) += q2d * sol_x[dx];
            }
         }
      }
   });
}

template<const int T_D1D = 0, const int T_Q1D = 0> static
void PAMassApply3D(const int NE,
                   const Array<double> &_B,
                   const Array<double> &_Bt,
                   const Vector &_op,
                   const Vector &_x,
                   Vector &_y,
                   const int d1d = 0,
                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");

   auto B = Reshape(_B.ReadAccess(), Q1D, D1D);
   auto Bt = Reshape(_Bt.ReadAccess(), D1D, Q1D);
   auto op = Reshape(_op.ReadAccess(), Q1D, Q1D, Q1D, NE);
   auto x = Reshape(_x.ReadAccess(), D1D, D1D, D1D, NE);
   auto y = Reshape(_y.WriteAccess(), D1D, D1D, D1D, NE);

   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int m_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int m_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double sol_xyz[m_Q1D][m_Q1D][m_Q1D];
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xyz[qz][qy][qx] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < D1D; ++dz)
      {
         double sol_xy[m_Q1D][m_Q1D];
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xy[qy][qx] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            double sol_x[m_Q1D];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_x[qx] = 0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double s = x(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_x[qx] += B(qx,dx) * s;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double wy = B(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xy[qy][qx] += wy * sol_x[qx];
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            const double wz = B(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
               }
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               sol_xyz[qz][qy][qx] *= op(qx,qy,qz,e);
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz)
      {
         double sol_xy[m_D1D][m_D1D];
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_xy[dy][dx] = 0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            double sol_x[m_D1D];
            for (int dx = 0; dx < D1D; ++dx)
            {
               sol_x[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double s = sol_xyz[qz][qy][qx];
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_x[dx] += Bt(dx,qx) * s;
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double wy = Bt(dy,qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  sol_xy[dy][dx] += wy * sol_x[dx];
               }
            }
         }
         for (int dz = 0; dz < D1D; ++dz)
         {
            const double wz = Bt(dz,qz);
            for (int dy = 0; dy < D1D; ++dy)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  y(dx,dy,dz,e) += wz * sol_xy[dy][dx];
               }
            }
         }
      }
   });
}

static void PAMassApply(const int dim,
                        const int D1D,
                        const int Q1D,
                        const int NE,
                        const Array<double> &B,
                        const Array<double> &Bt,
                        const Vector &op,
                        const Vector &x,
                        Vector &y)
{
#ifdef MFEM_USE_OCCA
   if (DeviceUseOcca())
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
         case 0x22: PAMassApply2D<2,2>(NE, B, Bt, op, x, y); break;
         case 0x24: PAMassApply2D<2,4>(NE, B, Bt, op, x, y); break;
         case 0x33: PAMassApply2D<3,3>(NE, B, Bt, op, x, y); break;
         case 0x34: PAMassApply2D<3,4>(NE, B, Bt, op, x, y); break;
         case 0x35: PAMassApply2D<3,5>(NE, B, Bt, op, x, y); break;
         case 0x36: PAMassApply2D<3,6>(NE, B, Bt, op, x, y); break;
         case 0x44: PAMassApply2D<4,4>(NE, B, Bt, op, x, y); break;
         case 0x45: PAMassApply2D<4,5>(NE, B, Bt, op, x, y); break;
         case 0x46: PAMassApply2D<4,6>(NE, B, Bt, op, x, y); break;
         case 0x48: PAMassApply2D<4,8>(NE, B, Bt, op, x, y); break;
         case 0x55: PAMassApply2D<5,5>(NE, B, Bt, op, x, y); break;
         case 0x58: PAMassApply2D<5,8>(NE, B, Bt, op, x, y); break;
         default: PAMassApply2D(NE, B, Bt, op, x, y, D1D, Q1D);
      }
      return;
   }
   if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: PAMassApply2D<2,2>(NE, B, Bt, op, x, y); break;
         case 0x23: PAMassApply3D<2,3>(NE, B, Bt, op, x, y); break;
         case 0x24: PAMassApply3D<2,4>(NE, B, Bt, op, x, y); break;
         case 0x34: PAMassApply3D<3,4>(NE, B, Bt, op, x, y); break;
         case 0x45: PAMassApply3D<4,5>(NE, B, Bt, op, x, y); break;
         case 0x56: PAMassApply3D<5,6>(NE, B, Bt, op, x, y); break;
         default: PAMassApply3D(NE, B, Bt, op, x, y, D1D, Q1D);
      }
      return;
   }
   MFEM_ABORT("Unknown kernel.");
}

void MassIntegrator::MultAssembled(Vector &x, Vector &y)
{
   // FIXME: this function actually performs '+=', not '='
   PAMassApply(dim, dofs1D, quad1D, ne, maps->B, maps->Bt, pa_data, x, y);
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
   // FIXME: use mfem::CalcShapes() from tfe.hpp
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
   // FIXME: copy quadrature weights from 'ir'
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
   maps->B = B1d;
   maps->G = G1d;
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
      MFEM_WARNING("using hashed DofToQuad ...");
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
   // FIXME: use mfem::CalcShapes() from tfe.hpp
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
      maps->W = W;
   }
   maps->B = B;
   maps->G = G;
   return maps;
}


// FIXME: should not use a single global sequence
static long sequence = -1;
// FIXME: should not use a single global GeometryExtension?
static GeometryExtension *geom = NULL;

static void GeomFill(const int vdim,
                     const int NE, const int ND,
                     const Memory<int> &elementMap,
                     Array<int> &eMap,
                     const Vector &X, Array<double> &meshNodes)
{
   auto d_elementMap = ReadAccess(elementMap, ND*NE);
   auto d_eMap = eMap.WriteAccess();
   auto d_X = X.ReadAccess();
   auto d_meshNodes = meshNodes.WriteAccess();
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
            d_meshNodes[moffset] = d_X[xoffset];
         }
      }
   });
}

static void NodeCopyByVDim(const int elements,
                           const int numDofs,
                           const int ndofs,
                           const int dims,
                           const Array<int> &eMap,
                           const Vector &Sx,
                           Array<double> &nodes)
{
   auto d_eMap = eMap.ReadAccess();
   auto d_Sx = Sx.ReadAccess();
   auto d_nodes = nodes.WriteAccess();
   MFEM_FORALL(e,elements,
   {
      for (int dof = 0; dof < numDofs; ++dof)
      {
         const int lid = dof+numDofs*e;
         const int gid = d_eMap[lid];
         for (int v = 0; v < dims; ++v)
         {
            const int moffset = v+dims*lid;
            const int voffset = gid+v*ndofs;
            d_nodes[moffset] = d_Sx[voffset];
         }
      }
   });
}

// FIXME: the algorithm in this function does not use tensor product structure,
// so D1D and Q1D are not the right template parameters. The number of element
// dofs (ND) and number of element quadrature points (NQ) are more appropriate
// and then this function can be used with any 2D element type.
template<int T_D1D = 0, int T_Q1D = 0> static
void PAGeom2D(const int NE,
              const Array<double> &_B,
              const Array<double> &_G,
              const Array<double> &_X,
              Array<double> &_Xq,
              Array<double> &_J,
              Array<double> &_invJ,
              Array<double> &_detJ,
              const int d1d = 0,
              const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   const int ND = D1D*D1D;
   const int NQ = Q1D*Q1D;

   auto B = Reshape(_B.ReadAccess(), NQ, ND);
   auto G = Reshape(_G.ReadAccess(), 2, NQ, ND);
   auto X = Reshape(_X.ReadAccess(), 2, ND, NE);
   auto Xq = Reshape(_Xq.WriteAccess(), 2, NQ, NE);
   auto J = Reshape(_J.WriteAccess(), 2, 2, NQ, NE);
   auto invJ = Reshape(_invJ.WriteAccess(), 2, 2, NQ, NE);
   auto detJ = Reshape(_detJ.WriteAccess(), NQ, NE);

   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int ND = D1D*D1D;
      const int NQ = Q1D*Q1D;
      constexpr int m_D1D = T_D1D ? T_D1D : MAX_D1D;

      double s_X[2*m_D1D*m_D1D];
      for (int d = 0; d < ND; d++)
      {
         s_X[0+d*2] = X(0,d,e);
         s_X[1+d*2] = X(1,d,e);
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
            J11 += (x * wx); J12 += (x * wy);
            J21 += (y * wx); J22 += (y * wy);
            X0 += b*x; X1 += b*y;
         }
         Xq(0,q,e) = X0; Xq(1,q,e) = X1;
         const double r_detJ = (J11 * J22)-(J12 * J21);
         J(0,0,q,e) = J11;
         J(1,0,q,e) = J21;
         J(0,1,q,e) = J12;
         J(1,1,q,e) = J22;
         const double r_idetJ = 1.0 / r_detJ;
         invJ(0,0,q,e) =  J22 * r_idetJ;
         invJ(1,0,q,e) = -J21 * r_idetJ;
         invJ(0,1,q,e) = -J12 * r_idetJ;
         invJ(1,1,q,e) =  J11 * r_idetJ;
         detJ(q,e) = r_detJ;
      }
   });
}

// FIXME: the algorithm in this function does not use tensor product structure,
// so D1D and Q1D are not the right template parameters. The number of element
// dofs (ND) and number of element quadrature points (NQ) are more appropriate
// and then this function can be used with any 3D element type.
template<int T_D1D = 0, int T_Q1D = 0> static
void PAGeom3D(const int NE,
              const Array<double> &_B,
              const Array<double> &_G,
              const Array<double> &_X,
              Array<double> &_Xq,
              Array<double> &_J,
              Array<double> &_invJ,
              Array<double> &_detJ,
              const int d1d = 0,
              const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   const int ND = D1D*D1D*D1D;
   const int NQ = Q1D*Q1D*Q1D;

   auto B = Reshape(_B.ReadAccess(), NQ, ND);
   auto G = Reshape(_G.ReadAccess(), 3, NQ, ND);
   auto X = Reshape(_X.ReadAccess(), 3, ND, NE);
   auto Xq = Reshape(_Xq.WriteAccess(), 3, NQ, NE);
   auto J = Reshape(_J.WriteAccess(), 3, 3, NQ, NE);
   auto invJ = Reshape(_invJ.WriteAccess(), 3, 3, NQ, NE);
   auto detJ = Reshape(_detJ.WriteAccess(), NQ, NE);

   MFEM_FORALL(e,NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d; // nvcc workaround
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int ND = D1D*D1D*D1D;
      const int NQ = Q1D*Q1D*Q1D;
      constexpr int m_D1D = T_D1D ? T_D1D : MAX_D1D;

      double s_nodes[3*m_D1D*m_D1D*m_D1D];
      for (int d = 0; d < ND; d++)
      {
         s_nodes[0+d*3] = X(0,d,e);
         s_nodes[1+d*3] = X(1,d,e);
         s_nodes[2+d*3] = X(2,d,e);
      }
      for (int q = 0; q < NQ; ++q)
      {
         double X0 = 0, X1 = 0, X2 = 0;
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
            J11 += (x * wx); J12 += (x * wy); J13 += (x * wz);
            J21 += (y * wx); J22 += (y * wy); J23 += (y * wz);
            J31 += (z * wx); J32 += (z * wy); J33 += (z * wz);
            X0 += b*x; X1 += b*y; X2 += b*z;
         }
         Xq(0,q,e) = X0; Xq(1,q,e) = X1; Xq(2,q,e) = X2;
         const double r_detJ = J11 * (J22 * J33 - J32 * J23) -
                               J21 * (J12 * J33 - J32 * J13) +
                               J31 * (J12 * J23 - J22 * J13);
         J(0,0,q,e) = J11;
         J(1,0,q,e) = J21;
         J(2,0,q,e) = J31;
         J(0,1,q,e) = J12;
         J(1,1,q,e) = J22;
         J(2,1,q,e) = J32;
         J(0,2,q,e) = J13;
         J(1,2,q,e) = J23;
         J(2,2,q,e) = J33;
         const double r_idetJ = 1.0 / r_detJ;
         invJ(0,0,q,e) = r_idetJ * ((J22 * J33)-(J23 * J32));
         invJ(1,0,q,e) = r_idetJ * ((J23 * J31)-(J33 * J21));
         invJ(2,0,q,e) = r_idetJ * ((J21 * J32)-(J31 * J22));
         invJ(0,1,q,e) = r_idetJ * ((J32 * J13)-(J12 * J33));
         invJ(1,1,q,e) = r_idetJ * ((J33 * J11)-(J31 * J13));
         invJ(2,1,q,e) = r_idetJ * ((J31 * J12)-(J11 * J32));
         invJ(0,2,q,e) = r_idetJ * ((J12 * J23)-(J22 * J13));
         invJ(1,2,q,e) = r_idetJ * ((J13 * J21)-(J23 * J11));
         invJ(2,2,q,e) = r_idetJ * ((J11 * J22)-(J12 * J21));
         detJ(q,e) = r_detJ;
      }
   });
}

// FIXME: the algorithm in this function does not use tensor product structure,
// so D1D and Q1D are not really needed.
static void PAGeom(const int dim,
                   const int D1D,
                   const int Q1D,
                   const int NE,
                   const Array<double> &B,
                   const Array<double> &G,
                   const Array<double> &X,
                   Array<double> &Xq,
                   Array<double> &J,
                   Array<double> &invJ,
                   Array<double> &detJ)
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
   // TODO: use tensor product evaluations for tensor product elements
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

   // FIXME
   // const bool dev_enabled = Device::IsEnabled();
   // if (dev_enabled) { Device::Disable(); }
   mesh->EnsureNodes();
   // if (dev_enabled) { Device::Enable(); }

   GridFunction *nodes = mesh->GetNodes();
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
   const Table& e2dTable = fespace->GetElementToDofTable();
   const Memory<int> &elementMap = e2dTable.GetJMemory();
   if (geom_to_allocate)
   {
      geom->nodes.SetSize(dims*numDofs*elements);
      geom->eMap.SetSize(numDofs*elements);
   }
   GeomFill(dims,
            elements,
            numDofs,
            elementMap,
            geom->eMap,
            *nodes,
            geom->nodes);
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

void GeometryExtension::ReorderByVDim(GridFunction *nodes)
{
   const mfem::FiniteElementSpace *fes = nodes->FESpace();
   const int size = nodes->Size();
   const int vdim = fes->GetVDim();
   const int ndofs = fes->GetNDofs();
   double *data = nodes->ReadWriteAccess(false);
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

void GeometryExtension::ReorderByNodes(GridFunction *nodes)
{
   const mfem::FiniteElementSpace *fes = nodes->FESpace();
   const int size = nodes->Size();
   const int vdim = fes->GetVDim();
   const int ndofs = fes->GetNDofs();
   double *data = nodes->ReadWriteAccess(false);
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
