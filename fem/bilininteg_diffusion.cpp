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

using namespace std;

namespace mfem
{

// PA Diffusion Integrator

// OCCA 2D Assemble kernel
#ifdef MFEM_USE_OCCA
static void OccaPADiffusionSetup2D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const Array<double> &W,
                                   const Vector &J,
                                   const Vector &C,
                                   Vector &op)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_W = OccaMemoryRead(W.GetMemory(), W.Size());
   const occa::memory o_J = OccaMemoryRead(J.GetMemory(), J.Size());
   const occa::memory o_C = OccaMemoryRead(C.GetMemory(), C.Size());
   occa::memory o_op = OccaMemoryWrite(op.GetMemory(), op.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
   static occa_kernel_t OccaDiffSetup2D_ker;
   if (OccaDiffSetup2D_ker.find(id) == OccaDiffSetup2D_ker.end())
   {
      const occa::kernel DiffusionSetup2D =
         mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                     "DiffusionSetup2D", props);
      OccaDiffSetup2D_ker.emplace(id, DiffusionSetup2D);
   }
   OccaDiffSetup2D_ker.at(id)(NE, o_W, o_J, o_C, o_op);
}

static void OccaPADiffusionSetup3D(const int D1D,
                                   const int Q1D,
                                   const int NE,
                                   const Array<double> &W,
                                   const Vector &J,
                                   const Vector &C,
                                   Vector &op)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_W = OccaMemoryRead(W.GetMemory(), W.Size());
   const occa::memory o_J = OccaMemoryRead(J.GetMemory(), J.Size());
   const occa::memory o_C = OccaMemoryRead(C.GetMemory(), C.Size());
   occa::memory o_op = OccaMemoryWrite(op.GetMemory(), op.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
   static occa_kernel_t OccaDiffSetup3D_ker;
   if (OccaDiffSetup3D_ker.find(id) == OccaDiffSetup3D_ker.end())
   {
      const occa::kernel DiffusionSetup3D =
         mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                     "DiffusionSetup3D", props);
      OccaDiffSetup3D_ker.emplace(id, DiffusionSetup3D);
   }
   OccaDiffSetup3D_ker.at(id)(NE, o_W, o_J, o_C, o_op);
}
#endif // MFEM_USE_OCCA

// PA Diffusion Assemble 2D kernel
static void PADiffusionSetup2D(const int Q1D,
                               const int NE,
                               const Array<double> &w,
                               const Vector &j,
                               const Vector &c,
                               Vector &d)
{
   const int NQ = Q1D*Q1D;
   auto W = w.Read();

   auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   auto C = Reshape(c.Read(), NQ, NE);
   auto D = Reshape(d.Write(), NQ, 3, NE);

   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double c_detJ = W[q] * C(q,e) / ((J11*J22)-(J21*J12));
         D(q,0,e) =  c_detJ * (J12*J12 + J22*J22); // 1,1
         D(q,1,e) = -c_detJ * (J12*J11 + J22*J21); // 1,2
         D(q,2,e) =  c_detJ * (J11*J11 + J21*J21); // 2,2
      }
   });
}

// PA Diffusion Assemble 3D kernel
static void PADiffusionSetup3D(const int Q1D,
                               const int NE,
                               const Array<double> &w,
                               const Vector &j,
                               const Vector &c,
                               Vector &d)
{
   const int NQ = Q1D*Q1D*Q1D;
   auto W = w.Read();
   auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   auto C = Reshape(c.Read(), NQ, NE);
   auto D = Reshape(d.Write(), NQ, 6, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(q,0,0,e);
         const double J21 = J(q,1,0,e);
         const double J31 = J(q,2,0,e);
         const double J12 = J(q,0,1,e);
         const double J22 = J(q,1,1,e);
         const double J32 = J(q,2,1,e);
         const double J13 = J(q,0,2,e);
         const double J23 = J(q,1,2,e);
         const double J33 = J(q,2,2,e);
         const double detJ = J11 * (J22 * J33 - J32 * J23) -
         /* */               J21 * (J12 * J33 - J32 * J13) +
         /* */               J31 * (J12 * J23 - J22 * J13);
         const double c_detJ = W[q] * C(q,e) / detJ;
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
         D(q,0,e) = c_detJ * (A11*A11 + A12*A12 + A13*A13); // 1,1
         D(q,1,e) = c_detJ * (A11*A21 + A12*A22 + A13*A23); // 2,1
         D(q,2,e) = c_detJ * (A11*A31 + A12*A32 + A13*A33); // 3,1
         D(q,3,e) = c_detJ * (A21*A21 + A22*A22 + A23*A23); // 2,2
         D(q,4,e) = c_detJ * (A21*A31 + A22*A32 + A23*A33); // 3,2
         D(q,5,e) = c_detJ * (A31*A31 + A32*A32 + A33*A33); // 3,3
      }
   });
}

static void PADiffusionSetup(const int dim,
                             const int D1D,
                             const int Q1D,
                             const int NE,
                             const Array<double> &W,
                             const Vector &J,
                             const Vector &C,
                             Vector &D)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADiffusionSetup"); }
   if (dim == 2)
   {
#ifdef MFEM_USE_OCCA
      if (DeviceCanUseOcca())
      {
         OccaPADiffusionSetup2D(D1D, Q1D, NE, W, J, C, D);
         return;
      }
#endif // MFEM_USE_OCCA
      PADiffusionSetup2D(Q1D, NE, W, J, C, D);
   }
   if (dim == 3)
   {
#ifdef MFEM_USE_OCCA
      if (DeviceCanUseOcca())
      {
         OccaPADiffusionSetup3D(D1D, Q1D, NE, W, J, C, D);
         return;
      }
#endif // MFEM_USE_OCCA
      PADiffusionSetup3D(Q1D, NE, W, J, C, D);
   }
}

void DiffusionIntegrator::Setup(const FiniteElementSpace &fes)
{
   // Assumes tensor-product elements
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFE(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);
   const int dims = el.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6
   const int nq = ir->GetNPoints();
   dim = mesh->Dimension();
   ne = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(symmDims * nq * ne, Device::GetMemoryType());
   Vector coeff(nq * ne);
   if (Q == nullptr)
   {
      coeff = 1.0;
   }
   else if (ConstantCoefficient* cQ = dynamic_cast<ConstantCoefficient*>(Q))
   {
      coeff = cQ->constant;
   }
   else
   {
      auto C = Reshape(coeff.Write(), nq, ne);
      for(int e = 0; e < ne; ++e)
      {
         ElementTransformation& T = *fes.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            C(q,e) = Q->Eval(T, ir->IntPoint(q));
         }
      }
   }
   PADiffusionSetup(dim, dofs1D, quad1D, ne, ir->GetWeights(), geom->J, coeff,
                    pa_data);
}


template<int T_D1D = 0, int T_Q1D = 0>
static void PADiffusionDiagonal2D(const int NE,
                                  const Array<double> &b,
                                  const Array<double> &g,
                                  const Vector &op,
                                  Vector &diag,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   // see eg PADiffusionApply2D
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   // note different shape for op, this is a (symmetric) matrix,
   // we only store necessary entries
   auto Q = Reshape(op.Read(), Q1D*Q1D, 3, NE);
   auto Y = Reshape(diag.ReadWrite(), D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      // gradphi \cdot Q \gradphi has four terms
      // we could probably use symmetry to make it three?
      // 4 terms:
      // one   Gx By O11 Gx By;
      // two   Gx By O12 Bx Gy;
      // three Bx Gy O21 Gx By;
      // four  Bx Gy O22 Bx Gy
      // below I do them all at once, but you could save memory by
      // doing them one at a time (with longer code...)
      double temp01[max_Q1D][max_D1D];
      double temp02[max_Q1D][max_D1D];
      double temp03[max_Q1D][max_D1D];
      double temp04[max_Q1D][max_D1D];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            temp01[qx][dy] = 0.0;
            temp02[qx][dy] = 0.0;
            temp03[qx][dy] = 0.0;
            temp04[qx][dy] = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const int q = qx + qy * Q1D;
               const double O11 = Q(q,0,e);
               const double O12 = Q(q,1,e);
               const double O22 = Q(q,2,e);
               temp01[qx][dy]   += B(qy, dy) * B(qy, dy) * O11;
               temp02[qx][dy]   += B(qy, dy) * G(qy, dy) * O12;
               temp03[qx][dy] += G(qy, dy) * B(qy, dy) * O12;
               temp04[qx][dy]  += G(qy, dy) * G(qy, dy) * O22;
            }
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         for (int dx = 0; dx < D1D; ++dx)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               Y(dx,dy,e) += G(qx, dx) * G(qx, dx) * temp01[qx][dy];
               Y(dx,dy,e) += G(qx, dx) * B(qx, dx) * temp02[qx][dy];
               Y(dx,dy,e) += B(qx, dx) * G(qx, dx) * temp03[qx][dy];
               Y(dx,dy,e) += B(qx, dx) * B(qx, dx) * temp04[qx][dy];
            }
         }
      }
   });
}

// Shared memory PA Diffusion Diagonal 2D kernel
template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void SmemPADiffusionDiagonal2D(const int NE,
                                      const Array<double> &_b,
                                      const Array<double> &_g,
                                      const Vector &_q,
                                      Vector &_y,
                                      const int d1d = 0,
                                      const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   auto b = Reshape(_b.Read(), Q1D, D1D);
   auto g = Reshape(_g.Read(), Q1D, D1D);
   auto Q = Reshape(_q.Read(), Q1D*Q1D, 3, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      MFEM_SHARED double BG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (BG+0);
      double (*G)[MD1] = (double (*)[MD1]) (BG+1);
      MFEM_SHARED double T[4][NBZ][MD1][MQ1];
      double (*T0)[MD1] = (double (*)[MD1])(T[0] + tidz);
      double (*T1)[MD1] = (double (*)[MD1])(T[1] + tidz);
      double (*T2)[MD1] = (double (*)[MD1])(T[2] + tidz);
      double (*T3)[MD1] = (double (*)[MD1])(T[3] + tidz);
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            T0[qx][dy] = 0.0;
            T1[qx][dy] = 0.0;
            T2[qx][dy] = 0.0;
            T3[qx][dy] = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const int q = qx + qy * Q1D;
               const double O11 = Q(q,0,e);
               const double O12 = Q(q,1,e);
               const double O22 = Q(q,2,e);
               const double By = B[qy][dy];
               const double Gy = G[qy][dy];
               T0[qx][dy] += By * By * O11;
               T1[qx][dy] += By * Gy * O12;
               T2[qx][dy] += Gy * By * O12;
               T3[qx][dy] += Gy * Gy * O22;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double Bx = B[qx][dx];
               const double Gx = G[qx][dx];
               y(dx,dy,e) += Gx * Gx * T0[qx][dy];
               y(dx,dy,e) += Gx * Bx * T1[qx][dy];
               y(dx,dy,e) += Bx * Gx * T2[qx][dy];
               y(dx,dy,e) += Bx * Bx * T3[qx][dy];
            }
         }
      }
   });
}


template<int T_D1D = 0, int T_Q1D = 0>
static void PADiffusionDiagonal3D(const int NE,
                                  const Array<double> &b,
                                  const Array<double> &g,
                                  const Vector &op,
                                  Vector &y,
                                  const int d1d = 0,
                                  const int q1d = 0)
{
   // see eg PADiffusionApply3D
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Q = Reshape(op.Read(), Q1D*Q1D*Q1D, 6, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      // gradphi \cdot OP \gradphi has nine terms
      // nine terms might be too many, but for proof of concept that's what I'll do
      // (you could use symmetry to only have six?)

      // nine terms:
      // one   Gx By Bz O11 Gx By Bz;
      // two   Gx By Bz O12 Bx Gy Bz;
      // three Gx By Bz O13 Bx By Gz;
      // four  Bx Gy Bz O21 Gx By Bz;
      // five  Bx Gy Bz O22 Bx Gy Bz;
      // six   Bx Gy Bz O23 Bx By Gz;
      // seven Bx By Gz O31 Gx By Bz;
      // eight Bx By Gz O32 Bx Gy Bz;
      // nine  Bx By Gz O33 Bx By Gz;

      double ztemp01[max_Q1D][max_Q1D][max_D1D];
      double ztemp02[max_Q1D][max_Q1D][max_D1D];
      double ztemp03[max_Q1D][max_Q1D][max_D1D];
      double ztemp04[max_Q1D][max_Q1D][max_D1D];
      double ztemp05[max_Q1D][max_Q1D][max_D1D];
      double ztemp06[max_Q1D][max_Q1D][max_D1D];
      double ztemp07[max_Q1D][max_Q1D][max_D1D];
      double ztemp08[max_Q1D][max_Q1D][max_D1D];
      double ztemp09[max_Q1D][max_Q1D][max_D1D];

      // first tensor contraction, along z direction
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int dz = 0; dz < D1D; ++dz)
            {
               ztemp01[qx][qy][dz] = 0.0;
               ztemp02[qx][qy][dz] = 0.0;
               ztemp03[qx][qy][dz] = 0.0;
               ztemp04[qx][qy][dz] = 0.0;
               ztemp05[qx][qy][dz] = 0.0;
               ztemp06[qx][qy][dz] = 0.0;
               ztemp07[qx][qy][dz] = 0.0;
               ztemp08[qx][qy][dz] = 0.0;
               ztemp09[qx][qy][dz] = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  const int q = qx + (qy + qz * Q1D) * Q1D;
                  const double O11 = Q(q,0,e);
                  const double O12 = Q(q,1,e);
                  const double O13 = Q(q,2,e);
                  const double O22 = Q(q,3,e);
                  const double O23 = Q(q,4,e);
                  const double O33 = Q(q,5,e);

                  ztemp01[qx][qy][dz] += B(qz, dz) * B(qz, dz) * O11;
                  ztemp02[qx][qy][dz] += B(qz, dz) * B(qz, dz) * O12;
                  ztemp03[qx][qy][dz] += B(qz, dz) * G(qz, dz) * O13;
                  ztemp04[qx][qy][dz] += B(qz, dz) * B(qz, dz) * O12;
                  ztemp05[qx][qy][dz] += B(qz, dz) * B(qz, dz) * O22;
                  ztemp06[qx][qy][dz] += B(qz, dz) * G(qz, dz) * O23;
                  ztemp07[qx][qy][dz] += G(qz, dz) * B(qz, dz) * O13;
                  ztemp08[qx][qy][dz] += G(qz, dz) * B(qz, dz) * O23;
                  ztemp09[qx][qy][dz] += G(qz, dz) * G(qz, dz) * O33;
               }
            }
         }
      }

      double ytemp01[max_Q1D][max_D1D][max_D1D];
      double ytemp02[max_Q1D][max_D1D][max_D1D];
      double ytemp03[max_Q1D][max_D1D][max_D1D];
      double ytemp04[max_Q1D][max_D1D][max_D1D];
      double ytemp05[max_Q1D][max_D1D][max_D1D];
      double ytemp06[max_Q1D][max_D1D][max_D1D];
      double ytemp07[max_Q1D][max_D1D][max_D1D];
      double ytemp08[max_Q1D][max_D1D][max_D1D];
      double ytemp09[max_Q1D][max_D1D][max_D1D];

      // second tensor contraction, along y direction
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int dz = 0; dz < D1D; ++dz)
         {
            for (int dy = 0; dy < D1D; ++dy)
            {
               ytemp01[qx][dy][dz] = 0.0;
               ytemp02[qx][dy][dz] = 0.0;
               ytemp03[qx][dy][dz] = 0.0;
               ytemp04[qx][dy][dz] = 0.0;
               ytemp05[qx][dy][dz] = 0.0;
               ytemp06[qx][dy][dz] = 0.0;
               ytemp07[qx][dy][dz] = 0.0;
               ytemp08[qx][dy][dz] = 0.0;
               ytemp09[qx][dy][dz] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  ytemp01[qx][dy][dz] += B(qy, dy) * B(qy, dy) * ztemp01[qx][qy][dz];
                  ytemp02[qx][dy][dz] += B(qy, dy) * G(qy, dy) * ztemp02[qx][qy][dz];
                  ytemp03[qx][dy][dz] += B(qy, dy) * B(qy, dy) * ztemp03[qx][qy][dz];
                  ytemp04[qx][dy][dz] += G(qy, dy) * B(qy, dy) * ztemp04[qx][qy][dz];
                  ytemp05[qx][dy][dz] += G(qy, dy) * G(qy, dy) * ztemp05[qx][qy][dz];
                  ytemp06[qx][dy][dz] += G(qy, dy) * B(qy, dy) * ztemp06[qx][qy][dz];
                  ytemp07[qx][dy][dz] += B(qy, dy) * B(qy, dy) * ztemp07[qx][qy][dz];
                  ytemp08[qx][dy][dz] += B(qy, dy) * G(qy, dy) * ztemp08[qx][qy][dz];
                  ytemp09[qx][dy][dz] += B(qy, dy) * B(qy, dy) * ztemp09[qx][qy][dz];
               }
            }
         }
      }

      // third tensor contraction, along x direction
      for (int dz = 0; dz < D1D; ++dz)
      {
         for (int dy = 0; dy < D1D; ++dy)
         {
            for (int dx = 0; dx < D1D; ++dx)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  Y(dx, dy, dz, e) += G(qx, dx) * G(qx, dx) * ytemp01[qx][dy][dz];
                  Y(dx, dy, dz, e) += G(qx, dx) * B(qx, dx) * ytemp02[qx][dy][dz];
                  Y(dx, dy, dz, e) += G(qx, dx) * B(qx, dx) * ytemp03[qx][dy][dz];
                  Y(dx, dy, dz, e) += B(qx, dx) * G(qx, dx) * ytemp04[qx][dy][dz];
                  Y(dx, dy, dz, e) += B(qx, dx) * B(qx, dx) * ytemp05[qx][dy][dz];
                  Y(dx, dy, dz, e) += B(qx, dx) * B(qx, dx) * ytemp06[qx][dy][dz];
                  Y(dx, dy, dz, e) += B(qx, dx) * G(qx, dx) * ytemp07[qx][dy][dz];
                  Y(dx, dy, dz, e) += B(qx, dx) * B(qx, dx) * ytemp08[qx][dy][dz];
                  Y(dx, dy, dz, e) += B(qx, dx) * B(qx, dx) * ytemp09[qx][dy][dz];
               }
            }
         }
      }

   });
}

// Shared memory PA Diffusion Diagonal 3D kernelt
// Still uses too many resources for launch if order >= 5
template<int T_D1D = 0, int T_Q1D = 0>
static void SmemPADiffusionDiagonal3D(const int NE,
                                      const Array<double> &_b,
                                      const Array<double> &_g,
                                      const Vector &_q,
                                      Vector &_y,
                                      const int d1d = 0,
                                      const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   auto b = Reshape(_b.Read(), Q1D, D1D);
   auto g = Reshape(_g.Read(), Q1D, D1D);
   auto Q = Reshape(_q.Read(), Q1D*Q1D*Q1D, 6, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      MFEM_SHARED double BG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (BG+0);
      double (*G)[MD1] = (double (*)[MD1]) (BG+1);
      // gradphi \cdot OP \gradphi has nine terms
      // (might be too many, you could use symmetry to only have six)
      // nine terms:
      // one   Gx By Bz O11 Gx By Bz;
      // two   Gx By Bz O12 Bx Gy Bz;
      // three Gx By Bz O13 Bx By Gz;
      // four  Bx Gy Bz O21 Gx By Bz;
      // five  Bx Gy Bz O22 Bx Gy Bz;
      // six   Bx Gy Bz O23 Bx By Gz;
      // seven Bx By Gz O31 Gx By Bz;
      // eight Bx By Gz O32 Bx Gy Bz;
      // nine  Bx By Gz O33 Bx By Gz;
      MFEM_SHARED double sm[9][MQ1*MQ1*MD1];
      double (*QQD0)[MQ1][MD1] = (double (*)[MQ1][MD1])(sm+0);
      double (*QQD1)[MQ1][MD1] = (double (*)[MQ1][MD1])(sm+1);
      double (*QQD2)[MQ1][MD1] = (double (*)[MQ1][MD1])(sm+2);
      double (*QQD3)[MQ1][MD1] = (double (*)[MQ1][MD1])(sm+3);
      double (*QQD4)[MQ1][MD1] = (double (*)[MQ1][MD1])(sm+4);
      double (*QQD5)[MQ1][MD1] = (double (*)[MQ1][MD1])(sm+5);
      double (*QQD6)[MQ1][MD1] = (double (*)[MQ1][MD1])(sm+6);
      double (*QQD7)[MQ1][MD1] = (double (*)[MQ1][MD1])(sm+7);
      double (*QQD8)[MQ1][MD1] = (double (*)[MQ1][MD1])(sm+8);
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      // first tensor contraction, along z direction
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dz,z,D1D)
            {
               QQD0[qx][qy][dz] = 0.0;
               QQD1[qx][qy][dz] = 0.0;
               QQD2[qx][qy][dz] = 0.0;
               QQD3[qx][qy][dz] = 0.0;
               QQD4[qx][qy][dz] = 0.0;
               QQD5[qx][qy][dz] = 0.0;
               QQD6[qx][qy][dz] = 0.0;
               QQD7[qx][qy][dz] = 0.0;
               QQD8[qx][qy][dz] = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  const int q = qx + (qy + qz * Q1D) * Q1D;
                  const double O11 = Q(q,0,e);
                  const double O12 = Q(q,1,e);
                  const double O13 = Q(q,2,e);
                  const double O22 = Q(q,3,e);
                  const double O23 = Q(q,4,e);
                  const double O33 = Q(q,5,e);
                  const double Bz = B[qz][dz];
                  const double Gz = G[qz][dz];
                  const double BB = Bz * Bz;
                  const double BG = Bz * Gz;
                  const double GG = Gz * Gz;
                  QQD0[qx][qy][dz] += BB * O11;
                  QQD1[qx][qy][dz] += BB * O12;
                  QQD2[qx][qy][dz] += BG * O13;
                  QQD3[qx][qy][dz] += BB * O12;
                  QQD4[qx][qy][dz] += BB * O22;
                  QQD5[qx][qy][dz] += BG * O23;
                  QQD6[qx][qy][dz] += BG * O13;
                  QQD7[qx][qy][dz] += BG * O23;
                  QQD8[qx][qy][dz] += GG * O33;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // temporary tensors in registers
      double QDD0[MQ1][MD1][MD1];
      double QDD1[MQ1][MD1][MD1];
      double QDD2[MQ1][MD1][MD1];
      double QDD3[MQ1][MD1][MD1];
      double QDD4[MQ1][MD1][MD1];
      double QDD5[MQ1][MD1][MD1];
      double QDD6[MQ1][MD1][MD1];
      double QDD7[MQ1][MD1][MD1];
      double QDD8[MQ1][MD1][MD1];
      // second tensor contraction, along y direction
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               QDD0[qx][dy][dz] = 0.0;
               QDD1[qx][dy][dz] = 0.0;
               QDD2[qx][dy][dz] = 0.0;
               QDD3[qx][dy][dz] = 0.0;
               QDD4[qx][dy][dz] = 0.0;
               QDD5[qx][dy][dz] = 0.0;
               QDD6[qx][dy][dz] = 0.0;
               QDD7[qx][dy][dz] = 0.0;
               QDD8[qx][dy][dz] = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double By = B[qy][dy];
                  const double Gy = G[qy][dy];
                  const double BB = By * By;
                  const double BG = By * Gy;
                  const double GG = Gy * Gy;
                  QDD0[qx][dy][dz] += BB * QQD0[qx][qy][dz];
                  QDD1[qx][dy][dz] += BG * QQD1[qx][qy][dz];
                  QDD2[qx][dy][dz] += BB * QQD2[qx][qy][dz];
                  QDD3[qx][dy][dz] += BG * QQD3[qx][qy][dz];
                  QDD4[qx][dy][dz] += GG * QQD4[qx][qy][dz];
                  QDD5[qx][dy][dz] += BG * QQD5[qx][qy][dz];
                  QDD6[qx][dy][dz] += BB * QQD6[qx][qy][dz];
                  QDD7[qx][dy][dz] += BG * QQD7[qx][qy][dz];
                  QDD8[qx][dy][dz] += BB * QQD8[qx][qy][dz];
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      // third tensor contraction, along x direction
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double Bx = B[qx][dx];
                  const double Gx = G[qx][dx];
                  const double BB = Bx * Bx;
                  const double BG = Bx * Gx;
                  const double GG = Gx * Gx;
                  y(dx, dy, dz, e) += GG * QDD0[qx][dy][dz];
                  y(dx, dy, dz, e) += BG * QDD1[qx][dy][dz];
                  y(dx, dy, dz, e) += BG * QDD2[qx][dy][dz];
                  y(dx, dy, dz, e) += BG * QDD3[qx][dy][dz];
                  y(dx, dy, dz, e) += BB * QDD4[qx][dy][dz];
                  y(dx, dy, dz, e) += BB * QDD5[qx][dy][dz];
                  y(dx, dy, dz, e) += BG * QDD6[qx][dy][dz];
                  y(dx, dy, dz, e) += BB * QDD7[qx][dy][dz];
                  y(dx, dy, dz, e) += BB * QDD8[qx][dy][dz];
               }
            }
         }
      }
   });
}

static void PADiffusionAssembleDiagonal(const int dim,
                                        const int D1D,
                                        const int Q1D,
                                        const int NE,
                                        const Array<double> &B,
                                        const Array<double> &G,
                                        const Vector &op,
                                        Vector &y)
{
#ifdef MFEM_USE_OCCA
   if (DeviceCanUseOcca())
   {
      MFEM_ABORT("OCCA PADiffusionAssembleDiagonal unknown kernel!");
   }
#endif // MFEM_USE_OCCA

#ifdef MFEM_USE_RAJA
   if (Device::Allows(Backend::RAJA_CUDA))
   {
      if (dim == 2)
      {
         switch ((D1D << 4 ) | Q1D)
         {
            default:   return PADiffusionDiagonal2D(NE,B,G,op,y,D1D,Q1D);
         }
      }
      if (dim == 3)
      {
         switch ((D1D << 4 ) | Q1D)
         {
            default:   return PADiffusionDiagonal3D(NE,B,G,op,y,D1D,Q1D);
         }
      }
   }
   else
#endif // MFEM_USE_RAJA
      if (dim == 2)
      {
         switch ((D1D << 4 ) | Q1D)
         {
            case 0x22: return SmemPADiffusionDiagonal2D<2,2,16>(NE,B,G,op,y);
            case 0x33: return SmemPADiffusionDiagonal2D<3,3,16>(NE,B,G,op,y);
            case 0x44: return SmemPADiffusionDiagonal2D<4,4,8>(NE,B,G,op,y);
            case 0x55: return SmemPADiffusionDiagonal2D<5,5,8>(NE,B,G,op,y);
            case 0x66: return SmemPADiffusionDiagonal2D<6,6,4>(NE,B,G,op,y);
            case 0x77: return SmemPADiffusionDiagonal2D<7,7,4>(NE,B,G,op,y);
            case 0x88: return SmemPADiffusionDiagonal2D<8,8,2>(NE,B,G,op,y);
            case 0x99: return SmemPADiffusionDiagonal2D<9,9,2>(NE,B,G,op,y);
            default: return PADiffusionDiagonal2D(NE,B,G,op,y,D1D,Q1D);
         }
      }
      else if (dim == 3)
      {
         switch ((D1D << 4 ) | Q1D)
         {
            case 0x23: return SmemPADiffusionDiagonal3D<2,3>(NE,B,G,op,y);
            case 0x34: return SmemPADiffusionDiagonal3D<3,4>(NE,B,G,op,y);
            case 0x45: return SmemPADiffusionDiagonal3D<4,5>(NE,B,G,op,y);
            case 0x56: return SmemPADiffusionDiagonal3D<5,6>(NE,B,G,op,y);
            // beyond: too many resources requested for launch
            case 0x67: return PADiffusionDiagonal3D<6,7>(NE,B,G,op,y);
            case 0x78: return PADiffusionDiagonal3D<7,8>(NE,B,G,op,y);
            case 0x89: return PADiffusionDiagonal3D<8,9>(NE,B,G,op,y);
            case 0x9A: return PADiffusionDiagonal3D<9,10>(NE,B,G,op,y);
            default: return PADiffusionDiagonal3D(NE,B,G,op,y,D1D,Q1D);
         }
      }
   MFEM_ABORT("Unknown kernel.");
}

void DiffusionIntegrator::AssembleDiagonalPA(Vector& diag) const
{
   PADiffusionAssembleDiagonal(dim, dofs1D, quad1D, ne,
                               maps->B, maps->G, pa_data, diag);
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
   occa::memory o_y = OccaMemoryReadWrite(y.GetMemory(), y.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
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
   occa::memory o_y = OccaMemoryReadWrite(y.GetMemory(), y.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
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
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto Gt = Reshape(gt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D*Q1D, 3, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      // the following variables are evaluated at compile time
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      double grad[max_Q1D][max_Q1D][2];
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
         double gradX[max_Q1D][2];
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

            const double O11 = op(q,0,e);
            const double O12 = op(q,1,e);
            const double O22 = op(q,2,e);

            const double gradX = grad[qy][qx][0];
            const double gradY = grad[qy][qx][1];

            grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
            grad[qy][qx][1] = (O12 * gradX) + (O22 * gradY);
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         double gradX[max_D1D][2];
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

// Shared memory PA Diffusion Apply 2D kernel
template<const int T_D1D = 0,
         const int T_Q1D = 0,
         const int T_NBZ = 0>
static void SmemPADiffusionApply2D(const int NE,
                                   const Array<double> &_b,
                                   const Array<double> &_g,
                                   const Array<double> &_bt,
                                   const Array<double> &_gt,
                                   const Vector &_op,
                                   const Vector &_x,
                                   Vector &_y,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   auto b = Reshape(_b.Read(), Q1D, D1D);
   auto g = Reshape(_g.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D*Q1D, 3, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, NE);
   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int NBZ = T_NBZ ? T_NBZ : 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      MFEM_SHARED double sBG[2][MQ1*MD1];
      double (*B)[MD1] = (double (*)[MD1]) (sBG+0);
      double (*G)[MD1] = (double (*)[MD1]) (sBG+1);
      double (*Bt)[MQ1] = (double (*)[MQ1]) (sBG+0);
      double (*Gt)[MQ1] = (double (*)[MQ1]) (sBG+1);
      MFEM_SHARED double Xz[NBZ][MD1][MD1];
      MFEM_SHARED double GD[2][NBZ][MD1][MQ1];
      MFEM_SHARED double GQ[2][NBZ][MD1][MQ1];
      double (*X)[MD1] = (double (*)[MD1])(Xz + tidz);
      double (*DQ0)[MD1] = (double (*)[MD1])(GD[0] + tidz);
      double (*DQ1)[MD1] = (double (*)[MD1])(GD[1] + tidz);
      double (*QQ0)[MD1] = (double (*)[MD1])(GQ[0] + tidz);
      double (*QQ1)[MD1] = (double (*)[MD1])(GQ[1] + tidz);
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            X[dy][dx] = x(dx,dy,e);
         }
      }
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
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
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
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
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            const int q = (qx + ((qy) * Q1D));
            const double O11 = op(q,0,e);
            const double O12 = op(q,1,e);
            const double O22 = op(q,2,e);
            const double gX = QQ0[qy][qx];
            const double gY = QQ1[qy][qx];
            QQ0[qy][qx] = (O11 * gX) + (O12 * gY);
            QQ1[qy][qx] = (O12 * gX) + (O22 * gY);
         }
      }
      MFEM_SYNC_THREAD;
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               Bt[d][q] = b(q,d);
               Gt[d][q] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
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
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
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


// PA Diffusion Apply 3D kernel
template<const int T_D1D = 0,
         const int T_Q1D = 0> static
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
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto G = Reshape(g.Read(), Q1D, D1D);
   auto Bt = Reshape(bt.Read(), D1D, Q1D);
   auto Gt = Reshape(gt.Read(), D1D, Q1D);
   auto op = Reshape(_op.Read(), Q1D*Q1D*Q1D, 6, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, D1D, NE);
   MFEM_FORALL(e, NE,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;
      double grad[max_Q1D][max_Q1D][max_Q1D][3];
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
         double gradXY[max_Q1D][max_Q1D][3];
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
            double gradX[max_Q1D][2];
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
               const double O11 = op(q,0,e);
               const double O12 = op(q,1,e);
               const double O13 = op(q,2,e);
               const double O22 = op(q,3,e);
               const double O23 = op(q,4,e);
               const double O33 = op(q,5,e);
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
         double gradXY[max_D1D][max_D1D][3];
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
            double gradX[max_D1D][3];
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

// Shared memory PA Diffusion Apply 3D kernel
template<const int T_D1D = 0,
         const int T_Q1D = 0>
static void SmemPADiffusionApply3D(const int NE,
                                   const Array<double> &_b,
                                   const Array<double> &_g,
                                   const Array<double> &_bt,
                                   const Array<double> &_gt,
                                   const Vector &_op,
                                   const Vector &_x,
                                   Vector &_y,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   auto b = Reshape(_b.Read(), Q1D, D1D);
   auto g = Reshape(_g.Read(), Q1D, D1D);
   auto op = Reshape(_op.Read(), Q1D*Q1D*Q1D, 6, NE);
   auto x = Reshape(_x.Read(), D1D, D1D, D1D, NE);
   auto y = Reshape(_y.ReadWrite(), D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MDQ = MQ1 > MD1 ? MQ1 : MD1;
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
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               X[dz][dy][dx] = x(dx,dy,dz,e);
            }
         }
      }
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
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
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
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
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
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
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const int q = qx + ((qy*Q1D) + (qz*Q1D*Q1D));
               const double O11 = op(q,0,e);
               const double O12 = op(q,1,e);
               const double O13 = op(q,2,e);
               const double O22 = op(q,3,e);
               const double O23 = op(q,4,e);
               const double O33 = op(q,5,e);
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
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               Bt[d][q] = b(q,d);
               Gt[d][q] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
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
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
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
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
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
                             const Array<double> &B,
                             const Array<double> &G,
                             const Array<double> &Bt,
                             const Array<double> &Gt,
                             const Vector &op,
                             const Vector &x,
                             Vector &y)
{
#ifdef MFEM_USE_OCCA
   if (DeviceCanUseOcca())
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
         case 0x22: return SmemPADiffusionApply2D<2,2,16>(NE,B,G,Bt,Gt,op,x,y);
         case 0x33: return SmemPADiffusionApply2D<3,3,16>(NE,B,G,Bt,Gt,op,x,y);
         case 0x44: return SmemPADiffusionApply2D<4,4,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x55: return SmemPADiffusionApply2D<5,5,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x66: return SmemPADiffusionApply2D<6,6,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x77: return SmemPADiffusionApply2D<7,7,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x88: return SmemPADiffusionApply2D<8,8,2>(NE,B,G,Bt,Gt,op,x,y);
         case 0x99: return SmemPADiffusionApply2D<9,9,2>(NE,B,G,Bt,Gt,op,x,y);
         default:   return PADiffusionApply2D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
      }
   }
   else if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return SmemPADiffusionApply3D<2,3>(NE,B,G,Bt,Gt,op,x,y);
         case 0x34: return SmemPADiffusionApply3D<3,4>(NE,B,G,Bt,Gt,op,x,y);
         case 0x45: return SmemPADiffusionApply3D<4,5>(NE,B,G,Bt,Gt,op,x,y);
         case 0x56: return SmemPADiffusionApply3D<5,6>(NE,B,G,Bt,Gt,op,x,y);
         case 0x67: return SmemPADiffusionApply3D<6,7>(NE,B,G,Bt,Gt,op,x,y);
         case 0x78: return SmemPADiffusionApply3D<7,8>(NE,B,G,Bt,Gt,op,x,y);
         case 0x89: return SmemPADiffusionApply3D<8,9>(NE,B,G,Bt,Gt,op,x,y);
         default:   return PADiffusionApply3D(NE,B,G,Bt,Gt,op,x,y,D1D,Q1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

// PA Diffusion Apply kernel
void DiffusionIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   PADiffusionApply(dim, dofs1D, quad1D, ne,
                    maps->B, maps->G, maps->Bt, maps->Gt,
                    pa_data, x, y);
}

DiffusionIntegrator* DiffusionIntegrator::Copy() const
{
   return new DiffusionIntegrator(*this);
}

} // namespace mfem
