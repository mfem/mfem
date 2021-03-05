// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#include <iostream>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

/// ****************************************************************************
#ifdef __GNUG__
#include <cxxabi.h>

#include <cstdlib>
#include <memory>

static inline std::string demangle(const char *name)
{
   int status = -1;
   std::unique_ptr<char, void (*)(void *)> res
   {
      abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
   return (status == 0) ? res.get() : name;
}
#else
std::string demangle(const char *name) { return name; }
#endif  // __GNUG__

#include "mfem.hpp"

//#define MFEM_DEBUG_COLOR 154
//#include "general/debug.hpp"

#include "general/forall.hpp"
#include "fem/kernels.hpp"
#include "linalg/kernels.hpp"

#if defined(MFEM_USE_MPI)
//extern mfem::MPI_Session *GlobalMPISession;
//#define PFesGetParMeshGetComm(pfes) pfes.GetParMesh()->GetComm()
//#define PFesGetParMeshGetComm0(pfes) pfes.GetParMesh()->GetComm()
#else
#error Not tested
#define HYPRE_Int int
typedef int MPI_Session;
#define ParMesh Mesh
#define GetParMesh GetMesh
#define GlobalTrueVSize GetVSize
#define ParBilinearForm BilinearForm
#define ParGridFunction GridFunction
#define ParFiniteElementSpace FiniteElementSpace
#define PFesGetParMeshGetComm(...)
#define PFesGetParMeshGetComm0(...) 0
#define MPI_Finalize()
#define MPI_Allreduce(src,dst,...) *dst = *src
#define MPI_Reduce(src, dst, n, T,...) *dst = *src
#endif

using namespace mfem;

namespace mfem
{

mfem::Mesh *CreateMeshEx7(int order);

using FE = mfem::FiniteElement;
using QI = mfem::QuadratureInterpolator;

// Kernels addons //////////////////////////////////////////////////////////////
namespace kernels
{

template<typename T> MFEM_HOST_DEVICE inline
void HouseholderReflect(T *A, const T *v,
                        const T b, const int m, const int n,
                        const int row, const int col)
{
   for (int j = 0; j < n; j++)
   {
      T w = A[0*row + j*col];
      for (int i = 1; i < m; i++) { w += v[i] * A[i*row + j*col]; }
      A[0*row + j*col] -= b * w;
      for (int i = 1; i < m; i++) { A[i*row + j*col] -= b * w * v[i]; }
   }
}

template<int Q1D, typename T> MFEM_HOST_DEVICE inline
void HouseholderApplyQ(T *A, const T *Q, const T *tau,
                       const int k, const int row, const int col)
{
   T v[Q1D];
   for (int ii=0; ii<k; ii++)
   {
      const int i = k-1-ii;
      for (int j = i+1; j < Q1D; j++) { v[j] = Q[j*k+i]; }
      // Apply Householder reflector (I - tau v v^T) coG^T
      HouseholderReflect(&A[i*row], &v[i], tau[i], Q1D-i, Q1D, row, col);
   }
}

template<int D1D, int Q1D, typename T> MFEM_HOST_DEVICE inline
void QRFactorization(T *mat, T *tau)
{
   T v[Q1D];
   DeviceMatrix B(mat, D1D, Q1D);
   for (int i = 0; i < D1D; i++)
   {
      // Calculate Householder vector, magnitude
      T sigma = 0.0;
      v[i] = B(i,i);
      for (int j = i + 1; j < Q1D; j++)
      {
         v[j] = B(i,j);
         sigma += v[j] * v[j];
      }
      T norm = std::sqrt(v[i]*v[i] + sigma); // norm of v[i:m]
      T Rii = -copysign(norm, v[i]);
      v[i] -= Rii;
      // norm of v[i:m] after modification above and scaling below
      //   norm = sqrt(v[i]*v[i] + sigma) / v[i];
      //   tau = 2 / (norm*norm)
      tau[i] = 2 * v[i]*v[i] / (v[i]*v[i] + sigma);
      for (int j=i+1; j<Q1D; j++) { v[j] /= v[i]; }
      // Apply Householder reflector to lower right panel
      HouseholderReflect(&mat[i*D1D+i+1], &v[i], tau[i],
                         Q1D-i, D1D-i-1, D1D, 1);
      // Save v
      B(i,i) = Rii;
      for (int j=i+1; j<Q1D; j++) { B(i,j) = v[j]; }
   }
}

template<int D1D, int Q1D, typename T = double> MFEM_HOST_DEVICE inline
void GetCollocatedGrad(DeviceTensor<2,const T> b,
                       DeviceTensor<2,const T> g,
                       DeviceTensor<2,T> CoG)
{
   T tau[Q1D];
   T B1d[Q1D*D1D];
   T G1d[Q1D*D1D];
   DeviceMatrix B(B1d, D1D, Q1D);
   DeviceMatrix G(G1d, D1D, Q1D);

   for (int d = 0; d < D1D; d++)
   {
      for (int q = 0; q < Q1D; q++)
      {
         B(d,q) = b(q,d);
         G(d,q) = g(q,d);
      }
   }
   QRFactorization<D1D,Q1D>(B1d, tau);
   // Apply Rinv, colograd1d = grad1d Rinv
   for (int i = 0; i < Q1D; i++)
   {
      CoG(0,i) = G(0,i)/B(0,0);
      for (int j = 1; j < D1D; j++)
      {
         CoG(j,i) = G(j,i);
         for (int k = 0; k < j; k++) { CoG(j,i) -= B(j,k)*CoG(k,i); }
         CoG(j,i) /= B(j,j);
      }
      for (int j = D1D; j < Q1D; j++) { CoG(j,i) = 0.0; }
   }
   // Apply Qtranspose, colograd = colograd Qtranspose
   HouseholderApplyQ<Q1D>((T*)CoG, B1d, tau, D1D, 1, Q1D);
}

/// Multiply a vector with the transpose matrix.
template <typename TA, typename TX, typename TY>
MFEM_HOST_DEVICE inline void MultTranspose(const int H, const int W, TA *data,
                                           const TX *x, TY *y)
{
   double *d_col = data;
   for (int col = 0; col < W; col++)
   {
      double y_col = 0.0;
      for (int row = 0; row < H; row++)
      {
         y_col += x[row] * d_col[row];
      }
      y[col] = y_col;
      d_col += H;
   }
}

/// Multiply the transpose of a matrix A with a matrix B: At*B
template <typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline void MultAtB(const int Aheight, const int Awidth,
                                     const int Bwidth, const TA *Adata,
                                     const TB *Bdata, TC *AtBdata)
{
   const int ah = Aheight;
   const int aw = Awidth;
   const int bw = Bwidth;
   const double *ad = Adata;
   const double *bd = Bdata;
   double *cd = AtBdata;

   for (int j = 0; j < bw; j++)
   {
      const double *ap = ad;
      for (int i = 0; i < aw; i++)
      {
         double d = 0.0;
         for (int k = 0; k < ah; k++)
         {
            d += ap[k] * bd[k];
         }
         *(cd++) = d;
         ap += ah;
      }
      bd += ah;
   }
}

static MFEM_HOST_DEVICE inline int qi(const int q, const int d, const int Q)
{
   return (q <= d) ? q : Q - 1 - q;
}
static MFEM_HOST_DEVICE inline int dj(const int q, const int d, const int D)
{
   return (q <= d) ? d : D - 1 - d;
}
static MFEM_HOST_DEVICE inline int qk(const int q, const int d, const int Q)
{
   return (q <= d) ? Q - 1 - q : q;
}
static MFEM_HOST_DEVICE inline int dl(const int q, const int d, const int D)
{
   return (q <= d) ? D - 1 - d : d;
}
static MFEM_HOST_DEVICE inline double sign(const int q, const int d)
{
   return (q <= d) ? -1.0 : 1.0;
}

/// Load half-&-half B/G matrice into shared memory
template <int D1D, int Q1D>
MFEM_HOST_DEVICE inline void LoadBG_ijkl(const ConstDeviceMatrix b,
                                         const ConstDeviceMatrix g,
                                         double BG[D1D * Q1D])
{
   DeviceMatrix B(BG, D1D, Q1D);
   DeviceMatrix G(BG, D1D, Q1D);

   MFEM_FOREACH_THREAD(dy, y, D1D)
   {
      MFEM_FOREACH_THREAD(qx, x, Q1D)
      {
         const int i = qi(qx, dy, Q1D);
         const int j = dj(qx, dy, D1D);
         const int k = qk(qx, dy, Q1D);
         const int l = dl(qx, dy, D1D);
         B(i, j) = b(qx, dy);
         G(k, l) = g(qx, dy) * sign(qx, dy);
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load half-&-half transpose B/G matrice into shared memory
template <int D1D, int Q1D>
MFEM_HOST_DEVICE inline void LoadBGt_ijkl(const ConstDeviceMatrix b,
                                          const ConstDeviceMatrix g,
                                          double BG[D1D * Q1D])
{
   DeviceMatrix Bt(BG, D1D, Q1D);
   DeviceMatrix Gt(BG, D1D, Q1D);

   MFEM_FOREACH_THREAD(d, y, D1D)
   {
      MFEM_FOREACH_THREAD(q, x, Q1D)
      {
         const int i = qi(q, d, Q1D);
         const int j = dj(q, d, D1D);
         const int k = qk(q, d, Q1D);
         const int l = dl(q, d, D1D);
         Bt(j, i) = b(q, d);
         Gt(l, k) = g(q, d) * sign(q, d);
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 2D scalar input into shared memory from map
template <int MD1, int NBZ>
MFEM_HOST_DEVICE inline void LoadXD(const int e, const int D1D,
                                    const DeviceTensor<3, const int> MAP,
                                    const DeviceTensor<1, const double> xd,
                                    double sX[NBZ][MD1 * MD1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix X(sX[tidz], MD1, MD1);

   MFEM_FOREACH_THREAD(dy, y, D1D)
   {
      MFEM_FOREACH_THREAD(dx, x, D1D)
      {
         const int gid = MAP(dx, dy, e);
         const int j = gid >= 0 ? gid : -1 - gid;
         X(dx, dy) = xd(j);
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 3D scalar input vector into shared memory from map
template <int MD1, typename T = double> MFEM_HOST_DEVICE
inline void LoadXD(const int e, const int D1D,
                   const DeviceTensor<4, const int> MAP,
                   const DeviceTensor<1, const double> xd,
                   T sm[MD1 * MD1 * MD1])
{
   DeviceTensor<3,T> X(sm, MD1, MD1, MD1);

   MFEM_FOREACH_THREAD(dz, z, D1D)
   {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD(dx, x, D1D)
         {
            const int gid = MAP(dx, dy, dz, e);
            const int j = gid >= 0 ? gid : -1 - gid;
            X(dx, dy, dz) = xd(j);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

template <int MD1, typename T = double, int SMS = 1> MFEM_HOST_DEVICE
inline void LoadXDGather(const int e, const int D1D,
                         const DeviceTensor<4, const int> MAP,
                         const DeviceTensor<1, const double> xd,
                         T sm[MD1 * MD1 * MD1])
{
   DeviceTensor<3,T> X(sm, MD1, MD1, MD1);

   MFEM_FOREACH_THREAD(dz, z, D1D)
   {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD(dx, x, D1D)
         {
            // Gather
            T XD;
            for (int i = 0; i < SMS; i++)
            {
               const int gid = MAP(dx, dy, dz, e + i);
               const int j = gid >= 0 ? gid : -1 - gid;
               XD[i] = xd(j);
            }
            X(dx, dy, dz) = XD;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Load 3D scalar input vector into shared memory from map
template <int D1D>
MFEM_HOST_DEVICE inline void LoadXD_ijkl(const int e,
                                         const DeviceTensor<4, const int> MAP,
                                         const DeviceTensor<1, const double> xd,
                                         double (*__restrict__ X)[D1D][D1D])
{
   MFEM_FOREACH_THREAD(dz, z, D1D)
   {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD(dx, x, D1D)
         {
            const int gid = MAP(dx, dy, dz, e);
            const int j = gid >= 0 ? gid : -1 - gid;
            X[dz][dy][dx] = xd(j);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Pull 2D Scalar Evaluation
template <int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PullEval1(const int qx, const int qy,
                                       const double sQQ[NBZ][MQ1 * MQ1],
                                       double &P)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix QQ(sQQ[tidz], MQ1, MQ1);
   P = QQ(qx, qy);
}

/// Push 2D Scalar Evaluation
template <int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PushEval1(const int qx, const int qy,
                                       const double &P,
                                       double sQQ[NBZ][MQ1 * MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix QQ(sQQ[tidz], MQ1, MQ1);
   QQ(qx, qy) = P;
}

/// 2D Scalar Transposed evaluation, 1/2
template <int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void EvalXt(const int D1D, const int Q1D,
                                    const double sB[MQ1 * MD1],
                                    const double sQQ[NBZ][MQ1 * MQ1],
                                    double sDQ[NBZ][MD1 * MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sB, MQ1, MD1);
   ConstDeviceMatrix QQ(sQQ[tidz], MQ1, MQ1);
   DeviceMatrix DQ(sDQ[tidz], MQ1, MD1);

   MFEM_FOREACH_THREAD(qy, y, Q1D)
   {
      MFEM_FOREACH_THREAD(dx, x, D1D)
      {
         double u = 0.0;
         for (int qx = 0; qx < Q1D; ++qx)
         {
            u += QQ(qx, qy) * Bt(qx, dx);
         }
         DQ(qy, dx) = u;
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Scalar Transposed evaluation, 2/2
template <int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void EvalYt(const int D1D, const int Q1D,
                                    const double sB[MQ1 * MD1],
                                    const double sDQ[NBZ][MD1 * MQ1],
                                    DeviceTensor<3, double> Y, const int e)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sB, MQ1, MD1);
   ConstDeviceMatrix DQ(sDQ[tidz], MQ1, MD1);

   MFEM_FOREACH_THREAD(dy, y, D1D)
   {
      MFEM_FOREACH_THREAD(dx, x, D1D)
      {
         double u = 0.0;
         for (int qy = 0; qy < Q1D; ++qy)
         {
            u += Bt(qy, dy) * DQ(qy, dx);
         }
         Y(dx, dy, e) += u;
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Scalar Transposed evaluation, 2/2 to map
template <int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void EvalYtD(const int D1D, const int Q1D,
                                     const double sB[MQ1 * MD1],
                                     const double sDQ[NBZ][MD1 * MQ1],
                                     const DeviceTensor<3, const int> MAP,
                                     DeviceTensor<1, double> YD, const int e)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sB, MQ1, MD1);
   ConstDeviceMatrix DQ(sDQ[tidz], MQ1, MD1);

   MFEM_FOREACH_THREAD(dy, y, D1D)
   {
      MFEM_FOREACH_THREAD(dx, x, D1D)
      {
         double u = 0.0;
         for (int qy = 0; qy < Q1D; ++qy)
         {
            u += Bt(qy, dy) * DQ(qy, dx);
         }
         const int gid = MAP(dx, dy, e);
         const int j = gid >= 0 ? gid : -1 - gid;
         AtomicAdd(YD(j), u);
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Scalar gradient, 1/2
template <int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Grad1X(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1 * MD1],
                                    const double XY[NBZ][MD1 * MD1],
                                    double DQ[2][NBZ][MD1 * MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sBG[0], MQ1, MD1);
   ConstDeviceMatrix G(sBG[1], MQ1, MD1);
   ConstDeviceMatrix X0(XY[tidz], MD1, MD1);
   DeviceMatrix QD0(DQ[0][tidz], MQ1, MD1);
   DeviceMatrix QD1(DQ[1][tidz], MQ1, MD1);

   MFEM_FOREACH_THREAD(dy, y, D1D)
   {
      MFEM_FOREACH_THREAD(qx, x, Q1D)
      {
         double u = 0.0;
         double v = 0.0;
         for (int dx = 0; dx < D1D; ++dx)
         {
            u += G(dx, qx) * X0(dx, dy);
            v += B(dx, qx) * X0(dx, dy);
         }
         QD0(qx, dy) = u;
         QD1(qx, dy) = v;
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Scalar gradient, 2/2
template <int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Grad1Y(const int D1D, const int Q1D,
                                    const double sBG[2][MQ1 * MD1],
                                    const double QD[2][NBZ][MD1 * MQ1],
                                    double sQQ[2][NBZ][MQ1 * MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sBG[0], MQ1, MD1);
   ConstDeviceMatrix G(sBG[1], MQ1, MD1);
   ConstDeviceMatrix QD0(QD[0][tidz], MQ1, MD1);
   ConstDeviceMatrix QD1(QD[1][tidz], MQ1, MD1);
   DeviceMatrix QQ0(sQQ[0][tidz], MQ1, MQ1);
   DeviceMatrix QQ1(sQQ[1][tidz], MQ1, MQ1);

   MFEM_FOREACH_THREAD(qy, y, Q1D)
   {
      MFEM_FOREACH_THREAD(qx, x, Q1D)
      {
         double u = 0.0;
         double v = 0.0;
         for (int dy = 0; dy < D1D; ++dy)
         {
            u += QD0(qx, dy) * B(dy, qy);
            v += QD1(qx, dy) * G(dy, qy);
         }
         QQ0(qx, qy) = u;
         QQ1(qx, qy) = v;
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Scalar gradient
template <int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Grad2D(const int D1D, const int Q1D,
                                    const double BG[2][MQ1 * MD1],
                                    const double XY[NBZ][MD1 * MD1],
                                    double QQ[2][NBZ][MQ1 * MQ1])
{
   double DQ[2][NBZ][MD1 * MQ1];
   Grad1X<MD1, MQ1, NBZ>(D1D, Q1D, BG, XY, DQ);
   Grad1Y<MD1, MQ1, NBZ>(D1D, Q1D, BG, DQ, QQ);
}

/// Pull 2D Scalar Gradient
template <int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PullGrad1(const int qx, const int qy,
                                       const double sQQ[2][NBZ][MQ1 * MQ1],
                                       double *A)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix X0(sQQ[0][tidz], MQ1, MQ1);
   ConstDeviceMatrix X1(sQQ[1][tidz], MQ1, MQ1);

   A[0] = X0(qx, qy);
   A[1] = X1(qx, qy);
}

/// Push 2D Scalar Gradient
template <int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PushGrad1(const int qx, const int qy,
                                       const double *A,
                                       double sQQ[2][NBZ][MQ1 * MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix X0(sQQ[0][tidz], MQ1, MQ1);
   DeviceMatrix X1(sQQ[1][tidz], MQ1, MQ1);

   X0(qx, qy) = A[0];
   X1(qx, qy) = A[1];
}

/// 2D Scalar Transposed gradient, 1/2
template <int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Grad1Yt(const int D1D, const int Q1D,
                                     const double sBG[2][MQ1 * MD1],
                                     const double GQ[2][NBZ][MQ1 * MQ1],
                                     double GD[2][NBZ][MD1 * MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   ConstDeviceMatrix Gt(sBG[1], MQ1, MD1);
   ConstDeviceMatrix QQ0(GQ[0][tidz], MQ1, MQ1);
   ConstDeviceMatrix QQ1(GQ[1][tidz], MQ1, MQ1);
   DeviceMatrix DQ0(GD[0][tidz], MQ1, MD1);
   DeviceMatrix DQ1(GD[1][tidz], MQ1, MD1);

   MFEM_FOREACH_THREAD(qy, y, Q1D)
   {
      MFEM_FOREACH_THREAD(dx, x, D1D)
      {
         double u = 0.0;
         double v = 0.0;
         for (int qx = 0; qx < Q1D; ++qx)
         {
            u += Gt(qx, dx) * QQ0(qx, qy);
            v += Bt(qx, dx) * QQ1(qx, qy);
         }
         DQ0(qy, dx) = u;
         DQ1(qy, dx) = v;
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Scalar Transposed gradient, 2/2
template <int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Grad1Xt(const int D1D, const int Q1D,
                                     const double sBG[2][MQ1 * MD1],
                                     const double GD[2][NBZ][MD1 * MQ1],
                                     mfem::DeviceTensor<3, double> Y,
                                     const int e)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   ConstDeviceMatrix Gt(sBG[1], MQ1, MD1);
   ConstDeviceMatrix DQ0(GD[0][tidz], MQ1, MD1);
   ConstDeviceMatrix DQ1(GD[1][tidz], MQ1, MD1);

   MFEM_FOREACH_THREAD(dy, y, D1D)
   {
      MFEM_FOREACH_THREAD(dx, x, D1D)
      {
         double u = 0.0;
         double v = 0.0;
         for (int qy = 0; qy < Q1D; ++qy)
         {
            u += DQ0(qy, dx) * Bt(qy, dy);
            v += DQ1(qy, dx) * Gt(qy, dy);
         }
         Y(dx, dy, e) += u + v;
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Scalar Transposed gradient to map, 2/2
template <int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Grad1XtD(const int D1D, const int Q1D,
                                      const double sBG[2][MQ1 * MD1],
                                      const double GD[2][NBZ][MD1 * MQ1],
                                      const DeviceTensor<3, const int> MAP,
                                      DeviceTensor<1, double> YD, const int e)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   ConstDeviceMatrix Gt(sBG[1], MQ1, MD1);
   ConstDeviceMatrix DQ0(GD[0][tidz], MQ1, MD1);
   ConstDeviceMatrix DQ1(GD[1][tidz], MQ1, MD1);

   MFEM_FOREACH_THREAD(dy, y, D1D)
   {
      MFEM_FOREACH_THREAD(dx, x, D1D)
      {
         double u = 0.0;
         double v = 0.0;
         for (int qy = 0; qy < Q1D; ++qy)
         {
            u += DQ0(qy, dx) * Bt(qy, dy);
            v += DQ1(qy, dx) * Gt(qy, dy);
         }
         const double value = u + v;
         const int gid = MAP(dx, dy, e);
         const int j = gid >= 0 ? gid : -1 - gid;
         AtomicAdd(YD(j), value);
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Scalar Transposed gradient
template <int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Grad2Dt(const int D1D, const int Q1D,
                                     const double BG[2][MQ1 * MD1],
                                     const double QQ[2][NBZ][MQ1 * MQ1],
                                     const DeviceTensor<3, const int> MAP,
                                     DeviceTensor<1, double> YD, const int e)
{
   double DQ[2][NBZ][MD1 * MQ1];
   Grad1Yt<MD1, MQ1, NBZ>(D1D, Q1D, BG, QQ, DQ);
   Grad1XtD<MD1, MQ1, NBZ>(D1D, Q1D, BG, DQ, MAP, YD, e);
}

///////////////////////////////////////////////////////////////////////////////
/// Push 3D Scalar Evaluation
template <int MQ1>
MFEM_HOST_DEVICE inline void PushEval(const int x, const int y, const int z,
                                      const double &A,
                                      double sQQQ[MQ1 * MQ1 * MQ1])
{
   DeviceCube XxBBB(sQQQ, MQ1, MQ1, MQ1);
   XxBBB(x, y, z) = A;
}

/// 3D Transposed Scalar Evaluation, 1/3
template <int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalXt(const int D1D, const int Q1D,
                                    const double sB[MQ1 * MD1],
                                    const double sQQQ[MQ1 * MQ1 * MQ1],
                                    double sDQQ[MD1 * MQ1 * MQ1])
{
   ConstDeviceMatrix Bt(sB, MQ1, MD1);
   ConstDeviceCube XxBBB(sQQQ, MQ1, MQ1, MQ1);
   DeviceCube XxBB(sDQQ, MQ1, MQ1, MD1);

   MFEM_FOREACH_THREAD(qz, z, Q1D)
   {
      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD(dx, x, D1D)
         {
            double u = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double Btx = Bt(qx, dx);
               u += XxBBB(qx, qy, qz) * Btx;
            }
            XxBB(qz, qy, dx) = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Transposed Scalar Evaluation, 2/3
template <int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalYt(const int D1D, const int Q1D,
                                    const double sB[MQ1 * MD1],
                                    const double sDQQ[MD1 * MQ1 * MQ1],
                                    double sDDQ[MD1 * MD1 * MQ1])
{
   ConstDeviceMatrix Bt(sB, MQ1, MD1);
   ConstDeviceCube XxBB(sDQQ, MQ1, MQ1, MD1);
   DeviceCube XxB(sDDQ, MQ1, MD1, MD1);

   MFEM_FOREACH_THREAD(qz, z, Q1D)
   {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD(dx, x, D1D)
         {
            double u = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double Bty = Bt(qy, dy);
               u += XxBB(qz, qy, dx) * Bty;
            }
            XxB(qz, dy, dx) = u;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Transposed Scalar Evaluation, 3/3
template <int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalZt(const int D1D, const int Q1D,
                                    const double sB[MQ1 * MD1],
                                    const double sDDQ[MD1 * MD1 * MQ1],
                                    DeviceTensor<4, double> Y, const int e)
{
   ConstDeviceMatrix Bt(sB, MQ1, MD1);
   ConstDeviceCube XxB(sDDQ, MQ1, MD1, MD1);

   MFEM_FOREACH_THREAD(dz, z, D1D)
   {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD(dx, x, D1D)
         {
            double u = 0.0;
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double Btz = Bt(qz, dz);
               u += XxB(qz, dy, dx) * Btz;
            }
            Y(dx, dy, dz, e) += u;
         }
      }
   }
}

/// 3D Transposed Scalar Evaluation, 3/3
template <int MD1, int MQ1>
MFEM_HOST_DEVICE inline void EvalZtD(const int D1D, const int Q1D,
                                     const double sB[MQ1 * MD1],
                                     const double sDDQ[MD1 * MD1 * MQ1],
                                     const DeviceTensor<4, const int> MAP,
                                     DeviceTensor<1, double> YD, const int e)
{
   ConstDeviceMatrix Bt(sB, MQ1, MD1);
   ConstDeviceCube XxB(sDDQ, MQ1, MD1, MD1);

   MFEM_FOREACH_THREAD(dz, z, D1D)
   {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD(dx, x, D1D)
         {
            double u = 0.0;
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double Btz = Bt(qz, dz);
               u += XxB(qz, dy, dx) * Btz;
            }
            const int gid = MAP(dx, dy, dz, e);
            const int j = gid >= 0 ? gid : -1 - gid;
            AtomicAdd(YD(j), u);
         }
      }
   }
}

/// 3D Scalar Gradient, 1/3
template <int MD1, int MQ1, typename T> MFEM_HOST_DEVICE inline
void Grad1X(const int D1D, const int Q1D,
            const double (*__restrict__ BG)[MQ1 * MD1],
            const T(*__restrict__ DDD),
            T (*__restrict__ DDQ)[MD1 * MD1 * MQ1])
{
   DeviceTensor<2,const double> B(BG[0], MD1, MQ1);
   DeviceTensor<2,const double> G(BG[1], MD1, MQ1);
   DeviceTensor<3,const T> X(DDD, MD1, MD1, MD1);
   DeviceTensor<3,T> XB(DDQ[0], MQ1, MD1, MD1);
   DeviceTensor<3,T> XG(DDQ[1], MQ1, MD1, MD1);

   MFEM_FOREACH_THREAD(dz, z, D1D)
   {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            T u; u = 0.0;
            T v; v = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               const T xx = X(dx, dy, dz);
               const double Bx = B(dx, qx);
               const double Gx = G(dx, qx);
               u += Bx * xx;
               v += Gx * xx;
            }
            XB(qx, dy, dz) = u;
            XG(qx, dy, dz) = v;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Scalar Gradient, 2/3
template <int MD1, int MQ1, typename T> MFEM_HOST_DEVICE
inline void Grad1Y(const int D1D, const int Q1D,
                   const double (*__restrict__ BG)[MQ1 * MD1],
                   const T (*__restrict__ DDQ)[MD1 * MD1 * MQ1],
                   T (*__restrict__ DQQ)[MD1 * MQ1 * MQ1])
{
   DeviceTensor<2,const double> B(BG[0], MD1, MQ1);
   DeviceTensor<2,const double> G(BG[1], MD1, MQ1);
   DeviceTensor<3,const T> XB(DDQ[0], MQ1, MD1, MD1);
   DeviceTensor<3,const T> XG(DDQ[1], MQ1, MD1, MD1);
   DeviceTensor<3,T> XBB(DQQ[0], MQ1, MQ1, MD1);
   DeviceTensor<3,T> XBG(DQQ[1], MQ1, MQ1, MD1);
   DeviceTensor<3,T> XGB(DQQ[2], MQ1, MQ1, MD1);

   MFEM_FOREACH_THREAD(dz, z, D1D)
   {
      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            T u; u = 0.0;
            T v; v = 0.0;
            T w; w = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double By = B(dy, qy);
               const double Gy = G(dy, qy);
               u += XB(qx, dy, dz) * By;
               v += XG(qx, dy, dz) * By;
               w += XB(qx, dy, dz) * Gy;
            }
            XBB(qx, qy, dz) = u;
            XBG(qx, qy, dz) = v;
            XGB(qx, qy, dz) = w;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Scalar Gradient, 3/3
template <int MD1, int MQ1, typename T = double> MFEM_HOST_DEVICE
inline void Grad1Z(const int D1D, const int Q1D,
                   const double (*__restrict__ BG)[MQ1 * MD1],
                   const T (*__restrict__ DQQ)[MD1 * MQ1 * MQ1],
                   T (*__restrict__ QQQ)[MQ1 * MQ1 * MQ1])
{
   DeviceTensor<2,const double> B(BG[0], MD1, MQ1);
   DeviceTensor<2,const double> G(BG[1], MD1, MQ1);
   DeviceTensor<3,const T> XBB(DQQ[0], MQ1, MQ1, MD1);
   DeviceTensor<3,const T> XBG(DQQ[1], MQ1, MQ1, MD1);
   DeviceTensor<3,const T> XGB(DQQ[2], MQ1, MQ1, MD1);
   DeviceTensor<3,T> XBBG(QQQ[0], MQ1, MQ1, MQ1);
   DeviceTensor<3,T> XBGB(QQQ[1], MQ1, MQ1, MQ1);
   DeviceTensor<3,T> XGBB(QQQ[2], MQ1, MQ1, MQ1);

   MFEM_FOREACH_THREAD(qz, z, Q1D)
   {
      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD(qx, x, Q1D)
         {
            T u; u = 0.0;
            T v; v = 0.0;
            T w; w = 0.0;
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double Bz = B(dz, qz);
               const double Gz = G(dz, qz);
               u += XBG(qx, qy, dz) * Bz;
               v += XGB(qx, qy, dz) * Bz;
               w += XBB(qx, qy, dz) * Gz;
            }
            XBBG(qx, qy, qz) = u;
            XBGB(qx, qy, qz) = v;
            XGBB(qx, qy, qz) = w;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// Pull 3D Scalar Gradient
template <int MQ1, typename T> MFEM_HOST_DEVICE inline
void PullGrad1(const int x, const int y, const int z,
               const T(*__restrict__ QQQ) /*[3]*/[MQ1 * MQ1 * MQ1],
               T *__restrict__ A)
{
   DeviceTensor<3,const T> BBG(QQQ[0], MQ1, MQ1, MQ1);
   DeviceTensor<3,const T> BGB(QQQ[1], MQ1, MQ1, MQ1);
   DeviceTensor<3,const T> GBB(QQQ[2], MQ1, MQ1, MQ1);

   A[0] = BBG(x, y, z);
   A[1] = BGB(x, y, z);
   A[2] = GBB(x, y, z);
}

template <int Q1D> MFEM_HOST_DEVICE inline
void PullGrad1_ijkl(const int x, const int y, const int z,
                    const double (*__restrict__ QQQ0)[Q1D][Q1D],
                    const double (*__restrict__ QQQ1)[Q1D][Q1D],
                    const double (*__restrict__ QQQ2)[Q1D][Q1D],
                    double *__restrict__ A)
{
   A[0] = QQQ0[z][y][x];
   A[1] = QQQ1[z][y][x];
   A[2] = QQQ2[z][y][x];
}

/// Push 3D Scalar Gradient
template <int MQ1, typename T> MFEM_HOST_DEVICE inline
void PushGrad1(const int x, const int y, const int z,
               const T *__restrict__ A,
               T(*__restrict__ QQQ) /*[3]*/[MQ1 * MQ1 * MQ1])
{
   DeviceTensor<3,T> BBG(QQQ[0], MQ1, MQ1, MQ1);
   DeviceTensor<3,T> BGB(QQQ[1], MQ1, MQ1, MQ1);
   DeviceTensor<3,T> GBB(QQQ[2], MQ1, MQ1, MQ1);

   BBG(x, y, z) = A[0];
   BGB(x, y, z) = A[1];
   GBB(x, y, z) = A[2];
}

template <int Q1D>
MFEM_HOST_DEVICE inline void PushGrad1_ijkl(
   const int x, const int y, const int z, const double *__restrict__ A,
   double (*__restrict__ QQQ0)[Q1D][Q1D],
   double (*__restrict__ QQQ1)[Q1D][Q1D],
   double (*__restrict__ QQQ2)[Q1D][Q1D])
{
   QQQ0[z][y][x] = A[0];
   QQQ1[z][y][x] = A[1];
   QQQ2[z][y][x] = A[2];
}

/// 3D Transposed Scalar Gradient, 1/3
template <int MD1, int MQ1, typename T> MFEM_HOST_DEVICE inline
void Grad1Zt(const int D1D, const int Q1D,
             const double (*__restrict__ BG) /*[2]*/[MQ1 * MD1],
             const T(*__restrict__ QQQ) /*[3]*/[MQ1 * MQ1 * MQ1],
             T(*__restrict__ DQQ) /*[3]*/[MD1 * MQ1 * MQ1])
{
   DeviceTensor<2,const double> Bt(BG[0], MQ1, MD1);
   DeviceTensor<2,const double> Gt(BG[1], MQ1, MD1);
   DeviceTensor<3,const T> XBBG(QQQ[0], MQ1, MQ1, MQ1);
   DeviceTensor<3,const T> XBGB(QQQ[1], MQ1, MQ1, MQ1);
   DeviceTensor<3,const T> XGBB(QQQ[2], MQ1, MQ1, MQ1);
   DeviceTensor<3,T> XBB(DQQ[0], MQ1, MQ1, MD1);
   DeviceTensor<3,T> XBG(DQQ[1], MQ1, MQ1, MD1);
   DeviceTensor<3,T> XGB(DQQ[2], MQ1, MQ1, MD1);

   MFEM_FOREACH_THREAD(qz, z, Q1D)
   {
      MFEM_FOREACH_THREAD(qy, y, Q1D)
      {
         MFEM_FOREACH_THREAD(dx, x, D1D)
         {
            T u; u = 0.0;
            T v; v = 0.0;
            T w; w = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double Btx = Bt(qx, dx);
               const double Gtx = Gt(qx, dx);
               u += XBBG(qx, qy, qz) * Gtx;
               v += XBGB(qx, qy, qz) * Btx;
               w += XGBB(qx, qy, qz) * Btx;
            }
            XBB(qz, qy, dx) = u;
            XBG(qz, qy, dx) = v;
            XGB(qz, qy, dx) = w;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Transposed Scalar Gradient, 2/3
template <int MD1, int MQ1, typename T> MFEM_HOST_DEVICE inline
void Grad1Yt(const int D1D, const int Q1D,
             const double(*__restrict__ BG) /*[2]*/[MQ1 * MD1],
             const T(*__restrict__ DQQ) /*[3]*/[MD1 * MQ1 * MQ1],
             T DDQ[3][MD1 * MD1 * MQ1])
{
   DeviceTensor<2,const double> Bt(BG[0], MQ1, MD1);
   DeviceTensor<2,const double> Gt(BG[1], MQ1, MD1);
   DeviceTensor<3,const T> XBB(DQQ[0], MQ1, MQ1, MD1);
   DeviceTensor<3,const T> XBG(DQQ[1], MQ1, MQ1, MD1);
   DeviceTensor<3,const T> XGB(DQQ[2], MQ1, MQ1, MD1);
   DeviceTensor<3,T> XB(DDQ[0], MQ1, MD1, MD1);
   DeviceTensor<3,T> XG(DDQ[1], MQ1, MD1, MD1);
   DeviceTensor<3,T> XC(DDQ[2], MQ1, MD1, MD1);

   MFEM_FOREACH_THREAD(qz, z, Q1D)
   {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD(dx, x, D1D)
         {
            T u; u = 0.0;
            T v; v = 0.0;
            T w; w = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double Bty = Bt(qy, dy);
               const double Gty = Gt(qy, dy);
               u += XBB(qz, qy, dx) * Bty;
               v += XBG(qz, qy, dx) * Gty;
               w += XGB(qz, qy, dx) * Bty;
            }
            XB(qz, dy, dx) = u;
            XG(qz, dy, dx) = v;
            XC(qz, dy, dx) = w;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

/// 3D Transposed Gradient, 3/3
template <int MD1, int MQ1>
MFEM_HOST_DEVICE inline void Grad1Xt(
   const int D1D, const int Q1D,
   const double(*__restrict__ BG) /*[2]*/[MQ1 * MD1],
   const double(*__restrict__ DDQ) /*[3]*/[MD1 * MD1 * MQ1],
   DeviceTensor<4, double> Y, const int e)
{
   ConstDeviceMatrix Bt(BG[0], MQ1, MD1);
   ConstDeviceMatrix Gt(BG[1], MQ1, MD1);
   ConstDeviceCube XB(DDQ[0], MQ1, MD1, MD1);
   ConstDeviceCube XG(DDQ[1], MQ1, MD1, MD1);
   ConstDeviceCube XC(DDQ[2], MQ1, MD1, MD1);

   MFEM_FOREACH_THREAD(dz, z, D1D)
   {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD(dx, x, D1D)
         {
            double u = 0.0;
            double v = 0.0;
            double w = 0.0;
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double Btz = Bt(qz, dz);
               const double Gtz = Gt(qz, dz);
               u += XB(qz, dy, dx) * Btz;
               v += XG(qz, dy, dx) * Btz;
               w += XC(qz, dy, dx) * Gtz;
            }
            Y(dx, dy, dz, e) += u + v + w;
         }
      }
   }
}

/// 3D Transposed Gradient, 3/3
template <int MD1, int MQ1, typename T> MFEM_HOST_DEVICE inline
void Grad1XtD(const int D1D, const int Q1D,
              const double(*__restrict__ BG) /*[2]*/[MQ1 * MD1],
              const T(*__restrict__ DDQ) /*[3]*/[MD1 * MD1 * MQ1],
              const DeviceTensor<4, const int> MAP,
              DeviceTensor<1, double> YD,
              const int e)
{
   DeviceTensor<2,const double> Bt(BG[0], MQ1, MD1);
   DeviceTensor<2,const double> Gt(BG[1], MQ1, MD1);
   DeviceTensor<3,const T> XB(DDQ[0], MQ1, MD1, MD1);
   DeviceTensor<3,const T> XG(DDQ[1], MQ1, MD1, MD1);
   DeviceTensor<3,const T> XC(DDQ[2], MQ1, MD1, MD1);

   MFEM_FOREACH_THREAD(dz, z, D1D)
   {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD(dx, x, D1D)
         {
            T u; u = 0.0;
            T v; v = 0.0;
            T w; w = 0.0;
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double Btz = Bt(qz, dz);
               const double Gtz = Gt(qz, dz);
               u += XB(qz, dy, dx) * Btz;
               v += XG(qz, dy, dx) * Btz;
               w += XC(qz, dy, dx) * Gtz;
            }
            const T value = u + v + w;
            const int gid = MAP(dx, dy, dz, e);
            const int j = gid >= 0 ? gid : -1 - gid;
            AtomicAdd(YD(j), value);
         }
      }
   }
}

/// 3D Transposed Gradient, 3/3
template <int MD1, int MQ1, typename T, int SMS>
MFEM_HOST_DEVICE inline
void Grad1XtDScatter(const int D1D, const int Q1D,
                     const double(*__restrict__ BG) /*[2]*/[MQ1 * MD1],
                     const T(*__restrict__ DDQ) /*[3]*/[MD1 * MD1 * MQ1],
                     const DeviceTensor<4, const int> MAP,
                     DeviceTensor<1, double> YD,
                     const int e)
{
   DeviceTensor<2,const double> Bt(BG[0], MQ1, MD1);
   DeviceTensor<2,const double> Gt(BG[1], MQ1, MD1);
   DeviceTensor<3,const T> XB(DDQ[0], MQ1, MD1, MD1);
   DeviceTensor<3,const T> XG(DDQ[1], MQ1, MD1, MD1);
   DeviceTensor<3,const T> XC(DDQ[2], MQ1, MD1, MD1);

   MFEM_FOREACH_THREAD(dz, z, D1D)
   {
      MFEM_FOREACH_THREAD(dy, y, D1D)
      {
         MFEM_FOREACH_THREAD(dx, x, D1D)
         {
            T u; u = 0.0;
            T v; v = 0.0;
            T w; w = 0.0;
            for (int qz = 0; qz < Q1D; ++qz)
            {
               const double Btz = Bt(qz, dz);
               const double Gtz = Gt(qz, dz);
               u += XB(qz, dy, dx) * Btz;
               v += XG(qz, dy, dx) * Btz;
               w += XC(qz, dy, dx) * Gtz;
            }
            const T value = u + v + w;
            for (int i = 0; i < SMS; i++)
            {
               const int gid = MAP(dx, dy, dz, e + i);
               const int j = gid >= 0 ? gid : -1 - gid;
               AtomicAdd(YD(j), value[i]);
            }
         }
      }
   }
}

}  // namespace kernels

// XFL addons //////////////////////////////////////////////////////////////////
namespace xfl
{

class XElementRestriction : public ElementRestriction
{
   const ParFiniteElementSpace &fes;

public:
   XElementRestriction(const ParFiniteElementSpace *fes,
                       ElementDofOrdering edo)
      : ElementRestriction(*fes, edo), fes(*fes) { }

   const Array<int> &ScatterMap() const { return offsets; }
   const Array<int> &ScatterIdx() const { return indices; }
   const Array<int> &GatherMap() const { return gatherMap; }

   void Mult(const Vector &x, Vector &y) const
   {
      const int ndof = fes.GetFE(0)->GetDof();
      const int ndofs = fes.GetNDofs();

      const auto d_x = Reshape(x.Read(), ndofs);
      const auto d_j = gatherMap.Read();

      auto d_y = Reshape(y.Write(), ndof, ne);

      MFEM_FORALL(i, ndof * ne, { d_y(i % ndof, i / ndof) = d_x(d_j[i]); });
   }

   void MultTranspose(const Vector &x, Vector &y) const
   {
      const int nd = fes.GetFE(0)->GetDof();

      const auto d_offsets = offsets.Read();
      const auto d_indices = indices.Read();

      const auto d_x = Reshape(x.Read(), nd, ne);
      auto d_y = Reshape(y.Write(), ndofs);
      MFEM_FORALL(i, ndofs,
      {
         double dofValue = 0.0;
         const int offset = d_offsets[i];
         const int nextOffset = d_offsets[i + 1];
         for (int j = offset; j < nextOffset; ++j)
         {
            const bool plus = d_indices[j] >= 0;
            const int idx_j = plus ? d_indices[j] : -1 - d_indices[j];
            const double value = d_x(idx_j % nd, idx_j / nd);
            dofValue += (plus) ? value : -value;
         }
         d_y(i) = dofValue;
      });
   }
};

/** ****************************************************************************
 * @brief The Operator class
 **************************************************************************** */
template <int DIM> class Operator;

/** ****************************************************************************
 * @brief The 2D Operator class
 **************************************************************************** */
template <>
class Operator<2> : public mfem::Operator
{
protected:
   static constexpr int DIM = 2;
   static constexpr int NBZ = 1;

   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const int flags = GeometricFactors::JACOBIANS |
                     GeometricFactors::COORDINATES |
                     GeometricFactors::DETERMINANTS;
   const ElementDofOrdering e_ordering = ElementDofOrdering::LEXICOGRAPHIC;

   mfem::ParMesh *mesh;
   const ParFiniteElementSpace *pfes;
   const GridFunction *nodes;
   const mfem::FiniteElementSpace *nfes;
   const int p, q;
   const xfl::XElementRestriction ER;
   const mfem::Operator *NR;
   const Geometry::Type type;
   const IntegrationRule &ir;
   const GeometricFactors *geom;
   const DofToQuad *maps;
   const QuadratureInterpolator *nqi;
   const int SDIM, VDIM, NDOFS, NE, NQ, D1D, Q1D;
   mutable Vector val_xq, grad_xq;
   Vector J0, dx;
   const mfem::Operator *P, *R;

public:
   Operator(const ParFiniteElementSpace *pfes)
      : mfem::Operator(pfes->GetVSize()),
        mesh(pfes->GetParMesh()),
        pfes(pfes),
        nodes((mesh->EnsureNodes(), mesh->GetNodes())),
        nfes(nodes->FESpace()),
        p(pfes->GetFE(0)->GetOrder()),
        q(2 * p + mesh->GetElementTransformation(0)->OrderW()),
        ER(pfes, e_ordering),
        NR(nfes->GetElementRestriction(e_ordering)),
        type(mesh->GetElementBaseGeometry(0)),
        ir(IntRules.Get(type, q)),
        geom(mesh->GetGeometricFactors(ir, flags)),//, mode)),  // add to cache
        maps(&pfes->GetFE(0)->GetDofToQuad(ir, mode)),
        nqi(nfes->GetQuadratureInterpolator(ir)),//, mode)),
        SDIM(mesh->SpaceDimension()),
        VDIM(pfes->GetVDim()),
        NDOFS(pfes->GetNDofs()),
        NE(mesh->GetNE()),
        NQ(ir.GetNPoints()),
        D1D(pfes->GetFE(0)->GetOrder() + 1),
        Q1D(IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints()),
        val_xq(NQ * VDIM * NE),
        grad_xq(NQ * VDIM * DIM * NE),
        J0(SDIM * DIM * NQ * NE),
        dx(NQ * NE),
        P(pfes->GetProlongationMatrix()),
        R(pfes->GetRestrictionMatrix())
   {
      MFEM_VERIFY(DIM == 2, "");
      MFEM_VERIFY(VDIM == 1, "");
      MFEM_VERIFY(SDIM == DIM, "");
      MFEM_VERIFY(NQ == Q1D * Q1D, "");
      MFEM_VERIFY(DIM == mesh->Dimension(), "");
      nqi->SetOutputLayout(QVectorLayout::byVDIM);
      const FiniteElement *fe = nfes->GetFE(0);
      const int vdim = nfes->GetVDim();
      const int nd = fe->GetDof();
      Vector Enodes(vdim * nd * NE);
      NR->Mult(*nodes, Enodes);
      dbg("VDIM:%d, SDIM:%d", VDIM, SDIM);
      nqi->Derivatives(Enodes, J0);
   }

   virtual void Mult(const mfem::Vector &, mfem::Vector &) const {}

   virtual void QMult(const mfem::Vector &, mfem::Vector &) const {}

   virtual const mfem::Operator *GetProlongation() const { return P; }

   virtual const mfem::Operator *GetRestriction() const { return R; }
};

/** ****************************************************************************
 * @brief The 3D Operator class
 **************************************************************************** */
template <>
class Operator<3> : public mfem::Operator
{
protected:
   static constexpr int DIM = 3;

   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const int flags = GeometricFactors::JACOBIANS |
                     GeometricFactors::COORDINATES |
                     GeometricFactors::DETERMINANTS;
   const ElementDofOrdering e_ordering = ElementDofOrdering::LEXICOGRAPHIC;

   mfem::ParMesh *mesh;
   const ParFiniteElementSpace *pfes;
   const GridFunction *nodes;
   const mfem::FiniteElementSpace *nfes;
   const int p, q;
   const xfl::XElementRestriction ER;
   const mfem::Operator *NR;
   const Geometry::Type type;
   const IntegrationRule &ir;
   const GeometricFactors *geom;
   const DofToQuad *maps;
   const QuadratureInterpolator *qi, *nqi;
   const int SDIM, VDIM, NDOFS, NE, NQ, D1D, Q1D;
   mutable Vector val_xq, grad_xq;
   Vector J0, dx;
   const mfem::Operator *P, *R;
   mutable Array<double> CoG;

public:
   Operator(const ParFiniteElementSpace *pfes)
      : mfem::Operator(pfes->GetVSize()),
        mesh(pfes->GetParMesh()),
        pfes(pfes),
        nodes((mesh->EnsureNodes(), mesh->GetNodes())),
        nfes(nodes->FESpace()),
        p(pfes->GetFE(0)->GetOrder()),
        q(2 * p + mesh->GetElementTransformation(0)->OrderW()),
        ER(pfes, e_ordering),
        NR(nfes->GetElementRestriction(e_ordering)),
        type(mesh->GetElementBaseGeometry(0)),
        ir(IntRules.Get(type, q)),
        geom(mesh->GetGeometricFactors(ir, flags, mode)),
        maps(&pfes->GetFE(0)->GetDofToQuad(ir, mode)),
        qi(pfes->GetQuadratureInterpolator(ir, mode)),
        nqi(nfes->GetQuadratureInterpolator(ir, mode)),
        SDIM(mesh->SpaceDimension()),
        VDIM(pfes->GetVDim()),
        NDOFS(pfes->GetNDofs()),
        NE(mesh->GetNE()),
        NQ(ir.GetNPoints()),
        D1D(pfes->GetFE(0)->GetOrder() + 1),
        Q1D(IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints()),
        val_xq(NQ * VDIM * NE),
        grad_xq(NQ * VDIM * DIM * NE),
        J0(SDIM * DIM * NQ * NE),
        dx(NQ * NE),
        P(pfes->GetProlongationMatrix()),
        R(pfes->GetRestrictionMatrix())
   {
      MFEM_VERIFY(DIM == 3, "");
      MFEM_VERIFY(VDIM == 1, "");
      MFEM_VERIFY(SDIM == DIM, "");
      MFEM_VERIFY(NQ == Q1D * Q1D * Q1D, "");
      MFEM_VERIFY(DIM == mesh->Dimension(), "");
      qi->SetOutputLayout(QVectorLayout::byVDIM);
      nqi->SetOutputLayout(QVectorLayout::byVDIM);
      const FiniteElement *fe = nfes->GetFE(0);
      const int vdim = nfes->GetVDim();
      const int nd = fe->GetDof();
      Vector Enodes(vdim * nd * NE);
      NR->Mult(*nodes, Enodes);
      dbg("VDIM:%d, SDIM:%d", VDIM, SDIM);
      dbg("p:%d q:%d OrderW:%d, D1D:%d, Q1D:%d", p, q,
          mesh->GetElementTransformation(0)->OrderW(), D1D, Q1D);
      nqi->Derivatives(Enodes, J0);
      ComputeDX();
   }

   /// 3D setup to compute DX: W * detJ
   void ComputeDX()
   {
      const int NE = this->NE;
      const int Q1D = this->Q1D;
      const auto W = mfem::Reshape(ir.GetWeights().Read(), Q1D, Q1D, Q1D);
      const auto J = mfem::Reshape(geom->J.Read(), Q1D, Q1D, Q1D, SDIM, DIM, NE);
      auto DX = mfem::Reshape(dx.Write(), Q1D, Q1D, Q1D, NE);
      MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qz,z,Q1D)
               {
                  const double J11 = J(qx,qy,qz,0,0,e);
                  const double J21 = J(qx,qy,qz,1,0,e);
                  const double J31 = J(qx,qy,qz,2,0,e);
                  const double J12 = J(qx,qy,qz,0,1,e);
                  const double J22 = J(qx,qy,qz,1,1,e);
                  const double J32 = J(qx,qy,qz,2,1,e);
                  const double J13 = J(qx,qy,qz,0,2,e);
                  const double J23 = J(qx,qy,qz,1,2,e);
                  const double J33 = J(qx,qy,qz,2,2,e);
                  const double detJ = J11 * (J22 * J33 - J32 * J23) -
                  /* */               J21 * (J12 * J33 - J32 * J13) +
                  /* */               J31 * (J12 * J23 - J22 * J13);
                  DX(qx,qy,qz,e) = W(qx,qy,qz) * detJ; // * coeff
               }
            }
         }  // namespace xfl
      }  // namespace mfem
                    );
   }

   virtual void Mult(const mfem::Vector &, mfem::Vector &) const {}

   virtual void QMult(const mfem::Vector &, mfem::Vector &) const {}

   virtual const mfem::Operator *GetProlongation() const { return P; }

   virtual const mfem::Operator *GetRestriction() const { return R; }
};

/** ****************************************************************************
 * @brief The Problem struct
 ******************************************************************************/
struct Problem
{
   mfem::Operator *QM {nullptr};
   mfem::ParLinearForm &b;
   Problem(mfem::ParLinearForm &b, mfem::Operator *QM) : QM(QM), b(b) {}
   ~Problem() { dbg(); delete QM; }
};

/** ****************************************************************************
 * @brief The QForm class
 ******************************************************************************/
class QForm
{
public:
   const char *qs;
   mfem::Operator *QM;
   mfem::ParLinearForm *b = nullptr;
   mfem::ConstantCoefficient *constant_coeff = nullptr;
   mfem::FunctionCoefficient *function_coeff = nullptr;
   mfem::ParFiniteElementSpace *pfes;

public:
   // Constructor
   QForm(mfem::ParFiniteElementSpace *pfes,
         const char *qs, mfem::Operator *QM)
      : qs(qs), QM(QM), pfes(pfes)
   {
      dbg("\033[33m%s", qs);
   }

   ~QForm() { dbg("\033[33m%s", qs); }

   // Create problem
   Problem *operator==(QForm &rhs)
   {
      assert(!b);
      mfem::ParLinearForm *b = new mfem::ParLinearForm(rhs.ParFESpace());
      assert(b);
      if (!rhs.ConstantCoeff() && !rhs.FunctionCoeff())
      {
         //dbg("\033[31m!rhs Coeffs");
         ConstantCoefficient *cst = new ConstantCoefficient(1.0);
         b->AddDomainIntegrator(new DomainLFIntegrator(*cst));
      }
      else if (rhs.ConstantCoeff())
      {
         //dbg("\033[31mrhs.ConstantCoeff()");
         ConstantCoefficient *cst = rhs.ConstantCoeff();
         b->AddDomainIntegrator(new DomainLFIntegrator(*cst));
      }
      else if (rhs.FunctionCoeff())
      {
         //dbg("\033[31mrhs.FunctionCoeff()");
         FunctionCoefficient *func = rhs.FunctionCoeff();
         b->AddDomainIntegrator(new DomainLFIntegrator(*func));
      }
      else
      {
         assert(false);
      }

      dbg("\033[31mProblem");
      return new Problem(*b, QM);
   }

   // + operator on QForms
   QForm &operator+(QForm &rhs)
   {
      assert(false);  // not supported
      return *this;
   }

   mfem::ParFiniteElementSpace *ParFESpace() { return pfes; }
   mfem::ConstantCoefficient *ConstantCoeff() const { return constant_coeff; }
   mfem::FunctionCoefficient *FunctionCoeff() const { return function_coeff; }
};

/** ****************************************************************************
 * @brief Function class
 ******************************************************************************/
class Function : public ParGridFunction
{
public:
   Function(ParFiniteElementSpace *pfes) : ParGridFunction(pfes)
   {
      assert(pfes);
      assert(pfes->GlobalTrueVSize() > 0);
   }
   void operator=(double value) { ParGridFunction::operator=(value); }
   int geometric_dimension() { return fes->GetMesh()->SpaceDimension(); }
   ParFiniteElementSpace *ParFESpace() { return ParGridFunction::ParFESpace(); }
   const ParFiniteElementSpace *ParFESpace() const { return ParGridFunction::ParFESpace(); }
   ConstantCoefficient *ConstantCoeff() const { return nullptr; }
   FunctionCoefficient *FunctionCoeff() const { return nullptr; }
};

/** ****************************************************************************
 * @brief TrialFunction class
 ******************************************************************************/
class TrialFunction : public Function
{
public:
   TrialFunction(ParFiniteElementSpace *pfes) : Function(pfes) { }
   ~TrialFunction() { }
};

/** ****************************************************************************
 * @brief TestFunction class
 ******************************************************************************/
class TestFunction : public Function
{
public:
   TestFunction(ParFiniteElementSpace *pfes) : Function(pfes) { }
   TestFunction(const TestFunction &) = default;
   ~TestFunction() { }
};

/** ****************************************************************************
 * @brief Constant class
 ******************************************************************************/
class Constant
{
   const double value = 0.0;
   ConstantCoefficient *cst = nullptr;

public:
   Constant(double val) : value(val), cst(new ConstantCoefficient(val)) { }
   ~Constant() { dbg(); /*delete cst;*/ }
   ParFiniteElementSpace *ParFESpace() const { return nullptr; }
   double Value() const { return value; }
   double Value() { return value; }
   double operator*(TestFunction &v) { return 0.0; }
   ConstantCoefficient *ConstantCoeff() const { return cst; }
   FunctionCoefficient *FunctionCoeff() const { return nullptr; }
   operator const double *() const { return nullptr; }  // qf eval
};

/** ****************************************************************************
 * @brief Expressions
 ******************************************************************************/
class Expression
{
   FunctionCoefficient *fct = nullptr;

public:
   Expression(std::function<double(const Vector &)> F)
      : fct(new FunctionCoefficient(F)) { }
   ~Expression() { /*delete fct;*/ }
   ParFiniteElementSpace *ParFESpace() const { return nullptr; }  // qf args
   ConstantCoefficient *ConstantCoeff() const { return nullptr; }
   FunctionCoefficient *FunctionCoeff() const { return fct; }
   operator const double *() const { return nullptr; }  // qf eval
   // double operator *(TestFunction &v) { dbg(); return 0.0;} // qf eval
};

/** ****************************************************************************
 * @brief Mesh
 ******************************************************************************/
static mfem::ParMesh *MeshToPMesh(mfem::Mesh *mesh)
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   int n111[3] {1, 1, 1};
   int n211[3] {2, 1, 1};
   int n221[3] {2, 2, 1};
   int n222[3] {2, 2, 2};
   int n422[3] {4, 2, 2};
   int n442[3] {4, 4, 2};
   int n444[3] {4, 4, 4};
   int *nxyz = (num_procs == 1 ? n111 :
                num_procs == 2 ? n211 :
                num_procs == 4 ? n221 :
                num_procs == 8 ? n222 :
                num_procs == 16 ? n422 :
                num_procs == 32 ? n442 :
                num_procs == 64 ? n444 : nullptr);
   assert(nxyz);
   const int mesh_p = 1;
   mesh->SetCurvature(mesh_p, false, -1, Ordering::byNODES);
   int *partitioning = mesh->CartesianPartitioning(nxyz);
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partitioning);
   delete mesh;
   return pmesh;
}

mfem::ParMesh &Mesh(const char *mesh_file)  // and & for mesh
{
   return *MeshToPMesh(new mfem::Mesh(mesh_file, 1, 1));
}

mfem::ParMesh &Mesh(mfem::Mesh *mesh) { return *MeshToPMesh(mesh); }
mfem::ParMesh &Mesh(mfem::ParMesh *pmesh) { return *pmesh; }

mfem::ParMesh &UnitSquareMesh(int nx, int ny)
{
   const double sx = 1.0, sy = 1.0;
   Element::Type quad = Element::Type::QUADRILATERAL;
   const bool edges = false, sfc = true;
   mfem::Mesh *mesh =
      new mfem::Mesh(nx, ny, quad, edges, sx, sy, sfc);
   return *MeshToPMesh(mesh);
}

mfem::ParMesh &UnitHexMesh(int nx, int ny, int nz)
{
   Element::Type hex = Element::Type::HEXAHEDRON;
   const bool edges = false, sfc = true;
   const double sx = 1.0, sy = 1.0, sz = 1.0;
   mfem::Mesh *mesh =
      new mfem::Mesh(nx, ny, nz, hex, edges, sx, sy, sz, sfc);
   return *MeshToPMesh(mesh);
}

/** ****************************************************************************
 * @brief Device
 ******************************************************************************/
mfem::Device Device(const char *device_config) { return {device_config}; }

/** ****************************************************************************
 * @brief FiniteElement
 ******************************************************************************/
FiniteElementCollection *FiniteElement(std::string family, int type, int p)
{
   MFEM_VERIFY(family == "Lagrange", "Unsupported family!");
   MFEM_VERIFY(type == Element::Type::QUADRILATERAL ||
               type == Element::Type::HEXAHEDRON, "Unsupported type!");
   const int dim = (type == Element::Type::QUADRILATERAL) ? 2 :
                   (type == Element::Type::HEXAHEDRON)  ? 3 : 0;
   const int btype = BasisType::GaussLobatto;
   return new H1_FECollection(p, dim, btype);
}

/** ****************************************************************************
 * @brief Function Spaces
 ******************************************************************************/
class FunctionSpace : public ParFiniteElementSpace {};

ParFiniteElementSpace *FunctionSpace(mfem::ParMesh *pmesh,
                                     std::string family,
                                     int p)
{
   assert(false);
   const int dim = pmesh->Dimension();
   MFEM_VERIFY(family == "P", "Unsupported FE!");
   FiniteElementCollection *fec = new H1_FECollection(p, dim);
   return new ParFiniteElementSpace(pmesh, fec);
}

ParFiniteElementSpace *FunctionSpace(mfem::ParMesh &pmesh, std::string f, int p)
{
   assert(false);
   return FunctionSpace(&pmesh, f, p);
}

ParFiniteElementSpace *FunctionSpace(mfem::ParMesh &pmesh,
                                     FiniteElementCollection *fec)
{
   const int vdim = 1;
   const Ordering::Type ordering = Ordering::byNODES;
   ParFiniteElementSpace *pfes =
      new ParFiniteElementSpace(&pmesh, fec, vdim, ordering);
   return pfes;
}

ParFiniteElementSpace *FunctionSpace(mfem::ParMesh &pmesh,
                                     FiniteElementCollection *fec,
                                     const int vdim)
{
   assert(false);
   return new ParFiniteElementSpace(&pmesh, fec, vdim);
}

/** ****************************************************************************
 * @brief Vector Function Space
 ******************************************************************************/
ParFiniteElementSpace *VectorFunctionSpace(mfem::ParMesh *pmesh,
                                           std::string family,
                                           const int p)
{
   const int dim = pmesh->Dimension();
   MFEM_VERIFY(family == "P", "Unsupported FE!");
   FiniteElementCollection *fec = new H1_FECollection(p, dim);
   return new ParFiniteElementSpace(pmesh, fec, dim);
}

ParFiniteElementSpace *VectorFunctionSpace(mfem::ParMesh &pmesh,
                                           std::string family,
                                           const int p)
{
   return VectorFunctionSpace(&pmesh, family, p);
}

ParFiniteElementSpace *VectorFunctionSpace(mfem::ParMesh &pmesh,
                                           FiniteElementCollection *fec)
{
   return new ParFiniteElementSpace(&pmesh, fec, pmesh.Dimension());
}

/** ****************************************************************************
 * @brief Boundary Conditions
 ******************************************************************************/
Array<int> DirichletBC(mfem::ParFiniteElementSpace *pfes)
{
   assert(pfes);
   Array<int> ess_tdof_list;
   mfem::ParMesh *pmesh = pfes->GetParMesh();
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      pfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   return ess_tdof_list;
}

/** ****************************************************************************
 * @brief Math namespace
 ******************************************************************************/
namespace math
{

Constant Pow(Function &gf, double exp)
{
   return Constant(gf.Vector::Normlp(exp));
}

}  // namespace math

/** ****************************************************************************
 * @brief solve with boundary conditions
 ******************************************************************************/
int solve(xfl::Problem *pb, xfl::Function &x, Array<int> ess_tdof_list)
{
   assert(x.FESpace());
   ParFiniteElementSpace *fes = x.ParFESpace();
   MFEM_VERIFY(UsesTensorBasis(*fes), "FE Space must Use Tensor Basis!");

   Vector B, X;
   pb->b.Assemble();
   mfem::Operator *A = nullptr;

   dbg("second solve with the inline QMult");
   mfem::Operator &op = *(pb->QM);
   op.FormLinearSystem(ess_tdof_list, x, pb->b, A, X, B);
   CG(*A, B, X, 1, 400, 1e-12, 0.0);
   op.RecoverFEMSolution(X, pb->b, x);
   x.HostReadWrite();

   delete pb;
   return 0;
}

/// solve with empty boundary conditions
int solve(xfl::Problem *pb, xfl::Function &x)
{
   Array<int> empty_tdof_list;
   return solve(pb, x, empty_tdof_list);
}

/** ****************************************************************************
 * @brief benchmark this prblem with boundary conditions
 ******************************************************************************/
int benchmark(xfl::Problem *pb, xfl::Function &x, Array<int> ess_tdof_list,
              const double rtol, const int max_it, const int print_lvl)
{
   assert(x.ParFESpace());
   ParFiniteElementSpace *pfes = x.ParFESpace();
   assert(pfes->GlobalTrueVSize() > 0);
   MFEM_VERIFY(UsesTensorBasis(*pfes), "FE Space must Use Tensor Basis!");

   mfem::ParLinearForm &b = pb->b;
   b.Assemble();

   Vector B, X;
   mfem::Operator *A;
   mfem::Operator &a = *(pb->QM);
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(rtol);
   cg.SetOperator(*A);

   // Warm-up CG solve (in case of JIT to avoid timing it)
   {
      Vector Y(X);
      cg.SetMaxIter(2);
      cg.SetPrintLevel(-1);
      cg.Mult(B, Y);
   }

   // benchmark this problem
   {
      tic_toc.Clear();
      cg.SetMaxIter(max_it);
      cg.SetPrintLevel(print_lvl);
      {
         tic_toc.Start();
         cg.Mult(B, X);
         tic_toc.Stop();
      }
   }

   // MFEM_VERIFY(cg.GetConverged(), "CG did not converged!");
   MFEM_VERIFY(cg.GetNumIterations() <= max_it, "");
   a.RecoverFEMSolution(X, b, x);
   x.HostReadWrite();

   int myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm comm = pfes->GetParMesh()->GetComm();

   const double rt = tic_toc.RealTime();
   double rt_min, rt_max;
   MPI_Reduce(&rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
   MPI_Reduce(&rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

   const HYPRE_Int dofs = pfes->GlobalTrueVSize();
   const int cg_iter = cg.GetNumIterations();
   const double mdofs_max = ((1e-6 * dofs) * cg_iter) / rt_max;
   const double mdofs_min = ((1e-6 * dofs) * cg_iter) / rt_min;

   if (myid == 0)
   {
      std::cout << "Number of finite element unknowns: " << dofs <<  std::endl;
      std::cout << "Total CG time:    " << rt_max << " (" << rt_min << ") sec."
                << std::endl;
      std::cout << "Time per CG step: "
                << rt_max / cg_iter << " ("
                << rt_min / cg_iter << ") sec." << std::endl;
      std::cout << "\"DOFs/sec\" in CG: " << mdofs_max << " ("
                << mdofs_min << ") million.";
      std::cout << std::endl;
   }
   delete pb;
   return 0;
}

int benchmark(xfl::Problem *pb, xfl::Function &x, Array<int> ess_tdof_list)
{
   return benchmark(pb, x, ess_tdof_list, 1e-12, 200, -1);
}

/** ****************************************************************************
 * @brief plot the x gridfunction
 ******************************************************************************/
int plot(xfl::Function &x)
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   ParFiniteElementSpace *fes = x.ParFESpace();
   assert(fes);
   mfem::ParMesh *pmesh = fes->GetParMesh();
   assert(pmesh);
   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << *pmesh << x << std::flush;
   return 0;
}

/** ****************************************************************************
 * @brief plot the mesh
 ******************************************************************************/
int plot(mfem::ParMesh *pmesh)
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "mesh\n" << *pmesh << std::flush;
   return 0;
}

/** ****************************************************************************
 * @brief save the x gridfunction
 ******************************************************************************/
int save(xfl::Function &x, const char *filename)
{
   int myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   std::ostringstream sol_name;
   sol_name << filename << "." << std::setfill('0') << std::setw(6) << myid;

   std::ofstream sol_ofs(sol_name.str().c_str());
   sol_ofs.precision(8);
   x.Save(sol_ofs);
   return 0;
}

/** ****************************************************************************
 * @brief save the x gridfunction
 ******************************************************************************/
int save(mfem::ParMesh &mesh, const char *filename)
{
   int myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   std::ostringstream mesh_name;
   mesh_name << filename << "." << std::setfill('0') << std::setw(6) << myid;

   std::ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   return 0;
}

constexpr int point = Element::Type::POINT;
constexpr int segment = Element::Type::SEGMENT;
constexpr int triangle = Element::Type::TRIANGLE;
constexpr int quadrilateral = Element::Type::QUADRILATERAL;
constexpr int tetrahedron = Element::Type::TETRAHEDRON;
constexpr int hexahedron = Element::Type::HEXAHEDRON;
constexpr int wedge = Element::Type::WEDGE;

}  // namespace xfl

template <typename... Args>
void print(const char *fmt, Args... args)
{
   std::cout << std::flush;
   std::printf(fmt, args...);
   std::cout << std::endl;
}

inline bool UsesTensorBasis(const FiniteElementSpace *fes)
{
   return mfem::UsesTensorBasis(*fes);
}

int sym(int u) { return u; }
int dot(int u, int v) { return u * v; }

/// CPP addons /////////////////////////////////////////////////////////////////
namespace cpp
{

// *****************************************************************************
struct Range : public std::vector<int>
{
   Range(const int n) : vector<int>(n)
   {
      // Fills the range with sequentially increasing values
      std::iota(std::begin(*this), std::end(*this), 0);
   }
};

}  // namespace cpp

}  // namespace mfem
