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

#include "../general/debug.hpp"
#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"

#ifdef MFEM_USE_SYCL
using namespace sycl;

namespace mfem
{

/// A SYCL buffer generic Tensor class
template<int Dim, typename XS> class BufferTensor
{
public:
   using val_t = typename XS::value_type;
   using ref_t = typename XS::reference;
   using const_ref_t = typename XS::const_reference;
protected:
   int capacity;
   XS xs;
   int sizes[Dim];

public:
   /// Default constructor
   BufferTensor() = delete;

   /// Constructor to initialize a tensor from the buffer _buf
   template <typename... Args>
   BufferTensor(XS _xs, Args... args): xs(_xs)
   {
      static_assert(sizeof...(args) == Dim, "Wrong number of arguments");
      // Initialize sizes, and compute the number of values
      const long int nb = Init<1, Dim, Args...>::result(sizes, args...);
      capacity = nb;
   }

   /// Conversion to `Scalar *`.
   inline operator val_t *() const { return *(xs.get_pointer()); }

   /// Const accessor for the data
   template <typename... Args> MFEM_HOST_DEVICE inline
   ref_t operator()(Args... args) const
   {
      static_assert(sizeof...(args) == Dim, "Wrong number of arguments");
      return xs[ TensorInd<1, Dim, Args...>::result(sizes, args...) ];
   }

   /// Subscript operator where the tensor is viewed as a 1D array.
   MFEM_HOST_DEVICE inline ref_t operator[](int i) const
   {
      return xs[i];
   }
};

template <typename XS, typename... Dims>
inline BufferTensor<sizeof...(Dims),XS> Reshape(XS xs, Dims... dims)
{
   return BufferTensor<sizeof...(Dims),XS>(xs, dims...);
}

class Ker1D {};

#define SYCL_KERNEL(...) { \
    sycl::queue Q;\
    Q.submit([&](sycl::handler &h) {__VA_ARGS__}); \
}

#define SYCL_NAME(LINE) Ker ## LINE

#define SYCL_FORALL(i,N,...)                             \
   ForallWrap<1>(N, h, [=] MFEM_DEVICE (int i) {__VA_ARGS__})

/// The forall kernel body wrapper
template <const int DIM, typename BODY>
inline void ForallWrap(const int N, sycl::handler &h, BODY &&body)
{
   if (DIM == 1)
   {
      h.parallel_for<Ker1D>(range<1>(N), [=](id<1> k) {body(k);});
      return;
   }
   MFEM_ABORT("Not yet supported!");
}

// PA Diffusion Apply 3D kernel
template<int D1D, int Q1D>
static void PADiffusionApply3D(const int NE,
                               const double *d_b,
                               const double *d_g,
                               const double *d_bt,
                               const double *d_gt,
                               const double *d_d,
                               const double *d_x,
                               double *d_y)
{
   dbg();

   constexpr size_t B_sz = Q1D*D1D;
   const size_t D_sz = Q1D*Q1D*Q1D * 6 * NE;
   const size_t X_sz = D1D*D1D*D1D * NE;
   const size_t Y_sz = D1D*D1D*D1D * NE;

   buffer<const double, 1> b_buf(d_b, range<1> {B_sz});
   buffer<const double, 1> g_buf(d_g, range<1> {B_sz});
   buffer<const double, 1> bt_buf(d_bt, range<1> {B_sz});
   buffer<const double, 1> gt_buf(d_gt, range<1> {B_sz});
   buffer<const double, 1> d_buf(d_d, range<1> {D_sz});
   buffer<const double, 1> x_buf(d_x, range<1> {X_sz});
   buffer<double, 1> y_buf(d_y, range<1> {Y_sz});

   SYCL_KERNEL( // Q.submit
   {
      const auto b_ = b_buf.get_access<access::mode::read>(h);
      const auto g_ = g_buf.get_access<access::mode::read>(h);
      const auto bt_ = bt_buf.get_access<access::mode::read>(h);
      const auto gt_ = gt_buf.get_access<access::mode::read>(h);
      const auto d_ = d_buf.get_access<access::mode::read>(h);
      const auto x_ = x_buf.get_access<access::mode::read>(h);
      auto y_ = y_buf.get_access<access::mode::write>(h);

      const auto B = Reshape(b_, Q1D, D1D);
      const auto G = Reshape(g_, Q1D, D1D);
      const auto Bt = Reshape(bt_, D1D, Q1D);
      const auto Gt = Reshape(gt_, D1D, Q1D);
      const auto D = Reshape(d_, Q1D*Q1D*Q1D, 6, NE);
      const auto X = Reshape(x_, D1D, D1D, D1D, NE);
      auto Y = Reshape(y_, D1D, D1D, D1D, NE);

      SYCL_FORALL(e, NE,
      {
         double grad[Q1D][Q1D][Q1D][3];
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
            double gradXY[Q1D][Q1D][3];
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
               double gradX[Q1D][2];
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] = 0.0;
                  gradX[qx][1] = 0.0;
               }
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double s = X(dx,dy,dz,e);
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
                  const double O11 = D(q,0,e);
                  const double O12 = D(q,1,e);
                  const double O13 = D(q,2,e);
                  const double O21 = O12;
                  const double O22 = D(q,3,e);
                  const double O23 = D(q,4,e);
                  const double O31 = O13;
                  const double O32 = O23;
                  const double O33 = D(q,5,e);
                  const double gradX = grad[qz][qy][qx][0];
                  const double gradY = grad[qz][qy][qx][1];
                  const double gradZ = grad[qz][qy][qx][2];
                  grad[qz][qy][qx][0] = (O11*gradX)+(O12*gradY)+(O13*gradZ);
                  grad[qz][qy][qx][1] = (O21*gradX)+(O22*gradY)+(O23*gradZ);
                  grad[qz][qy][qx][2] = (O31*gradX)+(O32*gradY)+(O33*gradZ);
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            double gradXY[D1D][D1D][3];
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
               double gradX[D1D][3];
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
                     Y(dx,dy,dz,e) +=
                        ((gradXY[dy][dx][0] * wz) +
                         (gradXY[dy][dx][1] * wz) +
                         (gradXY[dy][dx][2] * wDz));
                  }
               }
            }
         }
      }); // MFEM_FORALL
   }); // MFEM_KERNEL (Q.submit)
}

// SYCL PA Diffusion Apply 3D kernel
void SyclPADiffusionApply3D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const bool symm,
                            const Array<double> &b,
                            const Array<double> &g,
                            const Array<double> &bt,
                            const Array<double> &gt,
                            const Vector &d,
                            const Vector &x,
                            Vector &y)
{
   dbg();
   MFEM_VERIFY(symm, "Only symmetric is supported!");

   const double *B = b.HostRead();
   const double *G = g.HostRead();
   const double *Bt = bt.HostRead();
   const double *Gt = gt.HostRead();
   const double *D = d.HostRead();
   const double *X = x.HostRead();
   double *Y = y.HostReadWrite();

   const int ID = (D1D << 4) | Q1D;

   switch (ID)
   {
      case 0x23: return PADiffusionApply3D<2,3>(NE,B,G,Bt,Gt,D,X,Y);
      //case 0x34: return PADiffusionApply3D<3,4>(NE,B,G,Bt,Gt,D,X,Y);
      //case 0x45: return PADiffusionApply3D<4,5>(NE,B,G,Bt,Gt,D,X,Y);
      default:   MFEM_ABORT("Order D1D:"<<D1D<<", Q1D:"<<Q1D<<"!");
   }
   MFEM_ABORT("Unknown kernel.");
}

} // namespace mfem

#endif // MFEM_USE_SYCL
