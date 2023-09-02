// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#define MFEM_DEBUG_COLOR 50
#include "../../general/debug.hpp"

#include "../../general/array.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/tensor.hpp"
#include "../../linalg/ttensor.hpp"
#include "../../linalg/vector.hpp"
#include <cassert>

namespace mfem
{

namespace internal
{

#undef MFEM_SHARED_USE_CHAR
#define MFEM_SHARED_EXTRA_LOAD 64*1024

#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
template<typename T, std::size_t UID>
MFEM_DEVICE inline T& StaticSharedMemoryVariable()
{
   MFEM_SHARED uint8_t smem alignas(alignof(T))[sizeof(T)];
   return *(reinterpret_cast<T*>(smem));
}
#define MFEM_STATIC_SHARED_VAR(var, ...) \
__VA_ARGS__& var = StaticSharedMemoryVariable<__VA_ARGS__, __COUNTER__>()
#else
#define MFEM_STATIC_SHARED_VAR(var, ...) __VA_ARGS__ var;
#endif

#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
template<typename T, typename U>
MFEM_DEVICE inline T& DynamicSharedMemoryVariable(U* &smem)
{
   T* base = reinterpret_cast<T*>(smem);
   return (smem += sizeof(T)/sizeof(U), *base);
}
#define MFEM_DYNAMIC_SHARED_VAR(var, smem, ...) \
__VA_ARGS__& var = DynamicSharedMemoryVariable<__VA_ARGS__>(smem)
#else
#define MFEM_DYNAMIC_SHARED_VAR(var, sm, ...) \
__VA_ARGS__ var; sm += sizeof(__VA_ARGS__)/sizeof(*sm);
#endif

/**
 * @brief The mdsmem class
 */
template <int rank, typename T = double>
struct mdsmem
{
   static constexpr auto iseq = std::make_integer_sequence<int, rank> {};

   MFEM_HOST_DEVICE mdsmem() {}

   template<typename U>
   MFEM_HOST_DEVICE mdsmem(U* &smem, const int (& dimensions)[rank])
   {
      data = reinterpret_cast<T*>(smem);
      size = 1;
      for (int i = 0; i < rank; i++)
      {
         int id = rank - 1 - i;
         size *= dimensions[id];
         shape[id] = dimensions[id];
         strides[id] = (id == rank - 1) ? 1 : strides[id+1] * shape[id+1];
      }
      smem += size * sizeof(T) / sizeof(U);
   }

   template <int N, int R, typename S, typename... Args>
   struct Layout
   {
      MFEM_HOST_DEVICE
      static inline int ini(int* shape, int* strides, S k, Args... args)
      {
         shape[N - 1] = k;
         strides[N - 1] = Layout<N + 1, R, Args...>::ini(shape, strides, args...);
         return shape[N - 1] * strides[N - 1];
      }
   };

   template <int R, typename S, typename... Args>
   struct Layout<R, R, S, Args...>
   {
      MFEM_HOST_DEVICE
      static inline int ini(int* shape, int *strides, T k, Args...)
      {
         return (strides[R - 1] = 1, shape[R - 1] = k);
      }
   };

   template<typename U, typename... Args>
   MFEM_HOST_DEVICE mdsmem(U* &smem, Args... args)
   {
      data = reinterpret_cast<T*>(smem);
      MFEM_STATIC_ASSERT(sizeof...(args) == rank, "Wrong number of arguments");
      size = Layout<1, rank, Args...>::ini(shape, strides, args...);
      smem += size * sizeof(T) / sizeof(U);
   }

   template < typename ... index_types >
   MFEM_HOST_DEVICE auto & operator()(index_types ... indices)
   {
      static_assert(sizeof ... (indices) == rank);
      return data[index(iseq, indices...)];
   }

   template < typename ... index_types >
   MFEM_HOST_DEVICE auto & operator()(index_types ... indices) const
   {
      static_assert(sizeof ... (indices) == rank);
      return data[index(iseq, indices...)];
   }

   template < int ... I, typename ... index_types >
   MFEM_HOST_DEVICE auto index(std::integer_sequence<int, I...>, index_types
                               ... indices) const
   {
      return ((indices * strides[I]) + ...);
   }

   //MFEM_HOST_DEVICE inline operator T *() const { return data; }

   T* data alignas(alignof(T));
   int size;
   int shape[rank];
   int strides[rank];
};


/// CUDA backend
#if defined(MFEM_USE_CUDA)

template <typename Tsmem = double, typename BODY> __global__ static
void CuKernel2DSmem(const int N, BODY body)
{
   const int k = blockIdx.x*blockDim.z + threadIdx.z;
   if (k >= N) { return; }
   extern __shared__ Tsmem smem[];
   body(k, smem);
}

template <typename Tsmem = double, typename BODY> __global__ static
void CuKernel2DGmem(const int N, BODY body, Tsmem *smem, const int smem_size)
{
   const int k = blockIdx.x*blockDim.z + threadIdx.z;
   if (k >= N) { return; }
   body(k, smem + smem_size*blockIdx.x);
}

template <typename Tsmem = double, typename DBODY>
void CuWrapSmem2D(const int N, DBODY &&d_body, const int smem_size,
                  const int X, const int Y, const int BZ, const int G)
{
   if (N==0) { return; }
   MFEM_VERIFY(BZ > 0, "");
   MFEM_VERIFY(G == 0, "Grid not implemented!");
   MFEM_VERIFY(smem_size > 0, "No Shared memory!");

   const dim3 BLCK(X,Y,BZ);

   if (smem_size*sizeof(Tsmem) < 64*1024) // V100, w/o extra config
   {
      const int GRID = (N+BZ-1)/BZ;
      CuKernel2DSmem<Tsmem><<<GRID, BLCK, sizeof(Tsmem)*smem_size>>>(N, d_body);
   }
   else
   {
      constexpr int SM = 80;
      const int GRID = SM;
      dbg("\033[33mFolding back to GLOBAL memory!");
      static Memory<Tsmem> smem(smem_size*sizeof(Tsmem)*GRID);
      smem.UseDevice(true);
      CuKernel2DGmem<Tsmem><<<GRID,BLCK>>>(N, d_body,
                                           smem.Write(MemoryClass::DEVICE, smem_size),
                                           smem_size);
   }
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <typename Tsmem = double, typename BODY> __global__ static
void CuKernel3DSmem(const int N, BODY body)
{
   extern __shared__ Tsmem smem[];
   for (int k = blockIdx.x; k < N; k += gridDim.x) { body(k, smem); }
}

template <typename Tsmem = double, typename BODY> __global__ static
void CuKernel3DGmem(const int N, BODY body, Tsmem *smem, const int smem_size)
{
   for (int k = blockIdx.x; k < N; k += gridDim.x)
   {
      body(k, smem + smem_size*blockIdx.x);
   }
}

template <typename Tsmem = double, typename DBODY>
void CuWrapSmem3D(const int N, DBODY &&d_body, const int smem_size,
                  const int X, const int Y, const int Z, const int G)
{
   if (N==0) { return; }
   MFEM_VERIFY(smem_size > 0, "No Shared memory!");

   const dim3 BLCK(X,Y,Z);

   if (smem_size*sizeof(Tsmem) < 64*1024) // V100, w/o extra config
   {
      const int NB = X*Y*Z < 16 ? 4 : 1;
      const int GRID_X = (N + NB - 1) / NB;
      const int GRID = G == 0 ? GRID_X : G;
      CuKernel3DSmem<Tsmem><<<GRID, BLCK, sizeof(Tsmem)*smem_size>>>(N, d_body);
   }
   else
   {
      constexpr int SM = 80;
      const int GRID = G == 0 ? SM : G;
      dbg("\033[33mFolding back to GLOBAL memory (GRID:%d)!", GRID);
      static Memory<Tsmem> smem(smem_size*sizeof(Tsmem)*GRID);
      smem.UseDevice(true);
      CuKernel3DGmem<Tsmem><<<GRID,BLCK>>>(N, d_body,
                                           smem.Write(MemoryClass::DEVICE, smem_size),
                                           smem_size);
   }
   MFEM_GPU_CHECK(cudaGetLastError());
}

template <int Dim, typename Tsmem> struct CuWrapSmem;

template <typename Tsmem>
struct CuWrapSmem<2,Tsmem>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body, const int smem_size,
                   const int X, const int Y, const int Z, const int G)
   {
      CuWrapSmem2D<Tsmem>(N, d_body, smem_size, X, Y, Z, G);
   }
};

template <typename Tsmem>
struct CuWrapSmem<3,Tsmem>
{
   template <const int BLCK = MFEM_CUDA_BLOCKS, typename DBODY>
   static void run(const int N, DBODY &&d_body, const int smem_size,
                   const int X, const int Y, const int Z, const int G)
   {
      CuWrapSmem3D<Tsmem>(N, d_body, smem_size, X, Y, Z, G);
   }
};
#endif // MFEM_USE_CUDA

/// The forall kernel body wrapper
template <const int DIM, typename Tsmem = double, typename d_lambda, typename h_lambda>
inline void ForallWrapSmem(const bool use_dev, const int N,
                           d_lambda &&d_body, h_lambda &&h_body,
                           const int smem_size,
                           const int X=0, const int Y=0, const int Z=0,
                           const int G=0)
{
   MFEM_CONTRACT_VAR(X);
   MFEM_CONTRACT_VAR(Y);
   MFEM_CONTRACT_VAR(Z);
   MFEM_CONTRACT_VAR(G);
   MFEM_CONTRACT_VAR(d_body);
   MFEM_CONTRACT_VAR(smem_size);
   if (!use_dev) { goto backend_cpu; }

#ifdef MFEM_USE_CUDA
   // If Backend::CUDA is allowed, use it
   if (Device::Allows(Backend::CUDA))
   {
      return CuWrapSmem<DIM,Tsmem>::run(N, d_body, smem_size, X, Y, Z, G);
   }
#endif

   // If Backend::DEBUG_DEVICE is allowed, use it
   if (Device::Allows(Backend::DEBUG_DEVICE)) { goto backend_cpu; }

backend_cpu:
   // Handle Backend::CPU. This is also a fallback for any allowed backends not
   // handled above, e.g. OCCA_CPU with configuration 'occa-cpu,cpu', or
   // OCCA_OMP with configuration 'occa-omp,cpu'.
   assert(smem_size > 0);
   Tsmem smem[smem_size];
   //Tsmem *smem = new Tsmem[smem_size];
   for (int k = 0; k < N; k++) { h_body(k,smem); }
   //delete[] smem;
}

template <const int DIM, typename Tsmem = double, typename lambda>
inline void ForallWrapSmem(const bool use_dev, const int N, lambda &&body,
                           const int smem_bytes,
                           const int X=0, const int Y=0, const int Z=0,
                           const int G=0)
{
   ForallWrapSmem<DIM,Tsmem>(use_dev, N, body, body, smem_bytes, X, Y, Z, G);
}

template<typename Tsmem = double, typename lambda>
inline void forall_2D_batch_smem(int N, int X, int Y, int BZ, int smem_bytes,
                                 lambda &&body)
{
   ForallWrapSmem<2,Tsmem>(true, N, body, smem_bytes, X, Y, BZ, 0);
}

template<typename Tsmem = double, typename lambda>
inline void forall_3D_smem(int N, int X, int Y, int Z, int smem_bytes,
                           lambda &&body)
{
   ForallWrapSmem<3,Tsmem>(true, N, body, smem_bytes, X, Y, Z, 0);
}

void DynamicSmemPADiffusionApply2DKernel(const int NE,
                                         const bool symmetric,
                                         const Array<double> &b_,
                                         const Array<double> &g_,
                                         const Vector &d_,
                                         const Vector &x_,
                                         Vector &y_,
                                         const int d,
                                         const int q,
                                         int z)
{
   const auto b = Reshape(b_.Read(), q, d);
   const auto g = Reshape(g_.Read(), q, d);
   const auto D = Reshape(d_.Read(), q*q, symmetric ? 3 : 4, NE);
   const auto x = Reshape(x_.Read(), d, d, NE);

   auto Y = Reshape(y_.ReadWrite(), d, d, NE);

   const size_t smem_size = (MFEM_SHARED_EXTRA_LOAD + 4*q*d + z*
                             (d*d + 2*d*q+ 2*q*q));
   //dbg("smem_size:%d", smem_size);

#ifdef MFEM_SHARED_USE_CHAR
   const size_t smem_size_char = sizeof(double) * smem_size;
   dbg("smem_size_char:%d", smem_size_char);
   mfem::internal::forall_2D_batch_smem<char>(NE, q,q,z, smem_size_char,
                                              [=] MFEM_HOST_DEVICE(int e, char *sm)
#else
   mfem::internal::forall_2D_batch_smem(NE, q,q,z, smem_size,
                                        [=] MFEM_HOST_DEVICE(int e, double *sm)
#endif
   {
      const int tz = MFEM_THREAD_ID(z);
      const decltype(sm) base = sm;

      mdsmem<3> X(sm, d,d,z);

      mdsmem<2> B(sm, q,d), G(sm, q,d);

      //assert(q==3 && d==3); // if using tensor, TMatrix for shared variables
      //MFEM_STATIC_SHARED_VAR(B, tensor<double,3,3>); // q==3, d==3 !!
      //MFEM_STATIC_SHARED_VAR(G, tensor<double,3,3>); // q==3, d==3 !!

      //MFEM_DYNAMIC_SHARED_VAR(B, sm, TMatrix<3,3>); // q==3, d==3 !!
      //MFEM_DYNAMIC_SHARED_VAR(B, sm, tensor<double,3,3>); // q==3, d==3 !!

      mdsmem<2> Bt(sm, d,q), Gt(sm, d,q);
      mdsmem<4> DQ(sm, d,q,z,2), QQ(sm, q,q,z,2);

      mdsmem<1> Extra(sm, MFEM_SHARED_EXTRA_LOAD);

      // can be less if there are some static shared
      assert(sm <= base + smem_size*sizeof(double)/sizeof(*sm));

      MFEM_FOREACH_THREAD(dy,y,d)
      {
         MFEM_FOREACH_THREAD(dx,x,d)
         {
            X(dx,dy,tz) = x(dx,dy,e);
         }
      }
      if (tz == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(qx,x,q)
            {
               B(qx,dy) = b(qx,dy);
               G(qx,dy) = g(qx,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,d)
      {
         MFEM_FOREACH_THREAD(qx,x,q)
         {
            double u = 0.0;
            double v = 0.0;
            for (int dx = 0; dx < d; ++dx)
            {
               const double coords = X(dx,dy,tz);
               u += B(qx,dx) * coords;
               v += G(qx,dx) * coords;
            }
            DQ(dy,qx,tz,0) = u;
            DQ(dy,qx,tz,1) = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,q)
      {
         MFEM_FOREACH_THREAD(qx,x,q)
         {
            double u = 0.0;
            double v = 0.0;
            for (int dy = 0; dy < d; ++dy)
            {
               u += DQ(dy,qx,tz,1) * B(qy,dy);
               v += DQ(dy,qx,tz,0) * G(qy,dy);
            }
            QQ(qy,qx,tz,0) = u;
            QQ(qy,qx,tz,1) = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,q)
      {
         MFEM_FOREACH_THREAD(qx,x,q)
         {
            const int f = (qx + ((qy) * q));
            const double O11 = D(f,0,e);
            const double O21 = D(f,1,e);
            const double O12 = symmetric ? O21 : D(f,2,e);
            const double O22 = symmetric ? D(f,2,e) : D(f,3,e);
            const double gX = QQ(qy,qx,tz,0);
            const double gY = QQ(qy,qx,tz,1);
            QQ(qy,qx,tz,0) = (O11 * gX) + (O12 * gY);
            QQ(qy,qx,tz,1) = (O21 * gX) + (O22 * gY);
         }
      }
      MFEM_SYNC_THREAD;
      if (tz == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(qx,x,q)
            {
               Bt(dy,qx) = b(qx,dy);
               Gt(dy,qx) = g(qx,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,q)
      {
         MFEM_FOREACH_THREAD(dx,x,d)
         {
            double u = 0.0;
            double v = 0.0;
            for (int qx = 0; qx < q; ++qx)
            {
               u += Gt(dx,qx) * QQ(qy,qx,tz,0);
               v += Bt(dx,qx) * QQ(qy,qx,tz,1);
            }
            DQ(dx,qy,tz,0) = u;
            DQ(dx,qy,tz,1) = v;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,d)
      {
         MFEM_FOREACH_THREAD(dx,x,d)
         {
            double u = 0.0;
            double v = 0.0;
            for (int qy = 0; qy < q; ++qy)
            {
               u += DQ(dx,qy,tz,0) * Bt(dy,qy);
               v += DQ(dx,qy,tz,1) * Gt(dy,qy);
            }
            Y(dx,dy,e) += (u + v);
         }
      }
   });
}

void DynamicSmemPADiffusionApply2D(const int NE,
                                   const bool symm,
                                   const Array<double> &B,
                                   const Array<double> &G,
                                   const Vector &D,
                                   const Vector &X,
                                   Vector &Y,
                                   const int d,
                                   const int q,
                                   const int z)
{
   dbg("NE:%d d:%d q:%d z:%d",NE,d,q,z);
   DynamicSmemPADiffusionApply2DKernel(NE,symm,B,G,D,X,Y,d,q,z);
}

void DynamicSmemPADiffusionApply3DKernel(const int NE,
                                         const bool symmetric,
                                         const Array<double> &b_,
                                         const Array<double> &g_,
                                         const Vector &d_,
                                         const Vector &x_,
                                         Vector &y_,
                                         const int d,
                                         const int q)
{
   const auto b = Reshape(b_.Read(), q, d);
   const auto g = Reshape(g_.Read(), q, d);
   const auto D = Reshape(d_.Read(), q, q, q, symmetric ? 6 : 9, NE);
   const auto x = Reshape(x_.Read(), d, d, d, NE);

   auto y = Reshape(y_.ReadWrite(), d, d, d, NE);

   const size_t smem_size = 4*q*d + // B,Bt,G,Gt
                            d*d*d + // X
                            3*d*d*q + // DDQs
                            3*d*q*q + // DQQs
                            3*q*q*q +  // QQQs
                            MFEM_SHARED_EXTRA_LOAD;

#ifdef MFEM_SHARED_USE_CHAR
   const size_t smem_size_char = sizeof(double) * smem_size;
   mfem::internal::forall_3D_smem<char>(NE, q,q,q, smem_size_char,
                                        [=] MFEM_HOST_DEVICE(int e, char *sm)
#else
   forall_3D_smem(NE, q,q,q, smem_size, [=] MFEM_HOST_DEVICE (int e,
                                                              double *sm)
#endif
   {
      const decltype(sm) base = sm;

      mdsmem<2> B(sm, q,d), Bt(sm, d,q);
      mdsmem<2> G(sm, q,d), Gt(sm, d,q);

      mdsmem<3> X(sm, d,d,d);
      mdsmem<3> QDD0(sm, q,d,d), QDD1(sm, q,d,d), QDD2(sm, q,d,d);
      mdsmem<3> QQD0(sm, q,q,d), QQD1(sm, q,q,d), QQD2(sm, q,q,d);
      mdsmem<3> QQQ0(sm, q,q,q), QQQ1(sm, q,q,q), QQQ2(sm, q,q,q);

      mdsmem<1> Extra(sm, MFEM_SHARED_EXTRA_LOAD);

      assert(sm == base + smem_size*sizeof(double)/sizeof(*sm));

      MFEM_FOREACH_THREAD(dz,z,d)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(dx,x,d)
            {
               X(dx,dy,dz) = x(dx,dy,dz,e);
            }
         }
      }
      if (MFEM_THREAD_ID(z) == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(qx,x,q)
            {
               B(qx,dy) = b(qx,dy);
               G(qx,dy) = g(qx,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,d)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(qx,x,q)
            {
               double u = 0.0, v = 0.0;
               MFEM_UNROLL_DISABLED
               for (int dx = 0; dx < d; ++dx)
               {
                  const double coords = X(dx,dy,dz);
                  u += coords * B(qx,dx);
                  v += coords * G(qx,dx);
               }
               QDD0(qx,dy,dz) = u;
               QDD1(qx,dy,dz) = v;
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
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL_DISABLED
               for (int dy = 0; dy < d; ++dy)
               {
                  u += QDD1(qx,dy,dz) * B(qy,dy);
                  v += QDD0(qx,dy,dz) * G(qy,dy);
                  w += QDD0(qx,dy,dz) * B(qy,dy);
               }
               QQD0(qx,qy,dz) = u;
               QQD1(qx,qy,dz) = v;
               QQD2(qx,qy,dz) = w;
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
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL_DISABLED
               for (int dz = 0; dz < d; ++dz)
               {
                  u += QQD0(qx,qy,dz) * B(qz,dz);
                  v += QQD1(qx,qy,dz) * B(qz,dz);
                  w += QQD2(qx,qy,dz) * G(qz,dz);
               }
               const double O11 = D(qx,qy,qz,0,e);
               const double O12 = D(qx,qy,qz,1,e);
               const double O13 = D(qx,qy,qz,2,e);
               const double O21 = symmetric ? O12 : D(qx,qy,qz,3,e);
               const double O22 = symmetric ? D(qx,qy,qz,3,e) : D(qx,qy,qz,4,e);
               const double O23 = symmetric ? D(qx,qy,qz,4,e) : D(qx,qy,qz,5,e);
               const double O31 = symmetric ? O13 : D(qx,qy,qz,6,e);
               const double O32 = symmetric ? O23 : D(qx,qy,qz,7,e);
               const double O33 = symmetric ? D(qx,qy,qz,5,e) : D(qx,qy,qz,8,e);
               const double gX = u;
               const double gY = v;
               const double gZ = w;
               QQQ0(qx,qy,qz) = (O11*gX) + (O12*gY) + (O13*gZ);
               QQQ1(qx,qy,qz) = (O21*gX) + (O22*gY) + (O23*gZ);
               QQQ2(qx,qy,qz) = (O31*gX) + (O32*gY) + (O33*gZ);
            }
         }
      }
      MFEM_SYNC_THREAD;
      if (MFEM_THREAD_ID(z) == 0)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(qx,x,q)
            {
               Bt(dy,qx) = b(qx,dy);
               Gt(dy,qx) = g(qx,dy);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,q)
      {
         MFEM_FOREACH_THREAD(qy,y,q)
         {
            MFEM_FOREACH_THREAD(dx,x,d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL_DISABLED
               for (int qx = 0; qx < q; ++qx)
               {
                  u += QQQ0(qx,qy,qz) * Gt(dx,qx);
                  v += QQQ1(qx,qy,qz) * Bt(dx,qx);
                  w += QQQ2(qx,qy,qz) * Bt(dx,qx);
               }
               QQD0(qy,qz,dx) = u;
               QQD1(qy,qz,dx) = v;
               QQD2(qy,qz,dx) = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,q)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(dx,x,d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL_DISABLED
               for (int qy = 0; qy < q; ++qy)
               {
                  u += QQD0(qy,qz,dx) * Bt(dy,qy);
                  v += QQD1(qy,qz,dx) * Gt(dy,qy);
                  w += QQD2(qy,qz,dx) * Bt(dy,qy);
               }
               QDD0(qz,dy,dx) = u;
               QDD1(qz,dy,dx) = v;
               QDD2(qz,dy,dx) = w;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,d)
      {
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(dx,x,d)
            {
               double u = 0.0, v = 0.0, w = 0.0;
               MFEM_UNROLL_DISABLED
               for (int qz = 0; qz < q; ++qz)
               {
                  u += QDD0(qz,dy,dx) * Bt(dz,qz);
                  v += QDD1(qz,dy,dx) * Bt(dz,qz);
                  w += QDD2(qz,dy,dx) * Gt(dz,qz);
               }
               y(dx,dy,dz,e) += (u + v + w);
            }
         }
      }
   });
}

void DynamicSmemPADiffusionApply3D(const int NE,
                                   const bool symm,
                                   const Array<double> &B,
                                   const Array<double> &G,
                                   const Vector &D,
                                   const Vector &X,
                                   Vector &Y,
                                   const int d,
                                   const int q)
{
   static int first = 0;
   if (first++ == 0) { dbg("NE:%d D1D:%d Q1D:%d",NE,d,q); }
   DynamicSmemPADiffusionApply3DKernel(NE,symm,B,G,D,X,Y,d,q);
}

} // namespace internal

} // namespace mfem
