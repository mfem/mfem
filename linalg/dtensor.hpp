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

#ifndef MFEM_DTENSOR
#define MFEM_DTENSOR

#include "../general/backends.hpp"

namespace mfem
{

/// A Class to compute the real index from the multi-indices of a tensor
template <int N, int Dim, typename T, typename... Args>
class TensorInd
{
public:
   MFEM_HOST_DEVICE
   static inline int result(const int* sizes, T first, Args... args)
   {
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
      MFEM_ASSERT(first<sizes[N-1],"Trying to access out of boundary.");
#endif
      return first + sizes[N - 1] * TensorInd < N + 1, Dim, Args... >
             ::result(sizes, args...);
   }
};

// Terminal case
template <int Dim, typename T, typename... Args>
class TensorInd<Dim, Dim, T, Args...>
{
public:
   MFEM_HOST_DEVICE
   static inline int result(const int* sizes, T first, Args... args)
   {
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
      MFEM_ASSERT(first<sizes[Dim-1],"Trying to access out of boundary.");
#endif
      return first;
   }
};


/// A class to initialize the size of a Tensor
template <int N, int Dim, typename T, typename... Args>
class Init
{
public:
   static inline int result(int* sizes, T first, Args... args)
   {
      sizes[N - 1] = first;
      return first * Init < N + 1, Dim, Args... >::result(sizes, args...);
   }
};

// Terminal case
template <int Dim, typename T, typename... Args>
class Init<Dim, Dim, T, Args...>
{
public:
   static inline int result(int* sizes, T first, Args... args)
   {
      sizes[Dim - 1] = first;
      return first;
   }
};


/// A basic generic Tensor class, appropriate for use on the GPU
template<int Dim, typename Scalar = double>
class DeviceTensor
{
protected:
   int capacity;
   Scalar *data;
   int sizes[Dim];

public:
   /// Default constructor
   DeviceTensor() = delete;

   /// Constructor to initialize a tensor from the Scalar array _data
   template <typename... Args>
   DeviceTensor(Scalar* _data, Args... args)
   {
      static_assert(sizeof...(args) == Dim, "Wrong number of arguments");
      // Initialize sizes, and compute the number of values
      const long int nb = Init<1, Dim, Args...>::result(sizes, args...);
      capacity = nb;
      data = (capacity > 0) ? _data : NULL;
   }

   /// Copy constructor
   MFEM_HOST_DEVICE DeviceTensor(const DeviceTensor& t)
   {
      capacity = t.capacity;
      for (int i = 0; i < Dim; ++i)
      {
         sizes[i] = t.sizes[i];
      }
      data = t.data;
   }

   /// Conversion to `Scalar *`.
   inline operator Scalar *() const { return data; }

   /// Const accessor for the data
   template <typename... Args> MFEM_HOST_DEVICE inline
   Scalar& operator()(Args... args) const
   {
      static_assert(sizeof...(args) == Dim, "Wrong number of arguments");
      return data[ TensorInd<1, Dim, Args...>::result(sizes, args...) ];
   }

   /// Subscript operator where the tensor is viewed as a 1D array.
   MFEM_HOST_DEVICE inline Scalar& operator[](int i) const
   {
      return data[i];
   }
};


/** @brief Wrap a pointer as a DeviceTensor with automatically deduced template
    parameters */
template <typename T, typename... Dims>
inline DeviceTensor<sizeof...(Dims),T> Reshape(T *ptr, Dims... dims)
{
   return DeviceTensor<sizeof...(Dims),T>(ptr, dims...);
}


typedef DeviceTensor<1,int> DeviceArray;
typedef DeviceTensor<1,double> DeviceVector;
typedef DeviceTensor<2,double> DeviceMatrix;

// Getter for the N-th dimension value
template <int N, int... Dims>
struct Dim;

template <int Dim0, int... Dims>
struct Dim<0, Dim0, Dims...>
{
   static constexpr int val = Dim0;
};
template <int N, int Dim0, int... T>
struct Dim<N, Dim0, T...>
{
   static constexpr int val = Dim0;
};

// Compute the product of the dimensions
template<int Dim0, int... Dims>
struct Size
{
   static constexpr int val = Dim0 * Size<Dims...>::val;
};

template<int Dim0>
struct Size<Dim0>
{
   static constexpr int val = Dim0;
};

//Compute the index inside a Tensor
template<int Cpt, int rank, int... Dims>
struct Index
{
   template <typename... Idx>
   static inline int eval(int first, Idx... args)
   {
      return first + Dim<Cpt-1,Dims...>::val * Index<Cpt+1, rank, Dims...>::eval(args...);
   }
};

template<int rank, int... Dims>
struct Index<rank,rank,Dims...>
{
   static inline int eval(int first)
   {
      return first;
   }
};

template<int... Dims>
struct TensorIndex
{
   template <typename... Idx>
   static inline int eval(Idx... args)
   {
      return Index<1,sizeof...(Dims),Dims...>::eval(args...);
   }
};

/// A fixed size tensor class
template<typename T, int... Dims>
class Tensor{
private:
   MFEM_SHARED T data[Size<Dims...>::val];

public:
   template<typename... Idx> MFEM_HOST_DEVICE inline
   const T& operator()(Idx... args) const
   {
      static_assert(sizeof...(Dims)==sizeof...(Idx), "Wrong number of indices");
      return data[ TensorIndex<Dims...>::eval(args...) ];
   }

   template<typename... Idx> MFEM_HOST_DEVICE inline
   T& operator()(Idx... args)
   {
      static_assert(sizeof...(Dims)==sizeof...(Idx), "Wrong number of indices");
      return data[ TensorIndex<Dims...>::eval(args...) ];
   }

};

template <int... Dims>
using dTensor = Tensor<double,Dims...>;

// Functions to read values (dofs or values at quadrature point)
// Non-tensor read
template<int P> MFEM_HOST_DEVICE inline
void Read(const DeviceTensor<2> &e_vec, const int e, dTensor<P> &u)
{
   MFEM_FOREACH_THREAD(p,x,P)
   {
      u(p) = e_vec(p, e);
   }
   MFEM_SYNC_THREAD;
}

// Non-tensor read with VDIM components
template<int P, int VDIM> MFEM_HOST_DEVICE inline
void Read(const DeviceTensor<3> &e_vec, const int e,
          Tensor<dTensor<VDIM>,P> &u)
{
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(p,x,P)
      {
         u(p)(c) = e_vec(p, c, e);
      }
   }
   MFEM_SYNC_THREAD;
}

// Non-tensor read with VDIMxVDIM components
template<int P, int VDIM> MFEM_HOST_DEVICE inline
void Read(const DeviceTensor<4> &e_vec, const int e,
          Tensor<dTensor<VDIM,VDIM>,P> &u)
{
   for (int w = 0; w < VDIM; w++)
   {   
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(p,x,P)
         {
            u(p)(h,w) = e_vec(p, h, w, e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 3D tensor read
template <int D1d> MFEM_HOST_DEVICE inline
void Read(const DeviceTensor<4> &e_vec, const int e,
          dTensor<D1d,D1d,D1d> &u)
{
   for (int dz = 0; dz < D1d; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            u(dx,dy,dz) = e_vec(dx,dy,dz,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 3D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Read(const DeviceTensor<5> &e_vec, const int e,
          Tensor<dTensor<VDIM>,D1d,D1d,D1d> &u)
{
   for (int c = 0; c < VDIM; c++)
   {
      for (int dz = 0; dz < D1d; dz++)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            MFEM_FOREACH_THREAD(dx,x,D1d)
            {
               u(dx,dy,dz)(c) = e_vec(dx,dy,dz,c,e);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 3D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Read(const DeviceTensor<6> &e_vec, const int e,
          Tensor<dTensor<VDIM,VDIM>,D1d,D1d,D1d> &u)
{
   for (int w = 0; w < VDIM; w++)
   {
      for (int h = 0; h < VDIM; h++)
      {
         for (int dz = 0; dz < D1d; dz++)
         {
            MFEM_FOREACH_THREAD(dy,y,D1d)
            {
               MFEM_FOREACH_THREAD(dx,x,D1d)
               {
                  u(dx,dy,dz)(h,w) = e_vec(dx,dy,dz,h,w,e);
               }
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 2D tensor read
template <int D1d> MFEM_HOST_DEVICE inline
void Read(const DeviceTensor<3> &e_vec, const int e,
          dTensor<D1d,D1d> &u)
{
   MFEM_FOREACH_THREAD(dy,y,D1d)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         u(dx,dy) = e_vec(dx,dy,e);
      }
   }
   MFEM_SYNC_THREAD;
}

// 2D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Read(const DeviceTensor<4> &e_vec, const int e,
          Tensor<dTensor<VDIM>,D1d,D1d> &u)
{
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            u(dx,dy)(c) = e_vec(dx,dy,c,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 2D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Read(const DeviceTensor<4> &e_vec, const int e,
          Tensor<dTensor<VDIM,VDIM>,D1d,D1d> &u)
{
   for (int w = 0; w < VDIM; w++)
   {
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            MFEM_FOREACH_THREAD(dx,x,D1d)
            {
               u(dx,dy)(h,w) = e_vec(dx,dy,h,w,e);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 1D tensor read
template <int D1d> MFEM_HOST_DEVICE inline
void Read(const DeviceTensor<2> &e_vec, const int e,
          dTensor<D1d> &u)
{
   MFEM_FOREACH_THREAD(dx,x,D1d)
   {
      u(dx) = e_vec(dx,e);
   }
   MFEM_SYNC_THREAD;
}

// 1D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Read(const DeviceTensor<3> &e_vec, const int e,
          Tensor<dTensor<VDIM>,D1d> &u)
{
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         u(dx)(c) = e_vec(dx,c,e);
      }
   }
   MFEM_SYNC_THREAD;
}

// 1D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Read(const DeviceTensor<3> &e_vec, const int e,
          Tensor<dTensor<VDIM>,D1d> &u)
{
   for (int w = 0; w < VDIM; w++)
   {
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            u(dx)(h,w) = e_vec(dx,h,w,e);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// Functions to read dofs to quad matrix
template<int P, int Q> MFEM_HOST_DEVICE inline
void ReadMatrix(const DeviceTensor<2> &d_B, dTensor<Q,P> &s_B)
{
   for (int p = 0; p < P; p++)
   {
      MFEM_FOREACH_THREAD(q,x,Q)
      {
         s_B(q,p) = d_B(q,p);
      }    
   }
   MFEM_SYNC_THREAD;
}

// Functions to write values (dofs or values at quadrature point)
// Non-tensor write
template<int P> MFEM_HOST_DEVICE inline
void Write(const dTensor<P> &u, const int e, DeviceTensor<2> &e_vec)
{
   MFEM_FOREACH_THREAD(p,x,P)
   {
      e_vec(p, e) = u(p);
   }
   MFEM_SYNC_THREAD;
}

// Non-tensor read with VDIM components
template<int P, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM>,P> &u, const int e,
           DeviceTensor<3> &e_vec)
{
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(p,x,P)
      {
         e_vec(p, c, e) = u(p)(c);
      }
   }
   MFEM_SYNC_THREAD;
}

// Non-tensor read with VDIMxVDIM components
template<int P, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM,VDIM>,P> &u, const int e,
           DeviceTensor<4> &e_vec)
{
   for (int w = 0; w < VDIM; w++)
   {   
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(p,x,P)
         {
            e_vec(p, h, w, e) = u(p)(h,w);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 3D tensor read
template <int D1d> MFEM_HOST_DEVICE inline
void Write(const dTensor<D1d,D1d,D1d> &u, const int e,
           DeviceTensor<4> &e_vec)
{
   for (int dz = 0; dz < D1d; dz++)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            e_vec(dx,dy,dz,e) = u(dx,dy,dz);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 3D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM>,D1d,D1d,D1d> &u, const int e,
           DeviceTensor<5> &e_vec)
{
   for (int c = 0; c < VDIM; c++)
   {
      for (int dz = 0; dz < D1d; dz++)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            MFEM_FOREACH_THREAD(dx,x,D1d)
            {
               e_vec(dx,dy,dz,c,e) = u(dx,dy,dz)(c);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 3D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM,VDIM>,D1d,D1d,D1d> &u, const int e,
           DeviceTensor<6> &e_vec)
{
   for (int w = 0; w < VDIM; w++)
   {
      for (int h = 0; h < VDIM; h++)
      {
         for (int dz = 0; dz < D1d; dz++)
         {
            MFEM_FOREACH_THREAD(dy,y,D1d)
            {
               MFEM_FOREACH_THREAD(dx,x,D1d)
               {
                  e_vec(dx,dy,dz,h,w,e) = u(dx,dy,dz)(h,w);
               }
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 2D tensor read
template <int D1d> MFEM_HOST_DEVICE inline
void Write(const dTensor<D1d,D1d> &u, const int e,
           DeviceTensor<3> &e_vec)
{
   MFEM_FOREACH_THREAD(dy,y,D1d)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         e_vec(dx,dy,e) = u(dx,dy);
      }
   }
   MFEM_SYNC_THREAD;
}

// 2D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM>,D1d,D1d> &u, const int e,
           DeviceTensor<4> &e_vec)
{
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(dy,y,D1d)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            e_vec(dx,dy,c,e) = u(dx,dy)(c);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 2D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM,VDIM>,D1d,D1d> &u, const int e,
           DeviceTensor<4> &e_vec)
{
   for (int w = 0; w < VDIM; w++)
   {
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(dy,y,D1d)
         {
            MFEM_FOREACH_THREAD(dx,x,D1d)
            {
               e_vec(dx,dy,h,w,e) = u(dx,dy)(h,w);
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 1D tensor read
template <int D1d> MFEM_HOST_DEVICE inline
void Write(const dTensor<D1d> &u, const int e,
           DeviceTensor<2> &e_vec)
{
   MFEM_FOREACH_THREAD(dx,x,D1d)
   {
      e_vec(dx,e) = u(dx);
   }
   MFEM_SYNC_THREAD;
}

// 1D tensor read with VDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM>,D1d> &u, const int e,
           DeviceTensor<3> &e_vec)
{
   for (int c = 0; c < VDIM; c++)
   {
      MFEM_FOREACH_THREAD(dx,x,D1d)
      {
         e_vec(dx,c,e) = u(dx)(c);
      }
   }
   MFEM_SYNC_THREAD;
}

// 1D tensor read with VDIMxVDIM components
template <int D1d, int VDIM> MFEM_HOST_DEVICE inline
void Write(const Tensor<dTensor<VDIM>,D1d> &u, const int e,
           DeviceTensor<3> &e_vec)
{
   for (int w = 0; w < VDIM; w++)
   {
      for (int h = 0; h < VDIM; h++)
      {
         MFEM_FOREACH_THREAD(dx,x,D1d)
         {
            e_vec(dx,h,w,e) = u(dx)(h,w);
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// Functions to interpolate from degrees of freedom to quadrature points
// Non-tensor case
template<int P, int Q> MFEM_HOST_DEVICE inline
void Interp(const dTensor<Q,P> &B,
            const dTensor<P> &u,
            dTensor<Q> &u_q)
{
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v = 0.0;
      for (int d = 0; d < P; ++d)
      {
         const double b = B(q,d);
         v += b * u(d);
      }
      u_q(q) = v;
   }
}

// Non-tensor case with VDIM components
template<int Q, int P, int VDIM> MFEM_HOST_DEVICE inline
void Interp(const dTensor<Q,P> &B,
            const Tensor<dTensor<VDIM>,P> &u,
            Tensor<dTensor<VDIM>,Q> &u_q)
{
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      double v[VDIM];
      for (int c = 0; c < VDIM; c++)
      {
         v[c] = 0.0;
      }
      for (int d = 0; d < P; ++d)
      {
         const double b = B(q,d);
         for (int c = 0; c < VDIM; c++)
         {
            v[c] += b * u(d)(c);
         }
      }
      for (int c = 0; c < VDIM; c++)
      {
         u_q(q)(c) = v[c];
      }
   }
}

// 3D Tensor case
template<int Q1d, int P1d> MFEM_HOST_DEVICE inline
void Interp(const dTensor<Q1d,P1d> &B,
            const dTensor<P1d,P1d,P1d> &u,
            dTensor<Q1d,Q1d,Q1d> &u_q)
{
   dTensor<Q1d,P1d,P1d> Bu;
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double val = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double b = B(qx,dx);
               const double x = u(dx,dy,dz);
               val += b * x;
            }
            Bu(qx,dy,dz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   dTensor<Q1d,Q1d,P1d> BBu;
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            double val = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double b = B(qy,dy);
               const double x = Bu(qx,dy,dz);
               val += b * x;
            }
            BBu(qx,qy,dz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qz,z,D1D)
         {
            double val = 0.0;
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double b = B(qz,dz);
               const double x = Bu(qx,qy,dz);
               val += b * x;
            }
            u_q(qx,qy,qz) = val;
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 3D Tensor case with VDIM components
template<int Q1d, int P1d, int VDIM> MFEM_HOST_DEVICE inline
void Interp(const dTensor<Q1d,P1d> &B,
            const Tensor<dTensor<VDIM>,P1d,P1d,P1d> &u,
            Tensor<dTensor<VDIM>,Q1d,Q1d,Q1d> &u_q)
{
   Tensor<dTensor<VDIM>,Q1d,P1d,P1d> Bu;
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double val[VDIM];
            for (int c = 0; c < VDIM; c++)
            {
               val[c] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double b = B(qx,dx);
               for (int c = 0; c < count; c++)
               {
                  const double x = u(dx,dy,dz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDIM; c++)
            {
               Bu(qx,dy,dz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   Tensor<dTensor<VDIM>,Q1d,Q1d,P1d> BBu;
   MFEM_FOREACH_THREAD(dz,z,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            double val[VDIM];
            for (int c = 0; c < VDIM; c++)
            {
               val[c] = 0.0;
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const double b = B(qy,dy);
               for (int c = 0; c < count; c++)
               {
                  const double x = Bu(qx,dy,dz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDIM; c++)
            {
               BBu(qx,qy,dz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qz,z,D1D)
         {
            double val[VDIM];
            for (int c = 0; c < VDIM; c++)
            {
               val[c] = 0.0;
            }
            for (int dz = 0; dz < D1D; ++dz)
            {
               const double b = B(qz,dz);
               for (int c = 0; c < count; c++)
               {
                  const double x = Bu(qx,qy,dz)(c);
                  val[c] += b * x;
               }
            }
            for (int c = 0; c < VDIM; c++)
            {
               u_q(qx,qy,qz)(c) = val[c];
            }
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 2D Tensor case
template<int Q1d, int P1d> MFEM_HOST_DEVICE inline
void Interp(const dTensor<Q1d,P1d> &B,
            const dTensor<P1d,P1d> &u,
            dTensor<Q1d,Q1d> &u_q)
{
   dTensor<Q1d,P1d> Bu;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double val = 0.0;
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double b = B(qx,dx);
            const double x = u(dx,dy);
            val += b * x;
         }
         Bu(qx,dy) = val;
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         double val = 0.0;
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double b = B(qy,dy);
            const double x = Bu(qx,dy);
            val += b * x;
         }
         u_q(qx,qy) = val;
      }
   }
   MFEM_SYNC_THREAD;
}

// 2D Tensor case with VDIM components
template<int Q1d, int P1d, int VDIM> MFEM_HOST_DEVICE inline
void Interp(const dTensor<Q1d,P1d> &B,
            const Tensor<dTensor<VDIM>,P1d,P1d> &u,
            Tensor<dTensor<VDIM>,Q1d,Q1d> &u_q)
{
   Tensor<dTensor<VDIM>,Q1d,P1d> Bu;
   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double val[VDIM];
         for (int c = 0; c < VDIM; c++)
         {
            val[c] = 0.0;
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double b = B(qx,dx);
            for (int c = 0; c < VDIM; c++)
            {
               const double x = u(dx,dy)(c);
               val[c] += b * x;
            }
         }
         for (int c = 0; c < VDIM; c++)
         {
            Bu(qx,dy)(c) = val[c];
         }
      }
   }
   MFEM_SYNC_THREAD;
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         double val[VDIM];
         for (int c = 0; c < VDIM; c++)
         {
            val[c] = 0.0;
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double b = B(qy,dy);
            for (int c = 0; c < VDIM; c++)
            {
               const double x = Bu(qx,dy)(c);
               val[c] += b * x;
            }
         }
         for (int c = 0; c < VDIM; c++)
         {
            u_q(qx,qy)(c) = val[c];
         }
      }
   }
   MFEM_SYNC_THREAD;
}

// 1D Tensor case
template<int Q1d, int P1d> MFEM_HOST_DEVICE inline
void Interp(const dTensor<Q1d,P1d> &B,
            const dTensor<P1d> &u,
            dTensor<Q1d> &u_q)
{
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      double val = 0.0;
      for (int dx = 0; dx < D1D; ++dx)
      {
         const double b = B(qx,dx);
         const double x = u(dx);
         val += b * x;
      }
      u_q(qx) = val;
   }
   MFEM_SYNC_THREAD;
}

// 1D Tensor case with VDIM components
template<int Q1d, int P1d, int VDIM> MFEM_HOST_DEVICE inline
void Interp(const dTensor<Q1d,P1d> &B,
            const Tensor<dTensor<VDIM>,P1d> &u,
            Tensor<dTensor<VDIM>,Q1d> &u_q)
{
   MFEM_FOREACH_THREAD(qx,x,Q1D)
   {
      double val[VDIM];
      for (int c = 0; c < VDIM; c++)
      {
         val[c] = 0.0;
      }
      for (int dx = 0; dx < D1D; ++dx)
      {
         const double b = B(qx,dx);
         for (int c = 0; c < VDIM; c++)
         {
            const double x = u(dx)(c);
            val[c] += b * x;
         }
      }
      for (int c = 0; c < VDIM; c++)
      {
         u_q(qx)(c) = val[c];
      }
   }
   MFEM_SYNC_THREAD;
}

} // mfem namespace

#endif // MFEM_DTENSOR
