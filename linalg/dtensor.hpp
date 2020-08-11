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

template<int Q, int P> MFEM_HOST_DEVICE inline
void Interp(const dTensor<Q,P> &B,
            const dTensor<P> &u,
            dTensor<Q> &u_q)
{
   MFEM_FOREACH_THREAD(q,x,Q)
   {
      u_q(q) = 0;
      for (int d = 0; d < P; ++d)
      {
         u_q(q) += B(q,d) * u(d);
      }
   }
}

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

} // mfem namespace

#endif // MFEM_DTENSOR
