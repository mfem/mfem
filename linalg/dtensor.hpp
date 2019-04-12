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

#ifndef MFEM_DTENSOR
#define MFEM_DTENSOR

#include "../general/cuda.hpp"
#include "../general/mem_manager.hpp"

namespace mfem
{

/// A Class to compute the real index from the multi-indices of a tensor
template <int N, int Dim, typename T, typename... Args>
class TensorInd
{
public:
   MFEM_ATTR_HOST_DEVICE
   static inline int result(const int* sizes, T first, Args... args)
   {
#ifndef MFEM_USE_CUDA
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
   MFEM_ATTR_HOST_DEVICE
   static inline int result(const int* sizes, T first, Args... args)
   {
#ifndef MFEM_USE_CUDA
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
      data = (capacity > 0) ? mfem::Ptr(_data) : NULL;
   }

   /// Constructor to initialize a tensor from the Scalar array _data
   DeviceTensor(const Scalar* _data)
   {
      data = (Scalar*) mfem::Ptr(_data);
   }

   /// Constructor to initialize a tensor from the Scalar array _data
   DeviceTensor(Scalar* _data) { data = mfem::Ptr(_data); }

   /// Constructor to initialize a tensor from the const Scalar array _data
   template <typename... Args>
   DeviceTensor(const Scalar* _data, Args... args)
   {
      static_assert(sizeof...(args) == Dim, "Wrong number of arguments");
      // Initialize sizes, and compute the number of values
      const long int nb = Init<1, Dim, Args...>::result(sizes, args...);
      capacity = nb;
      data = (capacity > 0) ? (Scalar*)mfem::Ptr(_data) : NULL;
   }

   /// Copy constructor
   MFEM_ATTR_HOST_DEVICE DeviceTensor(const DeviceTensor& t)
   {
      for (int i = 0; i < Dim; ++i)
      {
         sizes[i] = t.sizes[i];
      }
      data = t.data;
   }

   /// Conversion to `Scalar *`.
   inline operator Scalar *() const { return data; }

   /// Const accessor for the data
   template <typename... Args> MFEM_ATTR_HOST_DEVICE inline
   Scalar& operator()(Args... args) const
   {
      static_assert(sizeof...(args) == Dim, "Wrong number of arguments");
      return data[ TensorInd<1, Dim, Args...>::result(sizes, args...) ];
   }

   MFEM_ATTR_HOST_DEVICE inline Scalar& operator[](int i) const
   {
      return data[i];
   }
};

typedef DeviceTensor<1,int> DeviceArray;
typedef DeviceTensor<1,double> DeviceVector;
typedef DeviceTensor<2,double> DeviceMatrix;

} // mfem namespace

#endif // MFEM_DTENSOR
