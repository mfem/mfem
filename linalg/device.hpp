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

#ifndef MFEM_DEVICE
#define MFEM_DEVICE

#include "../general/okina.hpp"

namespace mfem
{

// Simple vector and tensor classes for dense linear algebra on the device.

/// Simple vector of size 3, appropriate for use on the GPU
class DeviceVector3
{
private:
   double data[3];
public:
   DeviceVector3() {}
   DeviceVector3(const double *r)
   { data[0] = r[0]; data[1] = r[1]; data[2] = r[2]; }
   DeviceVector3(const double r0, const double r1, const double r2)
   { data[0] = r0; data[1] = r1; data[2] = r2; }
   ~DeviceVector3() {}
   inline operator double* () { return data; }
   inline operator const double* () const { return data; }
   inline double& operator[](const size_t x) { return data[x]; }
};

/// A Class to compute the real index from the multi-indices of a tensor
template <int N, int Dim, typename T, typename... Args>
class TensorInd
{
public:
   MFEM_HOST_DEVICE static int result(const int* sizes, T first, Args... args)
   {
#ifndef MFEM_USE_CUDA
      MFEM_ASSERT(first<sizes[N-1],"Trying to access out of boundary.");
#endif
      return first + sizes[N - 1] * TensorInd < N + 1, Dim, Args... >::result(sizes,
                                                                              args...);
   }
};
// Terminal case
template <int Dim, typename T, typename... Args>
class TensorInd<Dim, Dim, T, Args...>
{
public:
   MFEM_HOST_DEVICE static int result(const int* sizes, T first, Args... args)
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
   static int result(int* sizes, T first, Args... args)
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
   static int result(int* sizes, T first, Args... args)
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
   explicit DeviceTensor() : capacity(0), data(NULL)
   { mfem_error("No default constructor."); }

   /// Constructor to initialize a tensor from the Scalar array _data
   template <typename... Args>
   DeviceTensor(Scalar* _data, Args... args)
   {
      static_assert(sizeof...(args) == Dim, "Wrong number of arguments");
      // Initialize sizes, and compute the number of values
      const long int nb = Init<1, Dim, Args...>::result(sizes, args...);
      capacity = nb;
      data = capacity>0?(Scalar*)mfem::mm::ptr(_data):nullptr;
   }

   /// Constructor to initialize a tensor from the Scalar array _data
   DeviceTensor(const Scalar* _data)
   {
      data = (Scalar*)mfem::mm::ptr(_data);
   }

   /// Constructor to initialize a tensor from the Scalar array _data
   DeviceTensor(Scalar* _data)
   {
      data = (Scalar*)mfem::mm::ptr(_data);
   }

   /// Constructor to initialize a tensor from the const Scalar array _data
   template <typename... Args>
   DeviceTensor(const Scalar* _data, Args... args)
   {
      static_assert(sizeof...(args) == Dim, "Wrong number of arguments");
      // Initialize sizes, and compute the number of values
      const long int nb = Init<1, Dim, Args...>::result(sizes, args...);
      capacity = nb;
      data = (capacity>0)?const_cast<Scalar*>((Scalar*)mfem::mm::ptr(_data)):nullptr;
   }

   /// Copy constructor
   DeviceTensor(const DeviceTensor& t)
   {
      for (int i = 0; i < Dim; ++i)
      {
         sizes[i] = t.size(i);
      }
      data = const_cast<Scalar*>(t.getData());
   }

   /// Conversion to `double *`.
   inline operator Scalar *() { return data; }

   /// Conversion to `const double *`.
   inline operator const Scalar *() const { return data; }

   /// Returns the size of the i-th dimension #UNSAFE#
   int size(int i) const
   {
      return sizes[i];
   }

   /// Copy assignment operator (do not resize DeviceTensors)
   DeviceTensor& operator=(const DeviceTensor& t)
   {
      mfem_error("No assignment operator.");
      return *this;
   }

   /// Const accessor for the data
   template <typename... Args> MFEM_HOST_DEVICE
   const Scalar& operator()(Args... args) const
   {
      static_assert(sizeof...(args) == Dim, "Wrong number of arguments");
      return data[ TensorInd<1, Dim, Args...>::result(sizes, args...) ];
   }

   /// Reference accessor to the data
   template <typename... Args> MFEM_HOST_DEVICE
   Scalar& operator()(Args... args)
   {
      static_assert(sizeof...(args) == Dim, "Wrong number of arguments");
      return data[ TensorInd<1, Dim, Args...>::result(sizes, args...) ];
   }

   MFEM_HOST_DEVICE const Scalar& operator[](int i) const
   {
      return data[i];
   }

   MFEM_HOST_DEVICE Scalar& operator[](int i)
   {
      return data[i];
   }

   /// Returns the length of the DeviceTensor (number of values, may be
   /// different from capacity).
   int length() const
   {
      int res = 1;
      for (int i = 0; i < Dim; ++i)
      {
         res *= sizes[i];
      }
      return res;
   }

   /// Returns the dimension of the tensor.
   int dimension() const
   {
      return Dim;
   }

   /// Returns the Scalar array data (really unsafe). Mostly exists to remap
   /// DeviceTensors, so could be avoided by using more constructors...
   Scalar* getData()
   {
      return data;
   }

   const Scalar* getData() const
   {
      return data;
   }
};

typedef DeviceTensor<1,int> DeviceArray;
typedef DeviceTensor<1,double> DeviceVector;
typedef DeviceTensor<2,double> DeviceMatrix;

} // mfem namespace

#endif // MFEM_DEVICE
