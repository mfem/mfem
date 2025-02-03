// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// #ifndef MFEM_DTENSOR
// #define MFEM_DTENSOR

// #include <numeric>

// #include "../config/config.hpp"

// namespace mfem
// {

// /// ///////////////////////////////////////////////////////////////////////////
// template <int N, typename... Args>
// MFEM_HOST_DEVICE inline int ColMajor(const int (&dims)[N], Args... args)
// {
//    int offset = 0, i = 0, unused;
//    ((offset *= dims[N - ++i], offset += args, std::ignore = unused) = ...);
//    return offset;
// }

// /// ///////////////////////////////////////////////////////////////////////////
// template <int N, typename... Args>
// MFEM_HOST_DEVICE inline int RowMajor(const int (&dims)[N], Args... args)
// {
//    int offset = 0, i = 0;
//    return ((..., (offset *= dims[i++], offset += args, 0)), offset);
// }

// /// ///////////////////////////////////////////////////////////////////////////
// template <int N, typename T = real_t, bool Column = true>
// class DeviceTensor
// {
// protected:
//    int dims[N], size;
//    T *data;

// public:
//    /// Default constructor
//    // DeviceTensor() = delete;
//    MFEM_HOST_DEVICE
//    DeviceTensor() {}

//    /// Constructor to initialize a tensor from the Scalar array data_
//    template <typename... Args, typename = std::enable_if_t<(sizeof...(Args) == N)>>
//    MFEM_HOST_DEVICE DeviceTensor(T *data, Args... args):
//       dims{args...},
//       size(std::accumulate(dims, dims + N, 1, std::multiplies<int> {})),
//         data(size > 0 ? data : nullptr)
//    { }

//    /// Copy constructor (default)
//    // DeviceTensor(const DeviceTensor &) = default;

//    /// Copy assignment (default)
//    // DeviceTensor &operator=(const DeviceTensor &) = default;

//    /// Conversion to `Scalar *`.
//    MFEM_HOST_DEVICE inline operator T *() const { return data; }

//    /// Computes the offset of the tensor element at the given multi-indices
//    template <typename... Args, typename = std::enable_if_t<(sizeof...(Args) == N)>>
//    MFEM_HOST_DEVICE inline int Offset(Args... args) const
//    {
//       constexpr auto offset = Column ? ColMajor<N, Args...> : RowMajor<N, Args...>;
//       return offset(dims, args...);
//    }

//    /// Const accessor for the data
//    template <typename... Args, typename = std::enable_if_t<(sizeof...(Args) == N)>>
//    MFEM_HOST_DEVICE inline T &operator()(Args... args) const
//    {
//       return data[Offset(args...)];
//    }

//    /// Subscript operator where the tensor is viewed as a 1D array.
//    MFEM_HOST_DEVICE inline T &operator[](int i) const { return data[i]; }

//    /// Returns the size of the tensor
//    MFEM_HOST_DEVICE inline int Size() const { return size; }

//    MFEM_HOST_DEVICE inline std::array<int, N> GetShape() const
//    {
//       std::array<int, N> s;
//       for (int i = 0; i < N; i++)
//       {
//          s[i] = dims[i];
//       }
//       return s;
//    }
// };

// /** @brief Wrap a pointer as a DeviceTensor with automatically deduced template
//     parameters */
// template <typename T, typename... Dims>
// MFEM_HOST_DEVICE inline DeviceTensor<sizeof...(Dims), T> Reshape(T *ptr,
//                                                                  Dims... dims)
// {
//    return DeviceTensor<sizeof...(Dims), T>(ptr, dims...);
// }

// template <typename T, typename... Dims>
// MFEM_HOST_DEVICE inline DeviceTensor<sizeof...(Dims), T, false> RowReshape(
//    T *ptr,
//    Dims... dims)
// {
//    return DeviceTensor<sizeof...(Dims), T, false>(ptr, dims...);
// }

// using DeviceArray = DeviceTensor<1, int>;
// using ConstDeviceArray = DeviceTensor<1, const int>;

// using DeviceVector = DeviceTensor<1, real_t>;
// using ConstDeviceVector = DeviceTensor<1, const real_t>;

// using DeviceMatrix = DeviceTensor<2, real_t>;
// using ConstDeviceMatrix = DeviceTensor<2, const real_t>;

// using DeviceCube = DeviceTensor<3, real_t>;
// using ConstDeviceCube = DeviceTensor<3, const real_t>;

// } // namespace mfem

// #endif // MFEM_DTENSOR

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
      return static_cast<int>(first + sizes[N - 1] * TensorInd < N + 1, Dim, Args... >
                              ::result(sizes, args...));
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
      return static_cast<int>(first);
   }
};


/// A class to initialize the size of a Tensor
template <int N, int Dim, typename T, typename... Args>
class Init
{
public:
   MFEM_HOST_DEVICE
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
   MFEM_HOST_DEVICE
   static inline int result(int* sizes, T first, Args... args)
   {
      sizes[Dim - 1] = first;
      return first;
   }
};


/// A basic generic Tensor class, appropriate for use on the GPU
template<int Dim, typename Scalar = real_t>
class DeviceTensor
{
protected:
   int capacity;
   Scalar *data;
   int sizes[Dim];

public:
   /// Default constructor
   // DeviceTensor() = delete;
   MFEM_HOST_DEVICE
   DeviceTensor() {}

   /// Constructor to initialize a tensor from the Scalar array data_
   template <typename... Args> MFEM_HOST_DEVICE
   DeviceTensor(Scalar* data_, Args... args)
   {
      static_assert(sizeof...(args) == Dim, "Wrong number of arguments");
      // Initialize sizes, and compute the number of values
      const long int nb = Init<1, Dim, Args...>::result(sizes, args...);
      capacity = nb;
      data = (capacity > 0) ? data_ : NULL;
   }

   /// Copy constructor (default)
   DeviceTensor(const DeviceTensor&) = default;

   /// Copy assignment (default)
   DeviceTensor& operator=(const DeviceTensor&) = default;

   /// Conversion to `Scalar *`.
   MFEM_HOST_DEVICE inline operator Scalar *() const { return data; }

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

   MFEM_HOST_DEVICE inline std::array<int, Dim> GetShape() const
   {
      std::array<int, Dim> s;
      for (int i = 0; i < Dim; i++)
      {
         s[i] = sizes[i];
      }
      return s;
   }
};


/** @brief Wrap a pointer as a DeviceTensor with automatically deduced template
    parameters */
template <typename T, typename... Dims> MFEM_HOST_DEVICE
inline DeviceTensor<sizeof...(Dims),T> Reshape(T *ptr, Dims... dims)
{
   return DeviceTensor<sizeof...(Dims),T>(ptr, dims...);
}


typedef DeviceTensor<1,int> DeviceArray;
typedef DeviceTensor<1,const int> ConstDeviceArray;

typedef DeviceTensor<1,real_t> DeviceVector;
typedef DeviceTensor<1,const real_t> ConstDeviceVector;

typedef DeviceTensor<2,real_t> DeviceMatrix;
typedef DeviceTensor<2,const real_t> ConstDeviceMatrix;

typedef DeviceTensor<3,real_t> DeviceCube;
typedef DeviceTensor<3,const real_t> ConstDeviceCube;

} // mfem namespace

#endif // MFEM_DTENSOR
