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

#ifndef MFEM_TENSOR
#define MFEM_TENSOR

#include "../../general/backends.hpp"
#include "container.hpp"
#include "layout.hpp"
#include "util.hpp"

namespace mfem
{

template <int Rank,
          typename T = double,
          typename Container = DeviceContainer<T>,
          typename Layout = DynamicLayout<Rank>>
class Tensor
{
private:
   Container data;
   Layout layout;

public:
   template <typename... Sizes> MFEM_HOST_DEVICE
   Tensor(Sizes... sizes): data(sizes...), layout(sizes...) { }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   const T& operator()(Idx... args) const
   {
      static_assert(Rank==sizeof...(Idx), "Wrong number of indices");
      return data[ layout(args...) ];
   }

   template <typename... Idx> MFEM_HOST_DEVICE inline
   T& operator()(Idx... args)
   {
      static_assert(Rank==sizeof...(Idx), "Wrong number of indices");
      return data[ layout(args...) ];
   }

   MFEM_HOST_DEVICE inline
   Tensor& operator= (const T &val)
   {
      for (size_t i = 0; i < data.size(); i++)
      {
         data[i] = val;
      }
      return *this;
   }
};

template <typename T, int... Sizes>
using StaticSharedTensor = Tensor<sizeof...(Sizes),
                                    T,
                                    StaticSharedContainer<T, Sizes...>,
                                    StaticLayout<Sizes...> >;

template <int Rank, typename T, int MaxSize = 16>
using SharedTensor = Tensor<Rank,
                            T,
                            StaticSharedContainer<T, Rank*MaxSize>,
                            DynamicLayout<Rank> >;

template <typename T, int... Sizes>
using StaticTensor = Tensor<sizeof...(Sizes),
                            T,
                            StaticContainer<T, Sizes...>,
                            StaticLayout<Sizes...> >;

template <int... Sizes>
using dTensor = StaticTensor<double,Sizes...>;

/// A fixed size tensor class
// template<typename T, int... Dims>
// class Tensor{
// private:
//    MFEM_SHARED T data[Size<Dims...>::val];

// public:
//    MFEM_HOST_DEVICE
//    explicit Tensor() {}
   
//    MFEM_HOST_DEVICE
//    explicit Tensor(const T &val)
//    {
//       for (size_t i = 0; i < Size<Dims...>::val; i++)
//       {
//          data[i] = val;
//       }      
//    }

//    MFEM_HOST_DEVICE
//    Tensor(const Tensor &rhs)
//    {
//       for (size_t i = 0; i < Size<Dims...>::val; i++)
//       {
//          data[i] = rhs[i];
//       }
//    }

//    const int size() const
//    {
//       return Size<Dims...>::val;
//    }

//    template<typename... Idx> MFEM_HOST_DEVICE inline
//    const T& operator()(Idx... args) const
//    {
//       static_assert(sizeof...(Dims)==sizeof...(Idx), "Wrong number of indices");
//       return data[ TensorIndex<Dims...>::eval(args...) ];
//    }

//    template<typename... Idx> MFEM_HOST_DEVICE inline
//    T& operator()(Idx... args)
//    {
//       static_assert(sizeof...(Dims)==sizeof...(Idx), "Wrong number of indices");
//       return data[ TensorIndex<Dims...>::eval(args...) ];
//    }

//    MFEM_HOST_DEVICE inline
//    Tensor<T,Dims...>& operator= (const T &val)
//    {
//       for (size_t i = 0; i < Size<Dims...>::val; i++)
//       {
//          data[i] = val;
//       }
//       return *this;
//    }

// private:
//    MFEM_HOST_DEVICE inline
//    const T& operator[] (const int i) const
//    {
//       return data[i];
//    }

//    //Compute the index inside a Tensor
//    template<int Cpt, int rank, int... Sizes>
//    struct Index
//    {
//       template <typename... Idx>
//       static inline int eval(int first, Idx... args)
//       {
//          return first + Dim<Cpt-1,Sizes...>::val * Index<Cpt+1, rank, Sizes...>::eval(args...);
//       }
//    };

//    template<int rank, int... Sizes>
//    struct Index<rank,rank,Sizes...>
//    {
//       static inline int eval(int first)
//       {
//          return first;
//       }
//    };

//    template<int... Sizes>
//    struct TensorIndex
//    {
//       template <typename... Idx>
//       static inline int eval(Idx... args)
//       {
//          return Index<1,sizeof...(Sizes),Sizes...>::eval(args...);
//       }
//    };
// };

} // namespace mfem

#endif // MFEM_TENSOR
