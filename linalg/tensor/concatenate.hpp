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

#ifndef MFEM_CONCATENATE
#define MFEM_CONCATENATE

#include "tensor.hpp"

namespace mfem
{

template <int Rank, int N = 1>
struct AddSize
{
   template <typename... Sizes>
   static auto init(const DynamicLayout<Rank> &in, Sizes... sizes)
   {
      return AddSize<N+1,Rank>::init(in, in.template Size<Rank-N>(), sizes...);
   }
};

template <int Rank>
struct AddSize<Rank,Rank>
{
   template <typename... Sizes>
   static auto init(const DynamicLayout<Rank> &in, Sizes... sizes)
   {
      return DynamicLayout<sizeof...(Sizes)+1>(in.template Size<0>(), sizes...);
   }
};

template <int Rank, typename... ExtraSizes>
auto MakeLayout(const DynamicLayout<Rank> &in, ExtraSizes... sizes)
{
   return AddSize<Rank>::init(in, sizes...);
}

template <int Rank,
          typename T,
          template <typename,int...> class Container,
          template <int...> class Layout,
          int... Sizes>
auto Concatenate(Tensor<Rank,T,Container<T,Sizes...>,Layout<Rank>> &tl,
                 Tensor<Rank,T,Container<T,Sizes...>,Layout<Rank>> &tr)
{
   using ResContainer = Container<T,Sizes...,2>;
   using ResLayout = Layout<Rank+1>;
   ResLayout layout = MakeLayout(tl, 2);
   Tensor<Rank+1, T, ResContainer, ResLayout> res(layout);
   res.template Get<Rank>(0) = tl;
   res.template Get<Rank>(1) = tr;
   return res;
}

template <int Rank,
          typename T,
          template <typename,int...> class Container,
          template <int...> class Layout,
          int... Sizes>
auto Concatenate(Tensor<Rank,T,Container<T,Sizes...>,Layout<Rank>> &tl,
                 Tensor<Rank,T,Container<T,Sizes...>,Layout<Rank>> &tm,
                 Tensor<Rank,T,Container<T,Sizes...>,Layout<Rank>> &tr)
{
   using ResContainer = Container<T,Sizes...,2>;
   using ResLayout = Layout<Rank+1>;
   ResLayout layout = MakeLayout(tl, 2);
   Tensor<Rank+1, T, ResContainer, ResLayout> res(layout);
   res.template Get<Rank>(0) = tl;
   res.template Get<Rank>(1) = tm;
   res.template Get<Rank>(2) = tr;
   return res;
}

template <int Rank,
          typename T,
          template <typename,int...> class Container,
          template <int...> class Layout,
          int... Sizes>
auto Concatenate(Tensor<Rank,T,Container<T,Sizes...>,Layout<Sizes...>> &tl,
                 Tensor<Rank,T,Container<T,Sizes...>,Layout<Sizes...>> &tr)
{
   using ResContainer = Container<T,Sizes...,2>;
   using ResLayout = Layout<Sizes...,2>;
   Tensor<Rank+1, T, ResContainer, ResLayout> res;
   res.Get<Rank>(0) = tl;
   res.Get<Rank>(1) = tr;
   return res;
}

template <int Rank,
          typename T,
          template <typename,int...> class Container,
          template <int...> class Layout,
          int... Sizes>
auto Concatenate(Tensor<Rank,T,Container<T,Sizes...>,Layout<Sizes...>> &tl,
                 Tensor<Rank,T,Container<T,Sizes...>,Layout<Sizes...>> &tm,
                 Tensor<Rank,T,Container<T,Sizes...>,Layout<Sizes...>> &tr)
{
   using ResContainer = Container<T,Sizes...,3>;
   using ResLayout = Layout<Sizes...,3>;
   Tensor<Rank+1, T, ResContainer, ResLayout> res;
   res.Get<Rank>(0) = tl;
   res.Get<Rank>(1) = tm;
   res.Get<Rank>(2) = tr;
   return res;
}

// TODO write specialized versions for each main Tensor type?

} // namespace mfem

#endif // MFEM_CONCATENATE
