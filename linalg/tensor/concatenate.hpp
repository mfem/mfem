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

// template <int Rank, typename T, typename Container, typename Layout>
// auto Concatenate(Tensor<Rank, T, Container, Layout> &tl,
//                  Tensor<Rank, T, Container, Layout> &tr)
// {
//    Tensor<Rank+1, T, Container, Layout> res;
//    return res;
// }

// template <int Rank,
//           typename T,
//           typename Container,
//           typename Layout,
//           Tensor<Rank,T,Container,Layout>... Args>
// auto Concatenate(Args... ts)
// {
//    Tensor<Rank+1, T, Container, Layout> res;
//    return res;
// }

// namespace detail
// {
// template <int N>
// struct For
// {
//    template <typename TensorIn, typename TensorOut, typename... Args>
//    static void set(TensorOut &out, TensorIn &in0, Args... &ins)
//    {
//       out.Get<TensorOut::rank-1>(N) = in0;
//       For<N-1>::set(out,ins...);
//    }
// };

// template <>
// struct For<0>
// {
//    template <typename TensorIn, typename TensorOut>
//    static void set(TensorOut &out, TensorIn &in)
//    {
//       out.Get<TensorOut::rank-1>(0) = in;
//    }
// };

// }

// template <int Rank,
//           typename T,
//           template <typename,int...> class Container,
//           template <int...> class Layout,
//           int... Sizes,
//           Tensor<Rank,T,Container<T,Sizes...>,Layout<Sizes...>>... Args>
// auto Concatenate(Args... ts)
// {
//    using ResContainer = Container<T,Sizes...,sizeof...(Args)>;
//    using ResLayout = Layout<Sizes...,sizeof...(Args)>;
//    Tensor<Rank+1, T, ResContainer, ResLayout> res;
//    // detail::For<sizeof...(Args)>::set(res,ts...);
//    return res;
// }

template <int Rank, int N = 1>
struct AddSize
{
   template <template <int...> class Layout, typename... Sizes>
   auto init(const Layout<Rank> &in, Sizes... sizes)
   {
      return AddSize<N+1,Rank>::init(in, in.template Size<Rank-N>(), sizes...);
   }
};

template <int Rank>
struct AddSize<Rank,Rank>
{
   template <template <int...> class Layout, typename... Sizes>
   auto init(const Layout<Rank> &in, Sizes... sizes)
   {
      return Layout<sizeof...(Sizes)>(sizes...);
   }
};

template <int Rank, template <int...> class Layout, typename... ExtraSizes>
auto MakeLayout(const Layout<Rank> &in, ExtraSizes... sizes)
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
