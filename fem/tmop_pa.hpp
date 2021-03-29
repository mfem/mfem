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

#ifndef MFEM_TMOP_PA_HPP
#define MFEM_TMOP_PA_HPP

#include "../config/config.hpp"
#include "../linalg/dtensor.hpp"

#include "../fem/kernels.hpp"

#include <unordered_map>

namespace mfem
{

namespace kernels
{

/// Generic emplace
template<typename K, const int N,
         typename Key_t = typename K::Key_t,
         typename Kernel_t = typename K::Kernel_t>
void emplace(std::unordered_map<Key_t, Kernel_t> &map)
{
   constexpr Key_t key = K::template GetKey<N>();
   constexpr Kernel_t value = K::template GetValue<key>();
   map.emplace(key, value);
}

/// Instances
template<class K, typename T, T... idx>
struct instances
{
   static void Fill(std::unordered_map<typename K::Key_t,
                    typename K::Kernel_t> &map)
   {
      using unused = int[];
      (void) unused {0, (emplace<K,idx>(map), 0)... };
   }
};

/// Cat instances
template<class K, typename Offset, typename Lhs, typename Rhs> struct cat;
template<class K, typename T, T Offset, T... Lhs, T... Rhs>
struct cat<K, std::integral_constant<T, Offset>,
          instances<K, T, Lhs...>,
          instances<K, T, Rhs...> >
{ using type = instances<K, T, Lhs..., (Offset + Rhs)...>; };

/// Sequence, empty and one element terminal cases
template<class K, typename T, typename N>
struct sequence
{
   using Lhs = std::integral_constant<T, N::value/2>;
   using Rhs = std::integral_constant<T, N::value-Lhs::value>;
   using type = typename cat<K, Lhs,
         typename sequence<K, T, Lhs>::type,
         typename sequence<K, T, Rhs>::type>::type;
};

template<class K, typename T>
struct sequence<K, T, std::integral_constant<T,0> >
{ using type = instances<K,T>; };

template<class K, typename T>
struct sequence<K, T, std::integral_constant<T,1> >
{ using type = instances<K,T,0>; };

/// Make_sequence
template<class Instance, typename T = typename Instance::Key_t>
using make_sequence =
   typename sequence<Instance, T, std::integral_constant<T,Instance::N> >::type;

/// Instantiator class
template<class Instance,
         typename Key_t = typename Instance::Key_t,
         typename Return_t = typename Instance::Return_t,
         typename Kernel_t = typename Instance::Kernel_t>
class Instantiator
{
private:
   using map_t = std::unordered_map<Key_t, Kernel_t>;
   map_t map;

public:
   Instantiator() { make_sequence<Instance>().Fill(map); }

   bool Find(const Key_t id)
   {
      return (map.find(id) != map.end()) ? true : false;
   }

   Kernel_t At(const Key_t id) { return map.at(id); }
};

// MFEM_REGISTER_TMOP_KERNELS macro:
// - forward declaration of the kernel
// - kernel pointer declaration
// - struct K##name##_T definition
// - Instantiator definition
// - re-use kernel return type and name before its body
#define MFEM_REGISTER_TMOP_KERNELS(return_t, kernel, ...) \
template<int T_D1D = 0, int T_Q1D = 0, int T_MAX = 0> \
    return_t kernel(__VA_ARGS__);\
typedef return_t (*kernel##_p)(__VA_ARGS__);\
struct K##kernel##_T {\
   static const int N = 14;\
   using Key_t = std::size_t;\
   using Kernel_t = kernel##_p;\
   using Return_t = return_t;\
   template<Key_t I> static constexpr Key_t GetKey() noexcept { return \
     I==0 ? 0x22 : I==1 ? 0x23 : I==2 ? 0x24 : I==3 ? 0x25 : I==4 ? 0x26 :\
     I==5 ? 0x33 : I==6 ? 0x34 : I==7 ? 0x35 : I==8 ? 0x36  :\
     I==9 ? 0x44 : I==10 ? 0x45 : I==11 ? 0x46 :\
     I==12 ? 0x55 : I==13 ? 0x56 : 0; }\
   template<Key_t ID> static constexpr Kernel_t GetValue() noexcept\
   { return &kernel<(ID>>4)&0xF, ID&0xF>; }\
};\
static kernels::Instantiator<K##kernel##_T> K##kernel;\
template<int T_D1D, int T_Q1D, int T_MAX> return_t kernel(__VA_ARGS__)

// MFEM_LAUNCH_TMOP_KERNEL macro
#define MFEM_LAUNCH_TMOP_KERNEL(kernel, id, ...)\
if (K##kernel.Find(id)) { return K##kernel.At(id)(__VA_ARGS__,0,0); }\
else {\
   constexpr int T_MAX = 4;\
   MFEM_VERIFY(D1D <= MAX_D1D && Q1D <= MAX_Q1D, "Max size error!");\
   return kernel<0,0,T_MAX>(__VA_ARGS__,D1D,Q1D); }

} // namespace kernels

} // namespace mfem

#endif // MFEM_TMOP_PA_HPP
