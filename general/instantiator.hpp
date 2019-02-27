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

#ifndef MFEM_INSTANTIATOR_HPP
#define MFEM_INSTANTIATOR_HPP

#include <unordered_map>
using std::unordered_map;
using std::integral_constant;

// *****************************************************************************
// * Generic emplace
// *****************************************************************************
template<typename K, const int N,
         typename Key_t = typename K::Key_t,
         typename Kernel_t = typename K::Kernel_t>
void emplace(std::unordered_map<Key_t, Kernel_t> &map)
{
   constexpr Key_t key = K::template GetKey<N>();
   constexpr Kernel_t value = K::template GetValue<key>();
   map.emplace(key, value);
}

// *****************************************************************************
// * instances
// *****************************************************************************
template<class K, typename T, T... idx>
struct instances
{
   static void Fill(unordered_map<typename K::Key_t, typename K::Kernel_t> &map)
   {
      using unused = int[];
      (void) unused {0, (emplace<K,idx>(map), 0)... };
   }
};

// *****************************************************************************
// * cat instances
// *****************************************************************************
template<class K, typename Offset, typename Lhs, typename Rhs> struct cat;
template<class K, typename T, T Offset, T... Lhs, T... Rhs>
struct cat<K, integral_constant<T, Offset>,
          instances<K, T, Lhs...>, instances<K, T, Rhs...> >
{
   using type = instances<K, T, Lhs..., (Offset + Rhs)...>;
};

// *****************************************************************************
// * sequence, empty and one element terminal cases
// *****************************************************************************
template<class K, typename T, typename N>
struct sequence
{
   using Lhs = integral_constant<T, N::value/2>;
   using Rhs = integral_constant<T, N::value-Lhs::value>;
   using type = typename cat<K, Lhs,
         typename sequence<K, T, Lhs>::type,
         typename sequence<K, T, Rhs>::type>::type;
};
template<class K, typename T>
struct sequence<K, T, integral_constant<T,0> >
{
   using type = instances<K,T>;
};

template<class K, typename T>
struct sequence<K, T, integral_constant<T,1> >
{
   using type = instances<K,T,0>;
};

// *****************************************************************************
// * make_sequence
// *****************************************************************************
template<class I, typename T = typename I::Key_t>
using make_sequence = typename sequence<I, T, integral_constant<T,I::N> >::type;

// *****************************************************************************
// *****************************************************************************
template<class Instance,
         typename Key_t = typename Instance::Key_t,
         typename Kernel_t = typename Instance::Kernel_t>
class Instantiator
{
private:
   using map_t = std::unordered_map<Key_t, Kernel_t>;
   map_t map;
public:
   Instantiator()
   {
      make_sequence<Instance>().Fill(map);
   }

   bool Find(const Key_t id)
   {
      return (map.find(id) != map.end()) ? true : false;
   }

   Kernel_t At(const Key_t id)
   {
      return map.at(id);
   }
};

#endif // MFEM_INSTANTIATOR_HPP
