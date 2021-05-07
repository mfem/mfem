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

#include "../../config/config.hpp"
#include "../../linalg/dtensor.hpp"

#include "../kernels.hpp"

#include <unordered_map>

namespace mfem
{

namespace kernels
{

template <typename K> class KernelMap;

/// Instances
template<class K, int N = K::N> struct Instances
{
   static void Fill(KernelMap<K> &map)
   {
      map.template Emplace<N-1>();
      Instances<K,N-1>::Fill(map);
   }
};

// terminal case
template<class K> struct Instances<K,1>
{
   static void Fill(KernelMap<K> &map) { map.template Emplace<0>(); }
};

/// KernelMap class which creates an unordered_map of the Keys/Kernels
template<class K>
class KernelMap
{
   using Key_t = typename K::Key_t;
   using Return_t = typename K::Return_t;
   using Kernel_t = typename K::Kernel_t;
   using map_t = std::unordered_map<Key_t, Kernel_t>;
   map_t map;

public:
   // Fill all the map with the Keys/Kernels
   KernelMap() { Instances<K>::Fill(*this); }

   bool Find(const Key_t id) { return map.find(id) != map.end(); }

   Kernel_t At(const Key_t id) { return map.at(id); }

   template<int N> void Emplace()
   {
      constexpr Key_t key = K::template GetKey<N>();
      constexpr Kernel_t ker = K::template GetKer<key>();
      map.emplace(key, ker);
   }
};

// /////////////////////////////////////////////////////////////////////////////
// MFEM_REGISTER_TMOP_KERNELS macro:
//  - the first argument (return_t) is the return type of the kernel
//  - the second argument (kernel) is the name of the kernel
//  - the arguments of the kernel (...) captured as __VA_ARGS__
//
// This call will output the followings:
//  1. forward declaration of the kernel
//  2. kernel pointer declaration
//  3. struct K##name##_T definition which holds the keys/kernels map
//  4. KernelMap definition of the current kernel
//  5. the kernel signature by re-using all the arguments
//
// /////////////////////////////////////////////////////////////////////////////
// For example:
// MFEM_REGISTER_TMOP_KERNELS(void, Name,
//                            const int NE,
//                            const Array<double> &b,
//                            Vector &diagonal,
//                            const int d1d,
//                            const int q1d) {...}
//
// The resulting code would be:
//
// 1. forward declaration of the kernel
// template<int T_D1D = 0, int T_Q1D = 0, int T_MAX = 0>
// void Name(const int NE,
//           const Array<double> &b,
//           Vector &diagonal,
//           const int d1d,
//           const int q1d);
//
// 2. kernel pointer declaration
// typedef void (*Name_p)(const int NE,
//                        const Array<double> &b,
//                        Vector &diagonal,
//                        const int d1d,
//                        const int q1d);
//
// 3. struct K##Name##_T definition which holds the keys/kernels instance
// struct KName_T
// {
//    static const int N = 14;
//    using Key_t = int;
//    using Return_t = void;
//    using Kernel_t = Name_p;
//    template<Key_t I> static constexpr Key_t GetKey() noexcept
//    {
//       return I==0 ? 0x22 : I==1 ? 0x23 : I==2 ? 0x24 : I==3 ? 0x25 :
//              I==4 ? 0x26 : I== 5 ? 0x33 : I==6 ? 0x34 : I==7 ? 0x35 :
//              I==8 ? 0x36 : I==9 ? 0x44 : I==10 ? 0x45 : I==11 ? 0x46 :
//              I==12 ? 0x55 : I==13 ? 0x56 : 0;
//    }
//    template<Key_t K> static constexpr Kernel_t GetKer() noexcept
//    {
//       return &AssembleDiagonalPA_Kernel_2D<(K>>4)&0xF, K&0xF>;
//    }
// };
//
// 4. KernelMap definition of the current kernel
// static kernels::KernelMap<KName_T> KName;
//
// 5. the kernel signature by re-using all the arguments
// template<int T_D1D, int T_Q1D, int T_MAX>
// void Name(const int NE,
//           const Array<double> &b,
//           Vector &diagonal,
//           const int d1d,
//           const int q1d) {...}

// /////////////////////////////////////////////////////////////////////////////
// All of which allows to launch the kernel with a specific id ((D1D<<4)|Q1D).
//
// For example, a MFEM_LAUNCH_TMOP_KERNEL(Name,id,NE,B,D); call would result in:
//
// if (KName.Find(id)) { return KName.At(id)(NE,B,D,0,0); }
// else
// {
//    constexpr int T_MAX = 4;
//    const int D1D = (id>>4)&0xF, Q1D = id&0xF;
//    MFEM_VERIFY(D1D <= MAX_D1D && Q1D <= MAX_Q1D, "Max size error!");
//    return Name<0,0,T_MAX>(NE,B,D,D1D,Q1D);
// };

#define MFEM_REGISTER_TMOP_KERNELS(return_t, kernel, ...) \
template<int T_D1D = 0, int T_Q1D = 0, int T_MAX = 0> \
    return_t kernel(__VA_ARGS__);\
typedef return_t (*kernel##_p)(__VA_ARGS__);\
struct K##kernel##_T {\
   static const int N = 14;\
   using Key_t = int;\
   using Return_t = return_t;\
   using Kernel_t = kernel##_p;\
   template<Key_t I> static constexpr Key_t GetKey() noexcept { return \
     I==0 ? 0x22 : I==1 ? 0x23 : I==2 ? 0x24 : I==3 ? 0x25 : I==4 ? 0x26 :\
     I==5 ? 0x33 : I==6 ? 0x34 : I==7 ? 0x35 : I==8 ? 0x36  :\
     I==9 ? 0x44 : I==10 ? 0x45 : I==11 ? 0x46 :\
     I==12 ? 0x55 : I==13 ? 0x56 : 0; }\
   template<Key_t K> static constexpr Kernel_t GetKer() noexcept\
   { return &kernel<(K>>4)&0xF, K&0xF>; }\
};\
static kernels::KernelMap<K##kernel##_T> K##kernel;\
template<int T_D1D, int T_Q1D, int T_MAX> return_t kernel(__VA_ARGS__)

// MFEM_LAUNCH_TMOP_KERNEL macro
// This macro will try to find and launch the kernel with the id key and
// the templated arguments.
// If not, it will fall back to the kernel with the standard arguments.
#define MFEM_LAUNCH_TMOP_KERNEL(kernel, id, ...)\
if (K##kernel.Find(id)) { return K##kernel.At(id)(__VA_ARGS__,0,0); }\
else {\
   constexpr int T_MAX = 4;\
   const int D1D = (id>>4)&0xF, Q1D = id&0xF;\
   MFEM_VERIFY(D1D <= MAX_D1D && Q1D <= MAX_Q1D, "Max size error!");\
   return kernel<0,0,T_MAX>(__VA_ARGS__,D1D,Q1D); }

} // namespace kernels

} // namespace mfem

#endif // MFEM_TMOP_PA_HPP
