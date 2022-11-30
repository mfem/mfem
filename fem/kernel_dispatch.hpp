// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_KERNELDISPATCH_HPP
#define MFEM_KERNELDISPATCH_HPP

#include <unordered_map>

namespace mfem
{

template <typename KEY, typename F, typename HASH>
class DispatchTable
{
protected:
   std::unordered_map<KEY, F, HASH> table;
};

struct KernelDispatchKey
{
   const int dim, d1d, q1d;
   bool operator==(const KernelDispatchKey &k) const
   {
      return (dim == k.dim) && (d1d == k.d1d) && (q1d == k.q1d);
   }
};

struct KernelDispatchKeyHash
{
   std::hash<int> h;
   std::size_t operator()(const KernelDispatchKey &k) const
   {
      return h(k.dim + 4*k.d1d + 4*32*k.q1d);
   }
};

template <typename T>
class KernelDispatchTable : public
   DispatchTable<KernelDispatchKey, typename T::Kernel, KernelDispatchKeyHash>
{
private:
   // If the type U has member U::NBZ, this overload will be selected, and will
   // return U::NBZ(d1d, q1d).
   template <typename U>
   static constexpr int GetNBZ_(int d1d, int q1d, decltype(U::NBZ(0,0),nullptr))
   {
      return U::NBZ(d1d, q1d);
   }
   // If the type U does not have member U::NBZ, this "fallback" overload will
   // be selected
   template <typename U> static constexpr int GetNBZ_(int d1d, int q1d, ...)
   {
      return 0;
   }
   // Return T::NBZ(d1d, q1d) if T::NBZ is defined, 0 otherwise.
   static constexpr int GetNBZ(int d1d, int q1d)
   { return GetNBZ_<T>(d1d, q1d, nullptr); }
public:
   template <typename... Args>
   void Run(int dim, int d1d, int q1d, Args&&... args)
   {
      const KernelDispatchKey key = {dim, d1d, q1d};
      const auto it = this->table.find(key);
      if (it != this->table.end())
      {
         printf("Specialized.\n");
         it->second(args...);
      }
      else
      {
         printf("Fallback.\n");
         if (key.dim == 2)
         {
            T::Fallback2D()(args...);
         }
         else if (key.dim == 3)
         {
            T::Fallback3D()(args...);
         }
         else
         {
            MFEM_ABORT("Kernel not found.");
         }
      }
   }

   template <int DIM, int D1D, int Q1D>
   void AddSpecialization()
   {
      printf("Adding specialization.\n");
      constexpr KernelDispatchKey key = {DIM, D1D, Q1D};
      if (DIM == 2)
      {
         constexpr int NBZ = GetNBZ(D1D, Q1D);
         printf("NBZ = %d\n", NBZ);
         this->table[key] = T::template Kernel2D<D1D, Q1D, NBZ>();
      }
      else if (DIM == 3)
      {
         this->table[key] = T::template Kernel3D<D1D, Q1D>();
      }
      else
      {
         MFEM_ABORT("Unsupported dimension.");
      }
   }
};

}

#endif
