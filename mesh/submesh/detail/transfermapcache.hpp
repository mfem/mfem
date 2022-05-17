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

#ifndef MFEM_TRANSFERMAPCACHE
#define MFEM_TRANSFERMAPCACHE

#include <string>
#include <sstream>
#include <unordered_map>

namespace mfem
{
namespace detail
{

inline std::string ptrs_to_concat_string(const void *ptr1, const void *ptr2)
{
   std::stringstream ss;
   ss << ptr1;
   ss << ptr2;
   return ss.str();
}

template<class gridfunc_type, class transfermap_type>
class TransferMapCache
{
public:
   TransferMapCache() {}

   const transfermap_type *Find(const gridfunc_type &src,
                                const gridfunc_type &dst)
   {
      const auto search = forward_.find(ptrs_to_concat_string(&src, &dst));
      if (search != forward_.end())
      {
         return search->second;
      }
      else
      {
         return Insert(src, dst);
      }
   }

   void Clear() { forward_.clear(); }

   ~TransferMapCache() { Clear(); }

private:
   const transfermap_type *Insert(const gridfunc_type &src,
                                  const gridfunc_type &dst)
   {
      const auto *map = new transfermap_type(src, dst);
      forward_.insert({ptrs_to_concat_string(&src, &dst), map});
      return map;
   }

   std::unordered_map<std::string, const transfermap_type *> forward_;
};

} // namespace detail
} // namespace mfem

#endif // MFEM_TRANSFERMAPCACHE