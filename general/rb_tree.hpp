// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_RB_TREE_HPP
#define MFEM_RB_TREE_HPP

#include <cstddef>
#include <utility>

namespace mfem
{

/// Red-black tree
template <class ChildType> struct RBTree
{
   // @return the node which has value immediately after node @a idx, or 0 if
   // idx has the largest value.
   size_t Next(size_t idx);
   /// internal use only
   void InsertFixup(size_t &root, size_t curr);
   void LeftRotate(size_t &root, size_t curr);
   void RightRotate(size_t &root, size_t curr);
   size_t InsertImpl(size_t &root, size_t pos, size_t curr,
                     bool check_hint = false);
   void EraseSimpleOne(size_t &root, size_t idx, int child);

   /// insert node @a curr into tree, modifying @a root as needed. Uses @a pos
   /// as a hint for a node in the tree to start the BSP search from.
   size_t Insert(size_t &root, size_t pos, size_t curr)
   {
      return InsertImpl(root, pos, curr, true);
   }

   /// insert node @a curr into tree, modifying @root as needed
   size_t Insert(size_t &root, size_t curr) { return InsertImpl(root, root, curr, false); }

   /// erases node @a idx into tree, modifying @a root as needed.
   void Erase(size_t &root, size_t idx);

   /// @return node with smallest value (left-most child), if any. Otherwise
   /// returns 0.
   size_t First(size_t root) const;

   /// Depth-first visit the subtree starting at curr. Visits in NLR order
   /// @a visit_left(node_idx) -> bool: true to visit left child, false
   /// otherwise)
   /// @a visit_right(node_idx) -> bool: true to visit right child,
   /// false otherwise)
   /// @a func(node_idx) -> bool: true to early terminate visit, false
   /// otherwise
   template <class L, class R, class F>
   bool Visit(size_t curr, L &&visit_left, R &&visit_right, F &&func) const;
};
} // namespace mfem

#include "internal/rb_tree_impl.hpp"

#endif
