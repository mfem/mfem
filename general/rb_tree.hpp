#ifndef RB_TREE_HPP
#define RB_TREE_HPP

#include <cstddef>
#include <utility>

namespace mfem
{

/// Red-black tree
template <class ChildType> struct RBTree
{
   size_t Successor(size_t idx);

   void EraseFixup(size_t &root, size_t x, size_t x_parent, bool y_is_left);
   void InsertFixup(size_t &root, size_t curr);
   void LeftRotate(size_t &root, size_t curr);
   void RightRotate(size_t &root, size_t curr);

   size_t Insert(size_t &root, size_t curr);

   size_t Insert(size_t &root, size_t pos, size_t curr);

   void Erase(size_t &root, size_t idx);

   size_t First(size_t root) const;

   /// visit the subtree starting at curr.
   template <class L, class R, class F>
   bool Visit(size_t curr, L &&visit_left, R &&visit_right, F &&func) const;
};
} // namespace mfem

#include "internal/rb_tree_impl.hpp"

#endif
