#ifndef RB_TREE_HPP
#define RB_TREE_HPP

#include <cstddef>
#include <utility>

namespace mfem
{

/// Red-black tree
template <class ChildType> struct RBTree
{
   size_t successor(size_t idx);

   void erase_fixup(size_t &root, size_t x, size_t x_parent, bool y_is_left);
   void insert_fixup(size_t &root, size_t curr);
   void left_rotate(size_t &root, size_t curr);
   void right_rotate(size_t &root, size_t curr);

   size_t insert(size_t &root, size_t curr);

   size_t insert(size_t &root, size_t pos, size_t curr);

   void erase(size_t &root, size_t idx);

   size_t first(size_t root) const;

   /// visit the subtree starting at curr.
   template <class L, class R, class F>
   bool visit(size_t curr, L &&visit_left, R &&visit_right, F &&func) const;
};
} // namespace mfem

#include "internal/rb_tree_impl.hpp"

#endif
