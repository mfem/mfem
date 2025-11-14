#ifndef RB_TREE_IMPL_HPP
#define RB_TREE_IMPL_HPP

namespace mfem
{

template <class ChildType> size_t RBTree<ChildType>::successor(size_t idx)
{
   if (static_cast<ChildType &>(*this).get_node(idx).child[1])
   {
      // left-most child of right child
      idx = static_cast<ChildType &>(*this).get_node(idx).child[1];
      while (static_cast<ChildType &>(*this).get_node(idx).child[0])
      {
         idx = static_cast<ChildType &>(*this).get_node(idx).child[0];
      }
      return idx;
   }
   else
   {
      // need to search from parent
      size_t y = static_cast<ChildType &>(*this).get_node(idx).parent;
      while (y && idx == static_cast<ChildType &>(*this).get_node(y).child[1])
      {
         idx = y;
         y = static_cast<ChildType &>(*this).get_node(y).parent;
      }
      return y;
   }
}

template <class ChildType>
void RBTree<ChildType>::erase_fixup(size_t &root, size_t x, size_t x_parent,
                                    bool y_is_left)
{
   while (x != root && !static_cast<ChildType &>(*this).get_node(x).is_red())
   {
      size_t w;
      if (y_is_left)
      {
         w = static_cast<ChildType &>(*this).get_node(x).child[1];
         if (static_cast<ChildType &>(*this).get_node(w).is_red())
         {
            static_cast<ChildType &>(*this).get_node(w).set_black();
            static_cast<ChildType &>(*this).get_node(x_parent).set_red();
            left_rotate(root, x_parent);
            w = static_cast<ChildType &>(*this).get_node(x_parent).child[1];
         }
         if (!static_cast<ChildType &>(*this).get_node(w).is_red() &&
             !static_cast<ChildType &>(*this)
                 .get_node(static_cast<ChildType &>(*this).get_node(w).child[1])
                 .is_red())
         {
            static_cast<ChildType &>(*this).get_node(w).set_red();
            x = x_parent;
            x_parent = static_cast<ChildType &>(*this).get_node(x).parent;
            y_is_left =
               (x ==
                static_cast<ChildType &>(*this).get_node(x_parent).child[0]);
         }
         else
         {
            if (!static_cast<ChildType &>(*this)
                    .get_node(
                       static_cast<ChildType &>(*this).get_node(w).child[1])
                    .is_red())
            {
               static_cast<ChildType &>(*this)
                  .get_node(
                     static_cast<ChildType &>(*this).get_node(w).child[0])
                  .set_black();
               static_cast<ChildType &>(*this).get_node(w).set_red();
               right_rotate(root, w);
               w = static_cast<ChildType &>(*this).get_node(x_parent).child[1];
            }
            static_cast<ChildType &>(*this).get_node(w).copy_color(
               static_cast<ChildType &>(*this).get_node(x_parent));
            static_cast<ChildType &>(*this).get_node(x_parent).set_black();
            if (static_cast<ChildType &>(*this).get_node(w).child[1])
            {
               static_cast<ChildType &>(*this)
                  .get_node(
                     static_cast<ChildType &>(*this).get_node(w).child[1])
                  .set_black();
            }
            left_rotate(root, x_parent);
            x = root;
            x_parent = 0;
         }
      }
      else
      {
         w = static_cast<ChildType &>(*this).get_node(x_parent).child[0];
         if (static_cast<ChildType &>(*this).get_node(w).is_red())
         {
            static_cast<ChildType &>(*this).get_node(w).set_black();
            static_cast<ChildType &>(*this).get_node(x_parent).set_red();
            right_rotate(root, x_parent);
            w = static_cast<ChildType &>(*this).get_node(x_parent).child[0];
         }
         if (!static_cast<ChildType &>(*this)
                 .get_node(static_cast<ChildType &>(*this).get_node(w).child[1])
                 .is_red() &&
             !static_cast<ChildType &>(*this)
                 .get_node(static_cast<ChildType &>(*this).get_node(w).child[0])
                 .is_red())
         {
            static_cast<ChildType &>(*this).get_node(w).set_red();
            x = x_parent;
            x_parent = static_cast<ChildType &>(*this).get_node(x).parent;
            y_is_left =
               (x ==
                static_cast<ChildType &>(*this).get_node(x_parent).child[0]);
         }
         else
         {
            if (!static_cast<ChildType &>(*this)
                    .get_node(
                       static_cast<ChildType &>(*this).get_node(w).child[0])
                    .is_red())
            {
               static_cast<ChildType &>(*this)
                  .get_node(
                     static_cast<ChildType &>(*this).get_node(w).child[1])
                  .set_black();
               static_cast<ChildType &>(*this).get_node(w).set_red();
               left_rotate(root, w);
               w = static_cast<ChildType &>(*this).get_node(x_parent).child[0];
            }
            static_cast<ChildType &>(*this).get_node(w).copy_color(
               static_cast<ChildType &>(*this).get_node(x_parent));
            static_cast<ChildType &>(*this).get_node(x_parent).set_black();
            if (static_cast<ChildType &>(*this).get_node(w).child[0])
            {
               static_cast<ChildType &>(*this)
                  .get_node(
                     static_cast<ChildType &>(*this).get_node(w).child[0])
                  .set_black();
            }
            right_rotate(root, x_parent);
            x = root;
            x_parent = 0;
         }
      }
   }
   static_cast<ChildType &>(*this).get_node(x).set_black();
}

template <class ChildType>
void RBTree<ChildType>::insert_fixup(size_t &root, size_t curr)
{
   auto z = &static_cast<ChildType &>(*this).get_node(curr);
   while (z->parent)
   {
      auto p = &static_cast<ChildType &>(*this).get_node(z->parent);
      if (!p->is_red())
      {
         break;
      }
      if (!p->parent)
      {
         break;
      }
      auto gp = &static_cast<ChildType &>(*this).get_node(p->parent);
      if (z->parent == gp->child[0])
      {
         size_t y = gp->child[1];
         if (y && static_cast<ChildType &>(*this).get_node(y).is_red())
         {
            p->set_black();
            static_cast<ChildType &>(*this).get_node(y).set_black();
            gp->set_red();
            curr = p->parent;
            z = gp;
         }
         else
         {
            if (curr == p->child[1])
            {
               curr = z->parent;
               z = p;
               left_rotate(root, curr);
            }
            p = &static_cast<ChildType &>(*this).get_node(z->parent);
            gp = &static_cast<ChildType &>(*this).get_node(p->parent);
            p->set_black();
            gp->set_red();
            right_rotate(root, p->parent);
         }
      }
      else
      {
         size_t y = gp->child[0];
         if (y && static_cast<ChildType &>(*this).get_node(y).is_red())
         {
            p->set_black();
            static_cast<ChildType &>(*this).get_node(y).set_black();
            gp->set_red();
            curr = p->parent;
            z = gp;
         }
         else
         {
            if (p->child[0] == curr)
            {
               curr = z->parent;
               z = p;
               right_rotate(root, curr);
            }
            p = &static_cast<ChildType &>(*this).get_node(z->parent);
            gp = &static_cast<ChildType &>(*this).get_node(p->parent);
            p->set_black();
            gp->set_red();
            left_rotate(root, p->parent);
         }
      }
   }
   static_cast<ChildType &>(*this).get_node(root).set_black();
}

template <class ChildType>
void RBTree<ChildType>::left_rotate(size_t &root, size_t curr)
{
   auto x = &static_cast<ChildType &>(*this).get_node(curr);
   size_t y = x->child[1];
   auto yn = &static_cast<ChildType &>(*this).get_node(y);
   x->child[1] = yn->child[0];
   if (yn->child[0])
   {
      static_cast<ChildType &>(*this).get_node(yn->child[0]).parent = curr;
   }
   yn->parent = x->parent;
   if (!x->parent)
   {
      root = y;
   }
   else if (static_cast<ChildType &>(*this).get_node(x->parent).child[0] ==
            curr)
   {
      static_cast<ChildType &>(*this).get_node(x->parent).child[0] = y;
   }
   else
   {
      static_cast<ChildType &>(*this).get_node(x->parent).child[1] = y;
   }
   yn->child[0] = curr;
   x->parent = y;
   static_cast<ChildType &>(*this).post_left_rotate(curr);
}

template <class ChildType>
void RBTree<ChildType>::right_rotate(size_t &root, size_t curr)
{
   auto y = &static_cast<ChildType &>(*this).get_node(curr);
   size_t x = y->child[0];
   auto xn = &static_cast<ChildType &>(*this).get_node(x);
   y->child[0] = xn->child[1];
   if (xn->child[1])
   {
      static_cast<ChildType &>(*this).get_node(xn->child[1]).parent = curr;
   }
   xn->parent = y->parent;
   if (!y->parent)
   {
      root = x;
   }
   else if (static_cast<ChildType &>(*this).get_node(y->parent).child[0] ==
            curr)
   {
      static_cast<ChildType &>(*this).get_node(y->parent).child[0] = x;
   }
   else
   {
      static_cast<ChildType &>(*this).get_node(y->parent).child[1] = x;
   }
   xn->child[1] = curr;
   y->parent = x;
   static_cast<ChildType &>(*this).post_right_rotate(curr);
}

template <class ChildType>
size_t RBTree<ChildType>::insert(size_t &root, size_t pos, size_t curr)
{
   size_t y = 0;
   size_t x = pos;
   auto &z = static_cast<ChildType &>(*this).get_node(curr);
   while (x)
   {
      y = x;
      auto &nx = static_cast<ChildType &>(*this).get_node(x);
      auto cmp = static_cast<ChildType &>(*this).compare_nodes(x, curr);
      if (cmp < 0)
      {
         x = nx.child[0];
      }
      else if (cmp > 0)
      {
         x = nx.child[1];
      }
      else
      {
         // duplicate range, ignore
         static_cast<ChildType &>(*this).insert_duplicate(x);
         return x;
      }
   }
   z.parent = y;
   if (!y)
   {
      root = curr;
   }
   else if (static_cast<ChildType &>(*this).compare_nodes(y, curr) < 0)
   {
      static_cast<ChildType &>(*this).get_node(y).child[0] = curr;
   }
   else
   {
      static_cast<ChildType &>(*this).get_node(y).child[1] = curr;
   }
   z.set_red();
   insert_fixup(root, curr);

   return curr;
}

template <class ChildType>
size_t RBTree<ChildType>::insert(size_t &root, size_t curr)
{
   size_t y = 0;
   size_t x = root;
   auto &z = static_cast<ChildType &>(*this).get_node(curr);
   while (x)
   {
      y = x;
      auto &nx = static_cast<ChildType &>(*this).get_node(x);
      auto cmp = static_cast<ChildType &>(*this).compare_nodes(x, curr);
      if (cmp < 0)
      {
         x = nx.child[0];
      }
      else if (cmp > 0)
      {
         x = nx.child[1];
      }
      else
      {
         // duplicate range, ignore
         static_cast<ChildType &>(*this).insert_duplicate(x);
         return x;
      }
   }
   z.parent = y;
   if (!y)
   {
      root = curr;
   }
   else if (static_cast<ChildType &>(*this).compare_nodes(y, curr) < 0)
   {
      static_cast<ChildType &>(*this).get_node(y).child[0] = curr;
   }
   else
   {
      static_cast<ChildType &>(*this).get_node(y).child[1] = curr;
   }
   z.set_red();
   insert_fixup(root, curr);

   return curr;
}

template <class ChildType>
void RBTree<ChildType>::erase(size_t &root, size_t idx)
{
   if (!idx)
   {
      return;
   }

   auto curr = &static_cast<ChildType &>(*this).get_node(idx);

   size_t y;
   if (!curr->child[0])
   {
      y = idx;
   }
   else if (!curr->child[1])
   {
      y = idx;
   }
   else
   {
      y = successor(idx);
   }
   {
      if (y != idx)
      {
         // actually swap the nodes so all indices are valid except for idx
         // after erase
         auto &n = static_cast<ChildType &>(*this).get_node(y);
         // fixup connections
         if (curr->parent == y)
         {
            curr->parent = n.parent;
            if (n.parent)
            {
               if (static_cast<ChildType &>(*this)
                      .get_node(n.parent)
                      .child[0] == y)
               {
                  static_cast<ChildType &>(*this).get_node(n.parent).child[0] =
                     idx;
               }
               else
               {
                  static_cast<ChildType &>(*this).get_node(n.parent).child[1] =
                     idx;
               }
            }
            n.parent = idx;
            if (n.child[0] == idx)
            {
               n.child[0] = curr->child[0];
               curr->child[0] = y;
               std::swap(n.child[1], curr->child[1]);
               if (n.child[0])
               {
                  static_cast<ChildType &>(*this).get_node(n.child[0]).parent =
                     y;
               }
               if (n.child[1])
               {
                  static_cast<ChildType &>(*this).get_node(n.child[1]).parent =
                     y;
               }
               if (curr->child[1])
               {
                  static_cast<ChildType &>(*this)
                     .get_node(curr->child[1])
                     .parent = idx;
               }
            }
            else
            {
               n.child[1] = curr->child[1];
               curr->child[1] = y;
               std::swap(n.child[0], curr->child[0]);
               if (n.child[1])
               {
                  static_cast<ChildType &>(*this).get_node(n.child[1]).parent =
                     y;
               }
               if (n.child[0])
               {
                  static_cast<ChildType &>(*this).get_node(n.child[0]).parent =
                     y;
               }
               if (curr->child[0])
               {
                  static_cast<ChildType &>(*this)
                     .get_node(curr->child[0])
                     .parent = idx;
               }
            }
         }
         else if (n.parent == idx)
         {
            n.parent = curr->parent;
            if (curr->parent)
            {
               if (static_cast<ChildType &>(*this)
                      .get_node(curr->parent)
                      .child[0] == idx)
               {
                  static_cast<ChildType &>(*this)
                     .get_node(curr->parent)
                     .child[0] = y;
               }
               else
               {
                  static_cast<ChildType &>(*this)
                     .get_node(curr->parent)
                     .child[1] = y;
               }
            }
            curr->parent = y;
            if (curr->child[0] == y)
            {
               curr->child[0] = n.child[0];
               n.child[0] = idx;
               std::swap(n.child[1], curr->child[1]);
               if (curr->child[0])
               {
                  static_cast<ChildType &>(*this)
                     .get_node(curr->child[0])
                     .parent = idx;
               }
               if (curr->child[1])
               {
                  static_cast<ChildType &>(*this)
                     .get_node(curr->child[1])
                     .parent = idx;
               }
               if (n.child[1])
               {
                  static_cast<ChildType &>(*this).get_node(n.child[1]).parent =
                     y;
               }
            }
            else
            {
               curr->child[1] = n.child[1];
               n.child[1] = idx;
               std::swap(n.child[0], curr->child[0]);
               if (curr->child[1])
               {
                  static_cast<ChildType &>(*this)
                     .get_node(curr->child[1])
                     .parent = idx;
               }
               if (curr->child[0])
               {
                  static_cast<ChildType &>(*this)
                     .get_node(curr->child[0])
                     .parent = idx;
               }
               if (n.child[0])
               {
                  static_cast<ChildType &>(*this).get_node(n.child[0]).parent =
                     y;
               }
            }
         }
         else
         {
            // no direct connection between y and idx
            if (n.parent)
            {
               auto &pn = static_cast<ChildType &>(*this).get_node(n.parent);
               if (pn.child[0] == y)
               {
                  pn.child[0] = idx;
               }
               else
               {
                  pn.child[1] = idx;
               }
            }
            if (curr->parent)
            {
               auto &pn =
                  static_cast<ChildType &>(*this).get_node(curr->parent);
               if (pn.child[0] == idx)
               {
                  pn.child[0] = y;
               }
               else
               {
                  pn.child[1] = y;
               }
            }
            if (n.child[0])
            {
               static_cast<ChildType &>(*this).get_node(n.child[0]).parent =
                  idx;
            }
            if (n.child[1])
            {
               static_cast<ChildType &>(*this).get_node(n.child[1]).parent =
                  idx;
            }
            if (curr->child[0])
            {
               static_cast<ChildType &>(*this).get_node(curr->child[0]).parent =
                  y;
            }
            if (curr->child[1])
            {
               static_cast<ChildType &>(*this).get_node(curr->child[1]).parent =
                  y;
            }
            std::swap(n.parent, curr->parent);
            std::swap(n.child[0], curr->child[0]);
            std::swap(n.child[1], curr->child[1]);
         }

         if (root == idx)
         {
            root = y;
         }
         else if (root == y)
         {
            root = idx;
         }
         std::swap(idx, y);
         bool tflag = n.is_red();
         if (curr->is_red())
         {
            n.set_red();
         }
         else
         {
            n.set_black();
         }
         if (tflag)
         {
            curr->set_red();
         }
         else
         {
            curr->set_black();
         }
         curr = &n;
      }
      size_t x = static_cast<ChildType &>(*this).get_node(y).child[0];
      if (!x)
      {
         x = static_cast<ChildType &>(*this).get_node(y).child[1];
      }
      size_t x_parent = static_cast<ChildType &>(*this).get_node(y).parent;
      if (x)
      {
         static_cast<ChildType &>(*this).get_node(x).parent = x_parent;
      }
      if (!x_parent)
      {
         root = x;
      }
      else if (static_cast<ChildType &>(*this).get_node(x_parent).child[0] == y)
      {
         static_cast<ChildType &>(*this).get_node(x_parent).child[0] = x;
      }
      else
      {
         static_cast<ChildType &>(*this).get_node(x_parent).child[1] = x;
      }

      if (y != idx)
      {
         static_cast<ChildType &>(*this).erase_swap_hook(idx);
      }
      if (x)
      {
         auto &xn = static_cast<ChildType &>(*this).get_node(x);
         if (xn.is_red())
         {
            if (x_parent)
            {
               auto &yn = static_cast<ChildType &>(*this).get_node(y);
               auto &yp = static_cast<ChildType &>(*this).get_node(yn.parent);
               erase_fixup(root, x, x_parent, yp.child[0] == y);
            }
            else
            {
               xn.set_black();
            }
         }
      }
   }
}

template <class ChildType>
template <class L, class R, class F>
bool RBTree<ChildType>::visit(size_t curr, L &&visit_left, R &&visit_right,
                              F &&func) const
{
   if (curr)
   {
      auto &c = static_cast<const ChildType &>(*this).get_node(curr);
      if (func(curr))
      {
         return true;
      }
      if (visit_left(curr))
      {
         if (visit(c.child[0], std::forward<L>(visit_left),
                   std::forward<R>(visit_right), std::forward<F>(func)))
         {
            return true;
         }
      }
      if (visit_right(curr))
      {
         return visit(c.child[1], std::forward<L>(visit_left),
                      std::forward<R>(visit_right), std::forward<F>(func));
      }
   }
   return false;
}

template <class ChildType> size_t RBTree<ChildType>::first(size_t root) const
{
   if (root)
   {
      auto n = &static_cast<const ChildType &>(*this).get_node(root);
      while (n->child[0])
      {
         root = n->child[0];
         n = &static_cast<const ChildType &>(*this).get_node(root);
      }
      return root;
   }
   return 0;
}
} // namespace mfem

#endif
