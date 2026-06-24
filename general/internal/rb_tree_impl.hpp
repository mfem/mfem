#ifndef RB_TREE_IMPL_HPP
#define RB_TREE_IMPL_HPP

namespace mfem
{

template <class ChildType> size_t RBTree<ChildType>::Successor(size_t idx)
{
   if (static_cast<ChildType &>(*this).GetNode(idx).child[1])
   {
      // left-most child of right child
      idx = static_cast<ChildType &>(*this).GetNode(idx).child[1];
      while (static_cast<ChildType &>(*this).GetNode(idx).child[0])
      {
         idx = static_cast<ChildType &>(*this).GetNode(idx).child[0];
      }
      return idx;
   }
   else
   {
      // need to search from parent
      size_t y = static_cast<ChildType &>(*this).GetNode(idx).parent;
      while (y && idx == static_cast<ChildType &>(*this).GetNode(y).child[1])
      {
         idx = y;
         y = static_cast<ChildType &>(*this).GetNode(y).parent;
      }
      return y;
   }
}

template <class ChildType>
void RBTree<ChildType>::EraseFixup(size_t &root, size_t x, size_t x_parent,
                                   bool y_is_left)
{
   while (x != root && !static_cast<ChildType &>(*this).GetNode(x).IsRed())
   {
      size_t w;
      if (y_is_left)
      {
         w = static_cast<ChildType &>(*this).GetNode(x).child[1];
         if (static_cast<ChildType &>(*this).GetNode(w).IsRed())
         {
            static_cast<ChildType &>(*this).GetNode(w).SetBlack();
            static_cast<ChildType &>(*this).GetNode(x_parent).SetRed();
            LeftRotate(root, x_parent);
            w = static_cast<ChildType &>(*this).GetNode(x_parent).child[1];
         }
         if (!static_cast<ChildType &>(*this).GetNode(w).IsRed() &&
             !static_cast<ChildType &>(*this)
             .GetNode(static_cast<ChildType &>(*this).GetNode(w).child[1])
             .IsRed())
         {
            static_cast<ChildType &>(*this).GetNode(w).SetRed();
            x = x_parent;
            x_parent = static_cast<ChildType &>(*this).GetNode(x).parent;
            y_is_left =
               (x ==
                static_cast<ChildType &>(*this).GetNode(x_parent).child[0]);
         }
         else
         {
            if (!static_cast<ChildType &>(*this)
                .GetNode(
                   static_cast<ChildType &>(*this).GetNode(w).child[1])
                .IsRed())
            {
               static_cast<ChildType &>(*this)
               .GetNode(
                  static_cast<ChildType &>(*this).GetNode(w).child[0])
               .SetBlack();
               static_cast<ChildType &>(*this).GetNode(w).SetRed();
               RightRotate(root, w);
               w = static_cast<ChildType &>(*this).GetNode(x_parent).child[1];
            }
            static_cast<ChildType &>(*this).GetNode(w).CopyColor(
               static_cast<ChildType &>(*this).GetNode(x_parent));
            static_cast<ChildType &>(*this).GetNode(x_parent).SetBlack();
            if (static_cast<ChildType &>(*this).GetNode(w).child[1])
            {
               static_cast<ChildType &>(*this)
               .GetNode(
                  static_cast<ChildType &>(*this).GetNode(w).child[1])
               .SetBlack();
            }
            LeftRotate(root, x_parent);
            x = root;
            x_parent = 0;
         }
      }
      else
      {
         w = static_cast<ChildType &>(*this).GetNode(x_parent).child[0];
         if (static_cast<ChildType &>(*this).GetNode(w).IsRed())
         {
            static_cast<ChildType &>(*this).GetNode(w).SetBlack();
            static_cast<ChildType &>(*this).GetNode(x_parent).SetRed();
            RightRotate(root, x_parent);
            w = static_cast<ChildType &>(*this).GetNode(x_parent).child[0];
         }
         if (!static_cast<ChildType &>(*this)
             .GetNode(static_cast<ChildType &>(*this).GetNode(w).child[1])
             .IsRed() &&
             !static_cast<ChildType &>(*this)
             .GetNode(static_cast<ChildType &>(*this).GetNode(w).child[0])
             .IsRed())
         {
            static_cast<ChildType &>(*this).GetNode(w).SetRed();
            x = x_parent;
            x_parent = static_cast<ChildType &>(*this).GetNode(x).parent;
            y_is_left =
               (x ==
                static_cast<ChildType &>(*this).GetNode(x_parent).child[0]);
         }
         else
         {
            if (!static_cast<ChildType &>(*this)
                .GetNode(
                   static_cast<ChildType &>(*this).GetNode(w).child[0])
                .IsRed())
            {
               static_cast<ChildType &>(*this)
               .GetNode(
                  static_cast<ChildType &>(*this).GetNode(w).child[1])
               .SetBlack();
               static_cast<ChildType &>(*this).GetNode(w).SetRed();
               LeftRotate(root, w);
               w = static_cast<ChildType &>(*this).GetNode(x_parent).child[0];
            }
            static_cast<ChildType &>(*this).GetNode(w).CopyColor(
               static_cast<ChildType &>(*this).GetNode(x_parent));
            static_cast<ChildType &>(*this).GetNode(x_parent).SetBlack();
            if (static_cast<ChildType &>(*this).GetNode(w).child[0])
            {
               static_cast<ChildType &>(*this)
               .GetNode(
                  static_cast<ChildType &>(*this).GetNode(w).child[0])
               .SetBlack();
            }
            RightRotate(root, x_parent);
            x = root;
            x_parent = 0;
         }
      }
   }
   static_cast<ChildType &>(*this).GetNode(x).SetBlack();
}

template <class ChildType>
void RBTree<ChildType>::InsertFixup(size_t &root, size_t curr)
{
   auto z = &static_cast<ChildType &>(*this).GetNode(curr);
   while (z->parent)
   {
      auto p = &static_cast<ChildType &>(*this).GetNode(z->parent);
      if (!p->IsRed())
      {
         break;
      }
      if (!p->parent)
      {
         break;
      }
      auto gp = &static_cast<ChildType &>(*this).GetNode(p->parent);
      if (z->parent == gp->child[0])
      {
         size_t y = gp->child[1];
         if (y && static_cast<ChildType &>(*this).GetNode(y).IsRed())
         {
            p->SetBlack();
            static_cast<ChildType &>(*this).GetNode(y).SetBlack();
            gp->SetRed();
            curr = p->parent;
            z = gp;
         }
         else
         {
            if (curr == p->child[1])
            {
               curr = z->parent;
               z = p;
               LeftRotate(root, curr);
            }
            p = &static_cast<ChildType &>(*this).GetNode(z->parent);
            gp = &static_cast<ChildType &>(*this).GetNode(p->parent);
            p->SetBlack();
            gp->SetRed();
            RightRotate(root, p->parent);
         }
      }
      else
      {
         size_t y = gp->child[0];
         if (y && static_cast<ChildType &>(*this).GetNode(y).IsRed())
         {
            p->SetBlack();
            static_cast<ChildType &>(*this).GetNode(y).SetBlack();
            gp->SetRed();
            curr = p->parent;
            z = gp;
         }
         else
         {
            if (p->child[0] == curr)
            {
               curr = z->parent;
               z = p;
               RightRotate(root, curr);
            }
            p = &static_cast<ChildType &>(*this).GetNode(z->parent);
            gp = &static_cast<ChildType &>(*this).GetNode(p->parent);
            p->SetBlack();
            gp->SetRed();
            LeftRotate(root, p->parent);
         }
      }
   }
   static_cast<ChildType &>(*this).GetNode(root).SetBlack();
}

template <class ChildType>
void RBTree<ChildType>::LeftRotate(size_t &root, size_t curr)
{
   auto x = &static_cast<ChildType &>(*this).GetNode(curr);
   size_t y = x->child[1];
   auto yn = &static_cast<ChildType &>(*this).GetNode(y);
   x->child[1] = yn->child[0];
   if (yn->child[0])
   {
      static_cast<ChildType &>(*this).GetNode(yn->child[0]).parent = curr;
   }
   yn->parent = x->parent;
   if (!x->parent)
   {
      root = y;
   }
   else if (static_cast<ChildType &>(*this).GetNode(x->parent).child[0] ==
            curr)
   {
      static_cast<ChildType &>(*this).GetNode(x->parent).child[0] = y;
   }
   else
   {
      static_cast<ChildType &>(*this).GetNode(x->parent).child[1] = y;
   }
   yn->child[0] = curr;
   x->parent = y;
   static_cast<ChildType &>(*this).PostLeftRotate(curr);
}

template <class ChildType>
void RBTree<ChildType>::RightRotate(size_t &root, size_t curr)
{
   auto y = &static_cast<ChildType &>(*this).GetNode(curr);
   size_t x = y->child[0];
   auto xn = &static_cast<ChildType &>(*this).GetNode(x);
   y->child[0] = xn->child[1];
   if (xn->child[1])
   {
      static_cast<ChildType &>(*this).GetNode(xn->child[1]).parent = curr;
   }
   xn->parent = y->parent;
   if (!y->parent)
   {
      root = x;
   }
   else if (static_cast<ChildType &>(*this).GetNode(y->parent).child[0] ==
            curr)
   {
      static_cast<ChildType &>(*this).GetNode(y->parent).child[0] = x;
   }
   else
   {
      static_cast<ChildType &>(*this).GetNode(y->parent).child[1] = x;
   }
   xn->child[1] = curr;
   y->parent = x;
   static_cast<ChildType &>(*this).PostRightRotate(curr);
}

template <class ChildType>
size_t RBTree<ChildType>::Insert(size_t &root, size_t pos, size_t curr)
{
   size_t y = 0;
   size_t x = pos;
   auto &z = static_cast<ChildType &>(*this).GetNode(curr);
   while (x)
   {
      y = x;
      auto &nx = static_cast<ChildType &>(*this).GetNode(x);
      auto cmp = static_cast<ChildType &>(*this).CompareNodes(x, curr);
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
         static_cast<ChildType &>(*this).InsertDuplicate(x, curr);
         return x;
      }
   }
   z.parent = y;
   if (!y)
   {
      root = curr;
   }
   else if (static_cast<ChildType &>(*this).CompareNodes(y, curr) < 0)
   {
      static_cast<ChildType &>(*this).GetNode(y).child[0] = curr;
   }
   else
   {
      static_cast<ChildType &>(*this).GetNode(y).child[1] = curr;
   }
   z.SetRed();
   InsertFixup(root, curr);

   return curr;
}

template <class ChildType>
size_t RBTree<ChildType>::Insert(size_t &root, size_t curr)
{
   size_t y = 0;
   size_t x = root;
   auto &z = static_cast<ChildType &>(*this).GetNode(curr);
   while (x)
   {
      y = x;
      auto &nx = static_cast<ChildType &>(*this).GetNode(x);
      auto cmp = static_cast<ChildType &>(*this).CompareNodes(x, curr);
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
         static_cast<ChildType &>(*this).InsertDuplicate(x, curr);
         return x;
      }
   }
   z.parent = y;
   if (!y)
   {
      root = curr;
   }
   else if (static_cast<ChildType &>(*this).CompareNodes(y, curr) < 0)
   {
      static_cast<ChildType &>(*this).GetNode(y).child[0] = curr;
   }
   else
   {
      static_cast<ChildType &>(*this).GetNode(y).child[1] = curr;
   }
   z.SetRed();
   InsertFixup(root, curr);

   return curr;
}

template <class ChildType>
void RBTree<ChildType>::Erase(size_t &root, size_t idx)
{
   if (!idx)
   {
      return;
   }

   auto curr = &static_cast<ChildType &>(*this).GetNode(idx);

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
      y = Successor(idx);
   }
   {
      if (y != idx)
      {
         // actually swap the nodes so all indices are valid except for idx
         // after erase
         auto &n = static_cast<ChildType &>(*this).GetNode(y);
         // fixup connections
         if (curr->parent == y)
         {
            curr->parent = n.parent;
            if (n.parent)
            {
               if (static_cast<ChildType &>(*this)
                   .GetNode(n.parent)
                   .child[0] == y)
               {
                  static_cast<ChildType &>(*this).GetNode(n.parent).child[0] =
                     idx;
               }
               else
               {
                  static_cast<ChildType &>(*this).GetNode(n.parent).child[1] =
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
                  static_cast<ChildType &>(*this).GetNode(n.child[0]).parent =
                     y;
               }
               if (n.child[1])
               {
                  static_cast<ChildType &>(*this).GetNode(n.child[1]).parent =
                     y;
               }
               if (curr->child[1])
               {
                  static_cast<ChildType &>(*this)
                  .GetNode(curr->child[1])
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
                  static_cast<ChildType &>(*this).GetNode(n.child[1]).parent =
                     y;
               }
               if (n.child[0])
               {
                  static_cast<ChildType &>(*this).GetNode(n.child[0]).parent =
                     y;
               }
               if (curr->child[0])
               {
                  static_cast<ChildType &>(*this)
                  .GetNode(curr->child[0])
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
                   .GetNode(curr->parent)
                   .child[0] == idx)
               {
                  static_cast<ChildType &>(*this)
                  .GetNode(curr->parent)
                  .child[0] = y;
               }
               else
               {
                  static_cast<ChildType &>(*this)
                  .GetNode(curr->parent)
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
                  .GetNode(curr->child[0])
                  .parent = idx;
               }
               if (curr->child[1])
               {
                  static_cast<ChildType &>(*this)
                  .GetNode(curr->child[1])
                  .parent = idx;
               }
               if (n.child[1])
               {
                  static_cast<ChildType &>(*this).GetNode(n.child[1]).parent =
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
                  .GetNode(curr->child[1])
                  .parent = idx;
               }
               if (curr->child[0])
               {
                  static_cast<ChildType &>(*this)
                  .GetNode(curr->child[0])
                  .parent = idx;
               }
               if (n.child[0])
               {
                  static_cast<ChildType &>(*this).GetNode(n.child[0]).parent =
                     y;
               }
            }
         }
         else
         {
            // no direct connection between y and idx
            if (n.parent)
            {
               auto &pn = static_cast<ChildType &>(*this).GetNode(n.parent);
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
                  static_cast<ChildType &>(*this).GetNode(curr->parent);
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
               static_cast<ChildType &>(*this).GetNode(n.child[0]).parent =
                  idx;
            }
            if (n.child[1])
            {
               static_cast<ChildType &>(*this).GetNode(n.child[1]).parent =
                  idx;
            }
            if (curr->child[0])
            {
               static_cast<ChildType &>(*this).GetNode(curr->child[0]).parent =
                  y;
            }
            if (curr->child[1])
            {
               static_cast<ChildType &>(*this).GetNode(curr->child[1]).parent =
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
         bool tflag = n.IsRed();
         if (curr->IsRed())
         {
            n.SetRed();
         }
         else
         {
            n.SetBlack();
         }
         if (tflag)
         {
            curr->SetRed();
         }
         else
         {
            curr->SetBlack();
         }
         curr = &n;
      }
      size_t x = static_cast<ChildType &>(*this).GetNode(y).child[0];
      if (!x)
      {
         x = static_cast<ChildType &>(*this).GetNode(y).child[1];
      }
      size_t x_parent = static_cast<ChildType &>(*this).GetNode(y).parent;
      if (x)
      {
         static_cast<ChildType &>(*this).GetNode(x).parent = x_parent;
      }
      if (!x_parent)
      {
         root = x;
      }
      else if (static_cast<ChildType &>(*this).GetNode(x_parent).child[0] == y)
      {
         static_cast<ChildType &>(*this).GetNode(x_parent).child[0] = x;
      }
      else
      {
         static_cast<ChildType &>(*this).GetNode(x_parent).child[1] = x;
      }

      if (y != idx)
      {
         static_cast<ChildType &>(*this).EraseSwapHook(idx);
      }
      if (x)
      {
         auto &xn = static_cast<ChildType &>(*this).GetNode(x);
         if (xn.IsRed())
         {
            if (x_parent)
            {
               auto &yn = static_cast<ChildType &>(*this).GetNode(y);
               auto &yp = static_cast<ChildType &>(*this).GetNode(yn.parent);
               EraseFixup(root, x, x_parent, yp.child[0] == y);
            }
            else
            {
               xn.SetBlack();
            }
         }
      }
   }
}

template <class ChildType>
template <class L, class R, class F>
bool RBTree<ChildType>::Visit(size_t curr, L &&visit_left, R &&visit_right,
                              F &&func) const
{
   if (curr)
   {
      auto &c = static_cast<const ChildType &>(*this).GetNode(curr);
      if (func(curr))
      {
         return true;
      }
      if (visit_left(curr))
      {
         if (Visit(c.child[0], std::forward<L>(visit_left),
                   std::forward<R>(visit_right), std::forward<F>(func)))
         {
            return true;
         }
      }
      if (visit_right(curr))
      {
         return Visit(c.child[1], std::forward<L>(visit_left),
                      std::forward<R>(visit_right), std::forward<F>(func));
      }
   }
   return false;
}

template <class ChildType> size_t RBTree<ChildType>::First(size_t root) const
{
   if (root)
   {
      auto n = &static_cast<const ChildType &>(*this).GetNode(root);
      while (n->child[0])
      {
         root = n->child[0];
         n = &static_cast<const ChildType &>(*this).GetNode(root);
      }
      return root;
   }
   return 0;
}
} // namespace mfem

#endif
