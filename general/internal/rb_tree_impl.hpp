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

#ifndef MFEM_RB_TREE_IMPL_HPP
#define MFEM_RB_TREE_IMPL_HPP

namespace mfem
{

template <class ChildType> size_t RBTree<ChildType>::Next(size_t idx)
{
   ChildType &self = static_cast<ChildType &>(*this);
   if (self.GetNode(idx).child[1])
   {
      // left-most child of right child
      idx = self.GetNode(idx).child[1];
      while (self.GetNode(idx).child[0])
      {
         idx = self.GetNode(idx).child[0];
      }
      return idx;
   }
   else
   {
      // need to search from parent
      size_t y = self.GetNode(idx).parent;
      while (y && idx == self.GetNode(y).child[1])
      {
         idx = y;
         y = self.GetNode(y).parent;
      }
      return y;
   }
}

template <class ChildType>
void RBTree<ChildType>::InsertFixup(size_t &root, size_t curr)
{
   ChildType &self = static_cast<ChildType &>(*this);
   auto z = &self.GetNode(curr);
   while (z->parent)
   {
      auto p = &self.GetNode(z->parent);
      if (!p->IsRed())
      {
         // case 1
         break;
      }
      if (!p->parent)
      {
         // case 4
         break;
      }
      auto gp = &self.GetNode(p->parent);
      if (z->parent == gp->child[0])
      {
         // y is our uncle
         size_t y = gp->child[1];
         if (y && self.GetNode(y).IsRed())
         {
            // case 2
            p->SetBlack();
            self.GetNode(y).SetBlack();
            gp->SetRed();
            curr = p->parent;
            z = gp;
         }
         else
         {
            if (curr == p->child[1])
            {
               // case 5
               curr = z->parent;
               z = p;
               LeftRotate(root, curr);
            }
            // case 6
            p = &self.GetNode(z->parent);
            gp = &self.GetNode(p->parent);
            p->SetBlack();
            gp->SetRed();
            RightRotate(root, p->parent);
         }
      }
      else
      {
         // y is our uncle
         size_t y = gp->child[0];
         if (y && self.GetNode(y).IsRed())
         {
            // case 2
            p->SetBlack();
            self.GetNode(y).SetBlack();
            gp->SetRed();
            curr = p->parent;
            z = gp;
         }
         else
         {
            if (p->child[0] == curr)
            {
               // case 5
               curr = z->parent;
               z = p;
               RightRotate(root, curr);
            }
            // case 6
            p = &self.GetNode(z->parent);
            gp = &self.GetNode(p->parent);
            p->SetBlack();
            gp->SetRed();
            LeftRotate(root, p->parent);
         }
      }
   }
   self.GetNode(root).SetBlack();
}

template <class ChildType>
void RBTree<ChildType>::LeftRotate(size_t &root, size_t curr)
{
   ChildType &self = static_cast<ChildType &>(*this);
   auto x = &self.GetNode(curr);
   size_t y = x->child[1];
   auto yn = &self.GetNode(y);
   x->child[1] = yn->child[0];
   if (yn->child[0])
   {
      self.GetNode(yn->child[0]).parent = curr;
   }
   yn->parent = x->parent;
   if (!x->parent)
   {
      root = y;
   }
   else if (self.GetNode(x->parent).child[0] == curr)
   {
      self.GetNode(x->parent).child[0] = y;
   }
   else
   {
      self.GetNode(x->parent).child[1] = y;
   }
   yn->child[0] = curr;
   x->parent = y;
}

template <class ChildType>
void RBTree<ChildType>::RightRotate(size_t &root, size_t curr)
{
   ChildType &self = static_cast<ChildType &>(*this);
   auto y = &self.GetNode(curr);
   size_t x = y->child[0];
   auto xn = &self.GetNode(x);
   y->child[0] = xn->child[1];
   if (xn->child[1])
   {
      self.GetNode(xn->child[1]).parent = curr;
   }
   xn->parent = y->parent;
   if (!y->parent)
   {
      root = x;
   }
   else if (self.GetNode(y->parent).child[0] == curr)
   {
      self.GetNode(y->parent).child[0] = x;
   }
   else
   {
      self.GetNode(y->parent).child[1] = x;
   }
   xn->child[1] = curr;
   y->parent = x;
}

template <class ChildType>
size_t RBTree<ChildType>::InsertImpl(size_t &root, size_t pos, size_t curr,
                                     bool check_hint)
{
   ChildType &self = static_cast<ChildType &>(*this);
   auto &z = self.GetNode(curr);
   size_t y = 0;
   size_t x = pos;
   // insert into the binary tree (unbalanced)
   while (x)
   {
      y = x;
      auto &nx = self.GetNode(x);
      auto cmp = self.CompareNodes(x, curr);
      if (cmp < 0)
      {
         if (check_hint)
         {
            if (nx.parent)
            {
               auto &np = self.GetNode(nx.parent);
               if (np.child[1] == x)
               {
                  // x is a right child, ensure c is between p and x
                  // p
                  //     x
                  //  c?
                  auto cmp2 = self.CompareNodes(nx.parent, curr);
                  if (cmp2 < 0)
                  {
                     // curr is to the left of parent, bad hint. Try parent.
                     x = nx.parent;
                     continue;
                  }
                  else if (cmp2 == 0)
                  {
                     // duplicate with parent
                     self.InsertDuplicate(nx.parent, curr);
                     return nx.parent;
                  }
               }
            }
            // don't need to check for a valid hint anymore
            check_hint = false;
            continue;
         }
         x = nx.child[0];
      }
      else if (cmp > 0)
      {
         if (check_hint)
         {
            if (nx.parent)
            {
               auto &np = self.GetNode(nx.parent);
               if (np.child[0] == x)
               {
                  // x is a left child, ensure c is between x and p
                  //     p
                  // x
                  //   c?
                  auto cmp2 = self.CompareNodes(nx.parent, curr);
                  if (cmp2 > 0)
                  {
                     // curr is to the left of parent, bad hint. Try parent.
                     x = nx.parent;
                     continue;
                  }
                  else if (cmp2 == 0)
                  {
                     // duplicate with parent
                     self.InsertDuplicate(nx.parent, curr);
                     return nx.parent;
                  }
               }
            }
            // don't need to check for a valid hint anymore
            check_hint = false;
            continue;
         }
         x = nx.child[1];
      }
      else
      {
         // duplicate range, ignore
         self.InsertDuplicate(x, curr);
         return x;
      }
   }
   z.parent = y;
   if (!y)
   {
      root = curr;
   }
   else if (self.CompareNodes(y, curr) < 0)
   {
      self.GetNode(y).child[0] = curr;
   }
   else
   {
      self.GetNode(y).child[1] = curr;
   }
   z.SetRed();
   // fix red-black invariance
   InsertFixup(root, curr);

   return curr;
}

template <class ChildType>
void RBTree<ChildType>::EraseSimpleOne(size_t &root, size_t idx, int child)
{
   // simple case: only 1 child
   ChildType &self = static_cast<ChildType &>(*this);
   auto& a = self.GetNode(idx);
   auto cidx = a.child[child];
   auto& b = self.GetNode(cidx);
   // color b black, move b into a's spot
   MFEM_ASSERT(b.IsRed(), "");
   MFEM_ASSERT(!a.IsRed(), "");
   b.SetBlack();

   // internal connections
   b.parent = a.parent;

   // external connections
   if (a.parent)
   {
      auto &p = self.GetNode(a.parent);
      if (p.child[0] == idx)
      {
         p.child[0] = cidx;
      }
      else
      {
         p.child[1] = cidx;
      }
   }
   else
   {
      root = cidx;
   }
}

template <class ChildType>
void RBTree<ChildType>::Erase(size_t &root, size_t idx)
{
   if (!idx)
   {
      return;
   }
   ChildType &self = static_cast<ChildType &>(*this);

   auto curr = &self.GetNode(idx);

   if (!curr->child[0])
   {
      if (curr->child[1])
      {
         return EraseSimpleOne(root, idx, 1);
      }
   }
   else if (!curr->child[1])
   {
      return EraseSimpleOne(root, idx, 0);
   }
   else
   {
      // idx has two children
      // y is the left-most child of the right subtree of idx
      auto y = Next(idx);
      MFEM_ASSERT(y != idx, "");
      // swap nodes (idx, y)
      auto &n = self.GetNode(y);

      // change connections
      std::swap(n.child[0], curr->child[0]);
      if (y == curr->child[1])
      {
         // parent
         n.parent = curr->parent;
         curr->parent = y;
         // right child
         curr->child[1] = n.child[1];
         n.child[1] = idx;
      }
      else
      {
         std::swap(n.parent, curr->parent);
         std::swap(n.child[1], curr->child[1]);

         // fix y's old parent
         auto &p = self.GetNode(curr->parent);
         if (p.child[0] == y)
         {
            p.child[0] = idx;
         }
         else
         {
            p.child[1] = idx;
         }

         // fix idx's old right child
         if (n.child[1])
         {
            MFEM_ASSERT(self.GetNode(n.child[1]).parent == idx, "");
            self.GetNode(n.child[1]).parent = y;
         }
      }

      // fix idx's old parent
      if (n.parent)
      {
         auto &p = self.GetNode(n.parent);
         if (p.child[0] == idx)
         {
            p.child[0] = y;
         }
         else
         {
            p.child[1] = y;
         }
      }
      else
      {
         root = y;
      }
      // fix y's old children
      MFEM_ASSERT(!curr->child[0], "");
      if (curr->child[1])
      {
         MFEM_ASSERT(self.GetNode(curr->child[1]).parent == y, "");
         self.GetNode(curr->child[1]).parent = idx;
      }
      // fix idx's old left child
      if (n.child[0])
      {
         MFEM_ASSERT(self.GetNode(n.child[0]).parent == idx, "");
         self.GetNode(n.child[0]).parent = y;
      }

      // swap colors
      bool tflag = n.IsRed();
      n.CopyColor(*curr);
      if (tflag)
      {
         curr->SetRed();
      }
      else
      {
         curr->SetBlack();
      }
      return Erase(root, idx);
   }
   // idx has no children
   if (idx == root)
   {
      // idx was the last node
      root = 0;
      return;
   }
   auto p = &self.GetNode(curr->parent);
   int dir;
   if (p->child[0] == idx)
   {
      p->child[0] = 0;
      dir = 0;
   }
   else
   {
      p->child[1] = 0;
      dir = 1;
   }
   if (curr->IsRed())
   {
      // no rebalancing needed
      return;
   }
   // difficult case: delete black idx with no children and rebalance
   auto pidx = curr->parent;
   size_t nidx1, nidx2;
   size_t sidx;
   while (true)
   {
      sidx = p->child[1 - dir];
      MFEM_ASSERT(sidx != 0, "");
      auto sibling = &self.GetNode(sidx);
      nidx1 = sibling->child[1 - dir];
      nidx2 = sibling->child[dir];
      if (sibling->IsRed())
      {
         // case 3
         if (dir)
         {
            RightRotate(root, pidx);
         }
         else
         {
            LeftRotate(root, pidx);
         }
         p->SetRed();
         sibling->SetBlack();
         sidx = nidx2;
         MFEM_ASSERT(sidx, "");
         sibling = &self.GetNode(sidx);
         nidx1 = sibling->child[1 - dir];
         if (nidx1)
         {
            auto &nephew = self.GetNode(nidx1);
            if (nephew.IsRed())
            {
               goto case_6;
            }
         }
         nidx2 = sibling->child[dir];
         if (nidx2)
         {
            auto &nephew = self.GetNode(nidx2);
            if (nephew.IsRed())
            {
               goto case_5;
            }
         }

         // case 4
         self.GetNode(sidx).SetRed();
         p->SetBlack();
         return;
      }
      if (nidx1)
      {
         auto &nephew = self.GetNode(nidx1);
         if (nephew.IsRed())
         {
            goto case_6;
         }
      }
      if (nidx2)
      {
         auto &nephew = self.GetNode(nidx2);
         if (nephew.IsRed())
         {
            goto case_5;
         }
      }

      sibling->SetRed();
      if (p->IsRed())
      {
         // case 4
         p->SetBlack();
         return;
      }

      // case 2
      idx = pidx;
      curr = p;

      if (!curr->parent)
      {
         break;
      }
      pidx = curr->parent;
      MFEM_ASSERT(pidx, "");
      p = &self.GetNode(pidx);
      if (p->child[0] == idx)
      {
         dir = 0;
      }
      else
      {
         dir = 1;
      }
   }
   // case 1
   return;

case_5:
   if (dir)
   {
      LeftRotate(root, sidx);
   }
   else
   {
      RightRotate(root, sidx);
   }
   MFEM_ASSERT(sidx, "");
   MFEM_ASSERT(nidx2, "");
   self.GetNode(sidx).SetRed();
   self.GetNode(nidx2).SetBlack();
   nidx1 = sidx;
   sidx = nidx2;

case_6:
   if (dir)
   {
      RightRotate(root, pidx);
   }
   else
   {
      LeftRotate(root, pidx);
   }
   MFEM_ASSERT(sidx, "");
   self.GetNode(sidx).CopyColor(*p);
   p->SetBlack();
   MFEM_ASSERT(nidx1, "");
   self.GetNode(nidx1).SetBlack();
}

template <class ChildType>
template <class L, class R, class F>
bool RBTree<ChildType>::Visit(size_t curr, L &&visit_left, R &&visit_right,
                              F &&func) const
{
   if (curr)
   {
      const ChildType &self = static_cast<const ChildType &>(*this);
      auto &c = self.GetNode(curr);
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
      const ChildType &self = static_cast<const ChildType &>(*this);
      auto n = &self.GetNode(root);
      while (n->child[0])
      {
         root = n->child[0];
         n = &self.GetNode(root);
      }
      return root;
   }
   return 0;
}
} // namespace mfem

#endif
