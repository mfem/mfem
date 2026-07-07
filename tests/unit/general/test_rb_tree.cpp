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

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "general/rb_tree.hpp"
#include "general/internal/reusable_storage.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <set>
#include <vector>

using namespace mfem;

/// test fixture for RBTree
struct RBTreeFixture : RBTree<RBTreeFixture>
{
   struct Node
   {
      enum Flags
      {
         // used for red-black tree balancing, internal
         RED_COLOR = 1 << 0,
         VALID = 1 << 1,
         NONE = 0,
      };
      // 0 is null, i>0 refers to nodes[i-1]
      size_t parent = 0;
      size_t child[2] = {0, 0};
      ptrdiff_t offset;
      Flags flag = Flags::NONE;

      bool IsValid() const { return flag & VALID; }
      void SetValid() { flag = static_cast<Flags>(flag | VALID); }
      void SetInvalid() { flag = static_cast<Flags>(flag & ~VALID); }
      void SetBlack() { flag = static_cast<Flags>(flag & ~RED_COLOR); }
      void SetRed() { flag = static_cast<Flags>(flag | RED_COLOR); }
      void CopyColor(const Node &o)
      {
         flag = static_cast<Flags>((flag & ~RED_COLOR) | (o.flag & RED_COLOR));
      }

      bool IsRed() const { return flag & RED_COLOR; }
   };

   internal::ReusableStorage<Node> nodes;
   size_t root = 0;

   using RBTree<RBTreeFixture>::Insert;
   using RBTree<RBTreeFixture>::Erase;
   using RBTree<RBTreeFixture>::First;
   using RBTree<RBTreeFixture>::Visit;
   using RBTree<RBTreeFixture>::Next;

   Node &GetNode(size_t curr) { return nodes.Get(curr); }
   const Node &GetNode(size_t curr) const { return nodes.Get(curr); }

   auto CompareNodes(size_t a, size_t b) const
   {
      auto &na = GetNode(a);
      auto &nb = GetNode(b);
      return nb.offset - na.offset;
   }

   void InsertDuplicate(size_t a, size_t b) { nodes.Erase(b); }

   size_t AddNode(ptrdiff_t val)
   {
      auto idx = nodes.CreateNext();
      auto &n = nodes.Get(idx);
      n.offset = val;
      return Insert(root, idx);
   }

   size_t AddNode(ptrdiff_t val, size_t hint)
   {
      auto idx = nodes.CreateNext();
      auto &n = nodes.Get(idx);
      n.offset = val;
      return Insert(root, hint, idx);
   }

   void EraseNode(size_t idx)
   {
      Erase(root, idx);
      nodes.Erase(idx);
   }

   void PrintTree(std::ostream& out = std::cout) const
   {
      out << "strict digraph {" << std::endl;
      Visit(root, [](size_t) { return true; }, [](size_t) { return true; },
      [&](size_t idx)
      {
         REQUIRE(idx != 0);
         auto &node = GetNode(idx);
         if (node.IsRed())
         {
            out << node.offset << " [color=\"red\"];" << std::endl;
         }
         else
         {
            out << node.offset << " [color=\"black\"];" << std::endl;
         }
         for (int i = 0; i < 2; ++i)
         {
            if (node.child[i])
            {
               auto &child = GetNode(node.child[i]);
               out << node.offset << " -> ";
               out << child.offset << ";" << std::endl;
               out << child.offset << " -> " << node.offset << " [color=\"blue\"];" <<
                   std::endl;
            }
            else
            {
               out << "NULL_" << node.offset << "_" << i << " [shape=point];"
                   << std::endl;
               out << node.offset << " -> ";
               out << "NULL_" << node.offset << "_" << i << ";" << std::endl;
            }
         }
         return false;
      });
      out << "}" << std::endl;
   }

   void ValidateSubtree(
      size_t parent, size_t idx, size_t num_black,
      ptrdiff_t min_value = std::numeric_limits<ptrdiff_t>::min(),
      ptrdiff_t max_value = std::numeric_limits<ptrdiff_t>::max()) const
   {
      if (!idx)
      {
         REQUIRE(num_black == 0);
         return;
      }
      auto& n = GetNode(idx);
      REQUIRE(n.parent == parent);
      if (n.IsRed())
      {
         // red nodes cannot have red children
         if (n.child[0])
         {
            auto& child = GetNode(n.child[0]);
            REQUIRE(!child.IsRed());
         }
         if (n.child[1])
         {
            auto& child = GetNode(n.child[1]);
            REQUIRE(!child.IsRed());
         }
      }
      else
      {
         --num_black;
      }
      // make sure tree is a BSP
      REQUIRE(n.offset >= min_value);
      REQUIRE(n.offset <= max_value);
      ValidateSubtree(idx, n.child[0], num_black, min_value, n.offset - 1);
      ValidateSubtree(idx, n.child[1], num_black, n.offset + 1, max_value);
   }

   void ValidateTree() const
   {
      if (root)
      {
         size_t num_black = 0;
         // traverse down all left paths, count number of black nodes we passed
         // through
         Visit(root, [](size_t) { return true; }, [](size_t) { return false; },
         [&](size_t idx)
         {
            REQUIRE(idx != 0);
            auto &n = GetNode(idx);
            if (!n.IsRed())
            {
               ++num_black;
            }
            return false;
         });
         // now do a full tree traversal and validate all invariants
         ValidateSubtree(0, root, num_black);
      }
   }

   size_t Find(ptrdiff_t value) const
   {
      if (!root)
      {
         return 0;
      }

      size_t res = 0;
      Visit(root,
            [&](size_t idx)
      {
         auto &n = GetNode(idx);
         if (value < n.offset)
         {
            return true;
         }
         if (n.offset == value)
         {
            // found node
            res = idx;
            return false;
         }
         return false;
      }, [&](size_t idx)
      {
         auto &n = GetNode(idx);
         if (value > n.offset)
         {
            return true;
         }
         if (n.offset == value)
         {
            // found node
            res = idx;
            return false;
         }
         return false;
      }, [&](size_t idx) { return false; });
      return res;
   }
};

TEST_CASE("RB Tree Insert", "[RB Tree]")
{
   // special configurations of Insert to test
   SECTION("hint violation")
   {
      RBTreeFixture tree;
      REQUIRE(tree.AddNode(30) == 1);
      REQUIRE(tree.AddNode(10) == 2);
      REQUIRE(tree.AddNode(20) == 3);
      tree.ValidateTree();
      // tree:
      //    20
      // 10    30
      auto idx = tree.AddNode(40, 2);
      tree.ValidateTree();
      REQUIRE(idx == 4);

      // tree:
      //    20
      // 10    30
      //          40
      idx = tree.AddNode(5, 4);
      tree.ValidateTree();
      REQUIRE(idx == 5);

      // tree:
      //      20
      //   10    30
      // 5          40
      // duplicate with hint
      REQUIRE(tree.AddNode(10, 5) == 2);
      tree.ValidateTree();
      REQUIRE(tree.AddNode(20, 4) == 3);
      tree.ValidateTree();
   }
}

TEST_CASE("RB Tree Next", "[RB Tree]")
{
   constexpr int num = 100;
   std::vector<int> vals(num);
   for (int i = 0; i < num; ++i)
   {
      vals[i] = i;
   }
   std::minstd_rand engine(12345);
   std::shuffle(vals.begin(), vals.end(), engine);

   RBTreeFixture tree;
   for (int i = 0; i < num; ++i)
   {
      REQUIRE(tree.AddNode(vals[i]) == i + 1);
   }
   tree.ValidateTree();
   for (int i = 0; i < num; ++i)
   {
      auto next = tree.Next(i + 1);
      if (vals[i] + 1 == num)
      {
         // largest value has no successor
         REQUIRE(next == 0);
      }
      else
      {
         REQUIRE(tree.GetNode(next).offset == vals[i] + 1);
      }
   }
}

TEST_CASE("RB Tree Erase", "[RB Tree]")
{
   // special configurations of Erase to test
   SECTION("Erase black two children")
   {
      RBTreeFixture tree;
      for (int i = 0; i < 10; i += 2)
      {
         tree.AddNode(i);
      }
      for (int i = 1; i < 10; i += 2)
      {
         tree.AddNode(i);
      }
      tree.ValidateTree();
      auto idx = tree.Find(4);
      REQUIRE(idx != 0);
      tree.EraseNode(idx);
      for (int i = 0; i < 10; ++i)
      {
         idx = tree.Find(i);
         if (i == 4)
         {
            REQUIRE(idx == 0);
         }
         else
         {
            REQUIRE(idx != 0);
         }
      }
   }
   SECTION("Erase single child right")
   {
      RBTreeFixture tree;
      for (int i = 0; i < 10; i += 2)
      {
         tree.AddNode(i);
      }
      for (int i = 1; i < 10; i += 2)
      {
         tree.AddNode(i);
      }
      tree.ValidateTree();
      auto idx = tree.Find(0);
      REQUIRE(idx != 0);
      tree.EraseNode(idx);
      for (int i = 0; i < 10; ++i)
      {
         idx = tree.Find(i);
         if (i == 0)
         {
            REQUIRE(idx == 0);
         }
         else
         {
            REQUIRE(idx != 0);
         }
      }
   }
   SECTION("Erase single child left")
   {
      RBTreeFixture tree;
      for (int i = 0; i < 10; i += 2)
      {
         tree.AddNode(2 * i);
      }
      for (int i = 1; i < 10; i += 2)
      {
         tree.AddNode(2 * i);
      }
      tree.AddNode(17);
      tree.ValidateTree();
      auto idx = tree.Find(17);
      REQUIRE(idx != 0);
      tree.EraseNode(idx);
      for (int i = 0; i < 10; ++i)
      {
         idx = tree.Find(2 * i);
         REQUIRE(idx != 0);
      }
      idx = tree.Find(17);
      REQUIRE(idx == 0);
   }
}

TEST_CASE("RB Tree", "[RB Tree]")
{
   RBTreeFixture tree;
   std::set<ptrdiff_t> values;

   std::minstd_rand engine(12345);
   std::uniform_int_distribution<int> operation(0, 2);
   std::uniform_int_distribution<ptrdiff_t> value_distr(0, 2000);
   for (int i = 0; i < 1000; ++i)
   {
      int op = 0;
      if (values.size())
      {
         op = operation(engine);
      }
      if (op < 2)
      {
         // random insert, possibly a duplicate value
         ptrdiff_t v = value_distr(engine);
         // std::cout << "insert " << v << std::endl;
         tree.AddNode(v);
         values.emplace(v);
         tree.ValidateTree();
         // ensure v is in tree
         REQUIRE(tree.Find(v) != 0);
      }
      else
      {
         // random erase, not uniformly distributed
         ptrdiff_t v = value_distr(engine);
         auto iter = values.upper_bound(v);
         if (iter != values.begin())
         {
            --iter;
         }
         // iter points to first value < v (if any), otherwise values.begin()
         v = *iter;
         values.erase(iter);
         auto node = tree.Find(v);
         REQUIRE(node != 0);
         // std::cout << "erase " << v << std::endl;
         tree.EraseNode(node);
         tree.ValidateTree();
         // ensure v is not in tree
         REQUIRE(tree.Find(v) == 0);
      }
   }
   for (int i = 0; i < 1000; ++i)
   {
      int op = 0;
      if (values.size())
      {
         op = operation(engine);
      }
      if (op < 1)
      {
         // random insert, possibly a duplicate value
         ptrdiff_t v = value_distr(engine);
         // std::cout << "insert " << v << std::endl;
         tree.AddNode(v);
         values.emplace(v);
         tree.ValidateTree();
         // ensure v is in tree
         REQUIRE(tree.Find(v) != 0);
      }
      else
      {
         // random erase, not uniformly distributed
         ptrdiff_t v = value_distr(engine);
         auto iter = values.upper_bound(v);
         if (iter != values.begin())
         {
            --iter;
         }
         // iter points to first value < v (if any), otherwise values.begin()
         v = *iter;
         values.erase(iter);
         auto node = tree.Find(v);
         REQUIRE(node != 0);
         // std::cout << "erase " << v << std::endl;
         tree.EraseNode(node);
         tree.ValidateTree();
         // ensure v is not in tree
         REQUIRE(tree.Find(v) == 0);
      }
   }
   // now erase everything
   for (auto &v : values)
   {
      auto node = tree.Find(v);
      REQUIRE(node != 0);
      tree.EraseNode(node);
      tree.ValidateTree();
      // ensure v is not in tree
      REQUIRE(tree.Find(v) == 0);
   }
}
