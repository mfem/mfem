// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "gecko.hpp"

// This file collects the sources of the Gecko library as a single module.
// The original library can be found at https://github.com/LLNL/gecko
// Used here with permission.

// ------------------------------------------------------------------------------

// BSD 3-Clause License
//
// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// ------------------------------------------------------------------------------

// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and other
// gecko project contributors. See the above license for details.
// SPDX-License-Identifier: BSD-3-Clause
// LLNL-CODE-800597

#include <algorithm>
#include <functional>
#include <cstdlib>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <map>

// ----- options.h --------------------------------------------------------------

// ratio of max to min weight for aggregation
#ifndef GECKO_PART_FRAC
#define GECKO_PART_FRAC 4
#endif

// number of compatible relaxation sweeps
#ifndef GECKO_CR_SWEEPS
#define GECKO_CR_SWEEPS 1
#endif

// number of Gauss-Seidel relaxation sweeps
#ifndef GECKO_GS_SWEEPS
#define GECKO_GS_SWEEPS 1
#endif

// max number of nodes in subgraph
#ifndef GECKO_WINDOW_MAX
#define GECKO_WINDOW_MAX 16
#endif

// use adjacency list (1) or adjacency matrix (0)
#ifndef GECKO_WITH_ADJLIST
#define GECKO_WITH_ADJLIST 0
#endif

// use nonrecursive permutation algorithm
#ifndef GECKO_WITH_NONRECURSIVE
#define GECKO_WITH_NONRECURSIVE 0
#endif

// use double-precision computations
#ifndef GECKO_WITH_DOUBLE_PRECISION
#define GECKO_WITH_DOUBLE_PRECISION 0
#endif

// ----- heap.h -----------------------------------------------------------------

template <
   typename T,                             // data type
   typename P,                             // priority type
   class    C = std::less<P>,              // comparator for priorities
   class    M = std::map<T, unsigned int>  // maps type T to unsigned integer
   >
class DynamicHeap
{
public:
   DynamicHeap(size_t count = 0);
   ~DynamicHeap() {}
   void insert(T data, P priority);
   void update(T data, P priority);
   bool top(T& data);
   bool top(T& data, P& priority);
   bool pop();
   bool extract(T& data);
   bool extract(T& data, P& priority);
   bool erase(T data);
   bool find(T data) const;
   bool find(T data, P& priority) const;
   bool empty() const { return heap.empty(); }
   size_t size() const { return heap.size(); }
private:
   struct HeapEntry
   {
      HeapEntry(P p, T d) : priority(p), data(d) {}
      P priority;
      T data;
   };
   std::vector<HeapEntry> heap;
   M index;
   C lower;
   void ascend(unsigned int i);
   void descend(unsigned int i);
   void swap(unsigned int i, unsigned int j);
   bool ordered(unsigned int i, unsigned int j) const
   {
      return !lower(heap[i].priority, heap[j].priority);
   }
   unsigned int parent(unsigned int i) const { return (i - 1) / 2; }
   unsigned int left(unsigned int i) const { return 2 * i + 1; }
   unsigned int right(unsigned int i) const { return 2 * i + 2; }
};

template < typename T, typename P, class C, class M >
DynamicHeap<T, P, C, M>::DynamicHeap(size_t count)
{
   heap.reserve(count);
}

template < typename T, typename P, class C, class M >
void
DynamicHeap<T, P, C, M>::insert(T data, P priority)
{
   if (index.find(data) != index.end())
   {
      update(data, priority);
   }
   else
   {
      unsigned int i = (unsigned int)heap.size();
      heap.push_back(HeapEntry(priority, data));
      ascend(i);
   }
}

template < typename T, typename P, class C, class M >
void
DynamicHeap<T, P, C, M>::update(T data, P priority)
{
   unsigned int i = index[data];
   heap[i].priority = priority;
   ascend(i);
   descend(i);
}

template < typename T, typename P, class C, class M >
bool
DynamicHeap<T, P, C, M>::top(T& data)
{
   if (!heap.empty())
   {
      data = heap[0].data;
      return true;
   }
   else
   {
      return false;
   }
}

template < typename T, typename P, class C, class M >
bool
DynamicHeap<T, P, C, M>::top(T& data, P& priority)
{
   if (!heap.empty())
   {
      data = heap[0].data;
      priority = heap[0].priority;
      return true;
   }
   else
   {
      return false;
   }
}

template < typename T, typename P, class C, class M >
bool
DynamicHeap<T, P, C, M>::pop()
{
   if (!heap.empty())
   {
      T data = heap[0].data;
      swap(0, (unsigned int)heap.size() - 1);
      index.erase(data);
      heap.pop_back();
      if (!heap.empty())
      {
         descend(0);
      }
      return true;
   }
   else
   {
      return false;
   }
}

template < typename T, typename P, class C, class M >
bool
DynamicHeap<T, P, C, M>::extract(T& data)
{
   if (!heap.empty())
   {
      data = heap[0].data;
      return pop();
   }
   else
   {
      return false;
   }
}

template < typename T, typename P, class C, class M >
bool
DynamicHeap<T, P, C, M>::extract(T& data, P& priority)
{
   if (!heap.empty())
   {
      data = heap[0].data;
      priority = heap[0].priority;
      return pop();
   }
   else
   {
      return false;
   }
}

template < typename T, typename P, class C, class M >
bool
DynamicHeap<T, P, C, M>::erase(T data)
{
   if (index.find(data) == index.end())
   {
      return false;
   }
   unsigned int i = index[data];
   swap(i, heap.size() - 1);
   index.erase(data);
   heap.pop_back();
   if (i < heap.size())
   {
      ascend(i);
      descend(i);
   }
   return true;
}

template < typename T, typename P, class C, class M >
bool
DynamicHeap<T, P, C, M>::find(T data) const
{
   return index.find(data) != index.end();
}

template < typename T, typename P, class C, class M >
bool
DynamicHeap<T, P, C, M>::find(T data, P& priority) const
{
   typename M::const_iterator p;
   if ((p = index.find(data)) == index.end())
   {
      return false;
   }
   unsigned int i = p->second;
   priority = heap[i].priority;
   return true;
}

template < typename T, typename P, class C, class M >
void
DynamicHeap<T, P, C, M>::ascend(unsigned int i)
{
   for (unsigned int j; i && !ordered(j = parent(i), i); i = j)
   {
      swap(i, j);
   }
   index[heap[i].data] = i;
}

template < typename T, typename P, class C, class M >
void
DynamicHeap<T, P, C, M>::descend(unsigned int i)
{
   for (unsigned int j, k;
        (j = ((k =  left(i)) < heap.size() && !ordered(i, k) ? k : i),
         j = ((k = right(i)) < heap.size() && !ordered(j, k) ? k : j)) != i;
        i = j)
   {
      swap(i, j);
   }
   index[heap[i].data] = i;
}

template < typename T, typename P, class C, class M >
void
DynamicHeap<T, P, C, M>::swap(unsigned int i, unsigned int j)
{
   std::swap(heap[i], heap[j]);
   index[heap[i].data] = i;
}

// ----- subgraph.h -------------------------------------------------------------

namespace Gecko
{

// Node in a subgraph.
class Subnode
{
public:
   typedef unsigned char Index;
   Float pos;        // node position
   WeightedSum cost; // external cost at this position
};

class Subgraph
{
public:
   Subgraph(Graph* g, uint n);
   ~Subgraph() { delete[] cache; }
   void optimize(uint k);

private:
   Graph* const g;                        // full graph
   const uint n;                          // number of subgraph nodes
   Functional* const f;                   // ordering functional
   WeightedSum min;                       // minimum cost so far
   Subnode::Index best[GECKO_WINDOW_MAX]; // best permutation so far
   Subnode::Index perm[GECKO_WINDOW_MAX]; // current permutation
   const Subnode* node[GECKO_WINDOW_MAX]; // pointers to precomputed nodes
   Subnode* cache;                        // precomputed node positions and costs
#if GECKO_WITH_ADJLIST
   Subnode::Index
   adj[GECKO_WINDOW_MAX][GECKO_WINDOW_MAX]; // internal adjacency list
#else
   uint adj[GECKO_WINDOW_MAX];            // internal adjacency matrix
#endif
   Float weight[GECKO_WINDOW_MAX][GECKO_WINDOW_MAX]; // internal arc weights
   WeightedSum cost(uint k) const;
   void swap(uint k);
   void swap(uint k, uint l);
   void optimize(WeightedSum c, uint i);
};

}

// ----- subgraph.cpp -----------------------------------------------------------

using namespace Gecko;

// Constructor.
Subgraph::Subgraph(Graph* g, uint n) : g(g), n(n), f(g->functional)
{
   if (n > GECKO_WINDOW_MAX)
   {
      throw std::out_of_range("optimization window too large");
   }
   cache = new Subnode[n << n];
}

// Cost of k'th node's edges to external nodes and nodes at {k+1, ..., n-1}.
WeightedSum
Subgraph::cost(uint k) const
{
   Subnode::Index i = perm[k];
   WeightedSum c = node[i]->cost;
   Float p = node[i]->pos;
#if GECKO_WITH_ADJLIST
   for (k = 0; adj[i][k] != i; k++)
   {
      Subnode::Index j = adj[i][k];
      Float l = node[j]->pos - p;
      if (l > 0)
      {
         Float w = weight[i][k];
         f->accumulate(c, WeightedValue(l, w));
      }
   }
#else
   uint m = adj[i];
   while (++k < n)
   {
      Subnode::Index j = perm[k];
      if (m & (1u << j))
      {
         Float l = node[j]->pos - p;
         Float w = weight[i][j];
         f->accumulate(c, WeightedValue(l, w));
      }
   }
#endif
   return c;
}

// Swap the two nodes in positions k and k + 1.
void
Subgraph::swap(uint k)
{
   uint l = k + 1;
   Subnode::Index i = perm[k];
   Subnode::Index j = perm[l];
   perm[k] = j;
   perm[l] = i;
   node[i] -= ptrdiff_t(1) << j;
   node[j] += ptrdiff_t(1) << i;
}

// Swap the two nodes in positions k and l, k <= l.
void
Subgraph::swap(uint k, uint l)
{
   Subnode::Index i = perm[k];
   Subnode::Index j = perm[l];
   perm[k] = j;
   perm[l] = i;
   // Update node positions.
   uint m = 0;
   while (++k < l)
   {
      Subnode::Index h = perm[k];
      node[h] += ptrdiff_t(1) << i;
      node[h] -= ptrdiff_t(1) << j;
      m += 1u << h;
   }
   node[i] -= (1u << j) + m;
   node[j] += (1u << i) + m;
}

#if GECKO_WITH_NONRECURSIVE
// Evaluate all permutations generated by Heap's nonrecursive algorithm.
void
Subgraph::optimize(WeightedSum, uint)
{
   WeightedSum c[GECKO_WINDOW_MAX + 1];
   uint j[GECKO_WINDOW_MAX + 1];
   j[n] = 1;
   c[n] = 0;
   uint i = n;
   do
   {
      i--;
      j[i] = i;
   loop:
      c[i] = f->sum(c[i + 1], cost(i));
   }
   while (i);
   if (f->less(c[0], min))
   {
      min = c[0];
      for (uint k = 0; k < n; k++)
      {
         best[k] = perm[k];
      }
   }
   do
   {
      if (++i == n)
      {
         return;
      }
      swap(i & 1 ? i - j[i] : 0, i);
   }
   while (!j[i]--);
   goto loop;
}
#else
// Apply branch-and-bound to permutations generated by Heap's algorithm.
void
Subgraph::optimize(WeightedSum c, uint i)
{
   i--;
   if (f->less(c, min))
   {
      if (i)
      {
         uint j = i;
         do
         {
            optimize(f->sum(c, cost(i)), i);
            swap(i & 1 ? i - j : 0, i);
         }
         while (j--);
      }
      else
      {
         f->accumulate(c, cost(0));
         if (f->less(c, min))
         {
            min = c;
            for (uint j = 0; j < n; j++)
            {
               best[j] = perm[j];
            }
         }
      }
   }
   else if (i & 1)
      do { swap(--i); }
      while (i);
}
#endif

// Optimize layout of nodes {p, ..., p + n - 1}.
void
Subgraph::optimize(uint p)
{
   // Initialize subgraph.
   const Float q = g->node[g->perm[p]].pos - g->node[g->perm[p]].hlen;
   min = WeightedSum(GECKO_FLOAT_MAX, 1);
   for (Subnode::Index k = 0; k < n; k++)
   {
      best[k] = perm[k] = k;
      Node::Index i = g->perm[p + k];
      // Copy i's outgoing arcs.  We distinguish between internal
      // and external arcs to nodes within and outside the subgraph,
      // respectively.
#if GECKO_WITH_ADJLIST
      uint m = 0;
#else
      adj[k] = 0;
#endif
      std::vector<Arc::Index> external;
      for (Arc::Index a = g->node_begin(i); a < g->node_end(i); a++)
      {
         Node::Index j = g->adj[a];
         Subnode::Index l;
         for (l = 0; l < n && g->perm[p + l] != j; l++);
         if (l == n)
         {
            external.push_back(a);
         }
         else
         {
            // Copy internal arc to subgraph.
#if GECKO_WITH_ADJLIST
            adj[k][m] = l;
            weight[k][m] = g->weight[a];
            m++;
#else
            adj[k] += 1u << l;
            weight[k][l] = g->weight[a];
#endif
         }
      }
#if GECKO_WITH_ADJLIST
      adj[k][m] = k;
#endif
      // Precompute external costs associated with all possible positions
      // of this node.  Since node lengths can be arbitrary, there are as
      // many as 2^(n-1) possible positions, each corresponding to an
      // (n-1)-bit string that specifies whether the remaining n-1 nodes
      // succeed this node or not.  Caching the
      //                n
      //   2^(n-1) n = sum k C(n, k) = A001787
      //               k=1
      // external costs is exponentially cheaper than recomputing the
      //      n-1         n
      //   n! sum 1/k! = sum k! C(n, k) = A007526
      //      k=0        k=1
      // costs associated with all permutations.
      node[k] = cache + (k << n);
      for (uint m = 0; m < (1u << n); m++)
         if (!(m & (1u << k)))
         {
            Subnode* s = cache + (k << n) + m;
            s->pos = q + g->node[i].hlen;
            for (Subnode::Index l = 0; l < n; l++)
               if (l != k && !(m & (1u << l)))
               {
                  s->pos += 2 * g->node[g->perm[p + l]].hlen;
               }
            s->cost = g->cost(external, s->pos);
         }
         else
         {
            m += (1u << k) - 1;
         }
      node[k] += (1u << n) - (2u << k);
   }

   // Find optimal permutation of the n nodes.
   optimize(0, n);

   // Apply permutation to original graph.
   for (uint i = 0; i < n; i++)
   {
      g->swap(p + i, p + best[i]);
      for (uint j = i + 1; j < n; j++)
         if (best[j] == i)
         {
            best[j] = best[i];
         }
   }
}

// ----- graph.cpp --------------------------------------------------------------

using namespace std;
using namespace Gecko;

// Constructor.
void
Graph::init(uint nodes)
{
   node.push_back(Node(-1, 0, 1, Node::null));
   adj.push_back(Node::null);
   weight.push_back(0);
   bond.push_back(0);
   while (nodes--)
   {
      insert_node();
   }
}

// Insert node.
Node::Index
Graph::insert_node(Float length)
{
   Node::Index p = Node::Index(node.size());
   perm.push_back(p);
   node.push_back(Node(-1, length));
   return p;
}

// Return nodes adjacent to i.
std::vector<Node::Index>
Graph::node_neighbors(Node::Index i) const
{
   std::vector<Node::Index> neighbor;
   for (Arc::Index a = node_begin(i); a < node_end(i); a++)
   {
      neighbor.push_back(adj[a]);
   }
   return neighbor;
}

// Insert directed edge (i, j).
Arc::Index
Graph::insert_arc(Node::Index i, Node::Index j, Float w, Float b)
{
   if (!i || !j || i == j || !(last_node <= i && i <= nodes()))
   {
      return Arc::null;
   }
   last_node = i;
   for (Node::Index k = i - 1; node[k].arc == Arc::null; k--)
   {
      node[k].arc = Arc::Index(adj.size());
   }
   adj.push_back(j);
   weight.push_back(w);
   bond.push_back(b);
   node[i].arc = Arc::Index(adj.size());
   return Arc::Index(adj.size() - 1);
}

// Remove arc a.
bool
Graph::remove_arc(Arc::Index a)
{
   if (a == Arc::null)
   {
      return false;
   }
   Node::Index i = arc_source(a);
   adj.erase(adj.begin() + a);
   weight.erase(weight.begin() + a);
   bond.erase(bond.begin() + a);
   for (Node::Index k = i; k < node.size(); k++)
   {
      node[k].arc--;
   }
   return true;
}

// Remove directed edge (i, j).
bool
Graph::remove_arc(Node::Index i, Node::Index j)
{
   return remove_arc(arc_index(i, j));
}

// Remove edge {i, j}.
bool
Graph::remove_edge(Node::Index i, Node::Index j)
{
   bool success = remove_arc(i, j);
   if (success)
   {
      success = remove_arc(j, i);
   }
   return success;
}

// Index of arc (i, j) or null if not a valid arc.
Arc::Index
Graph::arc_index(Node::Index i, Node::Index j) const
{
   for (Arc::Index a = node_begin(i); a < node_end(i); a++)
      if (adj[a] == j)
      {
         return a;
      }
   return Arc::null;
}

// Return source node i in arc a = (i, j).
Node::Index
Graph::arc_source(Arc::Index a) const
{
   Node::Index j = adj[a];
   for (Arc::Index b = node_begin(j); b < node_end(j); b++)
   {
      Node::Index i = adj[b];
      if (node_begin(i) <= a && a < node_end(i))
      {
         return i;
      }
   }
   // should never get here
   throw std::runtime_error("internal data structure corrupted");
}

// Return reverse arc (j, i) of arc a = (i, j).
Arc::Index
Graph::reverse_arc(Arc::Index a) const
{
   Node::Index j = adj[a];
   for (Arc::Index b = node_begin(j); b < node_end(j); b++)
   {
      Node::Index i = adj[b];
      if (node_begin(i) <= a && a < node_end(i))
      {
         return b;
      }
   }
   return Arc::null;
}

// Return first directed arc if one exists or null otherwise.
Arc::Index
Graph::directed() const
{
   for (Node::Index i = 1; i < node.size(); i++)
      for (Arc::Index a = node_begin(i); a < node_end(i); a++)
      {
         Node::Index j = adj[a];
         if (!arc_index(j, i))
         {
            return a;
         }
      }
   return Arc::null;
}

// Add contribution of fine arc to coarse graph.
void
Graph::update(Node::Index i, Node::Index j, Float w, Float b)
{
   Arc::Index a = arc_index(i, j);
   if (a == Arc::null)
   {
      insert_arc(i, j, w, b);
   }
   else
   {
      weight[a] += w;
      bond[a] += b;
   }
}

// Transfer contribution of fine arc a to coarse node p.
void
Graph::transfer(Graph* g, const vector<Float>& part, Node::Index p,
                Arc::Index a, Float f) const
{
   Float w = f * weight[a];
   Float m = f * bond[a];
   Node::Index j = arc_target(a);
   Node::Index q = node[j].parent;
   if (q == Node::null)
   {
      for (Arc::Index b = node_begin(j); b < node_end(j); b++)
         if (part[b] > 0)
         {
            q = node[adj[b]].parent;
            if (q != p)
            {
               g->update(p, q, w * part[b], m * part[b]);
            }
         }
   }
   else
   {
      g->update(p, q, w, m);
   }
}

// Compute cost of a subset of arcs incident on node placed at pos.
WeightedSum
Graph::cost(const vector<Arc::Index>& subset, Float pos) const
{
   WeightedSum c;
   for (Arc::ConstPtr ap = subset.begin(); ap != subset.end(); ap++)
   {
      Arc::Index a = *ap;
      Node::Index j = arc_target(a);
      Float l = fabs(node[j].pos - pos);
      Float w = weight[a];
      functional->accumulate(c, WeightedValue(l, w));
   }
   return c;
}

// Compute cost of graph layout.
Float
Graph::cost() const
{
   if (edges())
   {
      WeightedSum c;
      Node::Index i = 1;
      for (Arc::Index a = 1; a < adj.size(); a++)
      {
         while (node_end(i) <= a)
         {
            i++;
         }
         Node::Index j = arc_target(a);
         Float l = length(i, j);
         Float w = weight[a];
         functional->accumulate(c, WeightedValue(l, w));
      }
      return functional->mean(c);
   }
   else
   {
      return Float(0);
   }
}

// Swap the two nodes in positions k and l, k <= l.
void
Graph::swap(uint k, uint l)
{
   Node::Index i = perm[k];
   perm[k] = perm[l];
   perm[l] = i;
   Float p = node[i].pos - node[i].hlen;
   do
   {
      i = perm[k];
      p += node[i].hlen;
      node[i].pos = p;
      p += node[i].hlen;
   }
   while (k++ != l);
}

// Optimize continuous position of a single node.
Float
Graph::optimal(Node::Index i) const
{
   vector<WeightedValue> v;
   for (Arc::Index a = node_begin(i); a < node_end(i); a++)
   {
      Node::Index j = adj[a];
      if (placed(j))
      {
         v.push_back(WeightedValue(node[j].pos, weight[a]));
      }
   }
   return v.empty() ? -1 : functional->optimum(v);
}

// Compute coarse graph with roughly half the number of nodes.
Graph*
Graph::coarsen()
{
   progress->beginphase(this, string("coarse"));
   Graph* g = new Graph(0, level - 1);
   g->functional = functional;
   g->progress = progress;

   // Compute importance of nodes in fine graph.
   DynamicHeap<Node::Index, Float> heap;
   for (Node::Index i = 1; i < node.size(); i++)
   {
      node[i].parent = Node::null;
      Float w = 0;
      for (Arc::Index a = node_begin(i); a < node_end(i); a++)
      {
         w += bond[a];
      }
      heap.insert(i, w);
   }

   // Select set of important nodes from fine graph that will remain in
   // coarse graph.
   vector<Node::Index> child(1, Node::null);
   while (!heap.empty())
   {
      Node::Index i;
      Float w = 0;
      heap.extract(i, w);
      if (w < 0)
      {
         break;
      }
      child.push_back(i);
      node[i].parent = g->insert_node(2 * node[i].hlen);

      // Reduce importance of neighbors.
      for (Arc::Index a = node_begin(i); a < node_end(i); a++)
      {
         Node::Index j = adj[a];
         if (heap.find(j, w))
         {
            heap.update(j, w - 2 * bond[a]);
         }
      }
   }

   // Assign parts of remaining nodes to aggregates.
   vector<Float> part = bond;
   for (Node::Index i = 1; i < node.size(); i++)
      if (!persistent(i))
      {
         // Find all connections to coarse nodes.
         Float w = 0;
         Float max = 0;
         for (Arc::Index a = node_begin(i); a < node_end(i); a++)
         {
            Node::Index j = adj[a];
            if (persistent(j))
            {
               w += part[a];
               if (max < part[a])
               {
                  max = part[a];
               }
            }
            else
            {
               part[a] = -1;
            }
         }
         max /= GECKO_PART_FRAC;

         // Weed out insignificant connections.
         for (Arc::Index a = node_begin(i); a < node_end(i); a++)
            if (0 < part[a] && part[a] < max)
            {
               w -= part[a];
               part[a] = -1;
            }

         // Compute node fractions (interpolation matrix) and assign
         // partial nodes to aggregates.
         for (Arc::Index a = node_begin(i); a < node_end(i); a++)
            if (part[a] > 0)
            {
               part[a] /= w;
               Node::Index p = node[adj[a]].parent;
               g->node[p].hlen += part[a] * node[i].hlen;
            }
      }

   // Transfer arcs to coarse graph.
   for (Node::Index p = 1; p < g->node.size(); p++)
   {
      Node::Index i = child[p];
      for (Arc::Index a = node_begin(i); a < node_end(i); a++)
      {
         transfer(g, part, p, a);
         Node::Index j = adj[a];
         if (!persistent(j))
         {
            Arc::Index b = arc_index(j, i);
            if (part[b] > 0)
               for (Arc::Index c = node_begin(j); c < node_end(j); c++)
               {
                  Node::Index k = adj[c];
                  if (k != i)
                  {
                     transfer(g, part, p, c, part[b]);
                  }
               }
         }
      }
   }

#if DEBUG
   if (g->directed())
   {
      throw runtime_error("directed edge found");
   }
#endif

   // Free memory.
   vector<Float> t = bond;
   bond.swap(t);

   progress->endphase(this, false);

   return g;
}

// Order nodes according to coarsened graph layout.
void
Graph::refine(const Graph* graph)
{
   progress->beginphase(this, string("refine"));

   // Place persistent nodes.
   DynamicHeap<Node::Index, Float> heap;
   for (Node::Index i = 1; i < node.size(); i++)
      if (persistent(i))
      {
         Node::Index p = node[i].parent;
         node[i].pos = graph->node[p].pos;
      }
      else
      {
         node[i].pos = -1;
         Float w = 0;
         for (Arc::Index a = node_begin(i); a < node_end(i); a++)
         {
            Node::Index j = adj[a];
            if (persistent(j))
            {
               w += weight[a];
            }
         }
         heap.insert(i, w);
      }

   // Place remaining nodes in order of decreasing connectivity with
   // already placed nodes.
   while (!heap.empty())
   {
      Node::Index i = 0;
      heap.extract(i);
      node[i].pos = optimal(i);
      for (Arc::Index a = node_begin(i); a < node_end(i); a++)
      {
         Node::Index j = adj[a];
         Float w;
         if (heap.find(j, w))
         {
            heap.update(j, w + weight[a]);
         }
      }
   }

   place(true);
   progress->endphase(this, true);
}

// Perform m sweeps of compatible or Gauss-Seidel relaxation.
void
Graph::relax(bool compatible, uint m)
{
   progress->beginphase(this, compatible ? string("crelax") : string("frelax"));
   while (m--)
      for (uint k = 0; k < perm.size() && !progress->quit(); k++)
      {
         Node::Index i = perm[k];
         if (!compatible || !persistent(i))
         {
            node[i].pos = optimal(i);
         }
      }
   place(true);
   progress->endphase(this, true);
}

// Optimize successive n-node subgraphs.
void
Graph::optimize(uint n)
{
   if (n > perm.size())
   {
      n = uint(perm.size());
   }
   ostringstream count;
   count << setw(2) << n;
   progress->beginphase(this, string("perm") + count.str());
   Subgraph* subgraph = new Subgraph(this, n);
   for (uint k = 0; k <= perm.size() - n && !progress->quit(); k++)
   {
      subgraph->optimize(k);
   }
   delete subgraph;
   progress->endphase(this, true);
}

// Place all nodes according to their positions.
void
Graph::place(bool sort)
{
   place(sort, 0, uint(perm.size()));
}

// Place nodes {k, ..., k + n - 1} according to their positions.
void
Graph::place(bool sort, uint k, uint n)
{
   // Place nodes.
   if (sort)
   {
      stable_sort(perm.begin() + k, perm.begin() + k + n,
                  Node::Comparator(node.begin()));
   }

   // Assign node positions according to permutation.
   for (Float p = k ? node[perm[k - 1]].pos + node[perm[k - 1]].hlen : 0; n--;
        k++)
   {
      Node::Index i = perm[k];
      p += node[i].hlen;
      node[i].pos = p;
      p += node[i].hlen;
   }
}

// Perform one V-cycle.
void
Graph::vcycle(uint n, uint work)
{
   if (n < nodes() && nodes() < edges() && level && !progress->quit())
   {
      Graph* graph = coarsen();
      graph->vcycle(n, work + edges());
      refine(graph);
      delete graph;
   }
   else
   {
      place();
   }
   if (edges())
   {
      relax(true, GECKO_CR_SWEEPS);
      relax(false, GECKO_GS_SWEEPS);
      for (uint w = edges(); w * (n + 1) < work; w *= ++n);
      n = std::min(n, uint(GECKO_WINDOW_MAX));
      if (n)
      {
         optimize(n);
      }
   }
}

// Custom random-number generator for reproducibility.
// LCG from doi:10.1090/S0025-5718-99-00996-5.
uint
Graph::random(uint seed)
{
   static uint state = 1;
   state = (seed ? seed : 0x1ed0675 * state + 0xa14f);
   return state;
}

// Generate a random permutation of the nodes.
void
Graph::shuffle(uint seed)
{
   random(seed);
   for (uint k = 0; k < perm.size(); k++)
   {
      uint r = random() >> 8;
      uint l = k + r % (uint(perm.size()) - k);
      std::swap(perm[k], perm[l]);
   }
   place();
}

// Recompute bonds for k'th V-cycle.
void
Graph::reweight(uint k)
{
   bond.resize(weight.size());
   for (Arc::Index a = 1; a < adj.size(); a++)
   {
      bond[a] = functional->bond(weight[a], length(a), k);
   }
}

// Linearly order graph.
void
Graph::order(Functional* functional, uint iterations, uint window, uint period,
             uint seed, Progress* progress)
{
   // Initialize graph.
   this->functional = functional;
   progress = this->progress = progress ? progress : new Progress;
   for (level = 0; (1u << level) < nodes(); level++);
   place();
   Float mincost = cost();
   vector<Node::Index> minperm = perm;
   if (seed)
   {
      shuffle(seed);
   }

   progress->beginorder(this, mincost);
   if (edges())
   {
      // Perform specified number of V-cycles.
      for (uint k = 1; k <= iterations && !progress->quit(); k++)
      {
         progress->beginiter(this, k, iterations, window);
         reweight(k);
         vcycle(window);
         Float c = cost();
         if (c < mincost)
         {
            mincost = c;
            minperm = perm;
         }
         progress->enditer(this, mincost, c);
         if (period && !(k % period))
         {
            window++;
         }
      }
      perm = minperm;
      place();
   }
   progress->endorder(this, mincost);

   if (!progress)
   {
      delete this->progress;
      this->progress = 0;
   }
}
